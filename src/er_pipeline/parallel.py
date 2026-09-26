"""Bounded, ordered CPU workers. Each writer owns a separate output file."""
import csv
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from .common import connect, open_text, dump_json


def bounded_map(pool, function, items, window):
    """Do not enqueue millions of futures (Executor.map on Python <3.14 does)."""
    iterator = iter(items)
    from collections import deque
    queue = deque()
    for _ in range(window):
        item = next(iterator, None)
        if item is None: break
        queue.append(pool.submit(function, item))
    while queue:
        yield queue.popleft().result()
        item = next(iterator, None)
        if item is not None: queue.append(pool.submit(function, item))


def merge_parquet(paths, destination, schema=None):
    paths = list(paths)
    schema = schema or pq.ParquetFile(paths[0]).schema_arrow
    partial = Path(str(destination) + '.partial')
    with pq.ParquetWriter(partial, schema, compression='zstd') as writer:
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
                writer.write_batch(batch)
    partial.replace(destination)


def _block_task(task):
    from .blocking import generate
    folder, tfidf, config, lo, hi, output = task
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    generate(Path(folder), Path(tfidf), config, start_rid=lo, end_rid=hi, output_dir=output)
    return str(output)


def block(work, tfidf, config):
    from .blocking import generate
    workers = config.get('workers', 1)
    if workers == 1: return generate(work, tfidf, config)
    conn = connect(work/'index.sqlite', readonly=True)
    lo, hi = conn.execute('SELECT MIN(rid),MAX(rid) FROM records WHERE source=1').fetchone(); conn.close()
    # Contiguous ranges preserve source order for labeling/export/submission.
    width = max(1, (hi-lo+1 + workers*4-1)//(workers*4))
    tasks = [(str(work), str(tfidf), config, start, min(hi+1,start+width),
              str(work/'block_shards'/f'{start:012d}')) for start in range(lo,hi+1,width)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        folders = [Path(p) for p in bounded_map(pool, _block_task, tasks, workers*2)]
    merge_parquet([p/'candidate_pairs.parquet' for p in folders], work/'candidate_pairs.parquet')
    partial = work/'queries.partial.tsv.gz'
    from collections import Counter
    totals = Counter()
    with open_text(partial, 'wt') as out:
        writer = None
        for folder in folders:
            with open_text(folder/'queries.tsv.gz') as inp:
                reader = csv.DictReader(inp, delimiter='\t')
                if writer is None:
                    writer=csv.DictWriter(out,fieldnames=reader.fieldnames,delimiter='\t');writer.writeheader()
                writer.writerows(reader)
            stats=json.loads((folder/'blocking_report.json').read_text())
            totals.update({k:v for k,v in stats.items() if k!='average_candidates'})
    partial.replace(work/'queries.tsv.gz')
    totals['average_candidates']=totals['pairs']/max(1,totals['queries'])
    dump_json(work/'blocking_report.json',dict(totals))
    import shutil
    shutil.rmtree(work/'block_shards')


def _feature_task(task):
    from .features import pair_features
    from .neural import vectors, unit
    import numpy as np
    work, source, group, output, schema, cache_mb = task
    work=Path(work)
    conn=connect(work/'index.sqlite',cache_mb,readonly=True)
    batch=pq.ParquetFile(source).read_row_group(group).to_pylist()
    ids=sorted({r[k] for r in batch for k in ('source1_entity_id','candidate_entity_id')})
    records={}
    for start in range(0,len(ids),800):
        values=ids[start:start+800]
        for eid,rid,payload in conn.execute('SELECT entity_id,rid,payload FROM records WHERE entity_id IN ('+','.join('?'*len(values))+')',values):
            records[eid]=(rid,json.loads(payload))
    if len(records)!=len(ids): raise ValueError('Candidate references missing from index')
    vec=vectors(work) if (work/'embeddings/manifest.json').exists() else None
    # Bounded vectorized gather/dot product; no per-pair SQL.
    if vec is not None and batch:
        for start in range(0,len(batch),2048):
            chunk=batch[start:start+2048]
            a=np.array([records[r['source1_entity_id']][0]-1 for r in chunk])
            b=np.array([records[r['candidate_entity_id']][0]-1 for r in chunk])
            scores=np.einsum('ij,ij->i',unit(vec[a]),unit(vec[b]))
            for row,score in zip(chunk,scores): row['neural_cosine']=float(score)
    for row in batch:
        row.update(pair_features(records[row['source1_entity_id']][1],records[row['candidate_entity_id']][1],
                                 row['name_retrieval_cosine'],row['address_retrieval_cosine']))
    pq.write_table(pa.Table.from_pylist(batch,schema=schema),output,compression='zstd')
    conn.close()
    if vec is not None: vec._mmap.close()
    return output,len(batch)


def features(work, tfidf, split, config):
    from .features import FEATURES
    from .common import LABEL_SCHEMA,PAIR_SCHEMA
    schema=pa.schema(list(LABEL_SCHEMA if split=='train' else PAIR_SCHEMA)+[pa.field(k,pa.float32()) for k in FEATURES])
    source=work/('labeled_pairs.parquet' if split=='train' else 'candidate_pairs.parquet')
    folder=work/'feature_shards';folder.mkdir(exist_ok=True)
    workers=config.get('workers',1)
    tasks=((str(work),str(source),g,str(folder/f'{g:08d}.parquet'),schema,config['sqlite_cache_mb'])
           for g in range(pq.ParquetFile(source).num_row_groups))
    outputs=[];count=0
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for path,n in bounded_map(pool,_feature_task,tasks,workers*2):
            outputs.append(path);count+=n
            if len(outputs)%20==0:print(f'Features: {count:,} pairs',flush=True)
    merge_parquet(outputs,work/'pair_features.parquet',schema)
    names=FEATURES+['candidate_source','candidate_rank','retrieval_score','name_retrieval_cosine','address_retrieval_cosine']+['block_'+c for c in 'ABCDEFG']+['neural_cosine','neural_rank']
    dump_json(work/'feature_columns.json',dict(features=names,target='label',pair_count=count,
        never_features=['source1_entity_id','candidate_entity_id','dataset_split'],missing_value='NaN'))
    import shutil
    shutil.rmtree(folder)
