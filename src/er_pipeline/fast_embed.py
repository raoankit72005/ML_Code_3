"""One process per GPU, disjoint writes, durable shard cursors and prefetch."""
import contextlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from pathlib import Path
import numpy as np
from .common import connect,dump_json
from .resources import check_disk


def ranges(count,workers):
    if count<1 or workers<1: raise ValueError('Need records and workers')
    workers=min(count,workers)
    return [(count*i//workers+1,count*(i+1)//workers+1) for i in range(workers)]


def _read_batch(conn,start,end,limit):
    return conn.execute('SELECT rid,payload FROM records WHERE rid>=? AND rid<? ORDER BY rid LIMIT ?',
                        (start,end,limit)).fetchall()


def _encode_shard(task):
    import torch
    from . import encoder
    folder,root,cfg,rank,lo,hi,dim=task
    folder=Path(folder);root=Path(root);directory=folder/'embeddings'
    torch.set_num_threads(cfg['cpu_threads'])
    if cfg['device']=='cuda':torch.cuda.set_device(rank)
    device=f'cuda:{rank}' if cfg['device']=='cuda' else 'cpu'
    local=dict(cfg,device=device)
    selected=json.loads((root/'encoder/complete.json').read_text())
    model=encoder.load(local,root/'encoder/best.pt' if selected['mode']=='lora' else None,selected['mode']=='lora')
    model.eval()
    marker=directory/f'shard_{rank}.json'
    cursor=json.loads(marker.read_text())['next_rid'] if marker.exists() else lo
    if not lo<=cursor<=hi:raise ValueError('Invalid embedding cursor')
    conn=connect(folder/'index.sqlite',readonly=True)
    # Only this thread uses its own SQLite connection for prefetch.
    def read(start):
        c=connect(folder/'index.sqlite',readonly=True)
        try:return _read_batch(c,start,hi,cfg.get('encode_chunk_rows',4096))
        finally:c.close()
    conn.close()
    batch_size=cfg['encode_batch_size'];started=time.monotonic();initial=cursor
    with ThreadPoolExecutor(max_workers=1) as pool,(directory/'vectors.f16').open('r+b') as out:
        future=pool.submit(read,cursor)
        while cursor<hi:
            records=future.result()
            if not records or records[0][0]!=cursor or records[-1][0]!=cursor+len(records)-1:raise ValueError('Noncontiguous embedding rows')
            next_rid=cursor+len(records)
            future=pool.submit(read,next_rid) if next_rid<hi else None
            texts=[encoder.text(json.loads(r[1])) for r in records]
            while True:
                try:
                    mixed=cfg['device']=='cuda' and torch.cuda.is_bf16_supported()
                    with torch.inference_mode(),(torch.autocast('cuda',dtype=torch.bfloat16) if mixed else contextlib.nullcontext()):
                        vec=model.encode(texts,batch_size=batch_size,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
                    break
                except torch.cuda.OutOfMemoryError:
                    if batch_size<=1:raise
                    batch_size=max(1,batch_size//2);torch.cuda.empty_cache()
            if vec.shape!=(len(records),dim) or not np.isfinite(vec).all():raise ValueError('Invalid embeddings')
            check_disk(directory,vec.nbytes,1)
            out.seek((cursor-1)*dim*2);out.write(vec.astype(np.float16).tobytes());out.flush();os.fsync(out.fileno())
            cursor=next_rid
            dump_json(marker,dict(next_rid=cursor,start=lo,end=hi))
            print(f'Embedding GPU {rank}: {cursor-lo:,}/{hi-lo:,}; {(cursor-initial)/max(.001,time.monotonic()-started):.1f} records/s',flush=True)
    return rank


def embed(folder,root,config):
    import torch
    from . import encoder
    from .modeling import file_identity
    folder=Path(folder);root=Path(root)
    requested=config.get('embedding_gpus',1) if config['device']=='cuda' else 1
    if config['device']=='cuda' and torch.cuda.device_count()<requested:raise RuntimeError(f'Need {requested} visible GPUs for this config')
    manifest=json.loads((root/'encoder/complete.json').read_text())
    conn=connect(folder/'index.sqlite',readonly=True)
    count,maximum=conn.execute('SELECT COUNT(*),MAX(rid) FROM records').fetchone();conn.close()
    if count!=maximum:raise ValueError('Records must have contiguous IDs')
    directory=folder/'embeddings';directory.mkdir(exist_ok=True)
    path=directory/'vectors.f16';marker=directory/'manifest.json'
    signature=dict(index=file_identity(folder/'index.sqlite'),encoder=manifest,config=config)
    old=json.loads(marker.read_text()) if marker.exists() else None
    if old and old['signature']!=signature:raise ValueError('Embedding signature changed: use new work directory')
    if old:
        dim=old['dimension']
        if not path.exists() or path.stat().st_size!=count*dim*2:raise ValueError('Embedding file missing/truncated')
        if old.get('complete'):return
    else:
        # Inspect dimension on CPU; no GPU context is inherited by spawn workers.
        model=encoder.load(dict(config,device='cpu'))
        dim=model.get_sentence_embedding_dimension();del model
        check_disk(directory,count*dim*2,5)
        with path.open('wb') as f:f.truncate(count*dim*2)
        for stale in directory.glob('shard_*.json'):stale.unlink()
        dump_json(marker,dict(signature=signature,rows=0,dimension=dim,complete=False))
    tasks=[(str(folder),str(root),config,i,lo,hi,dim) for i,(lo,hi) in enumerate(ranges(count,requested))]
    with ProcessPoolExecutor(max_workers=len(tasks),mp_context=mp.get_context('spawn')) as pool:list(pool.map(_encode_shard,tasks))
    dump_json(marker,dict(signature=signature,rows=count,dimension=dim,complete=True))
