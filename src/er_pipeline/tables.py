"""Export final model tables and challenge-format final candidate lists."""
import csv
import json
import os
import shutil
from collections import Counter
from contextlib import ExitStack

import pyarrow.parquet as pq

from .common import ParquetSink, parquet_rows, rows, dump_json


def export(work, split, config):
    schema = pq.ParquetFile(work / 'pair_features.parquet').schema_arrow
    counts = Counter()
    if split == 'test':
        destination = work / 'test_features.parquet'
        if destination.exists():
            destination.unlink()
        try:
            os.link(work / 'pair_features.parquet', destination)
        except OSError:
            from .resources import check_disk
            check_disk(work, (work/'pair_features.parquet').stat().st_size)
            shutil.copyfile(work/'pair_features.parquet', destination)
        counts['test_pairs'] = pq.ParquetFile(destination).metadata.num_rows
    else:
        import pyarrow.compute as pc
        paths={g:work/f'{g}_features.parquet.partial' for g in ('train','validation')}
        with ExitStack() as stack:
            writers={g:stack.enter_context(pq.ParquetWriter(p,schema,compression='zstd')) for g,p in paths.items()}
            for batch in pq.ParquetFile(work/'pair_features.parquet').iter_batches(batch_size=65536):
                for group in writers:
                    selected=batch.filter(pc.equal(batch.column(batch.schema.get_field_index('dataset_split')),group))
                    writers[group].write_batch(selected)
                    counts[group+'_pairs']+=selected.num_rows
                    counts[group+'_positive_pairs']+=pc.sum(selected.column(selected.schema.get_field_index('label'))).as_py() or 0
        for g,p in paths.items():p.replace(work/f'{g}_features.parquet')
    # Keep zero-candidate queries: the pair table alone cannot represent them.
    pairs = iter(parquet_rows(work / 'candidate_pairs.parquet'))
    pair = next(pairs, None)
    candidate_partial = work / 'candidate_pairs.tsv.partial'
    with candidate_partial.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['source1_entity_id', 'candidate_entity_ids'])
        for query in rows(work / 'queries.tsv.gz'):
            sid, ids = query['source1_entity_id'], []
            while pair is not None and pair['source1_entity_id'] == sid:
                ids.append(pair['candidate_entity_id'])
                pair = next(pairs, None)
            if len(ids) != int(query['n_candidates']) or len(ids) != len(set(ids)):
                raise ValueError('Candidate/manifest inconsistency for ' + sid)
            writer.writerow([sid, ','.join(ids)])
            counts['queries'] += 1
        if pair is not None:
            raise ValueError('Unconsumed candidate rows: query order changed')
        f.flush()
        os.fsync(f.fileno())
    if sum(1 for _ in rows(candidate_partial)) != counts['queries']:
        raise ValueError('Candidate TSV write did not preserve every query')
    candidate_partial.replace(work / 'candidate_pairs.tsv')
    if split == 'train':
        for query in rows(work / 'query_labels.tsv.gz'):
            counts[query['dataset_split'] + '_queries'] += 1
        for group in ('train', 'validation'):
            counts[group + '_negative_pairs'] = counts[group + '_pairs'] - counts[group + '_positive_pairs']
    dump_json(work / 'tables_report.json', dict(counts))
    print('Final feature tables and exact candidate lists exported.', flush=True)
