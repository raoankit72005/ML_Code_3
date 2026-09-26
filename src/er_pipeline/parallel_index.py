"""Parallel text/token preparation with one ordered SQLite writer."""
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import numpy as np
from .common import connect,find_source,rows,dump_json
from .text_features import Tfidf,block_keys,lexical_tokens,name_text,address_text,encoded,terms,bucket
from .indexing import FIELDS
from .parallel import bounded_map


def _prepare(task):
    batch,source,bits,fit=task
    output=[];counts={k:np.zeros(1<<bits,dtype=np.uint64) for k in ('name','address')} if fit else None
    for r in batch:
        if set(FIELDS)-r.keys():raise ValueError('Missing cleaned columns')
        record={k:r[k] for k in FIELDS};record['source']=source
        if not record['entity_id'].startswith(f'S{source}-'):raise ValueError('Wrong entity prefix')
        for flag in ('name_missing','address_missing'):record[flag]=int(record[flag])
        payload=json.dumps(record,ensure_ascii=False)
        search=None
        if source!=1:
            keys=block_keys(record);name=name_text(record);address=address_text(record)
            search=(encoded(record['country_key'] or 'UNKNOWN'),' '.join(k for group in keys.values() for k in group),lexical_tokens(name),lexical_tokens(address))
            if fit:
                for field,value in (('name',name),('address',address)):
                    indices=list({bucket(t,bits) for t in terms(value)})
                    counts[field][indices]+=1
        output.append((record['entity_id'],payload,search))
    return output,counts


def build(cleaned,work,split,config):
    path=work/'index.sqlite.partial';path.unlink(missing_ok=True)
    conn=connect(path,config['sqlite_cache_mb'])
    conn.executescript('''CREATE TABLE records(rid INTEGER PRIMARY KEY,entity_id TEXT UNIQUE NOT NULL,source INTEGER NOT NULL,payload TEXT NOT NULL);
    CREATE INDEX source_index ON records(source,rid);
    CREATE VIRTUAL TABLE search USING fts5(country,blocks,name_terms,address_terms,content='',detail='column',tokenize='ascii');''')
    tfidf=Tfidf(config['hash_bits']) if split=='train' else None
    counts={};rid=0;workers=config['workers']
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
            for source in (1,2,3):
                stream=iter(rows(find_source(cleaned,split,source)));count=0
                def tasks():
                    while batch:=list(islice(stream,2000)):
                        yield batch,source,config['hash_bits'],tfidf is not None and source!=1
                for prepared,df in bounded_map(pool,_prepare,tasks(),workers*2):
                    records=[];search=[]
                    for eid,payload,index in prepared:
                        rid+=1;records.append((rid,eid,source,payload))
                        if index is not None:search.append((rid,*index))
                    conn.executemany('INSERT INTO records VALUES(?,?,?,?)',records)
                    conn.executemany('INSERT INTO search(rowid,country,blocks,name_terms,address_terms) VALUES(?,?,?,?,?)',search)
                    count+=len(records)
                    if df is not None:
                        tfidf.n+=len(records)
                        for field in ('name','address'):tfidf.df[field]+=df[field]
                    conn.commit()
                    from .resources import check_disk
                    check_disk(work)
                    if count%10000==0:print(f'Index {split} S{source}: {count:,}',flush=True)
                counts[f'S{source}']=count
        if not counts['S1'] or not counts['S2']+counts['S3']:raise ValueError('Empty sources')
        conn.execute("INSERT INTO search(search) VALUES('optimize')");conn.commit()
        conn.execute("CREATE INDEX country_source ON records(source,json_extract(payload,'$.country_key'),rid)");conn.commit()
        if tfidf:tfidf.save(work/'tfidf.npz')
    finally:conn.close()
    path.replace(work/'index.sqlite')
    dump_json(work/'index_report.json',dict(counts=counts,index_bytes=(work/'index.sqlite').stat().st_size,
        tfidf_fit='Train reference text only; no held-out S1 or labels'))
