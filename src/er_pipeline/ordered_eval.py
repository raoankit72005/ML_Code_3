"""Full macro F0.5 without inserting every validation pair into another database."""
import json
import math
from collections import defaultdict
from itertools import zip_longest
import numpy as np
from .common import connect, rows, parquet_rows, dump_json


def evaluate(work, probabilities, thresholds):
    thresholds=sorted(set(thresholds))
    if not thresholds or any(not math.isfinite(t) or not 0<=t<=1 for t in thresholds):raise ValueError('Invalid thresholds')
    levels=np.asarray(thresholds)
    sums=defaultdict(lambda:np.zeros(len(levels),dtype=np.float64));counts=defaultdict(int)
    expected=parquet_rows(work/'validation_features.parquet',['source1_entity_id','candidate_entity_id'])
    actual=parquet_rows(probabilities,['source1_entity_id','candidate_entity_id','match_probability'])
    stream=iter(zip_longest(expected,actual));item=next(stream,None)
    conn=connect(work/'truth.sqlite',readonly=True)
    try:
        for query in rows(work/'queries.tsv.gz'):
            sid=query['source1_entity_id']
            label=conn.execute('SELECT ids,dataset_split FROM truth JOIN queries USING(sid) WHERE sid=?',(sid,)).fetchone()
            if label is None:raise ValueError('Missing truth query')
            if label[1]!='validation':continue
            truth=set(json.loads(label[0]));scores=[];hits=[];seen=set()
            while item is not None:
                pair,row=item
                if pair is None or row is None:raise ValueError('Probability row count mismatch')
                if any(pair[k]!=row[k] for k in ('source1_entity_id','candidate_entity_id')):raise ValueError('Probability pair/order mismatch')
                if pair['source1_entity_id']!=sid:break
                eid=row['candidate_entity_id'];p=float(row['match_probability'])
                if eid in seen or not math.isfinite(p) or not 0<=p<=1:raise ValueError('Duplicate pair or invalid probability')
                seen.add(eid);scores.append(p);hits.append(eid in truth);item=next(stream,None)
            if len(scores)!=int(query['n_candidates']):raise ValueError('Validation candidate coverage mismatch')
            order=np.argsort(-np.asarray(scores))
            sorted_scores=np.asarray(scores)[order]
            cumulative=np.r_[0,np.cumsum(np.asarray(hits,dtype=np.int64)[order])]
            n=np.searchsorted(-sorted_scores,-levels,side='right')
            tp=cumulative[n]
            denominator=4*n+len(truth)
            values=np.divide(5*tp,denominator,out=np.ones(len(levels)),where=denominator!=0)
            for key in ('all','country:'+query['country']):sums[key]+=values;counts[key]+=1
        if item is not None:raise ValueError('Unconsumed validation probabilities')
    finally:conn.close()
    if not counts['all']:raise ValueError('No validation queries')
    results=[dict(threshold=t,macro_f05=float(sums['all'][i]/counts['all']),by_country={k:float(sums[k][i]/n) for k,n in counts.items() if k!='all'}) for i,t in enumerate(thresholds)]
    best=max(results,key=lambda r:(r['macro_f05'],r['threshold']))
    result=dict(best=best,results=results,query_counts=dict(counts),note='Full validation, including singletons and missed positives; threshold selection is not an unbiased test estimate.')
    dump_json(work/'validation_f05.json',result);print(json.dumps(best,indent=2));return result
