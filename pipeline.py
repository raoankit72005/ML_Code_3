"""Resumable stage runner for persistent CPU/GPU worker Studios."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
PHASES={
    'prepare':['doctor','clean','train-index','test-index','clusters'],
    'train':['encoder'],
    'embed':['train-embed','test-embed'],
    'finish':['train-retrieve','train-block','train-label','train-features','train-export','matcher',
              'test-retrieve','test-block','test-features','test-export','predict','submit','audit'],
}
STAGES=[s for steps in PHASES.values() for s in steps]


def stage_input(value,work):
    if not value.startswith('s3://'):return Path(value).expanduser().resolve()
    import boto3
    from urllib.parse import urlparse
    uri=urlparse(value);key=uri.path.lstrip('/')
    if not key or not key.endswith('.zip'):raise ValueError('S3 input must be one ZIP object. For a prefix, download it to the worker first and use its local directory.')
    target=work/'input/dataset.zip';target.parent.mkdir(exist_ok=True)
    marker=target.with_suffix('.json')
    client=boto3.client('s3');head=client.head_object(Bucket=uri.netloc,Key=key)
    identity=dict(uri=value,etag=head['ETag'],size=head['ContentLength'])
    if marker.exists():
        if json.loads(marker.read_text())!=identity:raise ValueError('S3 object changed; use new work directory')
        if target.exists() and target.stat().st_size==identity['size']:return target
    partial=target.with_suffix('.partial')
    client.download_file(uri.netloc,key,str(partial));partial.replace(target)
    from er_pipeline.common import dump_json
    dump_json(marker,identity);return target


def execute(stage,args,source,cfg,prep,training):
    from er_pipeline import indexing,clusters,encoder,neural,labeling,tables,modeling,evaluation,coverage
    from er_pipeline import fast_clean,fast_embed,parallel
    from er_pipeline.common import find_truth
    work=args.work;cleaned=work/'cleaned'
    if stage=='doctor':
        from er_pipeline.resources import preflight,check_disk
        preflight(training);check_disk(work,reserve_gb=cfg.get('disk_reserve_gb',10))
        for split in ('train','test'):
            for i in (1,2,3):
                with coverage.raw_source(source,f'{split}_source{i}.tsv',args.sample) as rows:
                    r=next(iter(rows),None)
                    if r is None or not {'entity_id','business_name','business_address','country'}<=r.keys():raise ValueError('Invalid input schema')
        # Download pinned weights and test adapter targets before renting GPUs.
        model=encoder.load(dict(cfg,device='cpu'),adapters=True)
        model.encode(['name: setup | address: example | country: FR'],normalize_embeddings=True)
    elif stage=='clean':fast_clean.clean(source,cleaned,prep['workers'])
    elif stage=='clusters':clusters.build(work/'train/index.sqlite',find_truth(cleaned),work/'clusters.sqlite',prep['validation_fraction'],cfg['seed'])
    elif stage=='encoder':encoder.train(work,cfg)
    elif stage=='matcher':modeling.train(work,training)
    elif stage=='predict':modeling.predict(work,'test')
    elif stage=='submit':
        _,meta=modeling.load_model(work)
        evaluation.submit(work/'test',work/'test/test_probabilities.parquet',meta['decision_threshold'])
    elif stage=='audit':
        coverage.validate(work,source,allow_sample=args.sample)
        testdir=source
        if source.is_file():
            testdir=work/'official_test';testdir.mkdir(exist_ok=True)
            with zipfile.ZipFile(source) as z:
                for i in (1,2,3):
                    name=f'test_source{i}.tsv';matches=[n for n in z.namelist() if Path(n).name==name]
                    if len(matches)!=1:raise ValueError('Ambiguous test file')
                    with z.open(matches[0]) as inp,(testdir/name).open('wb') as out:shutil.copyfileobj(inp,out,1024**2)
        else:
            hits=list(source.rglob('test_source1.tsv'))
            if len(hits)!=1:raise ValueError('Ambiguous test directory')
            testdir=hits[0].parent
        subprocess.run([sys.executable,str(ROOT/'utils/validate_submission.py'),'--matching',str(work/'test/matching_results.tsv'),
            '--candidate',str(work/'test/candidate_pairs.tsv'),'--test-dir',str(testdir),'--check-ids'],check=True)
        output=work/('sample_output' if args.sample else 'output');output.mkdir(exist_ok=True)
        for name in ('matching_results.tsv','candidate_pairs.tsv'):shutil.copyfile(work/'test'/name,output/name)
        if args.sample:(output/'SAMPLE_ONLY.txt').write_text('TEST RUN ONLY. Do not submit sample predictions.\n')
        print(f'Validated output: {output}',flush=True)
    else:
        split,action=stage.split('-');folder=work/split;folder.mkdir(exist_ok=True)
        if action=='index':
            from er_pipeline.parallel_index import build
            build(cleaned,folder,split,prep)
        elif action=='embed':fast_embed.embed(folder,work,cfg)
        elif action=='retrieve':neural.retrieve(folder,work,cfg)
        elif action=='block':parallel.block(folder,work/'train/tfidf.npz',prep)
        elif action=='label':labeling.label(folder,find_truth(cleaned),prep)
        elif action=='features':parallel.features(folder,work/'train/tfidf.npz',split,prep)
        elif action=='export':tables.export(folder,split,prep)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--work',type=Path,default=Path('work_lightning'))
    p.add_argument('--config',type=Path,default=ROOT/'configs/lightning_model.json')
    p.add_argument('--phase',choices=list(PHASES)+['all'],default='all')
    p.add_argument('--stage',choices=STAGES,help=argparse.SUPPRESS)
    p.add_argument('--sample',action='store_true',help='Explicit sample test; outputs marked NOT FOR SUBMISSION')
    args=p.parse_args();args.work=args.work.expanduser().resolve();args.work.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    from er_pipeline.common import DEFAULTS,dump_json
    from er_pipeline.modeling import load_config,file_identity
    settings=json.loads(args.config.read_text());cfg=settings['encoder'];prep=dict(DEFAULTS,**settings['preparation'])
    training=load_config(overrides=settings['matcher'])
    if cfg['encoder_mode']!='lora':raise ValueError('This pipeline requires encoder fine-tuning')
    for k in ('batch_size','mini_batch_size','encode_batch_size','epochs','cpu_threads','embedding_gpus','encode_chunk_rows','search_batch_size'):
        if type(cfg[k]) is not int or cfg[k]<1:raise ValueError(f'Invalid {k}')
    if cfg['batch_size']<2 or cfg['device'] not in ('cuda','cpu'):raise ValueError('Invalid encoder settings')
    if not 1<=prep['workers']<=64:raise ValueError('workers must be 1..64')
    if not training['full_data']:raise ValueError('No implicit sampling of matcher training')
    if args.stage:
        # Internal subprocess: parent owns lock, signature and completion state.
        execute(args.stage,args,Path(args.input),cfg,prep,training);return
    lock=(args.work/'.pipeline.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another pipeline is writing this work directory')
    source=stage_input(args.input,args.work)
    if not source.exists():raise FileNotFoundError(source)
    files=[source] if source.is_file() else sorted(source.rglob('*.tsv'))
    code=hashlib.sha256()
    for path in sorted((ROOT/'src').rglob('*.py'))+[ROOT/'pipeline.py',ROOT/'clean_er_data.py',ROOT/'utils/validate_submission.py']:code.update(path.read_bytes())
    signature=dict(inputs=[file_identity(f) for f in files],settings=settings,code=code.hexdigest(),sample=args.sample)
    manifest=args.work/'pipeline_manifest.json'
    state=json.loads(manifest.read_text()) if manifest.exists() else dict(signature=signature,completed=[],timings={})
    if state['signature']!=signature:raise ValueError('Input/config/code changed: use a NEW work directory. Do not mix checkpoints.')
    dump_json(manifest,state)
    wanted=STAGES if args.phase=='all' else PHASES[args.phase]
    for stage in wanted:
        if stage in state['completed']:print('Skip completed '+stage,flush=True);continue
        missing=set(STAGES[:STAGES.index(stage)])-set(state['completed'])
        if missing:raise ValueError('Missing prior stages: '+str(sorted(missing)))
        command=[sys.executable,str(ROOT/'pipeline.py'),'--input',str(source),'--work',str(args.work),'--config',str(args.config.resolve()),'--stage',stage]
        if args.sample:command.append('--sample')
        started=time.monotonic();print('START '+stage,flush=True)
        env=dict(os.environ,OMP_NUM_THREADS=str(cfg['cpu_threads']),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        subprocess.run(command,check=True,env=env)
        state['completed'].append(stage);state['timings'][stage]=time.monotonic()-started;dump_json(manifest,state)
    print('Phase complete: '+args.phase,flush=True)

if __name__=='__main__':main()
