"""CPU controller switches a separate persistent worker Studio between CPU and A100.
Run --plan first. No infrastructure is started without --run.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time
import uuid

ROOT=Path(__file__).resolve().parent
PHASES=['prepare','train','embed','finish']


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(str(path)+'.partial');temporary.write_text(json.dumps(obj,indent=2));temporary.replace(path)


def validate(c):
    for key in ('teamspace','worker_studio','worker_repo','input','work'):
        if not c.get(key) or 'REPLACE' in c[key]:raise ValueError(f'Fill {key} in launch config')
    for key in ('worker_repo','work'):
        if not c[key].startswith('/teamspace/'):raise ValueError(f'{key} must be on persistent /teamspace storage')
    if c['worker_repo'].rstrip('/')==c['work'].rstrip('/'):raise ValueError('Use a separate work directory')
    for key in ('max_compute_usd','max_hours','cpu_rate_usd','gpu_rate_usd','embedding_rate_usd'):
        if not isinstance(c[key],(float,int)) or not math.isfinite(c[key]) or c[key]<=0:raise ValueError(f'Invalid {key}')
    if c['embedding_gpus'] not in (1,2):raise ValueError('embedding_gpus must be 1 or 2')
    for phase in PHASES:
        if not math.isfinite(c['phase_hours'][phase]) or c['phase_hours'][phase]<=0:raise ValueError('Phase hours must be positive and finite')


def plan(c):
    return [dict(phase=p,machine=c['cpu_machine'] if p in ('prepare','finish') else c['gpu_machine'] if p=='train' else c['embedding_machine'],
                 rate=c['cpu_rate_usd'] if p in ('prepare','finish') else c['gpu_rate_usd'] if p=='train' else c['embedding_rate_usd'],
                 hours=c['phase_hours'][p]) for p in PHASES]


def allowance(c,state,rate,requested):
    remaining_money=c['max_compute_usd']-state['estimated_compute_usd']
    remaining_time=c['max_hours']*3600-(time.time()-state['started_at'])
    seconds=min(requested*3600,remaining_time,remaining_money/rate*3600)
    # Leave release/polling overhead outside executable allowance.
    return int(seconds)-120


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/lightning_launch.json')
    p.add_argument('--model-config',type=Path,default=ROOT/'configs/lightning_model.json')
    p.add_argument('--state',type=Path,default=Path('launch_state.json'))
    p.add_argument('--run',action='store_true');p.add_argument('--plan',action='store_true')
    p.add_argument('--sample',action='store_true')
    a=p.parse_args();c=json.loads(a.config.read_text());validate(c)
    schedule=plan(c);print(json.dumps(schedule,indent=2))
    print('Rates are YOUR configured estimates. Storage, data transfer, controller costs and taxes are excluded.')
    if not a.run:return
    from lightning_sdk import Studio,Machine
    lock=Path(str(a.state)+'.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A controller already owns this state file')
    model=json.loads(a.model_config.read_text());model['encoder']['embedding_gpus']=c['embedding_gpus']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        # Configuration edits are expected; source edits need a commit to reproduce remotely.
        changed=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=ROOT,text=True).splitlines()
        if any(not x.startswith('configs/') for x in changed):raise ValueError('Commit source edits before launching')
    signature=hashlib.sha256(json.dumps(dict(config=c,model=model,revision=revision,sample=a.sample),sort_keys=True).encode()).hexdigest()
    state=json.loads(a.state.read_text()) if a.state.exists() else dict(signature=signature,started_at=time.time(),estimated_compute_usd=0.,completed=[],active=None)
    if state['signature']!=signature:raise ValueError('Launch settings changed; use a new state file. Worker still enforces model/data consistency.')
    worker=Studio(c['worker_studio'],teamspace=c['teamspace'],create_ok=False)
    if worker.id==os.environ.get('LIGHTNING_CLOUD_SPACE_ID'):raise ValueError('Controller must run OUTSIDE the worker Studio')
    # Never take over another active session. A stale active phase is charged conservatively.
    if state.get('active'):
        active=state['active'];worker.stop()
        state['estimated_compute_usd']+=(time.time()-active['started'])/3600*active['rate']
        state['active']=None;save(a.state,state)
    status=str(worker.status).lower()
    if 'running' in status or 'starting' in status:raise ValueError('Stop the dedicated worker Studio before launching; controller will start it')
    remote=c['worker_repo'].rstrip('/');work=c['work'].rstrip('/')
    for item in schedule:
        phase=item['phase']
        if phase in state['completed']:continue
        seconds=allowance(c,state,item['rate'],item['hours'])
        if seconds<60:raise RuntimeError('Time/estimated compute allowance exhausted. No new machine started.')
        state['active']=dict(phase=phase,started=time.time(),rate=item['rate']);save(a.state,state)
        try:
            machine=getattr(Machine,item['machine'])
            worker.start(machine=machine,interruptible=False,max_runtime=seconds+120)
            # Fail instead of silently using stale or different source code.
            output=worker.run('git -C '+shlex.quote(remote)+' rev-parse HEAD').strip()
            if output!=revision:raise ValueError(f'Worker checkout differs; use git checkout {revision} on worker before launching')
            # Installed once on CPU worker during setup, before orchestration.
            worker.run(shlex.quote(remote+'/.venv/bin/python')+' -c '+shlex.quote('import torch,lightgbm,faiss,sentence_transformers,peft,lightning_sdk'))
            local=Path(str(a.state)+'.model.json');local.write_text(json.dumps(model,indent=2))
            worker.upload_file(str(local),remote_path=remote+'/configs/active_model.json',progress_bar=False)
            worker.run('mkdir -p '+shlex.quote(work+'/logs'))
            result=work+'/logs/'+phase+'-'+uuid.uuid4().hex+'.json'
            command=[remote+'/.venv/bin/python',remote+'/pipeline.py','--phase',phase,'--input',c['input'],
                '--work',work,'--config',remote+'/configs/active_model.json']
            if a.sample:command.append('--sample')
            elapsed=time.time()-state['active']['started']
            remaining=seconds-int(elapsed)
            if remaining<60:raise RuntimeError('Machine provisioning consumed phase allowance')
            wrapped=[remote+'/.venv/bin/python',remote+'/scripts/phase_guard.py','--result',result,'--seconds',str(remaining),'--']+command
            shell=shlex.join(wrapped)+' > '+shlex.quote(work+'/logs/'+phase+'.log')+' 2>&1'
            _, early_code = worker.run_and_detach(shell,timeout=2,check_interval=1)
            if early_code is not None and early_code != 0:
                raise RuntimeError(f'{phase} failed to launch; check worker log')
            while True:
                if time.time()-state['active']['started']>seconds+30:raise TimeoutError(f'{phase} exceeded allowance; checkpoint retained')
                code='import pathlib; p=pathlib.Path('+repr(result)+'); print(p.read_text() if p.exists() else "WAIT")'
                raw=worker.run(shlex.quote(remote+'/.venv/bin/python')+' -c '+shlex.quote(code)).strip()
                if raw!='WAIT':
                    result_data=json.loads(raw)
                    if result_data['exit_code']!=0:
                        print(worker.run('tail -n 60 '+shlex.quote(work+'/logs/'+phase+'.log')))
                        raise RuntimeError(f'{phase} failed; see {work}/logs/{phase}.log')
                    break
                print(f'{phase}: {(time.time()-state["active"]["started"])/60:.1f} minutes; log {work}/logs/{phase}.log',flush=True)
                time.sleep(30)
            state['completed'].append(phase)
        finally:
            # Keep active state if releasing compute fails, so restart can retry stop.
            try:
                worker.stop()
            except Exception:
                save(a.state,state)
                print('URGENT: worker stop failed. Stop it in Lightning UI to end charges.',flush=True)
                raise
            else:
                state['estimated_compute_usd']+=(time.time()-state['active']['started'])/3600*item['rate']
                state['active']=None;save(a.state,state)
    print('Complete. Worker stopped. Outputs in '+work+('/sample_output' if a.sample else '/output'))
    print(f'Estimated compute: ${state["estimated_compute_usd"]:.2f}; check actual Lightning billing.')

if __name__=='__main__':main()
