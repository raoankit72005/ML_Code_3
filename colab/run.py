"""Colab single-GPU adapter. Active data stays local; portable artifacts are backed up."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
GIB=1024**3


def atomic_copy(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_name(dest.name+'.copying')
    shutil.copy2(source,temp);temp.replace(dest)


def backup(work,destination,final=False):
    # Checkpoints are published with rename by encoder.py, so an opened file is stable.
    selected=list((work/'encoder').glob('*.pt'))+list((work/'encoder').glob('*.json'))
    selected+=list((work/'logs').glob('*.jsonl'))+list((work/'colab_config').glob('*.json'))
    selected+=[work/'hybrid_manifest.json',work/'colab_hardware.json']
    if final:
        selected+=list((work/'model').glob('*.txt'))+list((work/'model').glob('*.json'))
        selected+=list((work/'test').glob('*.tsv'))+list((work/'test').glob('*.json'))
        selected+=list((work/'train').glob('*.json'))+[work/'train/tfidf.npz']
        selected+=list((work/'logs/tensorboard').rglob('*'))
    copied=0
    for src in selected:
        if not src.is_file():continue
        dest=destination/src.relative_to(work)
        if dest.exists() and (dest.stat().st_size,dest.stat().st_mtime_ns)==(src.stat().st_size,src.stat().st_mtime_ns):continue
        atomic_copy(src,dest);copied+=1
    env=Path('/content/ml_er_environment.txt')
    if env.exists():atomic_copy(env,destination/env.name)
    print(f'Backup: {copied} artifact files -> {destination}',flush=True)


def settings(vram_gib,threads,encoder_mode,matcher):
    h=json.loads((ROOT/'configs/hybrid.json').read_text())
    t=json.loads((ROOT/'configs/full_gpu.json').read_text())
    p=json.loads((ROOT/'configs/full_preparation.json').read_text())
    micro=4 if vram_gib<18 else 8 if vram_gib<35 else 16
    h.update(device='cuda',encoder_mode=encoder_mode,cpu_threads=threads,
             mini_batch_size=micro,encode_batch_size=micro*4,log_every=10,checkpoint_every=100)
    t.update(device_type=matcher,num_threads=threads,ram_fraction=.55,gpu_memory_fraction=.65,
             disk_reserve_gb=5,batch_size=2048,histogram_pool_size=128)
    p.update(sqlite_cache_mb=64,row_group_size=2000)
    return h,t,p


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='Original dataset ZIP already copied to local disk')
    p.add_argument('--work',type=Path,default=Path('/content/ml_er_work'))
    p.add_argument('--backup',type=Path,required=True,help='A unique run folder on mounted Drive')
    p.add_argument('--encoder-mode',choices=['lora','frozen'],default='lora')
    p.add_argument('--matcher',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--backup-seconds',type=int,default=180)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.backup_seconds<30:p.error('backup-seconds must be at least 30')
    if not a.input.is_file():p.error('Input ZIP does not exist')
    work=a.work.resolve();data=a.input.resolve();dest=a.backup.resolve()
    for path in (work,data):
        if path==Path('/content/drive') or Path('/content/drive') in path.parents:
            p.error('Copy dataset and work files to local /content; do not run SQLite on Drive')
    if dest==work or work in dest.parents or dest in work.parents:p.error('Backup and work must be separate trees')
    work.mkdir(parents=True,exist_ok=True)
    handle=(work/'.colab.lock').open('a')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another Colab launcher is using this work directory')
    sys.path.insert(0,str(ROOT/'src'))
    from er_pipeline.resources import available_ram,check_disk
    import torch
    if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU: select a GPU runtime')
    ram=available_ram();free,total=torch.cuda.mem_get_info()
    if ram<6*GIB:raise RuntimeError('Less than 6 GiB RAM available; this is only a minimum startup check, not a full-data fit guarantee')
    check_disk(work,reserve_gb=20)
    threads=max(1,min(4,os.cpu_count() or 1))
    hardware=dict(gpu=torch.cuda.get_device_name(0),vram_gib=total/GIB,free_vram_gib=free/GIB,
                  available_ram_gib=ram/GIB,free_disk_gib=shutil.disk_usage(work).free/GIB,threads=threads)
    print(json.dumps(hardware,indent=2),flush=True)
    (work/'colab_hardware.json').write_text(json.dumps(hardware,indent=2))
    cfg=work/'colab_config';cfg.mkdir(exist_ok=True)
    choices=dict(encoder_mode=a.encoder_mode,matcher=a.matcher,gpu=hardware['gpu'])
    marker=cfg/'selection.json'
    if marker.exists():
        if json.loads(marker.read_text())!=choices:raise RuntimeError('Device/mode changed: use a new work directory')
    else:
        if (work/'hybrid_manifest.json').exists():raise RuntimeError('Existing non-Colab work directory; automatic SageMaker migration is not supported')
        for name,value in zip(('hybrid.json','training.json','preparation.json'),settings(total/GIB,threads,a.encoder_mode,a.matcher)):
            (cfg/name).write_text(json.dumps(value,indent=2))
        marker.write_text(json.dumps(choices))
    for key,folder in [('TMPDIR','tmp'),('SQLITE_TMPDIR','tmp'),('HF_HOME','huggingface')]:
        target=work/folder;target.mkdir(exist_ok=True);os.environ[key]=str(target)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    if a.check_only:return
    dest.mkdir(parents=True,exist_ok=True)
    owner=dest/'backup_source.json'
    identity=dict(work=str(work),input=str(data),input_size=data.stat().st_size,selection=choices)
    # A backup is not automatically restored: SQLite/vector caches are not included.
    if owner.exists() and (not (work/'hybrid_manifest.json').exists() or json.loads(owner.read_text())!=identity):
        raise RuntimeError('Backup belongs to another or lost runtime. Use a new backup run name to preserve it.')
    owner.write_text(json.dumps(identity,indent=2))
    cmd=[sys.executable,'-u',str(ROOT/'hybrid.py'),'--input',str(data),'--work',str(work),
         '--config',str(cfg/'hybrid.json'),'--training-config',str(cfg/'training.json'),
         '--preparation-config',str(cfg/'preparation.json')]
    child=subprocess.Popen(cmd,cwd=ROOT,start_new_session=True)
    try:
        while True:
            try:
                status=child.wait(timeout=a.backup_seconds);break
            except subprocess.TimeoutExpired:
                try:backup(work,dest)
                except OSError as exc:print(f'BACKUP ERROR (local training continues): {exc}',flush=True)
    except KeyboardInterrupt:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            raise RuntimeError('Pipeline did not exit yet. Do not launch another process against this work directory.')
        raise
    finally:
        backup(work,dest,final=child.poll() is not None)
    if status:raise SystemExit(status)
    print('Complete: '+str(dest/'test/matching_results.tsv'),flush=True)

if __name__=='__main__':main()
