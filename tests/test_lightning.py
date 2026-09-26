"""No billed resources: controller lifecycle tests use a fake SDK worker."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import lightning_run

class FakeWorker:
    id='separate-worker';status='Stopped'
    def __init__(self,fail=False):self.started=[];self.stopped=0;self.fail=fail
    def start(self,**kwargs):self.started.append(str(kwargs['machine']))
    def stop(self):self.stopped+=1
    def run(self,command):
        if 'rev-parse' in command:return 'abc123'
        if 'p.read_text' in command:return json.dumps({'exit_code':1 if self.fail else 0})
        return ''
    def upload_file(self,*args,**kwargs):pass
    def run_and_detach(self,*args,**kwargs):return '',None

class LightningTests(unittest.TestCase):
    def config(self):
        c=json.loads((ROOT/'configs/lightning_launch.json').read_text())
        c.update(teamspace='owner/team',worker_studio='worker',input='/teamspace/data.zip')
        return c
    def invoke(self,root,worker):
        c=root/'launch.json';c.write_text(json.dumps(self.config()))
        args=['lightning_run.py','--config',str(c),'--state',str(root/'state.json'),'--run']
        def git(command,**kwargs):return 'abc123' if 'rev-parse' in command else ''
        with patch.object(sys,'argv',args),patch('lightning_sdk.Studio',return_value=worker),patch('lightning_run.subprocess.check_output',side_effect=git):
            lightning_run.main()
    def test_phase_order_stop_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);worker=FakeWorker();self.invoke(root,worker)
            self.assertEqual(worker.started,['DATA_PREP','A100_40GB','A100_40GB','DATA_PREP'])
            self.assertEqual(worker.stopped,4)
            self.invoke(root,worker);self.assertEqual(len(worker.started),4)
            self.assertEqual(json.loads((root/'state.json').read_text())['completed'],lightning_run.PHASES)
    def test_failed_phase_stops_and_does_not_advance(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);worker=FakeWorker(fail=True)
            with self.assertRaises(RuntimeError):self.invoke(root,worker)
            self.assertEqual(worker.stopped,1)
            self.assertEqual(json.loads((root/'state.json').read_text())['completed'],[])
    def test_budget_exhaustion(self):
        state=dict(estimated_compute_usd=35,started_at=time.time())
        self.assertLess(lightning_run.allowance(self.config(),state,2.19,3),0)
    def test_ranges_cover_every_record_once(self):
        from er_pipeline.fast_embed import ranges
        for n in (1,2,3,17):
            for workers in (1,2,4):
                actual=[i for lo,hi in ranges(n,workers) for i in range(lo,hi)]
                self.assertEqual(actual,list(range(1,n+1)))
    def test_parallel_index_matches_serial(self):
        import csv
        import numpy as np
        from clean_er_data import clean_record
        from er_pipeline.common import DEFAULTS,connect
        from er_pipeline import indexing,parallel_index
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cleaned=root/'cleaned';cleaned.mkdir()
            for source in (1,2,3):
                values=[clean_record(dict(entity_id=f'S{source}-{i}',business_name=f'Cafe {i}',business_address=f'{i} Main Road',country='France' if i%2 else 'India')) for i in range(4)]
                with (cleaned/f'train_source{source}.tsv').open('w') as out:
                    w=csv.DictWriter(out,fieldnames=list(values[0]),delimiter='\t');w.writeheader();w.writerows(values)
            serial=root/'serial';parallel=root/'parallel';serial.mkdir();parallel.mkdir()
            config=dict(DEFAULTS,workers=2,hash_bits=10)
            indexing.build(cleaned,serial,'train',config);parallel_index.build(cleaned,parallel,'train',config)
            a=connect(serial/'index.sqlite',readonly=True);b=connect(parallel/'index.sqlite',readonly=True)
            self.assertEqual(a.execute('SELECT * FROM records ORDER BY rid').fetchall(),b.execute('SELECT * FROM records ORDER BY rid').fetchall())
            a.close();b.close()
            with np.load(serial/'tfidf.npz') as x,np.load(parallel/'tfidf.npz') as y:
                for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    def test_guard_records_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as d:
            result=Path(d)/'result.json'
            run=subprocess.run([sys.executable,str(ROOT/'scripts/phase_guard.py'),'--result',str(result),'--seconds','10','--',sys.executable,'-c','raise SystemExit(7)'])
            self.assertEqual(run.returncode,7)
            self.assertEqual(json.loads(result.read_text())['exit_code'],7)
    def test_parallel_map_bounds_and_preserves_order(self):
        from concurrent.futures import ThreadPoolExecutor
        from er_pipeline.parallel import bounded_map
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(bounded_map(pool,abs,range(-10,0),4)),list(range(10,0,-1)))

if __name__=='__main__':unittest.main()
