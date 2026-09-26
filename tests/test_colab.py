import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('colab_runner',ROOT/'colab/run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class ColabTests(unittest.TestCase):
    def test_resource_settings_preserve_full_data(self):
        for ram,expected in [(15,4),(24,8),(40,16)]:
            h,t,p=runner.settings(ram,2,'lora','cuda')
            self.assertEqual(h['mini_batch_size'],expected)
            self.assertEqual(h['device'],'cuda');self.assertEqual(t['device_type'],'cuda')
            self.assertTrue(t['full_data']);self.assertEqual(p['sqlite_cache_mb'],64)
        h,t,_=runner.settings(15,2,'frozen','cpu')
        self.assertEqual(h['encoder_mode'],'frozen');self.assertEqual(t['device_type'],'cpu')

    def test_backup_keeps_artifacts_not_large_caches(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);work=root/'work';dest=root/'backup'
            for directory in ('encoder','logs','train','test','model/data_cache','colab_config'):
                (work/directory).mkdir(parents=True,exist_ok=True)
            for name in ('encoder/last.pt','logs/encoder.jsonl','train/index.sqlite','model/data_cache/train.f32','test/matching_results.tsv','model/lightgbm_model.txt','test/submission_coverage.json','train/tfidf.npz'):
                (work/name).write_bytes(b'example')
            runner.backup(work,dest)
            self.assertTrue((dest/'encoder/last.pt').exists())
            self.assertFalse((dest/'test/matching_results.tsv').exists())
            runner.backup(work,dest,final=True)
            self.assertTrue((dest/'test/matching_results.tsv').exists())
            self.assertTrue((dest/'train/tfidf.npz').exists())
            self.assertFalse((dest/'train/index.sqlite').exists())
            self.assertFalse((dest/'model/data_cache').exists())
            (work/'encoder/last.pt').write_bytes(b'updated checkpoint')
            runner.backup(work,dest)
            self.assertEqual((dest/'encoder/last.pt').read_bytes(),b'updated checkpoint')
            self.assertFalse(list(dest.rglob('*.copying')))

    def test_notebook_has_no_saved_output_and_python_cells_compile(self):
        nb=json.loads((ROOT/'colab/ML_Code_1_Colab.ipynb').read_text())
        self.assertEqual(nb['nbformat'],4)
        for c in nb['cells']:
            if c['cell_type']=='code':
                self.assertEqual(c['outputs'],[])
                s=''.join(c['source'])
                if not s.startswith('!'):compile(s,'notebook','exec')

if __name__=='__main__':unittest.main()
