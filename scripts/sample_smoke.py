"""Offline real-LoRA/FAISS/LightGBM wiring test on a user-supplied sample ZIP.
Random tiny encoder: this tests execution and format, NOT model quality.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--work',type=Path,default=Path('work_sample_smoke'));a=p.parse_args()
    root=a.work.resolve();root.mkdir(parents=True,exist_ok=True)
    import torch
    from transformers import BertConfig,BertModel,BertTokenizerFast
    from sentence_transformers import SentenceTransformer,models
    torch.set_num_threads(1);torch.manual_seed(42)
    base=root/'tiny_bert';saved=root/'tiny_sentence'
    if not saved.exists():
        base.mkdir(exist_ok=True)
        vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','name','address','country',':','|','business','road','india','us','france']+[str(i) for i in range(100)]
        (base/'vocab.txt').write_text('\n'.join(vocab))
        BertTokenizerFast(vocab_file=str(base/'vocab.txt')).save_pretrained(base)
        BertModel(BertConfig(vocab_size=len(vocab),hidden_size=24,num_hidden_layers=1,num_attention_heads=4,intermediate_size=32)).save_pretrained(base)
        sentence=SentenceTransformer(modules=[models.Transformer(str(base),max_seq_length=32),models.Pooling(24)])
        sentence.save(str(saved));del sentence
    c=json.loads((ROOT/'configs/lightning_model.json').read_text())
    c['encoder'].update(model_id=str(saved),revision=None,device='cpu',cpu_threads=1,max_length=32,batch_size=16,mini_batch_size=8,
        encode_batch_size=64,embedding_gpus=1,encode_chunk_rows=256,checkpoint_every=100,log_every=10,
        monitor_per_country=4,monitor_references=100,monitor_k=10,pq_m=6,disk_reserve_gb=.01)
    c['preparation'].update(workers=2,row_group_size=1000,sqlite_cache_mb=32)
    c['matcher'].update(num_threads=2,num_boost_round=20,early_stopping_rounds=5,min_data_in_leaf=5,disk_reserve_gb=.01,thresholds=[.1,.3,.5,.7,.9,1.])
    config=root/'sample_config.json';config.write_text(json.dumps(c,indent=2))
    command=[sys.executable,str(ROOT/'pipeline.py'),'--input',str(Path(a.input).resolve()),'--work',str(root/'run'),'--config',str(config),'--sample']
    subprocess.run(command,check=True,cwd=ROOT)
    subprocess.run(command,check=True,cwd=ROOT) # restart skips completed stages
    print('Sample wiring test PASSED. Random tiny encoder; NOT a quality benchmark or full submission.')
if __name__=='__main__':main()
