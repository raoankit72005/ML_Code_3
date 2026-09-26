"""Clean independent source files concurrently with per-file restart markers."""
import csv
import gzip
import io
import json
import shutil
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing as mp
from .common import dump_json


def clean_file(task):
    from clean_er_data import clean_record
    source, name, destination = task
    source=Path(source);target=Path(destination)/Path(name).name.split('_')[0]/(Path(name).name+'.gz')
    target.parent.mkdir(parents=True,exist_ok=True)
    marker=Path(str(target)+'.json')
    if marker.exists() and target.exists():return json.loads(marker.read_text())
    archive=zipfile.ZipFile(source) if source.is_file() else None
    counts=Counter();countries=Counter()
    try:
        binary=archive.open(name) if archive else (source/name).open('rb')
        partial=Path(str(target)+'.partial')
        with binary,io.TextIOWrapper(binary,encoding='utf-8-sig',newline='') as inp,gzip.open(partial,'wt',encoding='utf-8',newline='',compresslevel=1) as out:
            reader=csv.DictReader(inp,delimiter='\t');required={'entity_id','business_name','business_address','country'}
            if not required.issubset(reader.fieldnames or []):raise ValueError(f'Missing raw columns: {name}')
            derived=set(clean_record(dict.fromkeys(required,'')))-required
            if derived & set(reader.fieldnames):raise ValueError('Input is already cleaned')
            writer=csv.DictWriter(out,fieldnames=reader.fieldnames+sorted(derived),delimiter='\t');writer.writeheader()
            for row in reader:
                if None in row or any(v is None for v in row.values()) or not row['entity_id']:raise ValueError(f'Malformed row in {name}: {reader.line_num}')
                r=clean_record(row);writer.writerow(r);counts['rows']+=1;countries[r['country_key']]+=1
                if counts['rows']%100000==0:print(f'Clean {name}: {counts["rows"]:,}',flush=True)
        partial.replace(target)
        result=dict(file=name,rows=counts['rows'],countries=dict(countries));dump_json(marker,result)
        return result
    finally:
        if archive:archive.close()


def clean(source,destination,workers):
    source=Path(source);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    if source.is_file():
        with zipfile.ZipFile(source) as z: entries=z.namelist()
    else:entries=[str(p.relative_to(source)) for p in source.rglob('*.tsv')]
    wanted={f'{split}_source{i}.tsv' for split in ('train','test') for i in (1,2,3)}|{'train_ground_truth.tsv'}
    selected={}
    for n in entries:
        if Path(n).name in wanted:
            if Path(n).name in selected:raise ValueError('Duplicate dataset filename')
            selected[Path(n).name]=n
    if set(selected)!=wanted:raise ValueError(f'Missing dataset files: {wanted-set(selected)}')
    with ProcessPoolExecutor(max_workers=min(workers,6),mp_context=mp.get_context('spawn')) as pool:
        result=list(pool.map(clean_file,[(str(source),n,str(destination)) for k,n in sorted(selected.items()) if k!='train_ground_truth.tsv']))
    truth=destination/'train/train_ground_truth.tsv';partial=Path(str(truth)+'.partial')
    if source.is_file():
        with zipfile.ZipFile(source) as z,z.open(selected['train_ground_truth.tsv']) as inp,partial.open('wb') as out:shutil.copyfileobj(inp,out)
    else:shutil.copyfile(source/selected['train_ground_truth.tsv'],partial)
    partial.replace(truth)
    dump_json(destination/'cleaning_report.json',dict(files=result,all_rows_retained=True))
