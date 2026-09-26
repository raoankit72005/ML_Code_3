"""Package completed FULL-data outputs and reproducible code; never package sample runs."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--team',required=True);a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.team):raise ValueError('Use letters, numbers, underscores or hyphens in team name')
    work=a.work.resolve();manifest=json.loads((work/'pipeline_manifest.json').read_text())
    if manifest['signature']['sample'] or 'audit' not in manifest['completed']:raise ValueError('Only fully audited, full-data runs can be packaged')
    files=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    if not files:raise ValueError('Run from a committed Git checkout')
    metadata=json.loads((work/'model/model_metadata.json').read_text())
    method=(ROOT/'Documentation_template.md').read_text()
    method+='\n\n## Actual run metadata\n\n```json\n'+json.dumps(dict(timings=manifest['timings'],model_metadata=metadata),indent=2)+'\n```\n'
    target=work/f'{a.team}_submission.zip';partial=target.with_suffix('.partial')
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('matching_results.tsv','candidate_pairs.tsv'):z.write(work/'output'/name,'output/'+name)
        for name in files:
            path=ROOT/name
            if path.is_file() and not path.is_symlink():z.write(path,'code/business_entity_resolution/'+name)
        z.writestr('code/business_entity_resolution/configs/reproduction_model.json',json.dumps(manifest['signature']['settings'],indent=2))
        z.writestr('Documentation_template.md',method)
    partial.replace(target);print(target)
if __name__=='__main__':main()
