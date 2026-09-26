"""Generate private launch/model configs. Does not contact Lightning or rent hardware."""
import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightning_run import validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teamspace', required=True, help='owner/teamspace')
    p.add_argument('--worker', required=True, help='Name of a separate, dedicated worker Studio')
    p.add_argument('--input', required=True, help='Worker /teamspace ZIP/directory or S3 ZIP URI')
    p.add_argument('--worker-repo', default='/teamspace/studios/this_studio/ML_Code_3')
    p.add_argument('--work', help='New persistent worker work directory; keep it to resume')
    p.add_argument('--sample', action='store_true', help='Use sample safeguards and smaller CPU/LightGBM settings')
    p.add_argument('--embedding-gpus', type=int, choices=(1, 2), default=2)
    p.add_argument('--cpu-machine', default='DATA_PREP')
    p.add_argument('--train-machine', default='A100_40GB')
    p.add_argument('--embedding-machine', help='Default A100_40GB or A100_40GB_X_2')
    p.add_argument('--cpu-rate', type=float, default=1.48, help='Estimated USD/hour; verify in your account')
    p.add_argument('--train-rate', type=float, default=2.19, help='Estimated USD/hour; verify in your account')
    p.add_argument('--embedding-rate', type=float, help='TOTAL machine USD/hour; defaults to 2.19 per GPU')
    p.add_argument('--max-compute-usd', type=float, default=35)
    p.add_argument('--max-hours', type=float, default=12)
    p.add_argument('--workers', type=int, help='CPU processes: default 2 sample / 8 full')
    p.add_argument('--threads', type=int, help='LightGBM CPU threads: default 4 sample / 24 full')
    p.add_argument('--output', type=Path, help='Default configs/local_sample.json or configs/local_launch.json')
    a = p.parse_args()
    c = json.loads((ROOT / 'configs/lightning_launch.json').read_text())
    c.update(teamspace=a.teamspace, worker_studio=a.worker, worker_repo=a.worker_repo, input=a.input,
             work=a.work or '/teamspace/studios/this_studio/' + ('work_ml3_sample' if a.sample else 'work_ml3'),
             cpu_machine=a.cpu_machine, gpu_machine=a.train_machine,
             embedding_machine=a.embedding_machine or ('A100_40GB_X_2' if a.embedding_gpus == 2 else 'A100_40GB'),
             embedding_gpus=a.embedding_gpus, cpu_rate_usd=a.cpu_rate, gpu_rate_usd=a.train_rate,
             embedding_rate_usd=a.embedding_rate if a.embedding_rate is not None else 2.19 * a.embedding_gpus,
             max_compute_usd=a.max_compute_usd, max_hours=a.max_hours, sample=a.sample)
    validate(c)
    m = json.loads((ROOT / 'configs/lightning_model.json').read_text())
    m['encoder']['embedding_gpus'] = a.embedding_gpus
    m['preparation']['workers'] = a.workers if a.workers is not None else (2 if a.sample else 8)
    m['matcher']['num_threads'] = a.threads if a.threads is not None else (4 if a.sample else 24)
    if not 1 <= m['preparation']['workers'] <= 64 or m['matcher']['num_threads'] < 1:
        p.error('workers must be 1..64 and threads must be positive')
    if a.sample:
        # Same pretrained production encoder and real LoRA, not the random smoke encoder.
        m['encoder'].update(cpu_threads=2, monitor_per_country=20, monitor_references=1000)
        m['preparation'].update(row_group_size=1000, sqlite_cache_mb=32)
        m['matcher'].update(min_data_in_leaf=5, num_boost_round=300, early_stopping_rounds=30)
    output = (a.output or ROOT / 'configs' / ('local_sample.json' if a.sample else 'local_launch.json')).resolve()
    model = output.with_name(output.stem + '_model.json')
    if output.exists() or model.exists():
        p.error('Configuration already exists. Edit it in place to resume, or choose a new --output and --work.')
    output.parent.mkdir(parents=True, exist_ok=True)
    c['model_config'] = model.name
    model.write_text(json.dumps(m, indent=2) + '\n')
    output.write_text(json.dumps(c, indent=2) + '\n')
    output.chmod(0o600)
    model.chmod(0o600)
    print(f'Created {output}\nCreated {model}')
    print('One GPU for fine-tuning; ' + str(a.embedding_gpus) + ' GPU(s) for embeddings; CPU for preparation and matching.')
    print('Verify machine availability and ALL hourly rates in Lightning before --run.')
    command = 'python lightning_run.py --config ' + shlex.quote(str(output))
    print('Review: ' + command + ' --plan')
    print('Launch from a SEPARATE controller Studio: ' + command + ' --run')
    print('Status: ' + command + ' --status')
    if a.sample:
        print('Sample output is for testing only. Use a new config and work directory for the full dataset.')


if __name__ == '__main__':
    main()
