"""Control a separate persistent Lightning worker: CPU -> 1 GPU -> embedding GPUs -> CPU.

--plan and --status never start or stop machines. Run the controller outside the worker.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import signal
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parent
PHASES = ['prepare', 'train', 'embed', 'finish']
LIMIT_KEYS = {'max_compute_usd', 'max_hours', 'phase_hours'}
STATE_VERSION = 2


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + '.partial')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temporary.replace(path)


def positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def persistent_path(value):
    return (isinstance(value, str) and value.startswith('/teamspace/')
            and '..' not in PurePosixPath(value).parts and '\x00' not in value)


def validate(c):
    if 'sample' in c and type(c['sample']) is not bool:
        raise ValueError('sample must be a JSON boolean')
    for key in ('teamspace', 'worker_studio', 'worker_repo', 'input', 'work'):
        if not isinstance(c.get(key), str) or not c[key].strip() or 'REPLACE' in c[key]:
            raise ValueError(f'Fill {key} in launch config')
    for key in ('worker_repo', 'work'):
        if not persistent_path(c[key]):
            raise ValueError(f'{key} must be an absolute persistent /teamspace path without ..')
    repo, work = PurePosixPath(c['worker_repo']), PurePosixPath(c['work'])
    if repo == work or work in repo.parents:
        raise ValueError('Work must not contain the source checkout; use a separate work directory')
    if c['input'].startswith('s3://'):
        from urllib.parse import urlparse
        uri = urlparse(c['input'])
        if not uri.netloc or not uri.path.endswith('.zip') or uri.query or uri.fragment:
            raise ValueError('S3 input must name one .zip object, without query/fragment')
    elif not persistent_path(c['input']):
        raise ValueError('Input must be a persistent /teamspace path or s3://bucket/file.zip')
    for key in ('max_compute_usd', 'max_hours', 'cpu_rate_usd', 'gpu_rate_usd', 'embedding_rate_usd'):
        if not positive(c.get(key)):
            raise ValueError(f'{key} must be a positive finite number')
    if type(c.get('embedding_gpus')) is not int or c['embedding_gpus'] not in (1, 2):
        raise ValueError('embedding_gpus must be 1 or 2')
    for key in ('cpu_machine', 'gpu_machine', 'embedding_machine'):
        if not isinstance(c.get(key), str) or not c[key].strip():
            raise ValueError(f'Fill {key}')
    for phase in PHASES:
        if not positive(c.get('phase_hours', {}).get(phase)):
            raise ValueError(f'phase_hours.{phase} must be positive and finite')


def plan(c):
    return [dict(phase=p,
                 machine=c['cpu_machine'] if p in ('prepare', 'finish') else c['gpu_machine'] if p == 'train' else c['embedding_machine'],
                 rate=c['cpu_rate_usd'] if p in ('prepare', 'finish') else c['gpu_rate_usd'] if p == 'train' else c['embedding_rate_usd'],
                 hours=c['phase_hours'][p]) for p in PHASES]


def resolve_machines(c):
    """Validate EVERY choice before starting even the first CPU phase."""
    from lightning_sdk import Machine
    machines = {}
    for item in plan(c):
        machine = getattr(Machine, item['machine'], None)
        if not isinstance(machine, Machine):
            raise ValueError(f'Unknown SDK machine {item["machine"]!r}; use a Machine enum name')
        is_cpu = machine.family in ('CPU', 'DATA-PREP')
        if item['phase'] in ('prepare', 'finish'):
            if not is_cpu:
                raise ValueError('cpu_machine must be a CPU/DATA_PREP machine')
        else:
            expected = c['embedding_gpus'] if item['phase'] == 'embed' else 1
            if is_cpu or machine.accelerator_count != expected:
                raise ValueError(f'{item["phase"]} needs exactly {expected} GPU(s), not {item["machine"]}')
        machines[item['phase']] = machine
    return machines


def allowance(c, state, rate, requested, phase=None):
    remaining_money = c['max_compute_usd'] - state['estimated_compute_usd']
    remaining_time = c['max_hours'] * 3600 - state.get('elapsed_worker_seconds', 0)
    phase_time = requested * 3600 - state.get('phase_seconds', {}).get(phase, 0)
    seconds = min(phase_time, remaining_time, remaining_money / rate * 3600)
    return int(seconds) - 120  # Reserve provisioning/release/polling overhead.


def status_name(worker):
    status = worker.status
    return str(getattr(status, 'name', status)).rsplit('.', 1)[-1].lower()


def stop_worker(worker, timeout=180):
    """SDK stop() is not idempotent. Wait for confirmed shutdown before switching."""
    end = time.monotonic() + timeout
    requested = False
    while True:
        status = status_name(worker)
        if status in ('stopped', 'completed'):
            return
        if status in ('running', 'pending') and not requested:
            try:
                worker.stop()
            except Exception:
                # The remote guard/provider may have stopped it between status and stop.
                if status_name(worker) not in ('stopped', 'completed', 'stopping'):
                    raise
            requested = True
            continue
        elif status not in ('running', 'pending', 'stopping'):
            raise RuntimeError(f'Cannot confirm worker shutdown: status={status}; inspect Lightning UI')
        if time.monotonic() >= end:
            raise TimeoutError('Worker shutdown not confirmed; inspect Lightning UI before resuming')
        time.sleep(2)


def settle(state):
    active = state['active']
    elapsed = max(0, time.time() - active['started'])
    state['estimated_compute_usd'] += elapsed / 3600 * active['rate']
    state['elapsed_worker_seconds'] += elapsed
    phase = active['phase']
    state['phase_seconds'][phase] = state['phase_seconds'].get(phase, 0) + elapsed
    state['active'] = None


@contextmanager
def controller_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A controller already owns this state file') from None
        yield


@contextmanager
def shutdown_on_signal():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Controller received signal {signum}; releasing worker')
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def state_path(c, sample):
    # Stable across budget changes. Same worker/work must share a state file.
    identity = [c[k] for k in ('teamspace', 'worker_studio', 'work')] + [sample]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
    return ROOT / 'controller_state' / (key + '.json')


def run(a, c, machines, model):
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines()
    if any(not x.startswith('configs/') for x in changed):
        raise ValueError('Commit source edits before launching')
    identity = dict(config={k: v for k, v in c.items() if k not in LIMIT_KEYS},
                    model=model, revision=revision, sample=a.sample)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    state = json.loads(a.state.read_text()) if a.state.exists() else dict(
        version=STATE_VERSION, signature=signature, revision=revision, started_at=time.time(),
        estimated_compute_usd=0., elapsed_worker_seconds=0., phase_seconds={}, completed=[], active=None)
    if state.get('version') != STATE_VERSION:
        raise ValueError('Legacy controller state: inspect/stop the worker, then use a new --state file. Historical charges are not imported.')
    if state['signature'] != signature:
        raise ValueError('Model/source/machine/input changed; restore settings to resume or use a NEW work directory and state')
    state['limits'] = {k: c[k] for k in LIMIT_KEYS}
    save(a.state, state)
    if state['completed'] == PHASES and not state['active']:
        print('All phases already completed; no machine started.')
        return
    from lightning_sdk import Studio
    worker = Studio(c['worker_studio'], teamspace=c['teamspace'], create_ok=False)
    if worker.id == os.environ.get('LIGHTNING_CLOUD_SPACE_ID'):
        raise ValueError('Controller must run OUTSIDE the worker Studio')
    if state.get('active'):
        # Charge disconnected time conservatively; never discard an unfinished attempt.
        stop_worker(worker)
        settle(state)
        save(a.state, state)
    if status_name(worker) != 'stopped':
        raise ValueError('Stop the dedicated worker Studio before launching; controller will start it')
    remote, work = c['worker_repo'].rstrip('/'), c['work'].rstrip('/')
    for item in plan(c):
        phase = item['phase']
        if phase in state['completed']:
            continue
        seconds = allowance(c, state, item['rate'], item['hours'], phase)
        if seconds < 60:
            raise RuntimeError('Time/estimated compute allowance exhausted. Increase limits in the SAME config/state to resume.')
        state['active'] = dict(phase=phase, started=time.time(), rate=item['rate'])
        save(a.state, state)
        try:
            worker.start(machine=machines[phase], interruptible=False, max_runtime=seconds + 120)
            output = worker.run('git -C ' + shlex.quote(remote) + ' rev-parse HEAD').strip()
            if output != revision:
                raise ValueError(f'Worker checkout differs; use git checkout {revision} on worker before launching')
            dirty = worker.run('git -C ' + shlex.quote(remote) + ' diff --name-only HEAD').splitlines()
            if any(not x.startswith('configs/') for x in dirty):
                raise ValueError('Worker has uncommitted source edits; restore a clean checkout')
            local = Path(str(a.state) + '.model.json')
            local.write_text(json.dumps(model, indent=2))
            worker.upload_file(str(local), remote_path=remote + '/configs/active_model.json', progress_bar=False)
            worker.run('mkdir -p ' + shlex.quote(work + '/logs'))
            result = work + '/logs/' + phase + '-' + uuid.uuid4().hex + '.json'
            command = [remote + '/.venv/bin/python', remote + '/pipeline.py', '--phase', phase,
                       '--input', c['input'], '--work', work, '--config', remote + '/configs/active_model.json']
            if a.sample:
                command.append('--sample')
            remaining = seconds - int(time.time() - state['active']['started'])
            if remaining < 60:
                raise RuntimeError('Machine provisioning consumed phase allowance')
            wrapped = [remote + '/.venv/bin/python', remote + '/scripts/phase_guard.py', '--result', result,
                       '--seconds', str(remaining), '--'] + command
            shell = shlex.join(wrapped) + ' > ' + shlex.quote(work + '/logs/' + phase + '.log') + ' 2>&1'
            _, early_code = worker.run_and_detach(shell, timeout=2, check_interval=1)
            if early_code is not None and early_code != 0:
                raise RuntimeError(f'{phase} failed to launch; check {work}/logs/{phase}.log')
            while True:
                if time.time() - state['active']['started'] > seconds + 30:
                    raise TimeoutError(f'{phase} exceeded allowance; checkpoint retained')
                code = 'import pathlib; p=pathlib.Path(' + repr(result) + '); print(p.read_text() if p.exists() else "WAIT")'
                raw = worker.run(shlex.quote(remote + '/.venv/bin/python') + ' -c ' + shlex.quote(code)).strip()
                if raw != 'WAIT':
                    result_data = json.loads(raw)
                    if result_data['exit_code'] != 0:
                        print(worker.run('tail -n 60 ' + shlex.quote(work + '/logs/' + phase + '.log')))
                        raise RuntimeError(f'{phase} failed; see {work}/logs/{phase}.log')
                    break
                print(f'{phase}: {(time.time() - state["active"]["started"]) / 60:.1f} minutes; log {work}/logs/{phase}.log', flush=True)
                time.sleep(30)
            state['completed'].append(phase)
        finally:
            try:
                stop_worker(worker)
            except Exception:
                save(a.state, state)
                print('Worker shutdown could not be confirmed. Stop it in Lightning UI; active state retained.', flush=True)
                raise
            else:
                settle(state)
                save(a.state, state)
    print('Complete. Worker stopped. Outputs in ' + work + ('/sample_output' if a.sample else '/output'))
    print(f'Estimated compute: ${state["estimated_compute_usd"]:.2f}; check actual Lightning billing.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/lightning_launch.json')
    p.add_argument('--model-config', type=Path, help='Defaults to model_config in launch JSON, then configs/lightning_model.json')
    p.add_argument('--state', type=Path, help='Default: stable controller_state/<worker-work-id>.json')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--status', action='store_true', help='Read local controller state; no cloud calls')
    p.add_argument('--sample', action='store_true')
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    validate(c)
    a.sample = a.sample or bool(c.get('sample', False))
    a.state = (a.state or state_path(c, a.sample)).resolve()
    if a.status:
        print(a.state.read_text() if a.state.exists() else 'Not started. No controller state yet.')
        return
    model_path = a.model_config or (a.config.parent / c['model_config'] if c.get('model_config') else ROOT / 'configs/lightning_model.json')
    model = json.loads(model_path.read_text())
    model['encoder']['embedding_gpus'] = c['embedding_gpus']
    if model['encoder']['device'] != 'cuda' or model['encoder']['encoder_mode'] != 'lora':
        raise ValueError('Automatic GPU workflow requires encoder device=cuda and encoder_mode=lora')
    if model['matcher']['device_type'] != 'cpu' or not model['matcher']['full_data']:
        raise ValueError('Automatic CPU finish requires matcher device_type=cpu and full_data=true')
    machines = resolve_machines(c)
    print(json.dumps(plan(c), indent=2))
    print('Rates are configured estimates; storage, transfers, controller costs and taxes are excluded.')
    print('State: ' + str(a.state))
    if a.sample:
        print('SAMPLE RUN: outputs will be marked sample-only and cannot be packaged for submission.')
    if not a.run:
        return
    with controller_lock(str(a.state) + '.lock'), shutdown_on_signal():
        run(a, c, machines, model)


if __name__ == '__main__':
    main()
