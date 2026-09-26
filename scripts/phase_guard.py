"""Write an atomic result and enforce a remote phase timeout, independent of controller."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--seconds',type=int,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not command or a.seconds<=0:raise ValueError('Need command and positive timeout')
    result=Path(a.result);result.parent.mkdir(parents=True,exist_ok=True)
    result.unlink(missing_ok=True);started=time.time()
    proc=subprocess.Popen(command,start_new_session=True)
    timed_out=False
    try:code=proc.wait(timeout=a.seconds)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        code=124
    partial=Path(str(result)+'.partial')
    partial.write_text(json.dumps(dict(exit_code=code,timed_out=timed_out,elapsed_seconds=time.time()-started)))
    partial.replace(result)
    if timed_out:
        # Best effort independent release if the controller has disconnected.
        # This is NOT a provider-enforced billing cap.
        try:
            from lightning_sdk import Studio
            Studio(create_ok=False).stop()
        except Exception as exc:print(f'Automatic stop failed: {exc}. Stop worker in Lightning UI.',flush=True)
    raise SystemExit(code)

if __name__=='__main__':main()
