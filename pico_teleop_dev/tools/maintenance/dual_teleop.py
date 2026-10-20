"""Supervise two independent copies of the shared single-arm controller.

No robot SDK, kinematics or sender is imported here. Each child owns exactly
one arm and its own SDK runtime. Ending either child ends the other through
the existing signal/stop path; releasing a Grip only stops that arm.
"""
import argparse
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
import tempfile

from teleop_config import load_config
from session_stop import SessionStopFlag
from button_events import ExitGuard

from runtime_paths import APP_DIR, CONFIG_DIR

TASK_DIR=APP_DIR


def supervise(configs, execute=False, duration=0., popen=None, sleep=None):
    with tempfile.TemporaryDirectory(prefix='pico-dual-reference-') as directory:
        return _supervise(configs,execute,duration,popen,sleep,Path(directory)/'operator.json')


def _supervise(configs, execute, duration, popen, sleep, reference_file):
    real_processes = popen is None
    popen=subprocess.Popen if popen is None else popen
    sleep=time.sleep if sleep is None else sleep
    processes=[]
    handlers={}
    stop_requested=False
    outcome=0
    shared_stop = SessionStopFlag(reference_file.parent/'stop',create=True)
    exit_guard = ExitGuard()
    if real_processes:
        from robot_backend import read_pico
        try:exit_guard.observe(read_pico())
        except OSError:pass

    def interrupted(_signum,_frame):
        nonlocal stop_requested
        stop_requested=True
        shared_stop.set()

    try:
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            handlers[sig]=signal.signal(sig,interrupted)
        for side in ('left','right'):
            if stop_requested:break
            command=[sys.executable,str(TASK_DIR/'full_teleop.py'),
                '--config',str(configs[side]),'--execute' if execute else '--preview',
                '--duration',str(duration),'--operator-reference',str(reference_file),
                '--session-stop',str(reference_file.parent/'stop')]
            processes.append((side,popen(command,cwd=str(TASK_DIR))))
        while processes and not stop_requested:
            if real_processes:
                try:
                    if exit_guard.observe(read_pico()):shared_stop.set()
                except OSError:pass
            if shared_stop.is_set():break
            exited=[(side,p.poll()) for side,p in processes if p.poll() is not None]
            if exited:
                if any(code!=0 for _,code in exited):outcome=1
                break
            sleep(.02)
    finally:
        shared_stop.set()
        # Signal both workers before waiting. Do not force-kill a process
        # which may still be confirming its software stop.
        for side,p in processes:
            if p.poll() is None:
                try:p.terminate()
                except ProcessLookupError:pass
        for side,p in processes:
            try:
                code=p.wait(timeout=12.)
                if code!=0:outcome=1
            except subprocess.TimeoutExpired:
                outcome=1
                print(side+' worker is still confirming cleanup; it was not force-killed.',file=sys.stderr)
        for sig,handler in handlers.items():signal.signal(sig,handler)
        shared_stop.close()
    return outcome


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--preview',action='store_true')
    modes.add_argument('--execute',action='store_true')
    parser.add_argument('--duration',type=float,default=0.)
    parser.add_argument('--left-config',type=Path,default=CONFIG_DIR/'left_rotation_teleop.json')
    parser.add_argument('--right-config',type=Path,default=CONFIG_DIR/'right_rotation_teleop.json')
    args=parser.parse_args()
    if not math.isfinite(args.duration)or not(args.duration==0 or 1<=args.duration<=3600):
        parser.error('Duration must be 0, or 1..3600 seconds')
    configs={'left':args.left_config.resolve(),'right':args.right_config.resolve()}
    for side,path in configs.items():
        try:config=load_config(path)
        except (OSError,ValueError,RuntimeError)as exc:parser.error(str(exc))
        if config['arm']!=side:parser.error(side+' config selects the wrong arm')
    print(('EXECUTE' if args.execute else 'PREVIEW_NO_MOTION')+
        ': 左手控制左臂，右手控制右臂；B+Y（200ms内）或Ctrl+C结束整个会话。',flush=True)
    return supervise(configs,args.execute,args.duration)


if __name__=='__main__':raise SystemExit(main())
