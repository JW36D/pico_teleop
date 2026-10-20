"""Supervise the official XR service and a receive-only SDK client."""
import os
import signal
import subprocess
import time

stopping = False


def stop(_signum, _frame):
    global stopping
    stopping = True


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
os.makedirs('/state/runtime', exist_ok=True)
os.chmod('/state/runtime', 0o700)
os.environ['XDG_RUNTIME_DIR'] = '/state/runtime'
os.environ['HOME'] = '/state'
service_log = open('/state/service.log', 'a', buffering=1)
service = subprocess.Popen(['./RoboticsServiceProcess'], stdout=service_log,
                           stderr=subprocess.STDOUT)
receiver = subprocess.Popen(['python3', '/app/receiver.py'])
exit_code = 0
try:
    while not stopping:
        for name, child in [('service', service), ('receiver', receiver)]:
            if child.poll() is not None:
                print(f'{name} exited with {child.returncode}', flush=True)
                exit_code = 1
                stopping = True
        time.sleep(0.2)
finally:
    for child in (receiver, service):
        if child.poll() is None:
            child.terminate()
    for child in (receiver, service):
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    service_log.close()
raise SystemExit(exit_code)
