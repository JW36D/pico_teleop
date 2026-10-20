"""Query the receive-only PICO monitor from the Jetson host."""
import argparse
import json
from runtime_paths import RECEIVER_SOCKET
import socket
import time

parser = argparse.ArgumentParser()
parser.add_argument('--watch', action='store_true')
parser.add_argument('--socket', default=str(RECEIVER_SOCKET))
args = parser.parse_args()
while True:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(3)
        client.connect(args.socket)
        client.sendall(b'status\n')
        with client.makefile('rb') as stream:
            result = json.loads(stream.readline())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not args.watch:
        break
    time.sleep(1)
