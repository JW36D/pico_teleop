"""Receive PICO state through the official C SDK. No robot or send API.

ABI: bundled SDK/include/PXREARobotSDK.h from the SHA256-verified package.
SDK callbacks copy the data before returning, then replace one latest sample.
Local consumers request a snapshot via /state/pico.sock; old poses are never queued.
All interval/age measurements use the Jetson's monotonic clock.
"""
import collections
import ctypes
import json
import math
import os
import signal
import socketserver
import threading
import time
from button_events import ButtonEventRing


class DeviceState(ctypes.Structure):
    _fields_ = [('devID', ctypes.c_char * 32),
                ('stateJson', ctypes.c_char * 16352)]


CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                           ctypes.c_int, ctypes.c_void_p)
SERVER_ON, SERVER_OFF = 1 << 2, 1 << 3
DEVICE_FIND, DEVICE_MISSING, DEVICE_CONNECT = 1 << 4, 1 << 5, 1 << 9
DEVICE_STATE = 1 << 25
SOCKET_PATH = '/state/pico.sock'
lock = threading.Lock()
stop_event = threading.Event()
samples = collections.deque(maxlen=600)
stats = {'frames': 0, 'tracking_frames': 0, 'parse_errors': 0,
         'server_connected': False, 'device_connected': False}
latest = None
last_receive_ns = None
first_sample = None
position_min, position_max, input_min, input_max = {}, {}, {}, {}
gap_max_ms = 0.0
first_raw = None
button_events = ButtonEventRing()
HOST_BOOT_ID = open('/proc/sys/kernel/random/boot_id').read().strip()


def decode_json(value):
    for _ in range(3):
        if isinstance(value, str):
            value = json.loads(value)
        else:
            break
    return value


def normalize_pose(value):
    if isinstance(value, str):
        value = value.split(',')
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        return None
    pose = [float(x) for x in value]
    if not all(math.isfinite(x) for x in pose):
        return None
    return pose


def normalize_tracking(payload):
    result = {'pico_timestamp_ns': payload.get('timeStampNs'),
              'input_mode': payload.get('Input'),
              'app_focus': payload.get('appState', {}).get('focus'), 'head': None,
              'left': None, 'right': None}
    head = decode_json(payload.get('Head', {}))
    if isinstance(head, dict) and head.get('pose') is not None:
        result['head'] = {'pose': normalize_pose(head['pose']),
                          'status': head.get('status')}
    controllers = decode_json(payload.get('Controller', {}))
    if isinstance(controllers, dict):
        for side in ('left', 'right'):
            ctrl = decode_json(controllers.get(side, {}))
            if not isinstance(ctrl, dict) or ctrl.get('pose') is None:
                continue
            result[side] = {key: ctrl.get(key) for key in (
                'trigger', 'grip', 'axisX', 'axisY', 'axisClick', 'menuButton',
                'primaryButton', 'secondaryButton', 'status')}
            result[side]['pose'] = normalize_pose(ctrl['pose'])
    return result


def update_ranges(sample):
    for side in ('head', 'left', 'right'):
        item = sample.get(side)
        if not item:
            continue
        pose = item.get('pose')
        if pose:
            position_min.setdefault(side, pose[:3].copy())
            position_max.setdefault(side, pose[:3].copy())
            for i in range(3):
                position_min[side][i] = min(position_min[side][i], pose[i])
                position_max[side][i] = max(position_max[side][i], pose[i])
        for key in ('trigger', 'grip', 'axisX', 'axisY'):
            val = item.get(key)
            if isinstance(val, (int, float)) and math.isfinite(val):
                field = side + '_' + key
                input_min[field] = min(input_min.get(field, val), val)
                input_max[field] = max(input_max.get(field, val), val)


@CALLBACK
def callback(_context, event_type, status, user_data):
    global latest, last_receive_ns, gap_max_ms, first_sample, first_raw
    try:
        if event_type == DEVICE_STATE and user_data:
            msg = ctypes.cast(user_data, ctypes.POINTER(DeviceState)).contents
            raw = bytes(msg.stateJson)
            dev_id = bytes(msg.devID).decode('utf-8', errors='replace')
            data = json.loads(raw)
            payload = decode_json(data.get('value', data))
            if not isinstance(payload, dict):
                return
            now = time.monotonic_ns()
            is_tracking = any(k in payload for k in ('Head', 'Controller'))
            sample = normalize_tracking(payload) if is_tracking else None
            with lock:
                stats['frames'] += 1
                if sample is not None:
                    if last_receive_ns is not None:
                        gap_max_ms = max(gap_max_ms, (now - last_receive_ns) / 1e6)
                    last_receive_ns = now
                    samples.append(now)
                    sample.update({'receive_monotonic_ns': now,
                                   'device_id': dev_id,
                                   'sequence': stats['tracking_frames'] + 1})
                    button_events.observe(sample)
                    latest = sample
                    if first_sample is None:
                        first_sample = sample
                        first_raw = data
                    stats['tracking_frames'] += 1
                    stats['device_connected'] = True
                    update_ranges(sample)
            return
        with lock:
            if event_type == SERVER_ON:
                stats['server_connected'] = True
            elif event_type == SERVER_OFF:
                button_events.invalidate(time.monotonic_ns(), 'SERVER_OFF')
                stats['server_connected'] = False
                stats['device_connected'] = False
                latest = None
            elif event_type == DEVICE_MISSING:
                button_events.invalidate(time.monotonic_ns(), 'DEVICE_MISSING')
                stats['device_connected'] = False
                latest = None
            elif event_type == DEVICE_CONNECT:
                stats['device_connected'] = True
        print(json.dumps({'event': event_type, 'status': status}), flush=True)
    except Exception as exc:
        with lock:
            stats['parse_errors'] += 1
        print(json.dumps({'callback_error': str(exc)}), flush=True)


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))], 3)


def snapshot():
    now = time.monotonic_ns()
    with lock:
        recent = [t for t in samples if now - t < 2_000_000_000]
        intervals = [(b - a) / 1e6 for a, b in zip(recent, recent[1:])]
        hz = ((len(recent) - 1) * 1e9 / (recent[-1] - recent[0])
              if len(recent) > 1 and recent[-1] > recent[0] else 0.0)
        age = (now - latest['receive_monotonic_ns']) / 1e6 if latest else None
        return {'schema': 'rm_pico_receive_v1', 'read_only': True,
                'button_stream': button_events.summary(),
                'host_boot_id': HOST_BOOT_ID,
                'receiver_monotonic_ns': now, **stats,
                'fresh': bool(latest and age < 300 and stats['server_connected']
                              and stats['device_connected']),
                'age_ms': round(age, 3) if age is not None else None,
                'receive_hz_2s': round(hz, 2),
                'interval_ms_2s': {'p50': percentile(intervals, .5),
                                   'p95': percentile(intervals, .95),
                                   'p99': percentile(intervals, .99)},
                'longest_gap_ms': round(gap_max_ms, 3),
                'position_span_m': {side: [round(b - a, 5) for a, b in
                    zip(position_min[side], position_max[side])]
                    for side in position_min},
                'input_range': {field: [input_min[field], input_max[field]]
                                for field in input_min},
                'latest': latest, 'first_raw': first_raw}


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.request.settimeout(2)
        command = self.rfile.readline(512).strip()
        if command == b'status':
            response = snapshot()
        else:
            try:
                request = json.loads(command)
                if request.get('command') != 'button_events':
                    return
                cursor = request.get('cursor', 0)
                if type(cursor) is not int or cursor < 0:
                    return
                with lock:
                    response = button_events.read(cursor, request.get('receiver_id'))
                response['receiver_monotonic_ns'] = time.monotonic_ns()
            except (ValueError, TypeError):
                return
        self.wfile.write(json.dumps(response, allow_nan=False).encode() + b'\n')


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def main():
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop_event.set())
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    server = Server(SOCKET_PATH, Handler)
    os.chmod(SOCKET_PATH, 0o660)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sdk = ctypes.CDLL('/opt/apps/roboticsservice/SDK/arm64/libPXREARobotSDK.so')
    sdk.PXREAInit.argtypes = [ctypes.c_void_p, CALLBACK, ctypes.c_uint]
    sdk.PXREAInit.restype = ctypes.c_int
    sdk.PXREADeinit.argtypes = []
    sdk.PXREADeinit.restype = ctypes.c_int
    rc = sdk.PXREAInit(None, callback, 0xffffffff)
    if rc != 0:
        raise RuntimeError(f'PXREAInit returned {rc}')
    try:
        while not stop_event.wait(1):
            status = snapshot()
            temporary = '/state/status.json.tmp'
            with open(temporary, 'w') as f:
                json.dump(status, f, allow_nan=False)
            os.replace(temporary, '/state/status.json')
            print(json.dumps({k: status[k] for k in (
                'server_connected', 'device_connected', 'fresh', 'age_ms',
                'tracking_frames', 'parse_errors', 'receive_hz_2s',
                'position_span_m', 'input_range')}), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        os.unlink(SOCKET_PATH)
        sdk.PXREADeinit()


if __name__ == '__main__':
    main()
