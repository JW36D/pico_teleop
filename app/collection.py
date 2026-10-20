"""Collection coordinator: reliable buttons, independent workers, raw episodes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from collection_ipc import IngestServer
from collection_session import ButtonCursor, CollectionSession
from raw_episode_writer import RawEpisodeWriter, atomic_json
from session_stop import SessionStopFlag
from teleop_config import load_config
from runtime_paths import CONFIG_DIR

TASK_DIR = Path(__file__).resolve().parent
CAMERA_SERVICE = TASK_DIR/'collection_camera_service.py'
from collection_camera_service import load_camera_config


def request_receiver(path, command='status'):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
        client.settimeout(.1)
        client.connect(str(path))
        client.sendall((command if isinstance(command,str) else json.dumps(command)).encode()+b'\n')
        with client.makefile('rb') as stream:
            return json.loads(stream.readline(1024*1024))


def load_collection_config(path):
    path = path.resolve()
    config = json.loads(path.read_text())
    for name in ('data_root','camera_config','left_config','right_config','receiver_socket'):
        value = Path(config[name])
        config[name] = str(value if value.is_absolute() else (path.parent/value).resolve())
    for name in ('chord_window_ms','max_event_age_ms','source_stale_ms','min_free_gb'):
        if type(config.get(name)) not in (int,float) or not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError('invalid collection '+name)
    return config


def clock_check(status):
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    # The request/response bracket bounds a same-host monotonic timestamp.
    return {'host_boot_id':boot, 'receiver_boot_id':status.get('host_boot_id'),
            'verified_same_boot':status.get('host_boot_id') == boot,
            'axis':'host CLOCK_MONOTONIC nanoseconds', 'camera_alignment':'host receive, not exposure',
            'wall_time_role':'date/log lookup only'}


def preflight(config, require_camera=True, require_headset=True):
    checks = []
    def check(name, work):
        try:
            detail = work()
            checks.append({'name':name,'ok':True,'detail':detail})
            return detail
        except Exception as exc:
            checks.append({'name':name,'ok':False,'error':str(exc)})
    for side in ('left','right'):
        def arm_check(side=side):
            arm = load_config(Path(config[side+'_config']))
            if arm['arm'] != side:
                raise ValueError('config arm mismatch')
            return {'arm':side,'follow_mode':arm['follow_mode']}
        check(side+'_config',arm_check)
    def storage():
        path = Path(config['data_root'])
        while not path.exists():
            path = path.parent
        free = shutil.disk_usage(path).free/1e9
        if free < config['min_free_gb']:
            raise RuntimeError('insufficient free disk: %.2fGB'%free)
        if not os.access(path,os.W_OK):
            raise RuntimeError('data parent is not writable')
        return {'free_gb':round(free,2),'root':config['data_root']}
    check('storage',storage)
    def receiver():
        before = time.monotonic_ns()
        result = request_receiver(config['receiver_socket'])
        after = time.monotonic_ns()
        if not result.get('button_stream'):
            raise RuntimeError('receiver must be updated to pico_button_events_v1')
        event_reply = request_receiver(config['receiver_socket'],{'command':'button_events','cursor':0})
        if 'event_watermark_ns' not in event_reply:
            raise RuntimeError('receiver lacks boundary watermarks; rebuild collection receiver')
        if not before <= result['receiver_monotonic_ns'] <= after:
            raise RuntimeError('receiver CLOCK_MONOTONIC outside host request bracket')
        clocks = clock_check(result)
        if not clocks['verified_same_boot']:
            raise RuntimeError('receiver/host boot identity differs')
        if result['button_stream']['chord_window_ms'] != config['chord_window_ms']:
            raise RuntimeError('receiver chord window differs from collection config')
        return dict(clocks, fresh=result['fresh'],button_stream=result['button_stream'])
    check('receiver_clock_and_events',receiver)
    def camera_environment():
        result = subprocess.run([config['camera_python'],'-c',
            'import cv2, numpy, pyrealsense2'+('; import gi' if config.get('video',{}).get('enabled') else '')+
            '; print("camera dependencies available")'],
            capture_output=True,text=True,timeout=10)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()
    check('camera_environment',camera_environment)
    if require_camera:
        def camera_check():
            cameras = load_camera_config(config['camera_config'])
            result = subprocess.run([config['camera_python'],str(CAMERA_SERVICE),'--config',config['camera_config'],'--enumerate'],
                capture_output=True,text=True,timeout=10,check=True)
            devices = json.loads(result.stdout)
            found = {d['serial_number'] for d in devices}
            missing = {v['serial'] for v in cameras['cameras'].values()}-found
            if missing:
                raise RuntimeError('missing cameras: '+str(missing))
            return {'mapping':cameras['cameras'],'devices':devices}
        check('camera_roles_and_devices',camera_check)
    if require_headset:
        def headset():
            h = config['headset']
            if not h.get('switch_with_a_disabled_verified') or not h.get('mapping_verified') or not h.get('apk_version'):
                raise RuntimeError('现场 APK 版本、A不切Send、ABXY映射尚未确认；可先运行 --buttons')
            return h
        check('headset_operator_verification',headset)
    return {'ok':all(c['ok'] for c in checks),'checks':checks,'motion_commands':0}


def run_buttons(config, duration):
    cursor = ButtonCursor()
    deadline = time.monotonic()+duration if duration else float('inf')
    print('无运动按键查看器：右primary=A，左primary=X，右secondary=B，左secondary=Y；Ctrl+C退出',flush=True)
    next_health=0
    try:
        while time.monotonic() < deadline:
            try:
                reply = request_receiver(config['receiver_socket'],cursor.request())
            except (OSError,ValueError) as exc:
                print(json.dumps({'event':'receiver_unavailable','error':str(exc)}),flush=True)
                cursor=ButtonCursor()
                time.sleep(.2)
                continue
            events, issue = cursor.consume(reply)
            if issue:
                print(json.dumps({'event':issue,'stream':{k:v for k,v in reply.items() if k!='events'}},ensure_ascii=False),flush=True)
            for event in events:
                print(json.dumps(event,ensure_ascii=False),flush=True)
            if time.monotonic()>=next_health:
                status=request_receiver(config['receiver_socket'])
                print(json.dumps({'event':'input_health','time_monotonic_ns':time.monotonic_ns(),
                    'fresh':status.get('fresh'),'age_ms':status.get('age_ms'),
                    'hz':status.get('receive_hz_2s'),'sequence':(status.get('latest')or{}).get('sequence')}),flush=True)
                next_health=time.monotonic()+1
            time.sleep(.01)
    except KeyboardInterrupt:
        pass
    return 0


def run_collection(config, config_path, execute, task, duration, status_path=None):
    metadata = {'session_id':str(uuid.uuid4()), 'mode':'EXECUTE' if execute else 'PREVIEW_NO_MOTION',
        'required_sources':['left','right','front','left_wrist','right_wrist'],
        'boundary_watermarks':True,
        'task':task,'collection_config':config, 'camera_config':json.loads(Path(config['camera_config']).read_text()),
        'clock':clock_check(request_receiver(config['receiver_socket'])),
        'hostname':platform.node(), 'platform':platform.platform(),
        'source_hashes':{str(p.relative_to(TASK_DIR.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in TASK_DIR.glob('*.py')},
        'task_success_semantics':'null until independently annotated',
        'units':{'raw_joints':'deg','export_joints':'rad','gripper':'controller opening / 1000, not mm'}}
    writer = RawEpisodeWriter(config['data_root'],metadata,**config['writer'])
    session_id = writer.session_id
    workers = []
    source_lock = threading.Lock()
    sources = {}
    failure = []
    outcome = 0
    with tempfile.TemporaryDirectory(prefix='pico-collect-') as runtime:
        runtime = Path(runtime)
        status_path = Path(status_path) if status_path is not None else runtime/'status.json'
        shared = SessionStopFlag(runtime/'stop',create=True)
        def stop_workers():
            shared.set()
            for name,worker in workers:
                if name in ('left','right') and worker.poll() is None:
                    try:
                        worker.terminate()
                    except ProcessLookupError:
                        pass
        session = CollectionSession(writer,task,stop_workers,max_event_age_ms=config['max_event_age_ms'])
        def accept(record,blob):
            if record.get('session_id') != session_id or record.get('source') not in ('left','right','front','left_wrist','right_wrist','display'):
                return
            source = record['source']
            now = time.monotonic_ns()
            with source_lock:
                previous = sources.get(source,{})
                dropped = record.get('transport_dropped',0)
                if dropped > previous.get('transport_dropped',0) and source!='display':
                    failure.append(source+':transport_dropped='+str(dropped))
                state = dict(previous,transport_dropped=dropped,heartbeat_ns=now)
                state['publisher_queue_peak'] = max(previous.get('publisher_queue_peak',0),
                    record.get('publisher_queue_peak',0))
                if record['kind'] in ('telemetry','frame'):
                    state['latest_ns'] = record['receive_monotonic_ns']
                    state['received_records'] = previous.get('received_records',0)+1
                    state['source_sequence'] = record.get('source_sequence')
                    state.setdefault('first_receive_ns',record['receive_monotonic_ns'])
                    if record['kind']=='telemetry':
                        state['control_interval_ms'] = record.get('control_interval_ms')
                        state['send_count'] = record.get('send_count')
                        state['feedback_age_ms'] = record.get('robot_feedback_age_ms')
                    else:
                        state['frame_number'] = record.get('frame_number')
                        state['frame_number_gaps'] = previous.get('frame_number_gaps',0)+record.get('frame_number_gap',0)
                if record['kind'] == 'metadata':
                    state['metadata'] = record['metadata']
                    writer.update_metadata(source,record['metadata'])
                if record['kind'] == 'failure':
                    if source!='display':
                        failure.append(source+':'+record['error'])
                    state['error']=record['error']
                    writer.event(dict(record,event='source_failure'))
                if record['kind'] == 'source_closed':
                    state['closed'] = True
                    state['final_sequence'] = record['last_source_sequence']
                    writer.update_metadata(source+'_transport_final',record)
                    if source!='display' and (record.get('undelivered') or dropped):
                        writer.fail(source+':transport_not_fully_delivered')
                if record['kind'] == 'display_status':
                    if state.get('display_status') != record['status']:
                        writer.event(dict(record,event='display_status'))
                    state['display_status'] = record['status']
                sources[source] = state
            if record['kind'] in ('telemetry','frame'):
                writer.submit(record,blob)
        server = IngestServer(runtime/'ingest.sock',accept)
        cursor = ButtonCursor()
        cursor.consume(request_receiver(config['receiver_socket'],cursor.request()))
        handlers = {}
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            handlers[sig] = signal.signal(sig,lambda *_:session.exit(reason='SIGNAL'))
        print('SESSION '+session_id+' '+str(writer.directory),flush=True)
        print('实时状态 '+str(status_path),flush=True)
        try:
            video_config_path=runtime/'video.json'
            atomic_json(video_config_path,{'video':config.get('video',{})})
            workers.append(('cameras',subprocess.Popen([config['camera_python'],str(CAMERA_SERVICE),
                '--config',config['camera_config'],'--ipc',str(runtime/'ingest.sock'),'--session-id',session_id,
                '--video-config',str(video_config_path),'--status-path',str(status_path)])))
            for side in ('left','right'):
                workers.append((side,subprocess.Popen([sys.executable,str(TASK_DIR/'full_teleop.py'),
                    '--config',config[side+'_config'],'--execute' if execute else '--preview',
                    '--operator-reference',str(runtime/'operator.json'),'--session-stop',str(runtime/'stop'),
                    '--collection-socket',str(runtime/'ingest.sock'),'--session-id',session_id])))
            start = time.monotonic()
            last_message = None
            last_problems = None
            last_display = None
            next_status = 0
            while not session.exiting:
                now = time.monotonic_ns()
                dead = [(name,w.poll()) for name,w in workers if w.poll() is not None]
                if any(name in ('left','right') for name,_ in dead):
                    session.exit(reason='CONTROL_WORKER_EXIT')
                    outcome = 1
                    break
                with source_lock:
                    snapshot = {s:v.copy() for s,v in sources.items()}
                    problems, failure[:] = list(failure), []
                if any(name=='cameras' for name,_ in dead):
                    problems.append('CAMERA_WORKER_EXIT')
                now=time.monotonic_ns()
                fresh = all(s in snapshot and 0<=now-snapshot[s].get('latest_ns',0) <= config['source_stale_ms']*1e6
                    and 'metadata' in snapshot[s] for s in ('left','right','front','left_wrist','right_wrist'))
                ready = fresh and not problems and writer.ready
                if problems and problems!=last_problems:
                    print('采集故障：'+';'.join(problems),flush=True)
                    last_problems=problems
                if config.get('video',{}).get('enabled'):
                    video_status=snapshot.get('display',{}).get('display_status',{})
                    display_state=(video_status.get('connected',False),video_status.get('error'))
                    if display_state!=last_display:
                        print('头显视频：'+('已连接' if display_state[0] else
                            '连接失败，请重新打开头显 Listen：'+str(display_state[1]) if display_state[1] else
                            '等待头显 Listen'),flush=True)
                        last_display=display_state
                if session.active_episode_id and not ready:
                    reason = ';'.join(problems) or writer.error or 'SOURCE_STALE'
                    writer.fail(reason)
                    session.interrupt(reason,failed=True)
                try:
                    reply = request_receiver(config['receiver_socket'],cursor.request())
                    events, issue = cursor.consume(reply)
                    if issue:
                        session.interrupt(issue)
                    session.handle_batch(events,ready,'采集源缺失/过期或写入故障')
                    writer.advance_boundary_watermark(reply.get('event_watermark_ns'))
                    status = request_receiver(config['receiver_socket'])
                    stream = status['button_stream']
                    input_fresh = status.get('fresh') and status.get('age_ms',float('inf')) <= config['max_event_age_ms']
                    if session.active_episode_id and (not input_fresh or not all(stream['valid'].values())):
                        session.interrupt('PICO_INPUT_DISCONTINUITY')
                except (OSError,ValueError,KeyError) as exc:
                    if session.active_episode_id:
                        session.interrupt('PICO_DISCONNECTED:'+str(exc))
                while True:
                    try:
                        notice = writer.notices.get_nowait()
                    except queue.Empty:
                        break
                    print(json.dumps(notice,ensure_ascii=False),flush=True)
                if writer.error and session.active_episode_id:
                    session.interrupt(writer.error,failed=True)
                if time.monotonic() >= next_status:
                    atomic_json(status_path,dict(session.status(),ready=ready,writer_error=writer.error,
                        session_id=session_id,session_directory=str(writer.directory),
                        writer_queue_peak=writer.queue_peak,writer_dropped=writer.dropped,
                        sources=snapshot,source_problems=problems,time_monotonic_ns=now))
                    next_status = time.monotonic()+.1
                if session.message != last_message:
                    print(session.message,flush=True)
                    last_message = session.message
                if duration and time.monotonic()-start >= duration:
                    session.exit(reason='DURATION_ELAPSED')
                time.sleep(.01)
        except BaseException:
            session.exit(reason='COORDINATOR_EXCEPTION')
            raise
        finally:
            stop_workers()
            # Signal both controllers first; wait for their existing stop paths
            # before shutting down camera encoding or the raw writer.
            for name,worker in workers:
                if name in ('left','right'):
                    try:
                        if worker.wait(timeout=12) != 0:
                            outcome = 1
                    except subprocess.TimeoutExpired:
                        outcome = 1
                        print(name+' controller cleanup still pending; not force-killed',file=sys.stderr)
            for name,worker in workers:
                if name=='cameras':
                    if worker.poll() is None:
                        worker.terminate()
                    try:
                        worker.wait(timeout=12)
                    except subprocess.TimeoutExpired:
                        outcome = 1
            server.close()
            try:
                writer.close()
            except RuntimeError as exc:
                print('Raw writer: '+str(exc),file=sys.stderr)
                outcome = 1
            summary = dict(session.status(),session_id=session_id,
                mode='EXECUTE' if execute else 'PREVIEW_NO_MOTION',sources=sources,
                writer_queue_peak=writer.queue_peak,writer_dropped=writer.dropped,writer_error=writer.error,
                outcome=outcome,ended_monotonic_ns=time.monotonic_ns(),session_directory=str(writer.directory))
            atomic_json(writer.directory/'runtime_summary.json',summary)
            atomic_json(status_path,dict(summary,ready=False,time_monotonic_ns=time.monotonic_ns()))
            shared.close()
            for sig,handler in handlers.items():
                signal.signal(sig,handler)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG_DIR/'collection.json')
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('check','buttons','identify-cameras','preview','execute'):
        modes.add_argument('--'+name,action='store_true')
    parser.add_argument('--task')
    parser.add_argument('--duration',type=float,default=0)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--no-video',action='store_true',help='Disable auxiliary headset video; retain all three recorded cameras')
    parser.add_argument('--status-file',type=Path,help='Publish the existing live snapshot at this path for a read-only viewer')
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error('duration must be finite and >=0')
    if args.status_file:
        if not (args.preview or args.execute):
            parser.error('--status-file requires --preview or --execute')
        args.status_file=args.status_file.resolve()
        if args.status_file.exists():
            parser.error('status file exists; choose a new path')
        args.status_file.parent.mkdir(parents=True,exist_ok=True)
    config = load_collection_config(args.config)
    if args.no_video:
        config.setdefault('video',{})['enabled']=False
    if args.identify_cameras:
        command = [config['camera_python'],str(CAMERA_SERVICE),'--config',config['camera_config'],'--identify']
        if args.output:
            command.extend(['--output',str(args.output)])
        return subprocess.call(command)
    if args.buttons:
        return run_buttons(config,args.duration)
    result = preflight(config)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if args.check:
        return 0 if result['ok'] else 1
    if not result['ok']:
        return 1
    task = args.task or config.get('task')
    if not isinstance(task,str) or not task.strip():
        parser.error('--task or nonempty config.task is required')
    return run_collection(config,args.config.resolve(),args.execute,task,args.duration,status_path=args.status_file)


if __name__=='__main__':
    raise SystemExit(main())
