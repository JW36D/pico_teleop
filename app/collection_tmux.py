"""One tmux screen: independent control and read-only collection panes."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import shlex
import subprocess
import sys
import tempfile
import time
import uuid

from runtime_paths import PROJECT_DIR, CONFIG_DIR
from collection_viewer import mode_label, read_json


TASK_DIR = Path(__file__).resolve().parent


def write_json(path, data):
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(data, ensure_ascii=False) + '\n')
    temporary.replace(path)


def wait_ready(path, alive, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if read_json(path).get('ready'):
                return
        except (OSError, ValueError):
            pass
        if not alive():
            raise RuntimeError('分屏进程启动失败，未启动设备控制。')
        time.sleep(.05)
    raise RuntimeError('分屏启动确认超时，未授权启动设备控制。')


def launch_split(manifest, layout='horizontal', call=None):
    directory = Path(manifest['directory'])
    write_json(directory / 'run.json', manifest)
    if call is None:
        def call(arguments):
            result = subprocess.run(['tmux'] + arguments, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or 'tmux命令失败')
            return result.stdout.strip()
    session = manifest['tmux_session']
    created = False
    try:
        def worker(role):
            # User task text stays in run.json, never in shell source.
            return shlex.join([sys.executable, str(Path(__file__).resolve()), '--_worker', role, str(directory)])
        size = shutil.get_terminal_size((180, 40))
        control = call(['new-session', '-d', '-P', '-F', '#{pane_id}', '-s', session, '-n', 'PICO',
                        '-x', str(size.columns), '-y', str(size.lines), '-c', manifest['cwd'], worker('control')])
        created = True
        # Once both workers finish, close the session and return to the shell.
        # Keeping dead panes makes a successful Ctrl+C look unresponsive.
        call(['set-window-option', '-t', session + ':0', 'remain-on-exit', 'off'])
        split = '-h' if layout == 'horizontal' or layout == 'auto' and size.columns >= 150 else '-v'
        monitor = call(['split-window', split, '-p', '50', '-P', '-F', '#{pane_id}',
                        '-t', control, '-c', manifest['cwd'], worker('monitor')])
        call(['set-option', '-t', session, 'mouse', 'on'])
        call(['set-window-option', '-t', session + ':0', 'pane-border-status', 'top'])
        call(['set-window-option', '-t', session + ':0', 'pane-border-format', ' #{pane_title} '])
        call(['select-pane', '-t', control, '-T', '遥操采集：Ctrl+C / B+Y 停止'])
        call(['select-pane', '-t', monitor, '-T', '录制状态与操作记录'])
        for role, pane in (('monitor', monitor), ('control', control)):
            wait_ready(directory / (role + '_ready.json'),
                       lambda pane=pane: call(['display-message', '-p', '-t', pane, '#{pane_dead}']) == '0')
        call(['select-pane', '-t', control])
        write_json(directory / 'start.json', {'start': True})
    except BaseException as exc:
        write_json(directory / 'abort.json', {'error': str(exc)})
        write_json(directory / 'control.json', {'phase': 'failed', 'exit_code': 1, 'error': str(exc)})
        if created and not (directory / 'start.json').exists():
            call(['kill-session', '-t', session])
        raise
    print('已打开遥操窗口和录制状态窗口。', flush=True)
    print('重连命令：tmux attach -t ' + session, flush=True)


def final_result(manifest, code):
    """Runs after the coordinator has stopped devices and drained the writer."""
    result = {'phase': 'finished', 'exit_code': code}
    if manifest['mode'] == 'DEMO_NO_DEVICES':
        result['summary_text'] = '\n界面演示已结束，未采集真实数据。'
        return result
    directory = Path(manifest['directory'])
    try:
        source = read_json(directory / 'status.json')['session_directory']
    except (OSError, ValueError, KeyError):
        result['summary_text'] = '\n本次未生成 raw 记录。请检查遥操窗口中的启动错误。'
        return result
    result['raw_directory'] = source
    try:
        from collection_summary import save_summary
        print('正在整理本次采集结果…', flush=True)
        result['summary_text'] = save_summary(source)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        result['summary_error'] = str(exc)
        result['summary_text'] = ('\n本次采集已停止，汇总生成失败。\nraw 目录：' + source +
            '\n重新查看汇总：' + shlex.join(['bash', str(PROJECT_DIR / 'run_collection.sh'),
                                           '--summary', source]))
    return result


def show_result(directory):
    """A detached client is not a stopped collection session."""
    directory = Path(directory)
    try:
        state = read_json(directory / 'control.json')
    except (OSError, ValueError):
        print('尚未确认采集是否结束，请用重连命令查看。', flush=True)
        return False
    if state.get('phase') not in ('finished', 'failed'):
        print('已离开分屏，采集仍在运行。', flush=True)
        return False
    print(state.get('summary_text') or '采集未能正常完成：' + str(state.get('error', '请检查启动错误')), flush=True)
    return True


def attach_and_report(manifest):
    directory = Path(manifest['directory'])
    session = manifest['tmux_session']
    if os.environ.get('TMUX'):
        # Return the client to its original session so its launch pane can show
        # the result even when invoked inside an existing tmux session.
        client = subprocess.run(['tmux', 'display-message', '-p', '#{client_name}'],
                                capture_output=True, text=True).stdout.strip()
        original = subprocess.run(['tmux', 'display-message', '-p', '#{session_name}'],
                                  capture_output=True, text=True).stdout.strip()
        subprocess.run(['tmux', 'set-option', '-t', session, 'detach-on-destroy', 'off'], check=True)
        attached = subprocess.call(['tmux', 'switch-client', '-t', session])
        if attached:
            return attached
        while True:
            try:
                if read_json(directory / 'control.json').get('phase') in ('finished', 'failed'):
                    break
            except (OSError, ValueError):
                pass
            if subprocess.run(['tmux', 'has-session', '-t', session],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
                break
            time.sleep(.25)
        if client and original:
            subprocess.run(['tmux', 'switch-client', '-c', client, '-t', original],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        attached = subprocess.call(['tmux', 'attach-session', '-t', session])
    finished = show_result(directory)
    if finished:
        state = read_json(directory / 'control.json')
        code = state.get('exit_code', 1)
        # Only discard this launcher's scratch state after a durable raw report
        # exists. No session data or another launcher's directory is removed.
        if state.get('raw_directory') and not state.get('summary_error'):
            try:
                shutil.rmtree(directory)
            except OSError:
                pass
        return 128 - code if code < 0 else code
    return attached


def control_command(manifest):
    directory = Path(manifest['directory'])
    if manifest['mode'] == 'DEMO_NO_DEVICES':
        return [sys.executable, str(Path(__file__).resolve()), '--_demo', str(directory)]
    return [sys.executable, str(TASK_DIR / 'collection.py')] + manifest['arguments'] + [
        '--status-file', str(directory / 'status.json')]


def run_window(role, directory):
    manifest = read_json(directory / 'run.json')
    child = None
    stopping = False

    def interrupted(signum, _frame):
        nonlocal stopping
        stopping = True
        # Only our own child receives this signal. The log window never
        # signals the other terminal, its coordinator or any robot worker.
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    code = 1
    try:
        if role == 'control':
            print('PICO 遥操与采集 | ' + mode_label(manifest['mode']), flush=True)
            print('A：开始/结束录制；X：删除上一条；B+Y 或 Ctrl+C：退出。', flush=True)
            write_json(directory / 'control_ready.json', {'ready': True, 'pid': os.getpid()})
            deadline = time.monotonic() + 15
            while not (directory / 'start.json').is_file():
                if stopping or (directory / 'abort.json').exists() or time.monotonic() > deadline:
                    raise RuntimeError('分屏启动未完成，未连接设备。')
                time.sleep(.05)
            if stopping or (directory / 'abort.json').exists():
                raise RuntimeError('启动已取消，未连接设备。')
            child = subprocess.Popen(control_command(manifest), cwd=manifest['cwd'])
            write_json(directory / 'control.json', {'phase': 'running', 'pid': child.pid})
        else:
            child = subprocess.Popen([sys.executable, str(TASK_DIR / 'collection_viewer.py'),
                '--status', str(directory / 'status.json'), '--control-state', str(directory / 'control.json'),
                '--ready-file', str(directory / 'monitor_ready.json'), '--mode', manifest['mode']],
                cwd=manifest['cwd'])
        code = child.wait()
        if role == 'control':
            write_json(directory / 'control.json', final_result(manifest, code))
    except Exception as exc:
        print('窗口任务失败：' + str(exc), file=sys.stderr, flush=True)
        if role == 'control':
            write_json(directory / 'control.json', {'phase': 'failed', 'exit_code': 1, 'error': str(exc)})
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            # Preserve the controller's existing stop/cleanup path; no kill.
            child.wait()
    label = '遥操采集' if role == 'control' else '录制状态显示'
    print(label + ('已结束。' if code == 0 else '异常结束，退出码 ' + str(code)), flush=True)
    return code


def demo(directory):
    """Synthetic UI smoke run: no receiver, SDK, camera or production data."""
    from raw_episode_writer import atomic_json
    step = read_json(directory / 'run.json').get('demo_step_s', .7)
    root = directory / 'demo_session'
    root.mkdir()
    (root / 'episodes').mkdir()
    (root / '.trash').mkdir()
    session_id = str(uuid.uuid4())
    episode_id = str(uuid.uuid4())
    state = {'session_id': session_id, 'session_directory': str(root), 'recording': False,
             'ready': True, 'episode_id': None, 'episode_number': 0,
             'writer_dropped': 0, 'writer_queue_peak': 0, 'sources': {}, 'mode': 'DEMO_NO_DEVICES'}
    print('仅演示分屏：不连接机器人、不打开相机、不读取PICO。', flush=True)

    def update(**changes):
        state.update(changes, time_monotonic_ns=time.monotonic_ns())
        atomic_json(directory / 'status.json', state)
        time.sleep(step)

    def event(name, **values):
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event=name, receive_monotonic_ns=time.monotonic_ns(), **values), ensure_ascii=False) + '\n')

    update()
    path = root / 'episodes' / episode_id
    path.mkdir()
    info = {'closed': False, 'retained': True, 'deleted': False, 'episode_number': 1, 'synthetic_ui_demo': True}
    atomic_json(path / 'episode.json', info)
    event('record_toggle', episode_id=episode_id, message='正在录制 ' + episode_id[:8])
    update(recording=True, episode_id=episode_id, episode_number=1, started_ns=time.monotonic_ns())
    event('record_toggle', episode_id=episode_id, message='已结束并保留 ' + episode_id[:8])
    update(recording=False, episode_id=None, deletable_episode_id=episode_id, started_ns=None)
    info['closed'] = True
    atomic_json(path / 'episode.json', info)
    atomic_json(path / 'integrity.json', {'closed': True, 'complete': True})
    update()
    event('delete_requested', episode_id=episode_id)
    update()
    info.update(deleted=True, retained=False)
    atomic_json(path / 'episode.json', info)
    path.rename(root / '.trash' / episode_id)
    update(deletable_episode_id=None)
    event('delete_button', message='没有可删除的新记录')
    update()
    update(exiting=True, ready=False, ended_monotonic_ns=time.monotonic_ns())
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '--_worker':
        return run_window(argv[1], Path(argv[2]))
    if argv and argv[0] == '--_demo':
        return demo(Path(argv[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('preview', 'execute', 'demo'):
        modes.add_argument('--' + mode, action='store_true')
    modes.add_argument('--summary', type=Path, metavar='RAW目录', help='查看指定会话的采集结果与转换命令')
    parser.add_argument('--config', type=Path, default=CONFIG_DIR/'collection.json')
    parser.add_argument('--task')
    parser.add_argument('--duration', type=float, default=0)
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--layout', choices=('auto','horizontal','vertical'), default='horizontal')
    parser.add_argument('--detach', action='store_true', help='Create panes without attaching this terminal')
    args = parser.parse_args(argv)
    if args.summary is not None:
        try:
            from collection_summary import save_summary
            print(save_summary(args.summary), flush=True)
            return 0
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            print('无法生成采集汇总：' + str(exc), file=sys.stderr)
            return 1
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error('duration must be finite and >=0')
    if not args.detach and not sys.stdin.isatty():
        parser.error('请在VS Code或SSH终端内运行；无设备演示测试可使用 --demo --detach')
    try:
        if not shutil.which('tmux'):
            raise RuntimeError('需要tmux。Ubuntu安装命令：sudo apt-get install tmux')
        mode = 'DEMO_NO_DEVICES' if args.demo else 'EXECUTE' if args.execute else 'PREVIEW_NO_MOTION'
        arguments = ['--execute' if args.execute else '--preview', '--config', str(args.config.resolve()),
                     '--duration', str(args.duration)]
        if args.task is not None:
            arguments.extend(['--task', args.task])
        if args.no_video:
            arguments.append('--no-video')
        directory = Path(tempfile.mkdtemp(prefix='pico-collection-split-'))
        manifest = dict(directory=str(directory), cwd=os.getcwd(), mode=mode, arguments=arguments,
                        tmux_session='pico-' + uuid.uuid4().hex[:10],
                        demo_step_s=args.duration / 8 if args.demo and args.duration else .7)
        launch_split(manifest, args.layout)
        if not args.detach:
            return attach_and_report(manifest)
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
