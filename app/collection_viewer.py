"""Read collection snapshots/events; never connect to devices or send commands."""
import argparse
from collections import OrderedDict, deque
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import uuid


READ_LIMIT = 256 * 1024
LOCAL_ZONE = timezone(timedelta(hours=8))


def read_json(path):
    with Path(path).open('rb') as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError('status record too large')
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError('expected a JSON object')
    return result


def clean(value):
    return ''.join(c if c.isprintable() else ' ' for c in str(value))


def mode_label(mode):
    return {'EXECUTE': '正式采集（可控制机械臂）',
            'PREVIEW_NO_MOTION': '采集预览（机械臂不动作）',
            'DEMO_NO_DEVICES': '界面演示（不连接设备）'}.get(mode, '等待启动')


def reason_label(reason):
    return {'SOURCE_STALE': '相机或机械臂数据中断，请检查设备连接',
            'PICO_INPUT_DISCONTINUITY': '手柄数据中断，请检查头显追踪和 Send 是否开启',
            'PICO_FOCUS_OR_BUTTON_VALIDITY_LOST': '手柄输入不可用，请将 XRoboToolkit 切回前台',
            'EVENT_OVERFLOW': '手柄按键数据接收异常，请检查连接',
            'CONNECTION_CHANGED': '头显连接已变化，请确认 Send 已开启',
            'EVENT_BASELINE': '已连接手柄',
            'CONTROL_WORKER_EXIT': '遥操已停止',
            'CAMERA_WORKER_EXIT': '相机已停止，请检查相机连接',
            'B_Y_CHORD': '已按 B+Y 退出',
            'SIGNAL': '已收到停止指令',
            'DURATION_ELAPSED': '已到设定结束时间',
            'COORDINATOR_EXCEPTION': '采集程序异常，请查看遥操窗口的错误提示'}.get(
                str(reason), '头显连接中断，请检查 Send 是否开启' if str(reason).startswith(
                    'PICO_DISCONNECTED:') else str(reason))


class CollectionObserver:
    def __init__(self, status_path, control_path=None):
        self.status_path = Path(status_path)
        self.control_path = Path(control_path) if control_path else None
        self.state = {}
        self.control = {}
        self.problem = None
        self.directory = None
        self.offset = 0
        self.inode = None
        self.partial = b''
        self.discard_line = False
        self.logs = deque(maxlen=40)
        self.pending = OrderedDict()
        self.wall_offset = time.time() - time.monotonic()
        self.backlog = False

    def notice(self, text, stamp=None):
        seconds = self.wall_offset + stamp / 1e9 if type(stamp) is int else time.time()
        label = datetime.fromtimestamp(seconds, LOCAL_ZONE).strftime('%H:%M:%S')
        self.logs.append(label + '  ' + clean(text))

    def watch(self, episode_id, operation):
        try:
            if str(uuid.UUID(episode_id)) != episode_id:
                return
        except (ValueError, TypeError, AttributeError):
            return
        self.pending[episode_id] = operation
        if len(self.pending) > 128:
            self.pending.popitem(last=False)
            self.notice('待确认的记录较多，请检查数据是否已保存')

    def event(self, row):
        name = row.get('event')
        message = row.get('message', '')
        episode = row.get('episode_id')
        stamp = row.get('receive_monotonic_ns')
        if name == 'record_toggle':
            if message.startswith('正在录制 '):
                self.watch(episode, 'save')
                message = '开始录制'
            elif message.startswith('已结束并保留 '):
                self.watch(episode, 'save')
                message = '结束录制，正在保存'
            self.notice(message, stamp)
        elif name == 'delete_requested':
            self.watch(episode, 'delete')
            self.notice('正在删除上一条记录', stamp)
        elif name == 'delete_button':
            if not message.startswith('已移入回收目录 '):
                self.notice('请先按 A 结束录制，再按 X 删除上一条' if message == '正在录制，X无效' else message, stamp)
        elif name == 'recording_interrupted':
            self.watch(episode, 'save')
            self.notice(('录制中断：' if episode else '') + reason_label(row.get('reason')), stamp)
        elif name == 'session_exit':
            self.notice('采集结束：' + reason_label(row.get('reason')), stamp)
        elif name == 'source_failure':
            self.notice('采集源异常：' + str(row.get('source')) + ' ' + str(row.get('error')), stamp)

    def read_events(self):
        if self.directory is None:
            return
        try:
            with (self.directory / 'events.jsonl').open('rb') as stream:
                stat = (self.directory / 'events.jsonl').stat()
                if self.inode != stat.st_ino or stat.st_size < self.offset:
                    if self.inode is not None:
                        self.notice('操作记录已重新加载')
                    self.offset = max(0, stat.st_size - READ_LIMIT)
                    self.partial = b''
                    self.discard_line = self.offset > 0
                    if self.offset:
                        self.notice('已加载最近操作记录，完整历史已保留')
                self.inode = stat.st_ino
                stream.seek(self.offset)
                chunk = stream.read(READ_LIMIT)
                self.offset += len(chunk)
                self.backlog = self.offset < stat.st_size
        except FileNotFoundError:
            return
        if self.discard_line:
            if b'\n' not in chunk:
                return
            chunk = chunk.split(b'\n', 1)[1]
            self.discard_line = False
        lines = (self.partial + chunk).split(b'\n')
        self.partial = lines.pop()
        if len(self.partial) > READ_LIMIT:
            self.partial = b''
            self.discard_line = True
            self.notice('有一条操作记录过长，无法显示')
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('not an object')
                self.event(row)
            except (ValueError, TypeError, AttributeError):
                self.notice('有一条操作记录无法显示')

    def confirm_files(self):
        if self.directory is None:
            return
        for episode, operation in list(self.pending.items()):
            for deleted, folder in ((True, '.trash'), (False, 'episodes')):
                path = self.directory / folder / episode
                try:
                    info = read_json(path / 'episode.json')
                    integrity = read_json(path / 'integrity.json')
                except (OSError, ValueError):
                    continue
                if not info.get('closed') or not integrity.get('closed'):
                    continue
                if deleted and info.get('deleted') and not info.get('retained'):
                    result = '删除完成'
                elif operation == 'delete' or info.get('deleted'):
                    continue
                elif integrity.get('complete'):
                    result = '保存完成'
                else:
                    result = '已保留不完整记录，请检查原因'
                self.notice(result + '（第%s条）' % info.get('episode_number', '?'))
                del self.pending[episode]
                break

    def poll(self, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        try:
            state = read_json(self.status_path)
            if type(state.get('time_monotonic_ns')) is not int or type(state.get('recording')) is not bool:
                raise ValueError('missing collection state')
            directory = Path(state['session_directory'])
            if directory != self.directory:
                self.directory = directory
                self.offset = 0
                self.inode = None
                self.partial = b''
                self.discard_line = False
                self.pending.clear()
                self.notice('采集程序已连接')
                self.watch(state.get('episode_id') or state.get('deletable_episode_id'), 'save')
            self.state = state
            self.problem = None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.problem = str(exc)
        if self.control_path:
            try:
                self.control = read_json(self.control_path)
            except (OSError, ValueError):
                pass
        try:
            self.read_events()
            self.confirm_files()
        except (OSError, ValueError) as exc:
            self.problem = '操作记录读取失败：' + str(exc)
        age = (now_ns - self.state.get('time_monotonic_ns', 0)) / 1e9
        finished = self.control.get('phase') in ('finished', 'failed') or 'ended_monotonic_ns' in self.state
        stale = bool(self.problem) or not 0 <= age <= 2
        if finished:
            headline = '会话已结束' if not self.control.get('exit_code') else '采集启动失败或异常结束'
        elif not self.state:
            headline = '等待遥操采集启动'
        elif stale:
            headline = '状态不可用或已过期，无法确认当前录制状态'
        elif self.state.get('exiting'):
            headline = '正在停止，等待后台收尾'
        elif self.state['recording']:
            headline = '正在录制'
        elif self.state.get('ready'):
            headline = '未录制，可以按 A 开始'
        else:
            headline = '未录制，等待相机和机械臂连接'
        return {'headline': headline, 'stale': stale, 'finished': finished, 'age_s': age}

    def render(self, health, mode='', height=32):
        s = self.state
        elapsed = 0.
        if not health['stale'] and s.get('recording') and type(s.get('started_ns')) is int:
            elapsed = max(0., (s['time_monotonic_ns'] - s['started_ns']) / 1e9)
        display = s.get('sources', {}).get('display', {}).get('display_status', {})
        lines = ['PICO 遥操采集', '',
                 '>>> ' + health['headline'] + ' <<<',
                 '模式：' + mode_label(mode or s.get('mode')),
                 '本次已开始录制：%s 条' % s.get('episode_number', 0)]
        if s.get('recording') and not health['stale'] and not health['finished']:
            lines.append('当前第%s条    已录制：%.1f 秒' % (s.get('episode_number', '?'), elapsed))
        lines.append('头显画面：' + ('连接状态待确认' if health['stale'] or health['finished'] else
                     '已连接' if display.get('connected') else '未连接（可在头显中开启 Listen）'))
        if s.get('writer_error') or s.get('writer_dropped'):
            lines.append('数据保存异常，请停止录制并查看遥操窗口的错误提示')
        if s.get('source_problems'):
            lines.append('设备数据异常，请检查相机和机械臂连接')
        if health['stale'] and self.state and not health['finished']:
            lines.append('状态已 %.1f 秒未更新；下方仅为此前的操作记录' % health['age_s'])
        if self.control.get('error'):
            lines.append('启动/退出信息：' + clean(self.control['error']))
        lines.extend(['', '最近操作：'])
        if self.backlog:
            lines.append('正在加载操作记录…')
        count = max(3, height - len(lines) - 4)
        lines.extend(list(self.logs)[-count:] or ['尚无操作记录'])
        lines.extend(['', 'A：开始/结束录制    X：删除上一条（先结束录制）',
                      '退出：同时按 B+Y，或在遥操窗口按 Ctrl+C'])
        return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--control-state', type=Path)
    parser.add_argument('--ready-file', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--mode', default='')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    observer = CollectionObserver(args.status, args.control_state)
    terminal = sys.stdout.isatty() and not args.once
    try:
        if terminal:
            print('\033[?1049h\033[?25l', end='', flush=True)
        first = True
        while True:
            health = observer.poll()
            text = observer.render(health, args.mode, shutil.get_terminal_size((110, 32)).lines)
            print(('\033[H\033[2J' if terminal else '') + text, flush=True)
            if first and args.ready_file:
                args.ready_file.write_text(json.dumps({'ready': True}) + '\n')
            first = False
            if args.once or health['finished']:
                return 0
            time.sleep(.5)
    except KeyboardInterrupt:
        return 0
    finally:
        if terminal:
            print('\033[?25h\033[?1049l', end='', flush=True)
            print(observer.render(observer.poll(), args.mode), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
