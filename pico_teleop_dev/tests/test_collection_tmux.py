"""Observer truthfulness and terminal isolation; no devices are opened."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collection_viewer
import collection_tmux


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'episodes').mkdir()
        (self.root / '.trash').mkdir()
        self.status = self.root / 'live.json'
        self.episode = str(uuid.uuid4())
        self.state = dict(recording=False, ready=True, session_directory=str(self.root),
                          session_id=str(uuid.uuid4()), episode_number=1, episode_id=None)
        self.publish()
        self.observer = collection_viewer.CollectionObserver(self.status)

    def publish(self, **values):
        self.state.update(values, time_monotonic_ns=time.monotonic_ns())
        self.status.write_text(json.dumps(self.state))

    def event(self, name, **values):
        with (self.root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event=name, receive_monotonic_ns=time.monotonic_ns(), **values), ensure_ascii=False) + '\n')

    def finished_files(self):
        folder = self.root / 'episodes' / self.episode
        folder.mkdir()
        info = dict(closed=True, retained=True, deleted=False, episode_number=1)
        (folder / 'episode.json').write_text(json.dumps(info))
        (folder / 'integrity.json').write_text(json.dumps(dict(closed=True, complete=True)))
        return folder, info

    def test_fast_start_end_are_both_visible_and_observer_does_not_write_source(self):
        self.event('record_toggle', episode_id=self.episode, message='正在录制 ' + self.episode[:8])
        self.event('record_toggle', episode_id=self.episode, message='已结束并保留 ' + self.episode[:8])
        self.finished_files()
        def hashes():
            return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()}
        before = hashes()
        self.observer.poll()
        output = '\n'.join(self.observer.logs)
        for word in ('开始录制', '结束录制', '保存完成'):
            self.assertIn(word, output)
        self.assertEqual(before, hashes())

    def test_delete_is_confirmed_only_after_completed_move_to_trash(self):
        folder, info = self.finished_files()
        self.event('delete_requested', episode_id=self.episode)
        self.observer.poll()
        self.assertNotIn('删除完成', '\n'.join(self.observer.logs))
        info.update(deleted=True, retained=False)
        (folder / 'episode.json').write_text(json.dumps(info))
        folder.rename(self.root / '.trash' / self.episode)
        self.observer.poll()
        self.assertIn('删除完成', '\n'.join(self.observer.logs))

    def test_partial_utf8_line_is_buffered_and_repeated_noops_are_visible(self):
        line = json.dumps(dict(event='delete_button', message='没有可删除的新记录'), ensure_ascii=False).encode() + b'\n'
        split = line.index('没'.encode()) + 1
        path = self.root / 'events.jsonl'
        path.write_bytes(line[:split])
        self.observer.poll()
        self.assertNotIn('无法解析', '\n'.join(self.observer.logs))
        with path.open('ab') as stream:
            stream.write(line[split:] + line)
        self.observer.poll()
        self.assertEqual(sum('没有可删除' in row for row in self.observer.logs), 2)

    def test_stale_or_unreadable_snapshot_never_claims_live_recording(self):
        self.publish(recording=True, started_ns=time.monotonic_ns())
        stamp = self.state['time_monotonic_ns']
        self.assertEqual(self.observer.poll(stamp)['headline'], '正在录制')
        health = self.observer.poll(stamp + 3_000_000_000)
        self.assertTrue(health['stale'])
        self.assertIn('无法确认', health['headline'])
        self.status.write_text('{')
        self.assertTrue(self.observer.poll(stamp)['stale'])

    def test_attach_to_large_log_reads_bounded_recent_history(self):
        path=self.root/'events.jsonl'
        noise=json.dumps({'event':'display_status','padding':'x'*500})+'\n'
        path.write_text(noise*1000)
        self.event('delete_button',message='没有可删除的新记录')
        self.observer.poll()
        self.assertEqual(self.observer.offset,path.stat().st_size)
        self.assertIn('完整历史','\n'.join(self.observer.logs))
        self.assertIn('没有可删除','\n'.join(self.observer.logs))

    def test_incomplete_commit_is_never_reported_saved(self):
        folder, _ = self.finished_files()
        (folder / 'integrity.json').write_text(json.dumps(dict(closed=True, complete=False)))
        self.event('recording_interrupted', episode_id=self.episode, reason='SOURCE_STALE')
        self.observer.poll()
        output = '\n'.join(self.observer.logs)
        self.assertIn('不完整', output)
        self.assertNotIn('保存完成', output)

    def test_finished_worker_overrides_old_recording_snapshot(self):
        self.publish(recording=True)
        path = self.root / 'control.json'
        path.write_text(json.dumps(dict(phase='finished', exit_code=0)))
        observer = collection_viewer.CollectionObserver(self.status, path)
        self.assertEqual(observer.poll()['headline'], '会话已结束')


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tmux with spaces ')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.manifest = dict(directory=str(self.directory), cwd=str(self.directory), mode='EXECUTE',
                             arguments=['--execute', '--task', '分拣 $(touch SHOULD_NOT_EXIST) `echo test`'], tmux_session='pico-test')

    def test_start_permission_is_published_only_after_both_panes_acknowledge(self):
        commands = []
        def call(arguments):
            commands.append(arguments)
            self.assertFalse((self.directory / 'start.json').exists())
            if arguments[0] == 'new-session':
                collection_tmux.write_json(self.directory / 'control_ready.json', {'ready': True})
                return '%0'
            if arguments[0] == 'split-window':
                collection_tmux.write_json(self.directory / 'monitor_ready.json', {'ready': True})
                return '%1'
            return '0'
        with contextlib.redirect_stdout(io.StringIO()):
            collection_tmux.launch_split(self.manifest, call=call)
        self.assertTrue((self.directory / 'start.json').is_file())
        command = collection_tmux.control_command(self.manifest)
        self.assertIn(self.manifest['arguments'][-1], command)
        self.assertTrue(any(c[0] == 'split-window' for c in commands))
        self.assertFalse((self.directory / 'SHOULD_NOT_EXIST').exists())

    def test_monitor_failure_never_authorizes_devices_and_cleans_only_own_session(self):
        commands = []
        def call(arguments):
            commands.append(arguments)
            if arguments[0] == 'new-session': return '%0'
            if arguments[0] == 'split-window': return '%1'
            return '1'
        with self.assertRaisesRegex(RuntimeError, '启动失败'):
            collection_tmux.launch_split(self.manifest, call=call)
        self.assertFalse((self.directory / 'start.json').exists())
        self.assertTrue((self.directory / 'abort.json').exists())
        self.assertEqual(commands[-1], ['kill-session', '-t', 'pico-test'])

    def test_closing_monitor_signals_only_its_viewer_and_does_not_change_control_state(self):
        collection_tmux.write_json(self.directory / 'run.json', self.manifest)
        control_path = self.directory / 'control.json'
        original = b'{"phase":"running","pid":12345}\n'
        control_path.write_bytes(original)
        handlers = {}
        signals = []
        class Child:
            def poll(self): return None
            def send_signal(self, sig): signals.append(sig)
            def wait(self):
                handlers[signal.SIGINT](signal.SIGINT, None)
                return 0
        child = Child()
        with mock.patch.object(collection_tmux.signal, 'signal', side_effect=lambda sig, fn: handlers.update({sig: fn})), \
             mock.patch.object(collection_tmux.subprocess, 'Popen', return_value=child) as popen, \
             contextlib.redirect_stdout(io.StringIO()):
            child.poll = mock.Mock(side_effect=[None, 0])
            code = collection_tmux.run_window('monitor', self.directory)
        self.assertEqual(code, 0)
        self.assertEqual(signals, [signal.SIGINT])
        self.assertIn('collection_viewer.py', ' '.join(popen.call_args[0][0]))
        self.assertEqual(control_path.read_bytes(), original)

    def test_demo_command_cannot_include_execute(self):
        self.manifest['mode'] = 'DEMO_NO_DEVICES'
        command = collection_tmux.control_command(self.manifest)
        self.assertIn('--_demo', command)
        self.assertNotIn('--execute', command)
        self.assertNotIn('collection.py', command)


if __name__ == '__main__':
    unittest.main()
