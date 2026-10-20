import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import io
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from diagnostic_logger import AsyncDiagnosticLogger
from incremental import MappingError


class BlockedStream:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.rows = []
        self.closed = False

    def write(self, data):
        self.entered.set()
        self.release.wait(5.)
        self.rows.append(data)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FailingWriter:
    def write(self, _):
        raise OSError('disk full')

    def flush(self):
        pass

    def close(self):
        pass


class DiagnosticLoggerTests(unittest.TestCase):
    def logger(self, directory, **kwargs):
        return AsyncDiagnosticLogger(Path(directory)/'session.jsonl',
                                     Path(directory)/'full_status.json', **kwargs)

    def test_normal_output_in_order_and_final_state_after_close(self):
        with tempfile.TemporaryDirectory() as directory:
            console = io.StringIO()
            logger = self.logger(directory, console=console)
            for tick in range(30):
                logger.append({'event':'PAUSE','tick':tick}, critical=True)
            logger.append({'event':'SESSION_END','motion_streaming_active':False}, critical=True)
            result = logger.close()
            self.assertTrue(result['complete'])
            self.assertEqual(result['written_events'], 31)
            rows = [json.loads(row) for row in logger.log_path.read_text().splitlines()]
            self.assertEqual([row['tick'] for row in rows[:-1]], list(range(30)))
            self.assertEqual(rows[-1]['event'], 'SESSION_END')
            self.assertEqual(json.loads(logger.status_path.read_text()), rows[-1])

    def test_blocked_stdout_does_not_block_disk_or_append_or_close(self):
        with tempfile.TemporaryDirectory() as directory:
            console = BlockedStream()
            logger = self.logger(directory, console=console)
            try:
                logger.append({'event':'START'}, critical=True)
                self.assertTrue(console.entered.wait(1.))
                started = time.monotonic()
                for tick in range(1000):
                    logger.append({'mapping':{'state':'TRACKING'},'tick':tick})
                self.assertLess(time.monotonic()-started, .25)
                logger.append({'event':'SESSION_END'}, critical=True)
                started = time.monotonic()
                result = logger.close(timeout=1.)
                self.assertLess(time.monotonic()-started, 1.)
                self.assertTrue(result['complete'])
                self.assertEqual(json.loads(logger.status_path.read_text())['event'], 'SESSION_END')
                self.assertGreater(result['coalesced_regular_events'], 0)
                self.assertTrue(logger.console_thread.is_alive())
            finally:
                console.release.set()
                logger.close()
                logger.console_thread.join(1.)

    def test_blocked_writer_coalesces_regular_without_overwriting_critical(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = BlockedStream()
            logger = self.logger(directory, writer=writer, console=io.StringIO(), critical_queue_size=3)
            try:
                logger.append({'event':'START'}, critical=True)
                self.assertTrue(writer.entered.wait(1.))
                started = time.monotonic()
                logger.append({'event':'ANCHORED'}, critical=True)
                for tick in range(5000):
                    logger.append({'mapping':{'state':'FOLLOWER_WAIT'},'tick':tick})
                logger.append({'event':'SOFT_PAUSE'}, critical=True)
                logger.append({'event':'SESSION_END'}, critical=True)
                self.assertLess(time.monotonic()-started, .25)
                self.assertEqual(logger.stats()['coalesced_regular_events'], 4999)
                self.assertEqual(logger.stats()['pending_critical_events'], 3)
                writer.release.set()
                result = logger.close()
                self.assertTrue(result['complete'])
                rows = [json.loads(row) for row in writer.rows]
                self.assertEqual([row.get('event') for row in rows],
                                 ['START','ANCHORED',None,'SOFT_PAUSE','SESSION_END'])
                self.assertEqual(rows[2]['tick'], 4999)
                self.assertEqual(json.loads(logger.status_path.read_text())['event'], 'SESSION_END')
            finally:
                writer.release.set()
                logger.close()

    def test_critical_overflow_is_explicit_and_does_not_evict_any_event(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = BlockedStream()
            logger = self.logger(directory, writer=writer, console=io.StringIO(), critical_queue_size=1)
            try:
                logger.append({'event':'START'}, critical=True)
                self.assertTrue(writer.entered.wait(1.))
                logger.append({'event':'PAUSE'}, critical=True)
                with self.assertRaisesRegex(MappingError,'Critical diagnostic queue full'):
                    logger.append({'event':'ERROR'}, critical=True)
                writer.release.set()
                result = logger.close()
                self.assertEqual([json.loads(row)['event'] for row in writer.rows], ['START','PAUSE'])
                self.assertEqual(result['critical_queue_overflow'], 1)
                self.assertFalse(result['complete'])
            finally:
                writer.release.set()
                logger.close()

    def test_writer_failure_is_visible_to_tick_and_close(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = self.logger(directory, writer=FailingWriter(), console=io.StringIO())
            logger.append({'event':'START'}, critical=True)
            logger.thread.join(1.)
            with self.assertRaisesRegex(MappingError,'disk full'):
                logger.check_error()
            with self.assertRaisesRegex(MappingError,'disk full'):
                logger.append({'event':'ERROR'}, critical=True)
            result = logger.close()
            self.assertFalse(result['complete'])
            self.assertIn('disk full', result['error'])

    def test_shutdown_timeout_is_bounded_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = BlockedStream()
            logger = self.logger(directory, writer=writer, console=io.StringIO())
            try:
                logger.append({'event':'SESSION_END'}, critical=True)
                self.assertTrue(writer.entered.wait(1.))
                started = time.monotonic()
                result = logger.close(timeout=.03)
                self.assertLess(time.monotonic()-started, .3)
                self.assertFalse(result['complete'])
                self.assertIn('shutdown timeout', result['error'])
            finally:
                writer.release.set()
                logger.thread.join(1.)

    def test_loop_timing_finalized_before_background_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = self.logger(directory, console=io.StringIO())
            logger.append({'event':'SESSION_END','stage_timing_ms':{'controller':1.}},
                          critical=True, loop_start_ns=time.monotonic_ns()-2_000_000)
            self.assertTrue(logger.close()['complete'])
            stages = json.loads(logger.status_path.read_text())['stage_timing_ms']
            self.assertGreaterEqual(stages['loop'], 2.)
            self.assertIn('diagnostic_enqueue', stages)

    def test_real_console_descriptor_never_uses_python_stdio_lock(self):
        read_fd, write_fd = os.pipe()
        class DescriptorConsole:
            encoding = 'utf-8'
            def fileno(self):
                return write_fd
            def write(self, _):
                raise AssertionError('Real console must not acquire Python stdio lock')
            def flush(self):
                raise AssertionError('Real console must not flush Python stdio buffer')
        try:
            with tempfile.TemporaryDirectory() as directory:
                logger = self.logger(directory, console=DescriptorConsole())
                logger.append({'event':'SESSION_END','mode':'PREVIEW_NO_MOTION'}, critical=True)
                self.assertTrue(logger.close()['complete'])
                logger.console_thread.join(1.)
                self.assertFalse(logger.console_thread.is_alive())
                self.assertIsNone(logger.stats()['console_error'])
                self.assertEqual(os.read(read_fd, 4096), b'[PREVIEW_NO_MOTION] SESSION_END\n')
        finally:
            os.close(write_fd)
            os.close(read_fd)

    def test_pause_summary_retains_state_with_specific_short_hint(self):
        for state, hint in (
                ('CONTROL_GAP_PAUSED','保持 Grip 等待恢复'),
                ('WAIT_ROBOT_STATIONARY','等待机械臂停稳'),
                ('IK_REFUSED','可往回移动'),
                ('JOINT_SESSION_BOUND','软件关节范围'),
                ('WAIT_FRESH_INPUT','手柄输入失效'),
                ('CONTROL_GAP','控制长时间中断'),
                ('RESUMED','已恢复')):
            summary = AsyncDiagnosticLogger.summary({'mapping':{'state':state,'huge_payload':'x'*5000}})
            self.assertIn(state, summary)
            self.assertIn(hint, summary)
            self.assertLess(len(summary), 200)

    def test_main_offline_stops_before_drain_and_ignores_blocked_terminal(self):
        import full_teleop
        actions, loggers = [], []
        console = BlockedStream()
        config = {'follow_mode':'low','stream_trajectory_mode':0,'stream_radio':0,
                  'record_data':False,'control_hz':100,'log_interval_s':.5,
                  'rotation_mapping':'neutral_tool_axes','translation_mapping_frame':'work'}

        class Backend(full_teleop.Backend):
            def __init__(self):
                self.frames = {}
                self.installation = {'x':0.,'y':0.,'z':0.}
                self.algorithm_installation = [0.]*3
                self.initial_pose = [0.,0.,0.,1.,0.,0.,0.]
                self.initial_joints = [0.]*6
                self.minimum = [-180.]*6
                self.maximum = [180.]*6
                self.attempted = False
                self.robot = mock.Mock()
                self.robot.rm_algo_euler2quaternion.return_value = [1.,0.,0.,0.]
            def solve(self, target, seed):
                return seed.copy()
            def forward(self, joints):
                return self.initial_pose.copy()
            def close(self):
                actions.append('backend_close')

        class Feedback:
            def __init__(self, *_, **__):
                pass
            def start(self):
                return {'position_m':[0.]*3,'orientation_rad':[0.]*3}
            def close(self):
                actions.append('feedback_close')

        class Controller:
            def __init__(self, *_, **__):
                self.streaming = False
                self.send_count = 0
                self.ticks = 0
            def tick(self, *_, **__):
                self.ticks += 1
                return {'mapping':{'state':'SESSION_EXIT' if self.ticks==3 else 'TRACKING'}}
            def stop(self):
                self.streaming = False
                actions.append('stop')

        def make_logger(*args, **kwargs):
            logger = AsyncDiagnosticLogger(*args, console=console, **kwargs)
            original_close = logger.close
            def close(*args, **kwargs):
                actions.append('diagnostic_drain')
                return original_close(*args, **kwargs)
            logger.close = close
            loggers.append(logger)
            return logger

        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)/'pico_xr'
            task_dir.mkdir()
            try:
                with mock.patch.multiple(full_teleop, TASK_DIR=task_dir, Backend=Backend,
                        FeedbackMonitor=Feedback, TeleopController=Controller,
                        AsyncDiagnosticLogger=make_logger), \
                        mock.patch.object(full_teleop,'load_config',return_value=config), \
                        mock.patch.object(full_teleop,'resolve_translation_mapping',return_value=(config,{})), \
                        mock.patch.object(full_teleop,'checked_pair',return_value=[180.]*6), \
                        mock.patch.object(full_teleop,'check_known_controller_processes'), \
                        mock.patch.object(full_teleop,'read_pico',return_value={'button_stream':{'receiver_id':'fake','exit_sequence':0},'receive_hz_2s':72.,'age_ms':1.}), \
                        mock.patch.object(full_teleop.signal,'signal'), \
                        mock.patch.object(sys,'argv',['full_teleop.py','--preview']):
                    started = time.monotonic()
                    self.assertEqual(full_teleop.main(), 0)
                    self.assertLess(time.monotonic()-started, .5)
                self.assertLess(actions.index('stop'), actions.index('diagnostic_drain'))
                self.assertLess(actions.index('backend_close'), actions.index('diagnostic_drain'))
                logger = loggers[0]
                rows = [json.loads(row) for row in logger.log_path.read_text().splitlines()]
                self.assertEqual(rows[0]['event'], 'START')
                self.assertEqual(rows[-1]['event'], 'SESSION_END')
                self.assertFalse(rows[-1]['motion_streaming_active'])
                self.assertEqual(json.loads(logger.status_path.read_text())['event'], 'SESSION_END')
                self.assertTrue(any('stage_timing_ms' in row for row in rows))
                self.assertFalse((task_dir/'state'/'recordings').exists())
            finally:
                console.release.set()
                for logger in loggers:
                    logger.console_thread.join(1.)


if __name__=='__main__':
    unittest.main()
