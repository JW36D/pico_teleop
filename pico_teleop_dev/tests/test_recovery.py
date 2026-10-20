import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""离线恢复动作回归测试；假反馈/假 MoveJ，绝不连接实机。"""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from incremental import MappingError
from recovery_guard import (check_limits, execute_recovery, waypoints,
                            SEGMENT_DEG, SPEED_PERCENT)
import recover_arm_initial as entry
from recover_arm_initial import load_initial, main, validate_reviewed

LOW = [-178., -130., -135., -178., -128., -360.]
HIGH = [178., 130., 135., 178., 128., 360.]
START = [68.741, 68.028, 77.163, -86.509, -12.692, 13.430]
TARGET = load_initial()['joints_deg']


class Clock:
    def __init__(self):
        self.now = 0.
    def __call__(self):
        return self.now
    def sleep(self, value):
        self.now += value


class FakeBackend:
    def __init__(self, initial=START):
        self.current = list(initial)
        self.goal = None
        self.sent = []
        self.stops = 0
        self.error = None
        self.rc = 0
        self.stuck = False
        self.bad_corridor = False
        self.alarm = False
        self.active = False
        self.frozen_feedback = False

    def sample(self):
        if self.goal is not None:
            if self.error:
                raise MappingError(self.error)
            if self.alarm:
                raise MappingError('Nonzero arm_err')
            if not self.stuck:
                gap = max(abs(a-b) for a, b in zip(self.current, self.goal))
                fraction = min(1., .5/max(gap, 1e-9))
                self.current = [a+(b-a)*fraction for a, b in zip(self.current, self.goal)]
            if self.bad_corridor:
                self.current[5] += 1.
            if self.frozen_feedback:
                return {'joints': START.copy(), 'trajectory': 0}
        moving = self.goal is not None and max(abs(a-b) for a, b in zip(self.current, self.goal)) > .001
        return {'joints': self.current.copy(), 'trajectory': 1 if moving or self.active else 0}

    def movej(self, target, speed):
        self.sent.append((list(target), speed))
        self.goal = list(target)
        return self.rc

    def stop(self):
        self.stops += 1


class RecoveryTests(unittest.TestCase):
    def run_motion(self, backend, **kwargs):
        clock = Clock()
        return execute_recovery(backend, START, TARGET, LOW, HIGH,
                                clock=clock, sleep=clock.sleep, **kwargs)

    def test_original_target_matches_historical_record(self):
        root = Path(__file__).resolve().parent.parent
        history = [json.loads(line) for line in (root/'tests/fixtures/ik_verification.log').read_text().splitlines()
                   if line.startswith('{')]
        self.assertEqual(TARGET, history[0]['robot_joints_deg'])

    def test_partition_has_no_joint_jump_or_angle_wrapping(self):
        points = waypoints(START, TARGET, LOW, HIGH)
        self.assertEqual(len(points), 15)
        previous = START
        for current in points:
            self.assertLessEqual(max(abs(a-b) for a, b in zip(previous, current)), SEGMENT_DEG)
            check_limits(current, LOW, HIGH)
            previous = current
        self.assertEqual(points[-1], TARGET)

    def test_real_feedback_arrival_required_for_all_segments(self):
        backend = FakeBackend()
        actual = self.run_motion(backend)
        self.assertEqual(len(backend.sent), 15)
        self.assertTrue(all(speed == SPEED_PERCENT for _, speed in backend.sent))
        self.assertLessEqual(max(abs(a-b) for a, b in zip(actual, TARGET)), .05)

    def test_saved_pose_never_uses_left_joint_values(self):
        self.assertEqual(load_initial()['arm'], 'right')
        self.assertEqual(load_initial()['source'], 'ik_verification.log:2')

    def test_changed_start_refuses_before_any_motion(self):
        backend = FakeBackend([START[0]+.2]+START[1:])
        with self.assertRaisesRegex(MappingError, 'changed'):
            self.run_motion(backend)
        self.assertEqual(backend.sent, [])
        self.assertEqual(backend.stops, 0)

    def test_existing_trajectory_refuses_before_any_motion(self):
        backend = FakeBackend()
        backend.active = True
        with self.assertRaisesRegex(MappingError, 'active'):
            self.run_motion(backend)
        self.assertEqual(backend.sent, [])

    def test_acknowledgement_loss_stops_without_retry(self):
        backend = FakeBackend()
        backend.rc = -2
        with self.assertRaisesRegex(MappingError, 'no retry'):
            self.run_motion(backend)
        self.assertEqual(len(backend.sent), 1)
        self.assertEqual(backend.stops, 1)

    def test_timeout_stops_without_next_segment(self):
        backend = FakeBackend()
        backend.stuck = True
        with self.assertRaisesRegex(MappingError, 'timeout'):
            self.run_motion(backend, segment_timeout_s=.5)
        self.assertEqual(len(backend.sent), 1)
        self.assertEqual(backend.stops, 1)

    def test_feedback_failure_and_alarm_stop(self):
        for error in ('Feedback monitor: timed out', 'Nonzero arm_err'):
            with self.subTest(error=error):
                backend = FakeBackend()
                backend.error = error
                with self.assertRaisesRegex(MappingError, error):
                    self.run_motion(backend)
                self.assertEqual(len(backend.sent), 1)
                self.assertEqual(backend.stops, 1)

    def test_uncommanded_joint_or_overshoot_stops(self):
        backend = FakeBackend()
        backend.bad_corridor = True
        with self.assertRaisesRegex(MappingError, 'corridor'):
            self.run_motion(backend)
        self.assertEqual(len(backend.sent), 1)
        self.assertEqual(backend.stops, 1)

    def test_static_report_cannot_fake_arrival(self):
        backend = FakeBackend()
        backend.frozen_feedback = True
        with self.assertRaisesRegex(MappingError, 'timeout'):
            self.run_motion(backend, segment_timeout_s=.5)
        self.assertEqual(backend.stops, 1)

    def test_cancel_during_motion_stops(self):
        backend = FakeBackend()
        with self.assertRaises(KeyboardInterrupt):
            self.run_motion(backend, cancelled=lambda: bool(backend.sent))
        self.assertEqual(len(backend.sent), 1)
        self.assertEqual(backend.stops, 1)

    def test_cancel_in_record_before_send_emits_no_motion(self):
        backend = FakeBackend()
        cancel = [False]
        def record(event):
            if event['event'] == 'SEGMENT_START':
                cancel[0] = True
        with self.assertRaises(KeyboardInterrupt):
            self.run_motion(backend, record=record, cancelled=lambda: cancel[0])
        self.assertEqual(backend.sent, [])
        self.assertEqual(backend.stops, 0)

    def test_already_at_target_sends_nothing(self):
        backend = FakeBackend(TARGET)
        result = execute_recovery(backend, TARGET, TARGET, LOW, HIGH)
        self.assertEqual(result, TARGET)
        self.assertEqual(backend.sent, [])

    def test_invalid_targets_rejected(self):
        for bad in ([math.nan]+TARGET[1:], [179.]+TARGET[1:]):
            with self.subTest(target=bad), self.assertRaises(MappingError):
                waypoints(START, bad, LOW, HIGH)

    def test_current_right_pose_recovers_without_the_old_90_degree_cap(self):
        start = [115.308, 127.018, -107.236, -68.984, 128., -37.85]
        backend = FakeBackend(start)
        clock = Clock()
        execute_recovery(backend, start, TARGET, LOW, HIGH,
                         clock=clock, sleep=clock.sleep)
        self.assertEqual(len(backend.sent), 36)
        self.assertEqual(backend.sent[-1][0], TARGET)
        self.assertEqual(backend.stops, 0)

    def test_full_hardware_range_keeps_explicit_joint_turns(self):
        for start, target in ((LOW, HIGH), (HIGH, LOW)):
            with self.subTest(start=start):
                previous = start
                for point in waypoints(start, target, LOW, HIGH):
                    check_limits(point, LOW, HIGH)
                    self.assertLessEqual(max(abs(a-b) for a,b in zip(previous,point)), SEGMENT_DEG+1e-10)
                    previous = point
                self.assertEqual(previous, target)
        # 左臂初始 J6 的 188.838° 必须保留，不改写成 -171.162°。
        left = load_initial(arm='left')['joints_deg']
        self.assertGreater(left[5], 180.)
        self.assertEqual(waypoints(START, left, LOW, HIGH)[-1], left)

    def test_boundary_feedback_noise_never_expands_command_limits(self):
        start = TARGET.copy()
        start[4] = 128.001
        points = waypoints(start, TARGET, LOW, HIGH)
        self.assertLess(points[0][4], HIGH[4])
        for point in points:
            check_limits(point, LOW, HIGH)
        with self.assertRaises(MappingError):
            check_limits(start, LOW, HIGH)
        self.assertEqual(check_limits(start, LOW, HIGH, feedback=True), start)
        with self.assertRaises(MappingError):
            waypoints(START, start, LOW, HIGH)
        start[4] = 128.02
        with self.assertRaises(MappingError):
            waypoints(start, TARGET, LOW, HIGH)

    def test_marginal_feedback_can_leave_limit_and_finish(self):
        start = TARGET.copy()
        start[4] = 128.001
        backend = FakeBackend(start)
        clock = Clock()
        execute_recovery(backend, start, TARGET, LOW, HIGH,
                         clock=clock, sleep=clock.sleep)
        self.assertEqual(backend.sent[-1][0], TARGET)
        self.assertTrue(all(q[4] <= HIGH[4] for q,_ in backend.sent))

    def test_left_saved_target_is_its_own_readonly_baseline(self):
        root = Path(__file__).resolve().parent.parent
        source = json.loads((root/'tests/fixtures/left_right_readonly_inventory_20261015.json').read_text())['sides']['left']
        initial = load_initial(arm='left')
        self.assertEqual(initial['joints_deg'], source['rm_get_joint_degree'][1])
        self.assertEqual(initial['installation_deg'], [0.,45.,0.])
        with self.assertRaises(MappingError):
            load_initial(root/'configs/right_initial_pose.json', arm='left')

    def test_modified_plan_target_speed_or_points_rejected(self):
        initial = load_initial()
        good = {'schema': 1, 'arm': 'right', 'initial_file_sha256': initial['file_sha256'],
                'start_joints_deg': START, 'target_joints_deg': TARGET,
                'minimum_deg': LOW, 'maximum_deg': HIGH, 'speed_percent': 1,
                'waypoints_deg': waypoints(START, TARGET, LOW, HIGH)}
        validate_reviewed(good, initial)
        for key, value in [('target_joints_deg', START), ('arm', 'left'),
                           ('speed_percent', 10), ('initial_file_sha256', 'changed'),
                           ('waypoints_deg', [TARGET])]:
            with self.subTest(key=key):
                bad = copy.deepcopy(good)
                bad[key] = value
                with self.assertRaises(MappingError):
                    validate_reviewed(bad, initial)

    def test_default_and_offline_mode_do_not_connect(self):
        for args in ([], ['--offline']):
            with self.subTest(args=args), patch('socket.create_connection', side_effect=AssertionError('No network')):
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(main(args), 0)
                self.assertIn('"connected": false', out.getvalue())

    def test_execute_rejects_noninteractive_terminal_before_connect(self):
        with patch('sys.stdin.isatty', return_value=False), patch('socket.create_connection', side_effect=AssertionError('No network')):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(['--execute']), 1)
            self.assertIn('interactive SSH terminal', out.getvalue())


class EntryFlowTests(unittest.TestCase):
    """走完整预览/执行入口，用假硬件核对实际命令次数和上下文拒绝。"""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'pico_xr'
        self.root.mkdir()
        self.plan_file = self.root/'state/right_recovery_plan.json'
        self.initial = load_initial()
        self.position = START.copy()
        self.instances = []
        self.reads = []
        self.native_arms = []
        self.monitor_hosts = []
        harness = self

        class Native:
            def __init__(self, arm='right'):
                harness.native_arms.append(arm)
                self.minimum, self.maximum = LOW.copy(), HIGH.copy()
                self.frames = copy.deepcopy(harness.initial['frames'])
                self.installation = dict(zip(('x', 'y', 'z'), harness.initial['installation_deg']))
                self.robot = self
                self.attempted = False
                self.sent = []
                self.stops = self.closes = 0
                harness.instances.append(self)
            def joints(self, require_idle=False):
                return harness.position.copy()
            def forward(self, joints):
                return harness.initial['pose_wxyz'].copy()
            def rm_algo_euler2quaternion(self, euler):
                return harness.initial['pose_wxyz'][3:]
            def rm_get_arm_current_trajectory(self):
                return {'return_code': 0, 'trajectory_type': 0}
            def rm_movej(self, joints, speed, radius, connect, block):
                self.sent.append((joints.copy(), speed, radius, connect, block))
                harness.position = joints.copy()
                return 0
            def stop(self):
                if self.attempted:
                    self.stops += 1
                    self.attempted = False
            def close(self):
                self.stop()
                self.closes += 1

        class Monitor:
            def __init__(self, *args, **kwargs):
                harness.monitor_hosts.append(args[0])
            def snapshot(self):
                return {'joints': harness.position.copy(), 'age_ms': 0.,
                        'position_m': harness.initial['pose_wxyz'][:3],
                        'orientation_rad': [0., 0., 0.]}
            def start(self):
                return self.snapshot()
            def close(self):
                pass

        def query(host, command, timeout):
            harness.reads.append((host, command))
            return {'arm_state': {'joint': [0]*6, 'arm_err': 0, 'sys_err': 0}}

        native_module = types.ModuleType('robot_backend')
        native_module.Backend = Native
        native_module.ARM_IPS = {'left': 'left.example', 'right': 'right.example'}
        native_module.check_known_controller_processes = lambda: None
        feedback_module = types.ModuleType('robot_feedback')
        feedback_module.FeedbackMonitor, feedback_module.query = Monitor, query
        for patcher in (patch.dict('sys.modules', {'robot_backend': native_module,
                                                 'robot_feedback': feedback_module}),
                        patch.object(entry, 'ROOT', self.root),
                        patch('sys.stdin.isatty', return_value=True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_cli(self, mode, arm='right'):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = main(['--arm', arm, mode, '--plan-file', str(self.plan_file)])
        return code, out.getvalue()

    def test_plan_is_readonly_and_left_is_only_queried(self):
        code, output = self.run_cli('--plan')
        self.assertEqual(code, 0, output)
        self.assertEqual(self.instances[0].sent, [])
        self.assertEqual(self.instances[0].stops, 0)
        self.assertEqual(self.instances[0].closes, 1)
        self.assertEqual(self.reads, [('left.example', 'get_current_arm_state')])
        validate_reviewed(json.loads(self.plan_file.read_text()), self.initial)

    def test_plan_to_execute_uses_nonblocking_right_movej(self):
        self.assertEqual(self.run_cli('--plan')[0], 0)
        clock = Clock()
        def run(*args, **kwargs):
            return execute_recovery(*args, clock=clock, sleep=clock.sleep, **kwargs)
        with patch.object(entry, 'execute_recovery', side_effect=run):
            code, output = self.run_cli('--execute')
        self.assertEqual(code, 0, output)
        native = self.instances[-1]
        self.assertEqual(len(native.sent), 15)
        self.assertTrue(all(params[1:] == (1, 0, 0, 0) for params in native.sent))
        self.assertEqual(native.sent[-1][0], TARGET)
        self.assertEqual(native.stops, 1)
        self.assertEqual(native.closes, 1)
        self.assertIn('RECOVERY_DONE', output)

    def test_powercycle_position_change_demands_new_preview(self):
        self.assertEqual(self.run_cli('--plan')[0], 0)
        self.position[0] += 1.
        code, output = self.run_cli('--execute')
        self.assertEqual(code, 1)
        self.assertIn('changed since recovery plan', output)
        self.assertEqual(self.instances[-1].sent, [])

    def test_missing_plan_refused_before_native_initialization(self):
        code, output = self.run_cli('--execute')
        self.assertEqual(code, 1)
        self.assertEqual(self.instances, [])
        self.assertIn('No such file', output)

    def test_left_plan_and_execute_only_command_left_with_its_saved_angles(self):
        self.initial = load_initial(arm='left')
        self.position = [-103.379,-43.849,-94.361,-118.394,69.005,126.19]
        self.plan_file = self.root/'state/left_recovery_plan.json'
        self.assertEqual(self.run_cli('--plan', arm='left')[0], 0)
        self.assertEqual(self.instances[-1].sent, [])
        clock = Clock()
        def run(*args, **kwargs):
            return execute_recovery(*args, clock=clock, sleep=clock.sleep, **kwargs)
        with patch.object(entry, 'execute_recovery', side_effect=run):
            code, output = self.run_cli('--execute', arm='left')
        self.assertEqual(code, 0, output)
        self.assertEqual(self.native_arms, ['left', 'left'])
        self.assertEqual(self.monitor_hosts, ['left.example', 'left.example'])
        self.assertEqual(self.reads, [('right.example', 'get_current_arm_state')])
        self.assertEqual(self.instances[-1].sent[-1][0], self.initial['joints_deg'])
        self.assertTrue((self.root/'state/latest_left_recovery.txt').exists())
        self.assertFalse((self.root/'state/latest_right_recovery.txt').exists())

    def test_right_plan_cannot_be_executed_on_left_before_connecting(self):
        self.assertEqual(self.run_cli('--plan')[0], 0)
        count = len(self.instances)
        code, output = self.run_cli('--execute', arm='left')
        self.assertEqual(code, 1)
        self.assertIn('left-arm recovery plan', output)
        self.assertEqual(len(self.instances), count)

    def test_default_plan_paths_do_not_overwrite_the_other_arm(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['--arm','right','--plan']), 0, output.getvalue())
            self.initial = load_initial(arm='left')
            self.assertEqual(main(['--arm','left','--plan']), 0, output.getvalue())
        for arm in ('left','right'):
            saved = json.loads((self.root/('state/%s_recovery_plan.json' % arm)).read_text())
            self.assertEqual(saved['arm'], arm)


if __name__ == '__main__':
    unittest.main()
