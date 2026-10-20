"""Timing recovery uses fake feedback, input and clocks; no robot connections."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(TEST_DIR.parent))

from incremental import MappingError
from teleop_config import load_config
from test_full_teleop import ControllerHarness, ROOT, legacy_config
from test_incremental import frame


class TimingRecoveryTests(ControllerHarness, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.config.update(control_hz=100., follow_mode='low',
                           auto_resume_soft_limits=True, auto_resume_timing_gaps=True,
                           max_control_gap_s=.03, max_command_gap_s=.03,
                           max_recoverable_control_gap_s=.15, recovery_dwell_s=.15,
                           target_smoothing_tau_s=.01, tracking_governor=False)
        snapshot = self.feedback.snapshot
        self.feedback.snapshot = lambda: {**snapshot(), 'receive_monotonic_ns': self.ns}

    def start_stream(self):
        result = self.active()
        self.assertTrue(result['motion_command_sent_this_tick'])
        self.feedback.joints = self.control.last_sent_joints.copy()
        return copy.deepcopy(self.control.mapper.anchor_hand)

    def tick_after(self, milliseconds, grip=1, xyz=None, mutate=None):
        self.ns += int(milliseconds * 1_000_000)
        self.seq += 1
        status = frame(self.seq, self.ns, grip, xyz)
        if mutate is not None:
            mutate(status)
        return self.control.tick(status, lambda: status, self.ns)

    def assert_paused_without_send(self, result, state, anchor, sent_count):
        self.assertEqual(result['mapping']['state'], state)
        self.assertFalse(result['motion_command_sent_this_tick'])
        self.assertEqual(len(self.backend.sent), sent_count)
        self.assertTrue(self.control.mapper.active)
        self.assertTrue(self.control.mapper.paused)
        self.assertFalse(self.control.streaming)
        self.assertEqual(self.control.mapper.anchor_hand, anchor)

    def test_short_control_gaps_stop_and_keep_original_anchor(self):
        for milliseconds in (60, 100):
            with self.subTest(gap_ms=milliseconds):
                self.setUp()
                anchor = self.start_stream()
                count = len(self.backend.sent)
                result = self.tick_after(milliseconds, xyz=[0, 0, -.02])
                self.assert_paused_without_send(result, 'CONTROL_GAP_PAUSED', anchor, count)
                self.assertEqual(self.backend.stops, 1)

    def test_held_grip_recovers_after_settling_without_reanchor_or_jump(self):
        anchor = self.start_stream()
        count = len(self.backend.sent)
        self.tick_after(70, xyz=[0, 0, -.02])
        self.feedback.stationary = False
        for _ in range(20):
            result = self.tick(1, [0, 0, -.02])
            self.assertNotEqual(result['mapping']['state'], 'RESUMED')
            self.assertFalse(result['motion_command_sent_this_tick'])
        self.feedback.stationary = True
        valid_since = self.ns + int(1e9 / self.config['control_hz'])
        for _ in range(30):
            result = self.tick(1, [0, 0, -.02])
            if result['mapping']['state'] == 'RESUMED':
                break
        else:
            self.fail('A held Grip with fresh stationary feedback did not recover')
        self.assertGreaterEqual((self.ns - valid_since) / 1e9, .15 - 1e-9)
        self.assertFalse(result['motion_command_sent_this_tick'])
        self.assertEqual(len(self.backend.sent), count)
        self.assertEqual(self.control.mapper.anchor_hand, anchor)
        before = self.feedback.joints.copy()
        actual = self.backend.forward(before)
        self.assertEqual(result['mapping']['target_pose_wxyz'], actual)
        result = self.tick(1, [0, 0, -.02])
        self.assertTrue(result['motion_command_sent_this_tick'])
        after = self.backend.sent[-1]
        step_bound = self.config['max_joint_speed_deg_s'] / self.config['control_hz']
        self.assertGreater(after[0], before[0])
        self.assertLessEqual(max(abs(a-b) for a, b in zip(after, before)), step_bound + 1e-4)
        self.assertLess(after[0], result['mapping']['requested_pose_wxyz'][0] * 1000)

    def test_long_control_gap_requires_release_and_new_grip(self):
        self.start_stream()
        count = len(self.backend.sent)
        result = self.tick_after(151, xyz=[0, 0, -.02])
        self.assertEqual(result['mapping']['state'], 'CONTROL_GAP')
        self.assertFalse(result['motion_command_sent_this_tick'])
        self.assertFalse(self.control.mapper.active)
        self.assertFalse(self.control.mapper.paused)
        self.assertEqual(len(self.backend.sent), count)
        self.assertEqual(self.tick(1)['mapping']['state'], 'RELEASE_GRIP_TO_ARM')

    def test_high_follow_never_auto_recovers_a_timing_gap(self):
        self.start_stream()
        # Deliberately bypass load_config to test the runtime high-follow guard.
        self.config['follow_mode'] = 'high'
        result = self.tick_after(60, xyz=[0, 0, -.02])
        self.assertEqual(result['mapping']['state'], 'CONTROL_GAP')
        self.assertFalse(self.control.mapper.active)
        self.assertFalse(self.control.mapper.paused)
        self.assertFalse(result['motion_command_sent_this_tick'])

    def test_stop_inputs_still_cancel_a_soft_timing_pause(self):
        cases = (
            ('release', 0, None, 'RELEASED'),
            ('exit', 1, lambda s: s.update(session_exit_requested=True), 'SESSION_EXIT'),
            ('stale', 1, lambda s: s['latest'].update(receive_monotonic_ns=self.ns-200_000_000), 'WAIT_FRESH_INPUT'),
            ('focus', 1, lambda s: s['latest'].update(app_focus=False), 'WAIT_FRESH_INPUT'),
        )
        for name, grip, mutate, state in cases:
            with self.subTest(stop_input=name):
                self.setUp()
                self.start_stream()
                count = len(self.backend.sent)
                self.tick_after(60, xyz=[0, 0, -.02])
                result = self.tick_after(10, grip, [0, 0, -.02], mutate)
                self.assertEqual(result['mapping']['state'], state)
                self.assertFalse(self.control.mapper.active)
                self.assertFalse(self.control.mapper.paused)
                self.assertFalse(result['motion_command_sent_this_tick'])
                self.assertEqual(len(self.backend.sent), count)
                if name in ('stale', 'focus'):
                    self.assertEqual(self.tick(1)['mapping']['state'], 'RELEASE_GRIP_TO_ARM')

    def test_short_command_gap_stops_before_sending_and_keeps_anchor(self):
        anchor = self.start_stream()
        count = len(self.backend.sent)
        self.control.last_send_ns = self.ns - 60_000_000
        result = self.tick(1, [0, 0, -.02])
        self.assert_paused_without_send(result, 'COMMAND_GAP_PAUSED', anchor, count)

    def computation_gap(self, milliseconds):
        original_solve = self.backend.solve
        delayed = [False]

        def delayed_solve(target, seed):
            if not delayed[0]:
                delayed[0] = True
                self.ns += int(milliseconds * 1_000_000)
            return original_solve(target, seed)

        self.backend.solve = delayed_solve
        status = self.sample(1, [0, 0, -.02])
        now_ns = self.ns
        return self.control.tick(status, lambda: frame(self.seq, self.ns, 1, [0, 0, -.02]), now_ns)

    def test_short_computation_gap_stops_before_sending_and_keeps_anchor(self):
        anchor = self.start_stream()
        count = len(self.backend.sent)
        result = self.computation_gap(60)
        self.assert_paused_without_send(result, 'CONTROL_COMPUTATION_GAP_PAUSED', anchor, count)

    def test_long_computation_gap_disarms_without_sending(self):
        self.start_stream()
        count = len(self.backend.sent)
        # Either an explicit hard-gap report or a MappingError is acceptable;
        # neither may preserve the old armed session or send the delayed goal.
        try:
            result = self.computation_gap(151)
        except MappingError:
            result = None
        if result is not None:
            self.assertFalse(result['motion_command_sent_this_tick'])
            self.assertNotIn(result['mapping']['state'], ('TRACKING', 'RESUMED'))
        self.assertFalse(self.control.mapper.active)
        self.assertFalse(self.control.mapper.paused)
        self.assertEqual(len(self.backend.sent), count)
        self.assertEqual(self.backend.stops, 1)

    def test_phase_timings_are_finite_and_nonnegative(self):
        self.start_stream()
        result = self.tick(1, [0, 0, -.02])
        phases = result['control_phase_ms']
        self.assertTrue(phases)
        for name, milliseconds in phases.items():
            with self.subTest(phase=name):
                self.assertIsInstance(milliseconds, (int, float))
                self.assertGreaterEqual(milliseconds, 0)
                self.assertLess(milliseconds, float('inf'))


class TimingRecoveryConfigTests(unittest.TestCase):
    def test_config_rejects_high_follow_recovery_and_nonincreasing_gap_bound(self):
        raw = legacy_config()
        raw.update(auto_resume_timing_gaps=True, auto_resume_soft_limits=True)
        cases = (
            {'follow_mode': 'high', 'control_hz': 200, 'max_control_gap_s': .008,
             'max_command_gap_s': .01},
            {'follow_mode': 'low', 'max_control_gap_s': .03, 'max_command_gap_s': .03,
             'max_recoverable_control_gap_s': .03},
            {'follow_mode': 'low', 'max_control_gap_s': .03, 'max_command_gap_s': .08,
             'max_recoverable_control_gap_s': .05},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'timing-config.json'
            for overrides in cases:
                with self.subTest(overrides=overrides):
                    path.write_text(json.dumps({**raw, **overrides}))
                    with self.assertRaises(MappingError):
                        load_config(path)


if __name__ == '__main__':
    unittest.main()
