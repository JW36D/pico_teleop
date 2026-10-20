"""Small fault-context cache checks; no device or motion dependencies."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diagnostic_context import DiagnosticContext


def observation(tick, ns, state='TRACKING', feedback_sequence=7, pico_sequence=9):
    status = {'age_ms': 1., 'latest': {'sequence': pico_sequence,
        'receive_monotonic_ns': 900, 'app_focus': True,
        'right': {'grip': 1., 'pose': [1., 2., 3., 0., 0., 0., 1.]}}}
    report = {'control_ticks': tick, 'time_monotonic_ns': ns,
        'robot_feedback_joints_deg': [1.]*6,
        'robot_feedback_pose_wxyz': [0., 0., 0., 1., 0., 0., 0.],
        'robot_feedback_metadata': {'sequence': feedback_sequence,
            'request_monotonic_ns': 700, 'receive_monotonic_ns': 800,
            'joint_velocity_deg_s': [2.]*6, 'velocity_interval_s': .01},
        'last_actually_sent_command': {'sequence': 4, 'joints_deg': [3.]*6,
            'trajectory_sample_monotonic_ns': ns-5, 'send_monotonic_ns': ns-2},
        'mapping': {'state': state, 'target_joints_deg': [4.]*6,
            'joint_velocity_deg_s': [5.]*6, 'joint_acceleration_deg_s2': [6.]*6,
            'cartesian_achieved_velocity_m_s_rad_s': [7.]*6}}
    return status, report


class DiagnosticContextTests(unittest.TestCase):
    def test_two_second_window_includes_boundary_and_evicts_old_tick(self):
        cache = DiagnosticContext()
        for tick, ns in [(1, 1_000_000_000), (2, 2_000_000_000), (3, 3_000_000_000)]:
            cache.capture(*observation(tick, ns), True)
        self.assertEqual([x['control_tick'] for x in cache.snapshot('test')['samples']], [1, 2, 3])
        cache.capture(*observation(4, 3_000_000_001), True)
        self.assertEqual([x['control_tick'] for x in cache.snapshot('test')['samples']], [2, 3, 4])

    def test_tick_capacity_and_duplicate_tick_are_bounded(self):
        cache = DiagnosticContext(max_samples=2)
        for tick in (1, 2, 2, 3):
            cache.capture(*observation(tick, tick), True)
        snap = cache.snapshot('test')
        self.assertEqual([x['control_tick'] for x in snap['samples']], [2, 3])
        self.assertEqual(snap['capacity_evictions_total'], 1)
        cache.capture(*observation(4, 1), True)
        self.assertEqual(cache.snapshot('test')['out_of_order_captures_ignored_total'], 1)

    def test_selected_capture_and_snapshot_are_independent_copies(self):
        cache = DiagnosticContext()
        status, report = observation(1, 1000)
        cache.capture(status, report, True)
        status['latest']['right']['pose'][0] = 999
        report['robot_feedback_joints_deg'][0] = 999
        report['robot_feedback_metadata']['joint_velocity_deg_s'][0] = 999
        report['last_actually_sent_command']['joints_deg'][0] = 999
        report['mapping']['target_joints_deg'][0] = 999
        report['mapping']['cartesian_achieved_velocity_m_s_rad_s'][0] = 999
        snap = cache.snapshot({'why': ['fault']})
        row = snap['samples'][0]
        self.assertEqual(row['pico']['pose'][0], 1.)
        self.assertEqual(row['feedback']['joints_deg'][0], 1.)
        self.assertEqual(row['feedback']['joint_velocity_deg_s'][0], 2.)
        self.assertEqual(row['last_actually_sent_command']['joints_deg'][0], 3.)
        self.assertEqual(row['candidate']['joints_deg'][0], 4.)
        self.assertEqual(row['diagnostics']['cartesian_achieved_velocity_m_s_rad_s'][0], 7.)
        row['candidate']['joints_deg'][0] = 888
        self.assertEqual(cache.snapshot('again')['samples'][0]['candidate']['joints_deg'][0], 4.)
        self.assertTrue(snap['diagnostic_only'])
        self.assertFalse(snap['training_recording'])
        json.dumps(snap)

    def test_feedback_and_pico_reuse_keep_identity_and_real_intervals(self):
        cache = DiagnosticContext()
        cache.capture(*observation(1, 1000), True)
        cache.capture(*observation(2, 2000), True)
        cache.capture(*observation(3, 3000, feedback_sequence=8, pico_sequence=10), True)
        snap = cache.snapshot('test')
        self.assertEqual(snap['sample_count'], 3)
        self.assertEqual(snap['distinct_feedback_observations'], 2)
        self.assertEqual(snap['distinct_pico_frames'], 2)
        first, reused, new = snap['samples']
        self.assertIsNone(first['feedback']['reused_from_previous_control_tick'])
        self.assertTrue(reused['feedback']['reused_from_previous_control_tick'])
        self.assertTrue(reused['pico']['reused_from_previous_control_tick'])
        self.assertFalse(new['feedback']['reused_from_previous_control_tick'])
        self.assertEqual(reused['feedback']['velocity_interval_s'], .01)
        self.assertEqual(reused['feedback']['receive_monotonic_ns'], 800)

    def test_normal_release_and_b_button_do_not_trigger_fault_context(self):
        cases = [('RELEASED', None), ('SESSION_EXIT', None),
            ('INPUT_GATE_CLOSED', {'valid': True, 'grip': 0., 'reason': 'FRESH'})]
        for state, gate in cases:
            with self.subTest(state=state):
                cache = DiagnosticContext()
                cache.capture(*observation(1, 1000), True)
                status, report = observation(2, 2000, state)
                if gate is not None:
                    report['mapping']['gate_details'] = gate
                self.assertIsNone(cache.capture(status, report, False))

    def test_passive_loss_triggers_once_without_changing_inputs(self):
        for state in ('WAIT_FRESH_INPUT', 'INPUT_GATE_CLOSED'):
            with self.subTest(state=state):
                cache = DiagnosticContext()
                cache.capture(*observation(1, 1000), True)
                status, report = observation(2, 2000, state)
                report['mapping']['gate_details'] = {'valid': False, 'grip': 1., 'reason': 'INPUT_TIMEOUT'}
                event = cache.capture(status, report, False)
                self.assertEqual(event['event'], 'PASSIVE_CONTROL_DISARM')
                self.assertEqual(event['state'], state)
                self.assertEqual(report['mapping']['gate_details']['grip'], 1.)
                report['mapping']['gate_details']['grip'] = .3
                self.assertEqual(event['gate_details']['grip'], 1.)
                self.assertIsNone(cache.capture(*observation(3, 3000, state), False))


if __name__ == '__main__':
    unittest.main()
