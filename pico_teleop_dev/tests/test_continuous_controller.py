"""Continuous following and local boundary response with reversible fake FK."""
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from incremental import MappingError
from teleop_controller import TeleopController
from test_full_teleop import ControllerHarness, FakeBackend, FakeFeedback
from test_incremental import frame


class CartesianBackend(FakeBackend):
    """Three independent Cartesian axes, with joint speed checked separately."""
    def __init__(self, degrees_per_metre=100.):
        super().__init__()
        self.degrees_per_metre = degrees_per_metre

    def forward(self, joints):
        return [v / self.degrees_per_metre for v in joints[:3]] + [1., 0., 0., 0.]

    def solve(self, target, seed):
        return [v * self.degrees_per_metre for v in target[:3]] + [0., 0., 0.]


class ContinuousHarness(ControllerHarness):
    def setUp(self):
        super().setUp()
        # Use an already resolved work-frame fixture to isolate control behavior.
        # Production robot-body direction/calibration is tested independently.
        self.config.update(
            joint_session_limits_enabled=False, tracking_governor_mode='continuous',
            boundary_behavior='directional', position_scale=1., rotation_scale=0.,
            max_linear_speed_m_s=.2, max_angular_speed_rad_s=1.4,
            max_linear_accel_m_s2=3., max_angular_accel_rad_s2=12.,
            max_joint_speed_deg_s=90., feedback_hz=30.,
            max_command_lead_m=.02, max_command_lead_angle_rad=.12,
            target_smoothing_tau_s=.01, min_wrist_angle_deg=0.)
        self.backend = CartesianBackend()
        self.feedback = FakeFeedback()
        self.rebuild()

    def rebuild(self):
        self.feedback.joints = self.backend.initial_joints.copy()
        self.control = TeleopController(self.config, self.backend, self.feedback,
                                       True, lambda: self.ns)

    @staticmethod
    def hand(x=0., y=0., z=0.):
        # Existing matrix maps room -Z -> work +X, room -X -> work +Y.
        return [-y, z, -x]

    def engage(self):
        self.tick(0)
        self.assertEqual(self.tick(1)['mapping']['state'], 'ANCHORED')

    def follow_command(self):
        if self.backend.sent:
            self.feedback.joints = self.backend.sent[-1].copy()

    def assert_active(self):
        self.assertTrue(self.control.mapper.active)
        self.assertFalse(self.control.mapper.paused)
        self.assertEqual(self.backend.stops, 0)


class ContinuousControllerTests(ContinuousHarness, unittest.TestCase):
    def test_startup_joint_excursion_can_cross_old_90_degree_bound(self):
        self.backend = CartesianBackend(1000.)
        self.config['max_joint_session_deg_by_joint'] = [90.] * 6
        self.rebuild()
        self.engage()
        peak = 0.
        for k in range(240):
            self.follow_command()
            report = self.tick(1, self.hand(.09 if k < 20 else .12))
            self.assertEqual(report['mapping']['state'], 'TRACKING')
            peak = max(peak, self.backend.sent[-1][0])
        self.assertGreater(peak, 110.)
        self.assert_active()

    def test_disabled_session_bounds_are_null_and_hardware_limits_still_reported(self):
        self.engage()
        report = self.tick(1)
        self.assertFalse(report['joint_session_limits_enabled'])
        for key in ('joint_command_bound_deg', 'joint_command_bounds_deg',
                    'joint_session_bounds_deg'):
            self.assertIsNone(report[key])
        self.assertEqual(report['hardware_joint_limits_deg'],
                         {'min': self.backend.minimum, 'max': self.backend.maximum})

    def test_cartesian_target_accelerates_instead_of_starting_at_full_speed(self):
        self.engage()
        speeds = []
        for _ in range(10):
            self.follow_command()
            report = self.tick(1, self.hand(.08))
            mapping = report['mapping']
            self.assertTrue(mapping['continuous_shaping'])
            self.assertLessEqual(mapping['linear_accel_m_s2'], 3. + 1e-8)
            speeds.append(mapping['linear_speed_m_s'])
        self.assertGreater(speeds[0], 0.)
        self.assertLess(speeds[0], self.config['max_linear_speed_m_s'])
        self.assertGreater(speeds[-1], speeds[0])
        self.assertLessEqual(max(speeds), .2 + 1e-8)

    def test_rotation_accelerates_and_brakes_within_angular_bound(self):
        self.config['rotation_scale'] = 1.
        positional_forward = self.backend.forward
        positional_solve = self.backend.solve
        def rotational_forward(joints):
            result = positional_forward(joints)
            half_angle = math.radians(joints[3])/2
            return result[:3] + [math.cos(half_angle), 0., 0., math.sin(half_angle)]
        def rotational_solve(target, seed):
            result = positional_solve(target, seed)
            result[3] = math.degrees(2*math.atan2(target[6], target[3]))
            return result
        self.backend.forward = rotational_forward
        self.backend.solve = rotational_solve
        self.rebuild()
        self.engage()
        speeds = []
        for k in range(120):
            self.follow_command()
            status = self.sample(1)
            rotation = .4 if k == 0 else .8
            status['latest']['right']['pose'][3:] = [0., 0., math.sin(rotation/2),
                                                    math.cos(rotation/2)]
            report = self.control.tick(status, lambda: status, self.ns)
            self.assertEqual(report['mapping']['state'], 'TRACKING')
            self.assertLessEqual(report['mapping']['angular_accel_rad_s2'], 12. + 1e-8)
            speeds.append(report['mapping']['angular_speed_rad_s'])
        self.assertGreater(speeds[0], 0.)
        self.assertLess(speeds[0], 1.4)
        self.assertLessEqual(max(speeds), 1.4 + 1e-8)
        self.assertLess(speeds[-1], .01)
        self.assertAlmostEqual(math.radians(self.backend.sent[-1][3]), .8, places=4)
        self.assert_active()

    def test_slow_twenty_and_thirty_hz_feedback_continuously_streams(self):
        for feedback_hz in (20., 30.):
            with self.subTest(feedback_hz=feedback_hz):
                self.setUp()
                self.engage()
                physical = [0.] * 6
                next_feedback = 0.
                ratios = []
                peak_error = 0.
                for k in range(1, 1001):
                    command = self.backend.sent[-1] if self.backend.sent else [0.] * 6
                    physical = [a + (b-a)*(1-math.exp(-.01/.6))
                                for a, b in zip(physical, command)]
                    if k*.01 >= next_feedback:
                        self.feedback.joints = physical.copy()
                        next_feedback += 1/feedback_hz
                    report = self.tick(1, self.hand(.09*math.sin(k*.01*2*math.pi*.3)))
                    self.assertEqual(report['mapping']['state'], 'TRACKING')
                    self.assertTrue(report['motion_command_sent_this_tick'])
                    self.assertFalse(self.control.waiting_for_feedback)
                    ratios.append(report['mapping']['linear_governor_ratio'])
                    peak_error = max(peak_error, report['mapping']['command_tracking_error_mm'])
                    self.assertLessEqual(report['mapping']['linear_accel_m_s2'], 3. + 1e-8)
                self.assertLess(min(ratios), .8)
                self.assertGreater(max(ratios), .99)
                self.assertLess(peak_error, 30.)
                self.assert_active()

    def test_inward_command_brakes_then_moves_without_follower_wait(self):
        self.engage()
        for _ in range(160):
            report = self.tick(1, self.hand(.08))
        prior = self.backend.sent[-1][0]
        self.assertLess(report['mapping']['linear_governor_ratio'], .2)
        moved_inward = False
        for _ in range(10):
            report = self.tick(1, self.hand(-.01))
            self.assertEqual(report['mapping']['state'], 'TRACKING')
            self.assertLessEqual(report['mapping']['linear_accel_m_s2'], 3. + 1e-8)
            moved_inward |= self.backend.sent[-1][0] < prior - 1e-5
            prior = self.backend.sent[-1][0]
        self.assertTrue(moved_inward)
        self.assert_active()

    def test_real_hardware_feedback_limit_still_stops(self):
        self.engage()
        self.tick(1)
        self.feedback.joints[0] = self.backend.maximum[0] + .01
        with self.assertRaisesRegex(MappingError, 'hardware limit'):
            self.tick(1)
        self.assertFalse(self.control.mapper.active)
        self.assertEqual(self.backend.stops, 1)

    def test_real_tracking_error_still_stops(self):
        self.engage()
        self.tick(1)
        self.feedback.joints[0] = -.04 * self.backend.degrees_per_metre
        with self.assertRaisesRegex(MappingError, 'tracking error'):
            self.tick(1)
        self.assertFalse(self.control.mapper.active)
        self.assertEqual(self.backend.stops, 1)

    def test_release_b_invalid_pose_and_stale_input_stop_stream(self):
        for failure in ('release', 'b', 'invalid_pose', 'stale'):
            with self.subTest(failure=failure):
                self.setUp()
                self.engage()
                self.tick(1)
                count = len(self.backend.sent)
                status = self.sample(1)
                if failure == 'release': status['latest']['right']['grip'] = 0.
                if failure == 'b': status['session_exit_requested'] = True
                if failure == 'invalid_pose': status['latest']['right']['pose'] = [0.] * 6
                if failure == 'stale': status['latest']['receive_monotonic_ns'] -= 200_000_000
                report = self.control.tick(status, lambda: status, self.ns)
                self.assertFalse(report['motion_command_sent_this_tick'])
                self.assertEqual(len(self.backend.sent), count)
                self.assertFalse(self.control.mapper.active)
                self.assertEqual(self.backend.stops, 1)


class DirectionalBoundaryTests(ContinuousHarness, unittest.TestCase):
    def reject_positive_x(self):
        solve = self.backend.solve
        def blocked(target, seed):
            if target[0] > 1e-12:
                raise MappingError('No solution at this boundary',
                                   {'state': 'IK_REFUSED', 'retryable_ik': True})
            return solve(target, seed)
        self.backend.solve = blocked

    def hold_positive_x(self):
        self.engage()
        self.feedback.stationary = False
        report = self.tick(1, self.hand(.01))
        self.assertEqual(report['mapping']['state'], 'BOUNDARY_HOLD')
        self.assertFalse(report['mapping']['needs_regrip'])
        self.assert_active()
        return report

    def test_known_ik_boundary_holds_without_stop_or_regrip_then_moves_inward(self):
        self.reject_positive_x()
        self.hold_positive_x()
        for _ in range(8):
            report = self.tick(1, self.hand(.01))
            self.assertNotIn(report['mapping']['state'], ('RECOVERY_WAIT', 'RESUMED'))
            self.assert_active()
        report = self.tick(1, self.hand(.009))
        self.assertTrue(report['motion_command_sent_this_tick'])
        self.assertLess(self.backend.sent[-1][0], 0.)
        self.assert_active()

    def test_final_input_gate_still_stops_boundary_hold(self):
        for failure in ('release', 'b', 'focus', 'invalid_pose', 'stale'):
            with self.subTest(failure=failure):
                self.setUp()
                self.reject_positive_x()
                self.hold_positive_x()
                count = len(self.backend.sent)
                gate = frame(self.seq+1, self.ns+10_000_000, 1, self.hand(.01))
                if failure == 'release': gate['latest']['right']['grip'] = 0.
                if failure == 'b': gate['session_exit_requested'] = True
                if failure == 'focus': gate['latest']['app_focus'] = False
                if failure == 'invalid_pose': gate['latest']['right']['pose'] = [0.] * 6
                if failure == 'stale': gate['latest']['receive_monotonic_ns'] -= 200_000_000
                report = self.tick(1, self.hand(.01), gate=gate)
                self.assertFalse(report['motion_command_sent_this_tick'])
                self.assertEqual(len(self.backend.sent), count)
                self.assertFalse(self.control.mapper.active)
                self.assertEqual(self.backend.stops, 1)

    def test_tangential_motion_is_executable_on_next_tick(self):
        self.reject_positive_x()
        self.hold_positive_x()
        report = self.tick(1, self.hand(.01, .002))
        self.assertTrue(report['motion_command_sent_this_tick'])
        self.assertAlmostEqual(self.backend.sent[-1][0], 0., places=8)
        self.assertGreater(self.backend.sent[-1][1], 0.)
        self.assert_active()

    def test_hardware_target_refusal_holds_and_allows_inward_motion(self):
        self.backend.maximum[0] = 0.
        self.hold_positive_x()
        self.assertTrue(all(joints[0] <= 0. for joints in self.backend.sent))
        report = self.tick(1, self.hand(.009))
        self.assertTrue(report['motion_command_sent_this_tick'])
        self.assertLess(self.backend.sent[-1][0], 0.)
        self.assert_active()

    def test_wrist_boundary_preserves_grip_and_allows_escape_immediately(self):
        self.config['min_wrist_angle_deg'] = 8.
        self.backend.initial_joints[4] = 8.
        solve = self.backend.solve
        def wrist_solver(target, seed):
            joints = solve(target, seed)
            joints[4] = 8. - target[0]*1000.
            return joints
        self.backend.solve = wrist_solver
        self.rebuild()
        self.hold_positive_x()
        report = self.tick(1, self.hand(.009))
        self.assertTrue(report['motion_command_sent_this_tick'])
        self.assertGreater(self.backend.sent[-1][4], 8.)
        self.assert_active()

    def test_unknown_or_invalid_solver_results_are_not_boundary_holds(self):
        for failure in ('unknown_error', 'nonfinite_result'):
            with self.subTest(failure=failure):
                self.setUp()
                self.engage()
                self.tick(1)
                count = len(self.backend.sent)
                def invalid(target, seed):
                    if failure == 'unknown_error':
                        raise MappingError('Unexpected native result structure')
                    return [math.nan] * 6
                self.backend.solve = invalid
                with self.assertRaises(MappingError):
                    self.tick(1, self.hand(.01))
                self.assertEqual(len(self.backend.sent), count)
                self.assertFalse(self.control.mapper.active)
                self.assertEqual(self.backend.stops, 1)

    def test_task_workspace_bound_still_uses_existing_soft_pause(self):
        self.config['max_session_translation_m'] = .02
        self.engage()
        self.tick(1)
        report = self.tick(1, self.hand(.03))
        self.assertEqual(report['mapping']['state'], 'SESSION_BOUND')
        self.assertTrue(self.control.mapper.paused)
        self.assertTrue(self.control.mapper.active)
        self.assertEqual(self.backend.stops, 1)


if __name__ == '__main__':
    unittest.main()
