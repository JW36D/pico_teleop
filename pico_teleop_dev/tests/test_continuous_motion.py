import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continuous_motion import ContinuousMotionShaper
from incremental import MappingError, angle, inverse, multiply, norm


CONFIG = {
    'max_linear_speed_m_s': .2,
    'max_angular_speed_rad_s': 1.4,
    'max_linear_accel_m_s2': 3.,
    'max_angular_accel_rad_s2': 12.,
    'max_command_lead_m': .02,
    'max_command_lead_angle_rad': .12,
}
REST = [0., 0., 0., 1., 0., 0., 0.]


def z_pose(radians, xyz=None):
    return list(xyz or [0., 0., 0.])+[math.cos(radians/2), 0., 0., math.sin(radians/2)]


def proposed_rotation(prior, goal_radians, dt=.01):
    current = 2*math.atan2(prior[6], prior[3])
    remaining = goal_radians-current
    step = max(-CONFIG['max_angular_speed_rad_s']*dt,
               min(CONFIG['max_angular_speed_rad_s']*dt, remaining))
    return z_pose(current+step, prior[:3]), abs(remaining)


class ContinuousMotionTests(unittest.TestCase):
    def setUp(self):
        self.shaper = ContinuousMotionShaper(CONFIG)

    def advance(self, prior, goal, actual=None, dt=.01, governed=False, proposed=None, remaining=None):
        result, diagnostics = self.shaper.advance(prior, goal, proposed or goal,
            actual or prior, dt, governed=governed, angular_remaining_rad=remaining)
        self.shaper.sync(prior, result, dt)
        return result, diagnostics

    def assert_limits(self, diagnostics):
        self.assertLessEqual(diagnostics['linear_speed_m_s'], .2+1e-10)
        self.assertLessEqual(diagnostics['angular_speed_rad_s'], 1.4+1e-10)
        self.assertLessEqual(diagnostics['linear_accel_m_s2'], 3.+1e-8)
        self.assertLessEqual(diagnostics['angular_accel_rad_s2'], 12.+1e-8)
        self.assertFalse(diagnostics['acceleration_limit_overridden'])

    def test_fixed_translation_accelerates_cruises_and_brakes_without_overshoot(self):
        prior = REST.copy()
        goal = [.3, 0., 0.]+REST[3:]
        speeds = []
        for _ in range(250):
            previous = prior
            prior, diagnostics = self.advance(prior, goal)
            self.assert_limits(diagnostics)
            self.assertGreaterEqual(prior[0]+1e-12, previous[0])
            self.assertLessEqual(prior[0], .3+1e-12)
            speeds.append(diagnostics['linear_speed_m_s'])
        self.assertAlmostEqual(speeds[0], .03)
        self.assertAlmostEqual(max(speeds), .2)
        self.assertAlmostEqual(prior[0], .3, places=10)
        self.assertLess(speeds[-1], 1e-8)

    def test_speed_and_acceleration_use_vector_norm_not_each_axis(self):
        prior = REST.copy()
        goal = [.2, .2, .2]+REST[3:]
        for _ in range(100):
            prior, diagnostics = self.advance(prior, goal)
            self.assert_limits(diagnostics)
            self.assertAlmostEqual(prior[0], prior[1])
            self.assertAlmostEqual(prior[1], prior[2])

    def test_translation_reversal_brakes_before_changing_direction(self):
        prior = REST.copy()
        for _ in range(20):
            prior, _ = self.advance(prior, [.5, 0, 0]+REST[3:])
        reversed_at = prior[0]
        goal = [-.1, 0, 0]+REST[3:]
        speeds = []
        for _ in range(20):
            prior, diagnostics = self.advance(prior, goal)
            self.assert_limits(diagnostics)
            speeds.append(diagnostics['linear_velocity_m_s'][0])
        self.assertGreater(speeds[0], 0.)
        self.assertLess(speeds[-1], 0.)
        self.assertLess(prior[0], reversed_at)

    def test_stationary_target_never_integrates_drift(self):
        prior = REST.copy()
        for _ in range(1000):
            prior, diagnostics = self.advance(prior, REST)
            self.assertEqual(prior, REST)
            self.assert_limits(diagnostics)
            self.assertEqual(diagnostics['linear_speed_m_s'], 0.)
            self.assertEqual(diagnostics['angular_speed_rad_s'], 0.)

    def test_stopped_after_fixed_goal_has_no_later_drift(self):
        prior = REST.copy()
        goal = [.01, -.02, .03]+REST[3:]
        for _ in range(1000):
            prior, _ = self.advance(prior, goal)
        stopped = prior.copy()
        for _ in range(1000):
            prior, _ = self.advance(prior, goal)
        self.assertEqual(prior, stopped)
        self.assertLess(norm([a-b for a, b in zip(prior[:3], goal[:3])]), 1e-10)

    def test_signed_twist_over_180_uses_proposal_not_goal_shortest_path(self):
        for sign in (1, -1):
            for quaternion_flip in (1, -1):
                with self.subTest(sign=sign, quaternion_flip=quaternion_flip):
                    self.shaper.reset()
                    prior = z_pose(sign*math.radians(5))
                    goal = z_pose(sign*math.radians(190))
                    proposed = z_pose(sign*math.radians(5)+sign*.014)
                    proposed = proposed[:3]+[quaternion_flip*v for v in proposed[3:]]
                    result, diagnostics = self.advance(prior, goal, proposed=proposed,
                        remaining=math.radians(185))
                    self.assertGreater(sign*diagnostics['angular_velocity_rad_s'][2], 0.)
                    self.assertGreater(sign*(2*math.atan2(result[6], result[3])-sign*math.radians(5)), 0.)
                    self.assert_limits(diagnostics)

    def test_signed_180_rotation_converges_with_speed_and_acceleration_bounds(self):
        for goal_degrees in (180, -180, 190, -190):
            with self.subTest(degrees=goal_degrees):
                self.shaper.reset()
                prior = REST.copy()
                requested = math.radians(goal_degrees)
                for _ in range(400):
                    proposed, remaining = proposed_rotation(prior, requested)
                    prior, diagnostics = self.advance(prior, z_pose(requested), proposed=proposed,
                                                       remaining=remaining)
                    self.assert_limits(diagnostics)
                self.assertAlmostEqual(2*math.atan2(prior[6], prior[3]), requested, places=8)

    def test_rotation_direction_is_world_small_increment_on_tilted_tool(self):
        root = [math.cos(math.pi/4), math.sin(math.pi/4), 0., 0.]
        prior = REST[:3]+root
        # Small tool-Z motion maps to the world -Y axis at this tool orientation.
        step = [math.cos(.014/2), 0., 0., math.sin(.014/2)]
        goal = REST[:3]+multiply(root, [math.cos(3.2/2), 0., 0., math.sin(3.2/2)])
        proposed = REST[:3]+multiply(root, step)
        _, diagnostics = self.advance(prior, goal, proposed=proposed, remaining=3.2)
        self.assertLess(diagnostics['angular_velocity_rad_s'][1], 0.)
        self.assertAlmostEqual(diagnostics['angular_velocity_rad_s'][0], 0.)
        self.assertAlmostEqual(diagnostics['angular_velocity_rad_s'][2], 0.)

    def test_angular_reversal_has_bounded_deceleration(self):
        prior = REST.copy()
        for _ in range(30):
            proposal, remaining = proposed_rotation(prior, 1.)
            prior, _ = self.advance(prior, z_pose(1.), proposed=proposal, remaining=remaining)
        for _ in range(40):
            proposal, remaining = proposed_rotation(prior, -.5)
            prior, diagnostics = self.advance(prior, z_pose(-.5), proposed=proposal, remaining=remaining)
            self.assert_limits(diagnostics)
        self.assertLess(diagnostics['angular_velocity_rad_s'][2], 0.)

    def test_inward_translation_is_immediately_allowed_above_soft_lead(self):
        prior = [.023, 0, 0]+REST[3:]
        result, diagnostics = self.advance(prior, [-.1, 0, 0]+REST[3:], REST, governed=True)
        self.assertLess(result[0], prior[0])
        self.assertEqual(diagnostics['linear_governor_ratio'], 1.)
        self.assertFalse(diagnostics['continuous_slowdown'])

    def test_tangent_translation_is_not_blocked_by_radial_lead(self):
        prior = [.023, 0, 0]+REST[3:]
        result, diagnostics = self.advance(prior, [.023, .1, 0]+REST[3:], REST, governed=True)
        self.assertEqual(result[0], prior[0])
        self.assertGreater(result[1], 0.)
        self.assertEqual(diagnostics['linear_governor_ratio'], 1.)

    def test_outward_governor_is_continuous_and_directional(self):
        for distance, expected in ((.005, 1-.25**4), (.01, 1-.5**4), (.015, 1-.75**4), (.02, 0.)):
            with self.subTest(distance=distance):
                self.shaper.reset()
                prior = [distance, 0, 0]+REST[3:]
                _, diagnostics = self.advance(prior, [.5, 0, 0]+REST[3:], REST, governed=True)
                self.assertAlmostEqual(diagnostics['linear_governor_ratio'], expected)
        # At the radial cap a mixed command still retains its tangent component.
        self.shaper.reset()
        prior = [.02, 0, 0]+REST[3:]
        result, diagnostics = self.advance(prior, [.12, .1, 0]+REST[3:], REST, governed=True)
        self.assertAlmostEqual(result[0], prior[0])
        self.assertGreater(result[1], 0.)
        self.assertTrue(diagnostics['continuous_slowdown'])

    def test_inward_angular_motion_is_allowed_above_soft_lead(self):
        prior = z_pose(.14)
        result, diagnostics = self.advance(prior, z_pose(0), REST, governed=True,
            proposed=z_pose(.126), remaining=.14)
        self.assertLess(2*math.atan2(result[6], result[3]), .14)
        self.assertEqual(diagnostics['angular_governor_ratio'], 1.)

    def test_outward_angular_motion_slows_instead_of_full_clutch_pause(self):
        prior = z_pose(.11)
        result, diagnostics = self.advance(prior, z_pose(.5), REST, governed=True,
            proposed=z_pose(.124), remaining=.39)
        self.assertGreater(diagnostics['angular_governor_ratio'], 0.)
        self.assertLess(diagnostics['angular_governor_ratio'], 1.)
        self.assertGreater(2*math.atan2(result[6], result[3]), .11)

    def test_twenty_and_thirty_hz_delayed_feedback_stays_below_hard_bound(self):
        for feedback_hz in (20, 30):
            for robot_tau in (.15, .6):
                with self.subTest(feedback_hz=feedback_hz, robot_tau=robot_tau):
                    self.shaper.reset()
                    prior = REST.copy()
                    physical = 0.
                    actual = REST.copy()
                    peak = 0.
                    slowed = 0
                    for tick in range(1, 1201):
                        physical += (prior[0]-physical)*(1-math.exp(-.01/robot_tau))
                        if int(tick*feedback_hz/100) != int((tick-1)*feedback_hz/100):
                            actual[0] = physical
                        requested = .15*math.sin(tick*.01*2*math.pi*.3)
                        goal = [requested, 0, 0]+REST[3:]
                        prior, diagnostics = self.advance(prior, goal, actual, governed=True)
                        self.assert_limits(diagnostics)
                        peak = max(peak, abs(prior[0]-actual[0]))
                        slowed += diagnostics['continuous_slowdown']
                    self.assertLess(peak, .03)
                    self.assertGreater(slowed, 100)

    def test_delayed_angular_feedback_stays_below_hard_bound_and_smooth(self):
        for feedback_hz in (20, 30):
            with self.subTest(feedback_hz=feedback_hz):
                self.shaper.reset()
                prior = REST.copy()
                physical = 0.
                actual = REST.copy()
                peak = 0.
                for tick in range(1, 1001):
                    command = 2*math.atan2(prior[6], prior[3])
                    physical += (command-physical)*(1-math.exp(-.01/.2))
                    if int(tick*feedback_hz/100) != int((tick-1)*feedback_hz/100):
                        actual = z_pose(physical)
                    requested = .6*math.sin(tick*.01*2*math.pi*.3)
                    proposal, remaining = proposed_rotation(prior, requested)
                    prior, diagnostics = self.advance(prior, z_pose(requested), actual,
                        governed=True, proposed=proposal, remaining=remaining)
                    self.assert_limits(diagnostics)
                    peak = max(peak, angle(multiply(prior[3:], inverse(actual[3:]))))
                self.assertLess(peak, .25)

    def test_soft_lead_includes_finite_acceleration_limited_stopping_margin(self):
        prior = [.0195, 0, 0]+REST[3:]
        self.shaper.sync([.0175, 0, 0]+REST[3:], prior, .01)
        peak = prior[0]
        for _ in range(100):
            prior, diagnostics = self.advance(prior, [.5, 0, 0]+REST[3:], REST, governed=True)
            self.assert_limits(diagnostics)
            peak = max(peak, prior[0])
        self.assertGreater(peak, .02)
        self.assertLess(peak, .02+.2**2/(2*3)+.2*.01)
        self.assertLess(diagnostics['linear_speed_m_s'], 1e-8)

    def test_advance_is_preview_until_sync_and_rejected_proposal_has_no_windup(self):
        goal = [.5, 0, 0]+REST[3:]
        first, _ = self.shaper.advance(REST, goal, goal, REST, .01, governed=False)
        second, _ = self.shaper.advance(REST, goal, goal, REST, .01, governed=False)
        self.assertEqual(first, second)
        # IK accepts only a quarter of the first proposal.
        accepted = [first[0]/4, 0, 0]+REST[3:]
        self.shaper.sync(REST, accepted, .01)
        _, diagnostics = self.shaper.advance(accepted, goal, goal, accepted, .01, governed=False)
        self.assertAlmostEqual(diagnostics['linear_velocity_m_s'][0], .0075+.03)
        self.shaper.sync(accepted, accepted, .01)
        _, diagnostics = self.shaper.advance(accepted, goal, goal, accepted, .01, governed=False)
        self.assertAlmostEqual(diagnostics['linear_speed_m_s'], .03)

    def test_reset_zeros_previous_linear_and_angular_velocity(self):
        self.shaper.sync(REST, z_pose(.014, [.002, 0, 0]), .01)
        self.shaper.reset()
        _, diagnostics = self.advance(REST, z_pose(1., [.2, 0, 0]), proposed=z_pose(.014))
        self.assertAlmostEqual(diagnostics['linear_speed_m_s'], .03)
        self.assertAlmostEqual(diagnostics['angular_speed_rad_s'], .12)

    def test_external_overspeed_sync_is_explicitly_diagnosed(self):
        self.shaper.sync(REST, [.01, 0, 0]+REST[3:], .01)
        _, diagnostics = self.shaper.advance(REST, [.5, 0, 0]+REST[3:], REST, REST, .01,
                                            governed=False)
        self.assertTrue(diagnostics['acceleration_limit_overridden'])
        self.assertIn('linear_speed_clip', diagnostics['acceleration_override_reasons'])
        self.assertGreater(diagnostics['linear_accel_m_s2'], 3.)

    def test_invalid_pose_dt_and_path_budget_raise_instead_of_hold(self):
        for value in (math.nan, math.inf, True, 'bad'):
            bad = REST.copy()
            bad[0] = value
            for argument in ('prior', 'goal', 'proposed', 'actual'):
                with self.subTest(value=value, argument=argument):
                    values = {name: REST for name in ('prior', 'goal', 'proposed', 'actual')}
                    values[argument] = bad
                    with self.assertRaises(MappingError):
                        self.shaper.advance(values['prior'], values['goal'], values['proposed'], values['actual'], .01)
        for dt in (0, -1, True, math.nan, math.inf):
            with self.subTest(dt=dt), self.assertRaises(MappingError):
                self.shaper.advance(REST, REST, REST, REST, dt)
        for remaining in (-1, True, math.nan, math.inf):
            with self.subTest(remaining=remaining), self.assertRaises(MappingError):
                self.shaper.advance(REST, REST, REST, REST, .01, angular_remaining_rad=remaining)
        with self.assertRaises(MappingError):
            self.shaper.advance(REST, REST[:3]+[0, 0, 0, 0], REST, REST, .01)
        with self.assertRaises(MappingError):
            self.shaper.advance(REST, [1e308, 0, 0]+REST[3:], REST, REST, .01)

    def test_invalid_acceleration_bound_and_sync_are_rejected(self):
        for key in CONFIG:
            for value in (0, -1, True, math.nan, math.inf):
                with self.subTest(key=key, value=value):
                    config = CONFIG.copy()
                    config[key] = value
                    with self.assertRaises(MappingError):
                        ContinuousMotionShaper(config)
        with self.assertRaises(MappingError):
            self.shaper.sync(REST, [math.nan, 0, 0]+REST[3:], .01)
        self.assertEqual(self.shaper.linear_velocity, [0., 0., 0.])


if __name__ == '__main__':
    unittest.main()
