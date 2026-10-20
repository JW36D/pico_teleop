import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incremental import MappingError
from joint_trajectory import JointTrajectory
from robot_backend import Backend


class MathBackend:
    minimum = [-180., -180., -180., -180., -180., -720.]
    maximum = [180., 180., 180., 180., 180., 720.]
    initial_joints = [0.]*6
    speed_limits = [180., 180., 225., 225., 225., 225.]
    acceleration_limits = [600.]*6


CONFIG = {'max_joint_speed_deg_s': 120., 'max_joint_accel_deg_s2': 600.}


class JointTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.backend = MathBackend()
        self.trajectory = JointTrajectory(CONFIG, self.backend)

    def check_step(self, previous_joints, previous_velocity, emitted, dt, diagnostics):
        velocities = [(after-before)/dt for before, after in zip(previous_joints, emitted)]
        accelerations = [(after-before)/dt for before, after in zip(previous_velocity, velocities)]
        for index, (joint, velocity, acceleration) in enumerate(zip(emitted, velocities, accelerations)):
            self.assertGreaterEqual(joint, self.backend.minimum[index])
            self.assertLessEqual(joint, self.backend.maximum[index])
            self.assertLessEqual(abs(velocity), self.trajectory.speed_limits[index]+1e-7)
            self.assertLessEqual(abs(acceleration), self.trajectory.acceleration_limits[index]+1e-6)
            self.assertAlmostEqual(velocity, diagnostics['joint_velocity_deg_s'][index], places=8)
            self.assertAlmostEqual(acceleration, diagnostics['joint_acceleration_deg_s2'][index], places=7)

    def step(self, goal, dt=.01, velocity=None, scale=1.):
        previous_joints, previous_velocity = self.trajectory.joints.copy(), self.trajectory.velocity.copy()
        emitted, diagnostics = self.trajectory.advance(goal, dt,
            target_velocity_deg_s=velocity, speed_scale=scale)
        self.check_step(previous_joints, previous_velocity, emitted, dt, diagnostics)
        return emitted, diagnostics

    def test_static_goal_accelerates_cruises_brakes_and_settles(self):
        goal = [100., -80., 60., -40., 20., 190.]
        peak = [0.]*6
        for _ in range(500):
            emitted, diagnostics = self.step(goal)
            for index, value in enumerate(emitted):
                self.assertGreaterEqual(value*math.copysign(1., goal[index]), -1e-10)
                self.assertLessEqual(abs(value), abs(goal[index])+1e-10)
                peak[index] = max(peak[index], abs(diagnostics['joint_velocity_deg_s'][index]))
        self.assertAlmostEqual(peak[0], 120.)
        self.assertAlmostEqual(peak[5], 120.)
        for actual, target in zip(emitted, goal):
            self.assertAlmostEqual(actual, target, places=9)
        self.assertLess(max(abs(value) for value in self.trajectory.velocity), 1e-8)

    def test_local_ik_velocity_avoids_near_waypoint_second_braking(self):
        velocities = []
        for _ in range(60):
            goal = self.trajectory.joints.copy()
            goal[0] += 1.2
            emitted, diagnostics = self.step(goal, velocity=[120., 0., 0., 0., 0., 0.])
            velocities.append(diagnostics['joint_velocity_deg_s'][0])
        self.assertAlmostEqual(velocities[0], 6.)
        self.assertAlmostEqual(velocities[19], 120., places=7)
        self.assertAlmostEqual(velocities[-1], 120., places=7)
        self.assertGreater(emitted[0], 60.)

    def test_feedforward_velocity_is_limited_by_final_joint_acceleration(self):
        _, first = self.step([1.]*6, velocity=[120., -120., 120., -120., 120., -120.])
        self.assertEqual(first['joint_velocity_deg_s'], [6., -6., 6., -6., 6., -6.])
        self.assertTrue(all(first['joint_acceleration_limited']))

    def test_target_reversal_brakes_before_reversing_without_step_jump(self):
        for _ in range(40):
            self.step([100., 0., 0., 0., 0., 0.])
        prior = self.trajectory.joints[0]
        emitted, diagnostics = self.step([-100., 0., 0., 0., 0., 0.])
        self.assertGreater(emitted[0], prior)
        self.assertAlmostEqual(diagnostics['joint_velocity_deg_s'][0], 114., places=7)
        for _ in range(50):
            emitted, diagnostics = self.step([-100., 0., 0., 0., 0., 0.])
        self.assertLess(diagnostics['joint_velocity_deg_s'][0], 0.)
        self.assertLess(emitted[0], prior)

    def test_zero_velocity_goal_brakes_instead_of_freezing(self):
        for _ in range(25):
            goal = self.trajectory.joints.copy()
            goal[0] += 1.2
            self.step(goal, velocity=[120., 0., 0., 0., 0., 0.])
        goal = self.trajectory.joints.copy()
        emitted, diagnostics = self.step(goal, velocity=[0.]*6)
        self.assertGreater(emitted[0], goal[0])
        self.assertAlmostEqual(diagnostics['joint_velocity_deg_s'][0], 114., places=7)
        for _ in range(30):
            emitted, diagnostics = self.step(goal, velocity=[0.]*6)
        self.assertAlmostEqual(diagnostics['joint_velocity_deg_s'][0], 0.)
        stopped = emitted.copy()
        for _ in range(10):
            emitted, _ = self.step(goal, velocity=[0.]*6)
        self.assertEqual(emitted, stopped)

    def test_abrupt_speed_scale_zero_respects_deceleration(self):
        for _ in range(40):
            self.step([100., 0., 0., 0., 0., 0.])
        _, diagnostics = self.step([100., 0., 0., 0., 0., 0.], scale=0.)
        self.assertAlmostEqual(diagnostics['joint_velocity_deg_s'][0], 114., places=7)
        self.assertEqual(diagnostics['joint_desired_speed_scale'], 0.)

    def test_hardware_limit_brakes_in_advance_and_allows_reverse(self):
        self.backend.maximum = [20.]+MathBackend.maximum[1:]
        self.trajectory = JointTrajectory(CONFIG, self.backend)
        braking_seen = False
        for _ in range(200):
            emitted, diagnostics = self.step([20., 0., 0., 0., 0., 0.],
                                            velocity=[120., 0., 0., 0., 0., 0.])
            braking_seen |= diagnostics['joint_position_braking'][0]
        self.assertTrue(braking_seen)
        self.assertAlmostEqual(emitted[0], 20., places=9)
        held = emitted[0]
        emitted, _ = self.step([19., 0., 0., 0., 0., 0.],
                               velocity=[-120., 0., 0., 0., 0., 0.])
        self.assertLess(emitted[0], held)

    def test_infeasible_moving_state_at_hardware_bound_is_explicit_fault(self):
        position = [179.9, 0., 0., 0., 0., 0.]
        velocity = [100., 0., 0., 0., 0., 0.]
        self.trajectory.reset(position, velocity)
        with self.assertRaises(MappingError) as raised:
            self.trajectory.propose([180., 0., 0., 0., 0., 0.], .01)
        self.assertEqual(raised.exception.details['state'], 'JOINT_TRAJECTORY_INFEASIBLE')
        self.assertEqual(self.trajectory.joints, position)
        self.assertEqual(self.trajectory.velocity, velocity)

    def test_variable_period_preserves_limits_after_random_goal_reversals(self):
        generator = random.Random(73)
        for _ in range(1500):
            goal = [generator.uniform(-179., 179.) for _ in range(5)]+[generator.uniform(-700., 700.)]
            dt = generator.choice([.005, .008, .01, .02])
            feedforward = [generator.uniform(-240., 240.) for _ in range(6)]
            self.step(goal, dt=dt, velocity=feedforward)

    def test_period_changes_near_hardware_bound_keep_braking_feasible(self):
        self.backend.maximum = [5.]+MathBackend.maximum[1:]
        self.trajectory = JointTrajectory(CONFIG, self.backend)
        for index in range(600):
            dt = (.02, .005, .01, .008)[index % 4]
            emitted, _ = self.step([5., 0., 0., 0., 0., 0.], dt=dt,
                velocity=[120., 0., 0., 0., 0., 0.])
            # The remaining position budget covers continuous braking even
            # when the next controller period becomes shorter.
            stopping_distance = self.trajectory.velocity[0]**2/(2*600.)
            self.assertLessEqual(stopping_distance, 5.-emitted[0]+1e-10)
        self.assertAlmostEqual(emitted[0], 5., places=9)

    def test_real_device_speed_and_acceleration_reduce_configured_limits(self):
        self.backend.speed_limits = [60., 90., 180., 180., 180., 180.]
        self.backend.acceleration_limits = [150., 300., 900., 900., 900., 900.]
        self.trajectory = JointTrajectory(CONFIG, self.backend)
        for _ in range(100):
            _, diagnostics = self.step([120.]*6)
        self.assertEqual(diagnostics['effective_joint_speed_limits_deg_s'], [60., 90., 120., 120., 120., 120.])
        self.assertEqual(diagnostics['effective_joint_acceleration_limits_deg_s2'], [150., 300., 600., 600., 600., 600.])

    def test_moving_handover_does_not_reset_velocity(self):
        start = [40., -20., 0., 0., 0., 0.]
        velocity = [80., -40., 0., 0., 0., 0.]
        self.trajectory.reset(start, velocity)
        emitted, diagnostics = self.step([100., -100., 0., 0., 0., 0.])
        self.assertAlmostEqual((emitted[0]-40.)/.01, 86., places=8)
        self.assertAlmostEqual(diagnostics['joint_acceleration_deg_s2'][0], 600., places=6)

    def test_handover_above_task_speed_brakes_without_velocity_clip(self):
        self.trajectory.reset([0.]*6, [150., 0., 0., 0., 0., 0.])
        emitted, diagnostics = self.trajectory.advance([100., 0., 0., 0., 0., 0.], .01)
        self.assertAlmostEqual(emitted[0], 1.44)
        self.assertAlmostEqual(diagnostics['joint_velocity_deg_s'][0], 144.)
        self.assertAlmostEqual(diagnostics['joint_acceleration_deg_s2'][0], -600.)
        self.assertTrue(diagnostics['joint_startup_speed_recovery'][0])
        for _ in range(5):
            self.trajectory.advance([100., 0., 0., 0., 0., 0.], .01)
        self.assertLessEqual(self.trajectory.velocity[0], 120.+1e-8)

    def test_stationary_target_has_no_drift(self):
        for _ in range(1000):
            emitted, diagnostics = self.step([0.]*6)
        self.assertEqual(emitted, [0.]*6)
        self.assertEqual(diagnostics['joint_acceleration_deg_s2'], [0.]*6)

    def test_j6_continuous_190_degree_goal_does_not_wrap(self):
        self.trajectory.reset([0., 0., 0., 0., 0., 175.])
        prior = 175.
        crossed = False
        for _ in range(200):
            emitted, _ = self.step([0., 0., 0., 0., 0., 190.])
            self.assertGreaterEqual(emitted[5], prior-1e-10)
            self.assertLessEqual(emitted[5]-prior, 1.2+1e-8)
            crossed |= emitted[5] > 180.
            prior = emitted[5]
        self.assertTrue(crossed)
        self.assertAlmostEqual(emitted[5], 190., places=9)

    def test_preview_cancel_has_no_hidden_velocity_accumulation(self):
        first, _ = self.trajectory.preview([100., 0., 0., 0., 0., 0.], .01)
        self.assertEqual(self.trajectory.joints, [0.]*6)
        self.assertEqual(self.trajectory.velocity, [0.]*6)
        opposite, _ = self.trajectory.preview([-100., 0., 0., 0., 0., 0.], .01)
        self.assertAlmostEqual(first[0], -opposite[0])
        self.trajectory.commit(opposite, .01)
        self.assertEqual(self.trajectory.joints, opposite)

    def test_external_clipping_cannot_silently_break_final_acceleration(self):
        with self.assertRaises(MappingError):
            self.trajectory.commit([1., 0., 0., 0., 0., 0.], .01)
        self.assertEqual(self.trajectory.joints, [0.]*6)
        self.assertEqual(self.trajectory.velocity, [0.]*6)

    def test_missing_device_capability_is_reported_as_unknown(self):
        class UnknownDevice:
            minimum = MathBackend.minimum
            maximum = MathBackend.maximum
            initial_joints = MathBackend.initial_joints
        trajectory = JointTrajectory(CONFIG, UnknownDevice())
        _, diagnostics = trajectory.advance([0.]*6, .01)
        self.assertEqual(diagnostics['joint_acceleration_limit_source'],
                         'configured_only_device_limit_unavailable')

    def test_invalid_numeric_inputs_are_rejected_without_state_change(self):
        for value in (0., -1., math.inf, math.nan, True):
            with self.subTest(dt=value), self.assertRaises(MappingError):
                self.trajectory.propose([0.]*6, value)
        for value in (-.1, 1.1, math.nan, True):
            with self.subTest(scale=value), self.assertRaises(MappingError):
                self.trajectory.propose([0.]*6, .01, speed_scale=value)
        with self.assertRaises(MappingError):
            self.trajectory.propose([0.]*6, .01, target_velocity_deg_s=[0., math.nan, 0., 0., 0., 0.])
        with self.assertRaises(MappingError):
            self.trajectory.reset([0.]*6, [181., 0., 0., 0., 0., 0.])
        self.assertEqual(self.trajectory.joints, [0.]*6)


class CapabilityReaderTests(unittest.TestCase):
    def reader(self, robot):
        backend = object.__new__(Backend)
        backend.robot = robot
        return backend

    def test_read_capabilities_is_read_only_and_disabled_collision_is_explicit(self):
        class ReadOnlyDevice:
            def __init__(self):
                self.calls = []
            def rm_get_joint_max_speed(self):
                self.calls.append('get_speed')
                return 0, [180.]*6
            def rm_get_joint_max_acc(self):
                self.calls.append('get_acc')
                return 0, [600.]*6
            def rm_get_self_collision_enable(self):
                self.calls.append('get_collision')
                return 0, False
            def rm_set_self_collision_enable(self, *_):
                raise AssertionError('Unexpected setting change')
        robot = ReadOnlyDevice()
        backend = self.reader(robot)
        diagnostics = backend.read_motion_capabilities()
        self.assertEqual(robot.calls, ['get_speed', 'get_acc', 'get_collision'])
        self.assertEqual(backend.acceleration_limits, [600.]*6)
        self.assertEqual(diagnostics['joint_acceleration']['units'], 'deg/s^2')
        self.assertEqual(diagnostics['self_collision']['status'], 'available')
        self.assertFalse(diagnostics['self_collision']['enabled'])

    def test_missing_getters_are_explicitly_unsupported(self):
        diagnostics = self.reader(object()).read_motion_capabilities()
        for key in ('joint_speed', 'joint_acceleration', 'self_collision'):
            self.assertEqual(diagnostics[key]['status'], 'unsupported')

    def test_failed_queries_and_invalid_values_are_not_claimed_available(self):
        class BrokenDevice:
            def rm_get_joint_max_speed(self):
                return -2, [180.]*6
            def rm_get_joint_max_acc(self):
                return 0, [0.]*6
            def rm_get_self_collision_enable(self):
                return 0, 1
        backend = self.reader(BrokenDevice())
        diagnostics = backend.read_motion_capabilities()
        self.assertFalse(hasattr(backend, 'acceleration_limits'))
        for key in ('joint_speed', 'joint_acceleration', 'self_collision'):
            self.assertEqual(diagnostics[key]['status'], 'unavailable')
        self.assertNotIn('enabled', diagnostics['self_collision'])


if __name__ == '__main__':
    unittest.main()
