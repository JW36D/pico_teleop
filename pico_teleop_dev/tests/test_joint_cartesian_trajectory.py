"""Cartesian geometry and box-QP tests; FK callbacks are local fake math."""
import itertools
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incremental import MappingError, inverse, multiply, norm
from joint_trajectory import (JointTrajectory, _box_qp, _rotation_log,
                              _solve_positive_definite)
from test_joint_trajectory import CONFIG, MathBackend


def axis_quaternion(axis, degrees):
    half = math.radians(degrees)/2
    return [math.cos(half)]+[math.sin(half) if index == axis else 0. for index in range(3)]


class CoupledBackend(MathBackend):
    def __init__(self):
        self.calls = 0
        self.sign_flip = False
        self.rank_deficient = False
    def forward(self, joints):
        self.calls += 1
        if any(not low <= value <= high for value, low, high in
               zip(joints, self.minimum, self.maximum)):
            raise AssertionError('FK difference crossed a physical limit')
        # q1 must compensate q0 to keep a straight x path. Independent rate
        # clipping misses this coupling and can move the end effector sideways.
        rotation = multiply(axis_quaternion(2, 0. if self.rank_deficient else joints[5]),
                            multiply(axis_quaternion(1, joints[4]), axis_quaternion(0, joints[3])))
        if self.sign_flip and rotation[0] < 0:
            rotation = [-value for value in rotation]
        return [joints[0]/1000., (joints[0]+joints[1])/1000., joints[2]/1000.]+rotation
    def send(self, *_):
        raise AssertionError('Unexpected hardware command in arithmetic optimizer')


class BoxQPTests(unittest.TestCase):
    def objective(self, matrix, rhs, value):
        return .5*sum(value[i]*matrix[i][j]*value[j]
                       for i in range(len(value)) for j in range(len(value)))-sum(a*b for a, b in zip(rhs, value))

    def brute(self, matrix, rhs, lower, upper):
        size = len(rhs)
        best = None
        for status in itertools.product((-1, 0, 1), repeat=size):
            free = [index for index in range(size) if status[index] == 0]
            fixed = [index for index in range(size) if status[index] != 0]
            value = [low if state == -1 else high if state == 1 else 0.
                     for low, high, state in zip(lower, upper, status)]
            if free:
                reduced = [[matrix[i][j] for j in free] for i in free]
                reduced_rhs = [rhs[i]-sum(matrix[i][j]*value[j] for j in fixed) for i in free]
                solution = _solve_positive_definite(reduced, reduced_rhs)
                for index, result in zip(free, solution):
                    value[index] = result
            if any(not low-1e-8 <= x <= high+1e-8 for x, low, high in zip(value, lower, upper)):
                continue
            objective = self.objective(matrix, rhs, value)
            if best is None or objective < best[0]:
                best = objective, value
        return best

    def test_wrong_initial_active_bounds_are_released(self):
        value, diagnostic = _box_qp([[4., 3.], [3., 4.]], [1., 1.], [-1., -1.], [1., 1.], [1., 1.])
        self.assertGreaterEqual(diagnostic['bound_releases'], 2)
        self.assertAlmostEqual(value[0], 1./7.)
        self.assertAlmostEqual(value[1], 1./7.)
        self.assertLess(diagnostic['kkt_residual'], 1e-10)

    def test_coupled_bound_optimum_differs_from_independent_clipping(self):
        # The unconstrained optimum is [100,-100]. Limiting the first axis
        # forces the optimum second axis to -20 to preserve the y constraint.
        matrix = [[2., 1.], [1., 1.]]
        value, _ = _box_qp(matrix, [100., 0.], [-20., -120.], [20., 120.], [0., 0.])
        self.assertAlmostEqual(value[0], 20.)
        self.assertAlmostEqual(value[1], -20.)

    def test_random_three_axis_optima_match_exhaustive_faces(self):
        generator = random.Random(981)
        releases = 0
        for _ in range(60):
            factors = [[generator.uniform(-2., 2.) for _ in range(3)] for _ in range(3)]
            matrix = [[sum(row[i]*row[j] for row in factors)+(1e-4 if i == j else 0.)
                       for j in range(3)] for i in range(3)]
            rhs = [generator.uniform(-4., 4.) for _ in range(3)]
            lower, upper = [-1., -2., -.5], [2., 1., .75]
            initial = [generator.choice((low, high, 0.)) for low, high in zip(lower, upper)]
            result, diagnostics = _box_qp(matrix, rhs, lower, upper, initial)
            expected, _ = self.brute(matrix, rhs, lower, upper)
            self.assertAlmostEqual(self.objective(matrix, rhs, result), expected, places=8)
            self.assertLess(diagnostics['kkt_residual'], 1e-7)
            releases += diagnostics['bound_releases']
        self.assertGreater(releases, 0)

    def test_six_axis_singular_like_system_matches_exhaustive_faces(self):
        factors = [[1., 1., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 1., 1., 0.], [0., 0., 0., 0., 0., 1e-5]]
        matrix = [[sum(row[i]*row[j] for row in factors)+(1e-8 if i == j else 0.)
                   for j in range(6)] for i in range(6)]
        rhs = [1., 1., .2, -.5, -.5, 1e-8]
        lower, upper = [-.2]*6, [.8]*6
        result, diagnostics = _box_qp(matrix, rhs, lower, upper, upper)
        expected, _ = self.brute(matrix, rhs, lower, upper)
        self.assertAlmostEqual(self.objective(matrix, rhs, result), expected, places=9)
        self.assertTrue(diagnostics['converged'])


class JointCartesianTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.backend = CoupledBackend()
        self.config = {**CONFIG, 'max_linear_speed_m_s': .25,
                       'max_angular_speed_rad_s': 1.4,
                       'cartesian_rotation_weight_m': .25/1.4}
        self.trajectory = JointTrajectory(self.config, self.backend)

    def step(self, desired, dt=.01):
        previous = self.trajectory.joints.copy()
        previous_velocity = self.trajectory.velocity.copy()
        target = self.backend.forward(previous)
        candidate, diagnostics = self.trajectory.propose_cartesian(target, dt, self.backend,
                                                                  target_cartesian_velocity=desired)
        self.trajectory.commit(candidate, dt)
        for index, (before, after) in enumerate(zip(previous, candidate)):
            velocity = (after-before)/dt
            acceleration = (velocity-previous_velocity[index])/dt
            self.assertLessEqual(abs(velocity), self.trajectory.speed_limits[index]+1e-7)
            self.assertLessEqual(abs(acceleration), self.trajectory.acceleration_limits[index]+1e-6)
            self.assertGreaterEqual(after, self.backend.minimum[index])
            self.assertLessEqual(after, self.backend.maximum[index])
        self.assertTrue(diagnostics['joint_optimizer_converged'])
        self.assertLess(diagnostics['joint_optimizer_kkt_residual'], 1e-8)
        return candidate, diagnostics

    def test_jointly_compensates_coupled_axes_instead_of_sideways_drift(self):
        self.backend.speed_limits = [20., 120., 120., 120., 120., 120.]
        self.trajectory = JointTrajectory(self.config, self.backend)
        for _ in range(100):
            joints, diagnostics = self.step([.1, 0., 0., 0., 0., 0.])
            self.assertLess(abs(self.backend.forward(joints)[1]), 1e-7)
        self.assertAlmostEqual(diagnostics['joint_velocity_deg_s'][0], 20., places=7)
        self.assertAlmostEqual(diagnostics['joint_velocity_deg_s'][1], -20., places=5)
        self.assertLess(diagnostics['cartesian_linear_velocity_residual_m_s'], .081)

    def test_final_acceleration_slowdown_preserves_cartesian_direction(self):
        for index in range(100):
            desired = [.1 if index < 50 else -.1, 0., 0., 0., 0., 0.]
            joints, _ = self.step(desired)
            self.assertLess(abs(self.backend.forward(joints)[1]), 1e-6)
        self.assertLess(self.trajectory.velocity[0], 0.)

    def test_real_bound_braking_preserves_compensating_joint_motion(self):
        self.backend.maximum = [.5]+MathBackend.maximum[1:]
        self.trajectory = JointTrajectory(self.config, self.backend)
        for index in range(300):
            joints, _ = self.step([.1, 0., 0., 0., 0., 0.],
                                  dt=(.005, .008, .01)[index % 3])
            self.assertLess(abs(self.backend.forward(joints)[1]), 1e-6)
        self.assertAlmostEqual(joints[0], .5, places=8)
        prior = joints[0]
        joints, _ = self.step([-.1, 0., 0., 0., 0., 0.])
        self.assertLess(joints[0], prior)

    def test_zero_goal_decays_useless_singular_nullspace_motion(self):
        self.backend.rank_deficient = True
        self.trajectory.reset([0.]*6, [0., 0., 0., 0., 0., 30.])
        for _ in range(50):
            _, diagnostic = self.step([0.]*6)
        self.assertLess(abs(self.trajectory.velocity[5]), 1e-8)
        self.assertLess(diagnostic['cartesian_weighted_velocity_residual_m_s'], 1e-8)

    def test_preview_does_not_commit_and_uses_only_local_fk(self):
        target = [0., 0., 0., 1., 0., 0., 0.]
        self.trajectory.propose_cartesian(target, .01, self.backend,
                                          target_cartesian_velocity=[.1, 0., 0., 0., 0., 0.])
        self.assertEqual(self.trajectory.joints, [0.]*6)
        self.assertEqual(self.trajectory.velocity, [0.]*6)
        self.assertEqual(self.backend.calls, 13)

    def test_finite_difference_stays_inside_real_limits(self):
        self.trajectory.reset([180., -180., 0., 0., 0., 0.])
        pose = self.backend.forward(self.trajectory.joints)
        result, _ = self.trajectory.propose_cartesian(pose, .01, self.backend)
        self.assertEqual(result, self.trajectory.joints)

    def test_local_rotation_after_180_degrees_keeps_direction_with_canonical_fk(self):
        self.backend.sign_flip = True
        self.trajectory.reset([0., 0., 0., 0., 0., 179.])
        for _ in range(100):
            before = self.trajectory.joints[5]
            joints, diagnostics = self.step([0., 0., 0., 0., 0., .2])
            self.assertGreater(joints[5], before)
            self.assertGreater(diagnostics['cartesian_achieved_velocity_m_s_rad_s'][5], 0.)
        self.assertGreater(joints[5], 180.)

    def test_pose_argument_uses_local_world_rotation_log(self):
        self.trajectory.reset([0., 0., 0., 35., -20., 60.])
        prior = self.backend.forward(self.trajectory.joints)
        delta = axis_quaternion(0, .03)
        target = prior[:3]+multiply(delta, prior[3:])
        _, diagnostics = self.trajectory.propose_cartesian(target, .01, self.backend)
        self.assertAlmostEqual(diagnostics['cartesian_requested_velocity_m_s_rad_s'][3], math.radians(3.), places=8)
        self.assertLess(norm(diagnostics['cartesian_requested_velocity_m_s_rad_s'][4:]), 1e-8)

    def test_explicit_twist_prevents_sender_period_from_scaling_nominal_goal(self):
        target = [0., 0., 0., 1., 0., 0., 0.]
        _, diagnostics = self.trajectory.propose_cartesian(target, .02, self.backend,
            target_cartesian_velocity=[.1, 0., 0., 0., 0., 0.])
        self.assertEqual(diagnostics['cartesian_requested_velocity_m_s_rad_s'], [.1, 0., 0., 0., 0., 0.])
        self.assertAlmostEqual(diagnostics['cartesian_rotation_weight_m'], .25/1.4)

    def test_invalid_twist_and_duplicate_aliases_leave_state_unchanged(self):
        target = [0., 0., 0., 1., 0., 0., 0.]
        with self.assertRaises(MappingError):
            self.trajectory.propose_cartesian(target, .01, self.backend,
                target_cartesian_velocity=[math.nan, 0., 0., 0., 0., 0.])
        with self.assertRaises(MappingError):
            self.trajectory.propose_cartesian(target, .01, self.backend,
                target_cartesian_velocity=[0.]*6, desired_velocity=[0.]*6)
        self.assertEqual(self.trajectory.joints, [0.]*6)


if __name__ == '__main__':
    unittest.main()
