"""Startup IK diagnostics must never replace the real joint command seed."""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incremental import MappingError
from robot_backend import Backend


INITIAL_JOINTS = [80., 105., .001, -28., 72., 8.5]
INITIAL_POSE = [.1, -.4, .2, 1., 0., 0., 0.]


class StartupIKTests(unittest.TestCase):
    def backend(self, solved=None, fk_pose=None):
        backend = object.__new__(Backend)
        backend.initial_joints = INITIAL_JOINTS.copy()
        backend.initial_pose = INITIAL_POSE.copy()
        backend.minimum = [-178., -130., -135., -178., -128., -360.]
        backend.maximum = [178., 130., 135., 178., 128., 360.]
        backend.solve = Mock(return_value=(INITIAL_JOINTS.copy()
                                          if solved is None else solved))
        backend.forward = Mock(return_value=(INITIAL_POSE.copy()
                                            if fk_pose is None else fk_pose))
        backend.send = Mock(side_effect=AssertionError('Startup cannot send motion'))
        return backend

    def assert_original_seed_preserved(self, backend, diagnostic):
        self.assertEqual(backend.initial_joints, INITIAL_JOINTS)
        self.assertEqual(backend.initial_pose, INITIAL_POSE)
        self.assertEqual(diagnostic['initial_joints_deg'], INITIAL_JOINTS)
        self.assertEqual(diagnostic['command_seed_source'], 'actual_joint_feedback')
        backend.send.assert_not_called()

    def test_near_straight_elbow_numerical_joint_difference_is_diagnostic(self):
        solved = INITIAL_JOINTS.copy()
        solved[1] -= .035
        solved[2] = .08
        solved[4] -= .044
        backend = self.backend(solved)
        result = backend.validate_startup_ik()
        self.assertGreater(result['max_joint_delta_deg'], .03)
        self.assertEqual(result['ik_joints_deg'], solved)
        self.assertAlmostEqual(result['joint_delta_deg'][2], .079)
        self.assertAlmostEqual(result['fk_position_error_m'], 0.)
        self.assertAlmostEqual(result['fk_orientation_error_rad'], 0.)
        self.assert_original_seed_preserved(backend, result)

    def test_geometrically_correct_large_branch_difference_does_not_command_it(self):
        solved = [-150., -40., -20., 130., -90., -260.]
        backend = self.backend(solved)
        result = backend.validate_startup_ik()
        self.assertEqual(result['ik_joints_deg'], solved)
        self.assertAlmostEqual(result['max_joint_delta_deg'], 268.5)
        self.assert_original_seed_preserved(backend, result)

    def test_solver_receives_copies_of_real_startup_pose_and_joint_seed(self):
        backend = self.backend()
        seen = {}

        def solve(target, seed):
            seen['target'] = target.copy()
            seen['seed'] = seed.copy()
            target[0] += .5
            seed[0] += 100.
            return INITIAL_JOINTS.copy()

        backend.solve.side_effect = solve
        result = backend.validate_startup_ik()
        self.assertEqual(seen['target'], INITIAL_POSE)
        self.assertEqual(seen['seed'], INITIAL_JOINTS)
        self.assert_original_seed_preserved(backend, result)

    def test_fk_position_error_inside_tolerance_is_accepted_and_reported(self):
        fk_pose = INITIAL_POSE.copy()
        fk_pose[0] += .000099
        backend = self.backend(fk_pose=fk_pose)
        result = backend.validate_startup_ik()
        self.assertAlmostEqual(result['fk_position_error_m'], .000099)
        self.assert_original_seed_preserved(backend, result)

    def test_fk_position_error_outside_tolerance_is_rejected_with_diagnostics(self):
        fk_pose = INITIAL_POSE.copy()
        fk_pose[0] += .000101
        backend = self.backend(fk_pose=fk_pose)
        with self.assertRaises(MappingError) as raised:
            backend.validate_startup_ik()
        self.assertAlmostEqual(raised.exception.details['fk_position_error_m'], .000101)
        self.assertEqual(raised.exception.details['initial_joints_deg'], INITIAL_JOINTS)
        self.assertEqual(backend.initial_joints, INITIAL_JOINTS)
        backend.send.assert_not_called()

    def test_fk_orientation_error_inside_tolerance_is_accepted_and_reported(self):
        rotation = .00099
        fk_pose = INITIAL_POSE[:3] + [math.cos(rotation/2), 0., 0., math.sin(rotation/2)]
        backend = self.backend(fk_pose=fk_pose)
        result = backend.validate_startup_ik()
        self.assertAlmostEqual(result['fk_orientation_error_rad'], rotation)
        self.assert_original_seed_preserved(backend, result)

    def test_fk_orientation_error_outside_tolerance_is_rejected(self):
        rotation = .00101
        fk_pose = INITIAL_POSE[:3] + [math.cos(rotation/2), 0., 0., math.sin(rotation/2)]
        backend = self.backend(fk_pose=fk_pose)
        with self.assertRaises(MappingError) as raised:
            backend.validate_startup_ik()
        self.assertAlmostEqual(raised.exception.details['fk_orientation_error_rad'], rotation)
        backend.send.assert_not_called()

    def test_equivalent_quaternion_sign_is_accepted(self):
        backend = self.backend(fk_pose=INITIAL_POSE[:3] + [-1., 0., 0., 0.])
        result = backend.validate_startup_ik()
        self.assertAlmostEqual(result['fk_orientation_error_rad'], 0.)
        self.assert_original_seed_preserved(backend, result)

    def test_hardware_joint_limits_reject_geometrically_matching_invalid_solution(self):
        for index in range(6):
            for side in ('minimum', 'maximum'):
                with self.subTest(joint=index, side=side):
                    backend = self.backend()
                    solved = INITIAL_JOINTS.copy()
                    solved[index] = (backend.minimum[index] - .001 if side == 'minimum'
                                     else backend.maximum[index] + .001)
                    backend.solve.return_value = solved
                    with self.assertRaises(MappingError):
                        backend.validate_startup_ik()
                    backend.send.assert_not_called()

    def test_malformed_or_nonfinite_joint_solution_is_rejected(self):
        invalid_solutions = [[], INITIAL_JOINTS[:5], INITIAL_JOINTS + [0.],
                             [math.nan] + INITIAL_JOINTS[1:],
                             [math.inf] + INITIAL_JOINTS[1:],
                             [-math.inf] + INITIAL_JOINTS[1:]]
        for solved in invalid_solutions:
            with self.subTest(solved=solved):
                backend = self.backend(solved)
                with self.assertRaises(MappingError):
                    backend.validate_startup_ik()
                backend.send.assert_not_called()

    def test_nonfinite_fk_position_and_quaternion_are_rejected(self):
        for index in range(7):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(index=index, value=value):
                    fk_pose = INITIAL_POSE.copy()
                    fk_pose[index] = value
                    backend = self.backend(fk_pose=fk_pose)
                    with self.assertRaises(MappingError):
                        backend.validate_startup_ik()
                    backend.send.assert_not_called()

    def test_invalid_fk_pose_shape_and_zero_quaternion_are_rejected(self):
        for fk_pose in ([], INITIAL_POSE[:6], INITIAL_POSE + [0.],
                        INITIAL_POSE[:3] + [0., 0., 0., 0.]):
            with self.subTest(fk_pose=fk_pose):
                backend = self.backend(fk_pose=fk_pose)
                with self.assertRaises(MappingError):
                    backend.validate_startup_ik()
                backend.send.assert_not_called()

    def test_unknown_solver_error_propagates_without_fallback_or_motion(self):
        backend = self.backend()
        failure = RuntimeError('Unexpected SDK failure')
        backend.solve.side_effect = failure
        with self.assertRaises(RuntimeError) as raised:
            backend.validate_startup_ik()
        self.assertIs(raised.exception, failure)
        backend.forward.assert_not_called()
        backend.send.assert_not_called()

    def test_unknown_forward_error_propagates_without_motion(self):
        backend = self.backend()
        failure = RuntimeError('Unexpected FK failure')
        backend.forward.side_effect = failure
        with self.assertRaises(RuntimeError) as raised:
            backend.validate_startup_ik()
        self.assertIs(raised.exception, failure)
        backend.send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
