"""Pure-math reachability backtracking and pending Grip anchor regressions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import math
import unittest

from incremental import (AnchorUnavailable, IncrementalMapper, MappingError,
                         angle, limit_ik_step, position_priority_step)
from teleop_config import load_config
from test_incremental import ROBOT, frame


class IKBacktrackingTests(unittest.TestCase):
    def test_solver_refusal_retries_smaller_cartesian_and_quaternion_step(self):
        attempts = []
        goal = [.004, 0, 0, math.cos(.08), 0, 0, math.sin(.08)]
        def solver(target, seed):
            attempts.append(target.copy())
            if target[0] > .0011:
                raise MappingError('IK failed: 1; target is not sent')
            return [target[0]*1000, 0, 0, 0, 0, 0]
        target, joints, limited = limit_ik_step(solver, ROBOT, goal, [0.]*6, 2.)
        self.assertEqual(len(attempts), 3)
        self.assertTrue(limited)
        self.assertAlmostEqual(target[0], .001)
        self.assertAlmostEqual(angle(target[3:]), .04)
        self.assertEqual(joints, [1., 0, 0, 0, 0, 0])

    def test_declared_ik_refusal_is_compatible_without_message_matching(self):
        attempts = []
        def solver(target, seed):
            attempts.append(target[0])
            if target[0] > .0006:
                raise MappingError('Native solver returned no solution', {'retryable_ik': True})
            return [target[0]*1000, 0, 0, 0, 0, 0]
        target, _, limited = limit_ik_step(solver, ROBOT, [.001, 0, 0, 1, 0, 0, 0], [0.]*6, 1.)
        self.assertEqual(attempts, [.001, .0005])
        self.assertEqual(target[0], .0005)
        self.assertTrue(limited)

    def test_reserved_boundary_backtracks_without_crossing_and_allows_inward_motion(self):
        seen = []
        def solver(target, seed):
            return [target[0]*1000, 0, 0, 0, 0, 0]
        def validate(joints):
            seen.append(joints[0])
            if joints[0] > 1.:
                raise MappingError('Reserved boundary', {'state': 'JOINT_SESSION_BOUND'})
        prior = [.0009, 0, 0, 1, 0, 0, 0]
        target, joints, limited, rotation_limited = position_priority_step(solver, prior,
            [.0013, 0, 0, 1, 0, 0, 0], [.9, 0, 0, 0, 0, 0], [1.]*6, validate)
        self.assertGreater(len(seen), 1)
        self.assertLessEqual(joints[0], 1.)
        self.assertAlmostEqual(target[0], .001)
        self.assertTrue(limited)
        self.assertFalse(rotation_limited)
        inward, inward_joints, _, _ = position_priority_step(solver, target,
            [.0008, 0, 0, 1, 0, 0, 0], joints, [1.]*6, validate)
        self.assertAlmostEqual(inward[0], .0008)
        self.assertAlmostEqual(inward_joints[0], .8)

    def test_every_refused_step_preserves_boundary_classification(self):
        attempts = []
        def solver(target, seed):
            attempts.append(target.copy())
            return [target[0]*1000, 0, 0, 0, 0, 0]
        def validate(joints):
            if joints[0] > 1.:
                raise MappingError('Reserved boundary', {'state': 'JOINT_SESSION_BOUND'})
        with self.assertRaises(MappingError) as caught:
            limit_ik_step(solver, [.001, 0, 0, 1, 0, 0, 0],
                [.002, 0, 0, 1, 0, 0, 0], [1., 0, 0, 0, 0, 0], 1., validate)
        self.assertEqual(caught.exception.details['state'], 'JOINT_SESSION_BOUND')
        self.assertEqual(len(attempts), 8)

    def test_unreachable_ik_still_refuses_after_finite_number_of_attempts(self):
        attempts = []
        def solver(target, seed):
            attempts.append(target.copy())
            raise MappingError('IK failed: 1; target is not sent')
        with self.assertRaisesRegex(MappingError, 'IK failed'):
            limit_ik_step(solver, ROBOT, [.001, 0, 0, 1, 0, 0, 0], [0.]*6, 1.)
        self.assertEqual(len(attempts), 8)

    def test_bad_joint_result_is_never_retried(self):
        for bad in (None, [0.]*5, [math.nan]+[0.]*5, [True]+[0.]*5):
            with self.subTest(bad=bad):
                attempts = []
                def solver(target, seed):
                    attempts.append(target.copy())
                    return bad
                with self.assertRaises(MappingError):
                    limit_ik_step(solver, ROBOT, [.001, 0, 0, 1, 0, 0, 0], [0.]*6, 1.)
                self.assertEqual(len(attempts), 1)

    def test_unknown_solver_error_and_cancellation_propagate_immediately(self):
        for error in (MappingError('Nonfinite or invalid vector'),
                      MappingError('Unexpected native result structure'), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                attempts = []
                def solver(target, seed):
                    attempts.append(target.copy())
                    raise error
                with self.assertRaises(type(error)):
                    limit_ik_step(solver, ROBOT, [.001, 0, 0, 1, 0, 0, 0], [0.]*6, 1.)
                self.assertEqual(len(attempts), 1)

    def test_rotation_allocation_does_not_hide_malformed_solver_data(self):
        for bad in ([math.inf]+[0.]*5, [0.]*5):
            with self.subTest(bad=bad):
                def solver(target, seed):
                    return [target[0]*1000, 0, 0, 0, 0, 0] if angle(target[3:]) < 1e-7 else bad
                with self.assertRaises(MappingError):
                    position_priority_step(solver, ROBOT,
                        [.0005, 0, 0, math.cos(.05), 0, 0, math.sin(.05)],
                        [0.]*6, [1.]*6, lambda _: None)

    def test_validator_fault_is_not_retried_as_ik_refusal(self):
        attempts = []
        def solver(target, seed):
            attempts.append(target.copy())
            return [target[0]*1000, 0, 0, 0, 0, 0]
        def validate(joints):
            raise MappingError('Hardware limit crossed')
        with self.assertRaisesRegex(MappingError, 'Hardware limit'):
            limit_ik_step(solver, ROBOT, [.001, 0, 0, 1, 0, 0, 0], [0.]*6, 1., validate)
        self.assertEqual(len(attempts), 1)

    def test_translation_wrist_guard_is_not_retried_as_session_boundary(self):
        attempts = []
        def solver(target, seed):
            attempts.append(target.copy())
            return [target[0]*1000, 0, 0, 0, 0, 0]
        def validate(joints):
            raise MappingError('Wrist singularity guard', {'state': 'WRIST_SINGULARITY'})
        with self.assertRaises(MappingError) as caught:
            limit_ik_step(solver, ROBOT, [.001, 0, 0, 1, 0, 0, 0], [0.]*6, 1., validate)
        self.assertEqual(caught.exception.details['state'], 'WRIST_SINGULARITY')
        self.assertEqual(len(attempts), 1)


class PendingAnchorTests(unittest.TestCase):
    def setUp(self):
        config = load_config(Path(__file__).resolve().parent.parent/'configs/right_full_teleop.json')
        config.update(rotation_scale=0., translation_reference='operator_heading')
        self.mapper = IncrementalMapper(config, ROBOT)
        self.ns = 1_000_000_000
        self.seq = 0

    def step(self, grip, xyz=None, yaw=0., provider=None):
        self.ns += 10_000_000
        self.seq += 1
        data = frame(self.seq, self.ns, grip, xyz)
        data['latest']['head']['pose'] = [0, 0, 0, 0, math.sin(yaw/2), 0, math.cos(yaw/2)]
        return self.mapper.update(data, self.ns, provider or (lambda: ROBOT))

    @staticmethod
    def unsettled():
        raise AnchorUnavailable('Robot has not settled after release')

    def waiting(self):
        self.step(0)
        self.assertEqual(self.step(1, provider=self.unsettled)['state'], 'WAIT_ROBOT_STATIONARY')

    def test_held_grip_retries_latest_hand_heading_and_robot_without_jump(self):
        self.waiting()
        self.assertTrue(self.mapper.release_seen)
        self.assertFalse(self.mapper.active)
        self.assertIsNone(self.mapper.anchor_hand)
        self.assertIsNone(self.mapper.operator_basis)
        actual = [.012, 0, 0, 1, 0, 0, 0]
        result = self.step(1, [.5, 0, 0], math.pi/2, lambda: actual)
        self.assertEqual(result['state'], 'ANCHORED')
        self.assertFalse(result['send'])
        self.assertEqual(result['target_pose_wxyz'], actual)
        self.assertEqual(result['anchor_hand_pose_wxyz'][:3], [.5, 0, 0])
        result = self.step(1, [.5, 0, -.02], math.pi/2)
        self.assertEqual(result['state'], 'TRACKING')
        self.assertAlmostEqual(result['requested_delta_work_m'][0], 0., places=7)
        self.assertAlmostEqual(result['requested_delta_work_m'][1], -.01)

    def test_release_cancels_pending_anchor_and_next_press_uses_latest_input(self):
        self.waiting()
        self.assertEqual(self.step(0, [.1, 0, 0])['state'], 'RELEASED')
        self.assertIsNone(self.mapper.anchor_hand)
        self.assertFalse(self.mapper.active)
        result = self.step(1, [.2, 0, 0])
        self.assertEqual(result['state'], 'ANCHORED')
        self.assertEqual(result['anchor_hand_pose_wxyz'][:3], [.2, 0, 0])

    def test_neutral_rotation_calibration_waits_for_successful_robot_anchor(self):
        self.mapper.config.update(rotation_mapping='neutral_tool_axes', rotation_scale=1.)
        self.mapper = IncrementalMapper(self.mapper.config, ROBOT)
        self.waiting()
        self.assertIsNone(self.mapper.neutral_rotation_calibration)
        self.ns += 10_000_000
        data = frame(self.seq+1, self.ns, 1, [.3, 0, 0],
                     [math.sin(.2), 0, 0, math.cos(.2)])
        result = self.mapper.update(data, self.ns, lambda: ROBOT)
        self.assertEqual(result['state'], 'ANCHORED')
        self.assertFalse(result['send'])
        self.assertEqual(result['target_pose_wxyz'], ROBOT)
        self.assertAlmostEqual(self.mapper.anchor_hand[4], math.sin(.2))
        self.assertIsNotNone(self.mapper.neutral_rotation_calibration)

    def test_invalid_input_tracking_loss_and_stale_input_disarm_pending_anchor(self):
        for failure in ('grip', 'pose', 'head_tracking', 'stale', 'focus'):
            with self.subTest(failure=failure):
                self.setUp()
                self.waiting()
                self.ns += 10_000_000
                data = frame(self.seq+1, self.ns, 1)
                if failure == 'grip': data['latest']['right']['grip'] = math.nan
                if failure == 'pose': data['latest']['right']['pose'] = [0.]*6
                if failure == 'head_tracking': data['latest']['head']['status'] = 0
                if failure == 'stale': data['latest']['receive_monotonic_ns'] -= 200_000_000
                if failure == 'focus': data['latest']['app_focus'] = False
                result = self.mapper.update(data, self.ns, lambda: ROBOT)
                self.assertFalse(result['send'])
                self.assertFalse(self.mapper.active)
                self.assertFalse(self.mapper.release_seen)
                self.assertEqual(self.step(1)['state'], 'RELEASE_GRIP_TO_ARM')


if __name__ == '__main__':
    unittest.main()
