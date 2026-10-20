"""Math-only boundary sliding and continuous rotation command cache checks."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from incremental import (IncrementalMapper, MappingError, angle, inverse, multiply,
                         reachable_ik_step)
from teleop_config import load_config
from test_incremental import frame

ROBOT = [0.,0.,0.,1.,0.,0.,0.]


def roll(degrees):
    half = math.radians(degrees)/2
    return [math.cos(half),0.,0.,math.sin(half)]


def cartesian_solver(target, seed):
    return [value*1000 for value in target[:3]]+[0.,0.,math.degrees(angle(target[3:]))]


class ReachableBoundaryTests(unittest.TestCase):
    def hard_x_boundary(self, joints):
        if joints[0] > 1.+1e-10:
            raise MappingError('Declared boundary', {'state':'JOINT_SESSION_BOUND'})

    def test_outward_hold_and_next_inward_step_need_no_state_reset(self):
        prior = [.001,0.,0.,1.,0.,0.,0.]
        seed = [1.,0.,0.,0.,0.,0.]
        out = reachable_ik_step(cartesian_solver,prior,[.002,0.,0.,1.,0.,0.,0.],
                                seed,[2.]*6,self.hard_x_boundary)
        self.assertEqual(out[0],prior)
        self.assertEqual(out[1],seed)
        self.assertEqual(out[2:],(True,False,'JOINT_SESSION_BOUND'))
        inward = [.0005,0.,0.,1.,0.,0.,0.]
        recovered = reachable_ik_step(cartesian_solver,out[0],inward,out[1],
                                      [2.]*6,self.hard_x_boundary)
        self.assertEqual(recovered[0],inward)
        self.assertEqual(recovered[2:],(False,False,None))

    def test_boundary_keeps_largest_translation_plane_and_rotation(self):
        prior = [.001,0.,0.,1.,0.,0.,0.]
        target,joints,limited,rotation_limited,reason = reachable_ik_step(
            cartesian_solver,prior,[.002,.0008,.0006]+roll(2),[1.,0.,0.,0.,0.,0.],
            [3.]*6,self.hard_x_boundary)
        self.assertEqual(target[:3],[.001,.0008,.0006])
        self.assertAlmostEqual(angle(target[3:]),math.radians(2))
        self.assertTrue(limited)
        self.assertFalse(rotation_limited)
        self.assertEqual(reason,'JOINT_SESSION_BOUND')
        self.assertEqual(joints[:3],[1.,.8,.6])

    def test_near_boundary_does_not_shrink_tangent_with_blocked_axis(self):
        prior = [.00099,0.,0.,1.,0.,0.,0.]
        out = reachable_ik_step(cartesian_solver,prior,[.0011,.001,0.,1.,0.,0.,0.],
                                [.99,0.,0.,0.,0.,0.],[2.]*6,self.hard_x_boundary)
        self.assertAlmostEqual(out[0][0],prior[0])
        self.assertAlmostEqual(out[0][1],.001)
        self.assertEqual(out[-1],'JOINT_SESSION_BOUND')

    def test_no_boundary_refusal_never_authorizes_projection(self):
        calls = []
        def solve(target,seed):
            calls.append(target)
            return [100.]*6
        out = reachable_ik_step(solve,ROBOT,[.001,.001,0.,1.,0.,0.,0.],
                                [0.]*6,[.1]*6)
        self.assertEqual(out[0],ROBOT)
        self.assertTrue(out[2])
        self.assertIsNone(out[-1])
        self.assertEqual(len(calls),8)

    def test_coupled_pose_can_escape_translation_only_ik_refusal(self):
        goal = [.001,0.,0.]+roll(10)
        def solve(target, seed):
            if target[0] > angle(target[3:])*.01+1e-12:
                raise MappingError('Coupled orientation required', {'retryable_ik':True})
            return cartesian_solver(target,seed)
        target,joints,limited,rotation_limited,reason = reachable_ik_step(
            solve,ROBOT,goal,[0.]*6,[20.]*6)
        self.assertEqual(target,goal)
        self.assertEqual(joints[:3],[1.,0.,0.])
        self.assertFalse(rotation_limited)
        self.assertEqual(reason,'IK_REFUSED')

    def test_oversized_step_backtracks_to_valid_half_step(self):
        calls = []
        def solve(target,seed):
            calls.append(target.copy())
            if target[0] > .0011:
                raise MappingError('IK failed: reachability boundary')
            return cartesian_solver(target,seed)
        out = reachable_ik_step(solve,ROBOT,[.002,0.,0.,1.,0.,0.,0.],
                                [0.]*6,[2.]*6)
        self.assertEqual(out[0][:3],[.001,0.,0.])
        self.assertEqual(out[2:],(True,False,'IK_REFUSED'))
        self.assertEqual(len(calls),2)

    def test_rotation_only_remains_available_when_translation_fails(self):
        def solve(target,seed):
            if target[0] > 0.:
                raise MappingError('Unreachable translation')
            return cartesian_solver(target,seed)
        goal = [.001,0.,0.]+roll(2)
        out = reachable_ik_step(solve,ROBOT,goal,[0.]*6,[3.]*6)
        self.assertEqual(out[0][:3],ROBOT[:3])
        self.assertLess(angle(multiply(out[0][3:],inverse(goal[3:]))),1e-7)
        self.assertEqual(out[2:],(True,False,'IK_REFUSED'))

    def test_wrist_boundary_allows_tangential_motion(self):
        def validate(joints):
            if joints[0] > 0.:
                raise MappingError('Wrist refusal',{'state':'WRIST_SINGULARITY'})
        out = reachable_ik_step(cartesian_solver,ROBOT,[.001,.001,0.,1.,0.,0.,0.],
                                [0.]*6,[2.]*6,validate)
        self.assertEqual(out[0][:3],[0.,.001,0.])
        self.assertEqual(out[-1],'WRIST_SINGULARITY')

    def test_every_accepted_joint_respects_its_own_speed_bound(self):
        out = reachable_ik_step(cartesian_solver,ROBOT,[.001,.002,.003]+roll(10),
                                [0.]*6,[.5,.1,.2,.1,.1,.25])
        self.assertTrue(out[2])
        self.assertIsNone(out[-1])
        for value,bound in zip(out[1],[.5,.1,.2,.1,.1,.25]):
            self.assertLessEqual(abs(value),bound+.0001)

    def test_unknown_solver_errors_hard_faults_and_cancellation_propagate(self):
        errors = [MappingError('Unexpected native result'),ValueError('SDK error'),KeyboardInterrupt(),
                  MappingError('IK failed: hard fault',{'state':'HARD_FAULT'}),
                  MappingError('Declared refusal with hardware fault',
                               {'retryable_ik':True,'hard_fault':True})]
        for error in errors:
            with self.subTest(error=repr(error)):
                calls = []
                def solve(target,seed):
                    calls.append(target)
                    raise error
                with self.assertRaises(type(error)):
                    reachable_ik_step(solve,ROBOT,[.001,0.,0.,1.,0.,0.,0.],[0.]*6,[1.]*6)
                self.assertEqual(len(calls),1)

    def test_malformed_results_and_validator_faults_are_not_recovered(self):
        for invalid in (None,[0.]*5,[math.nan]+[0.]*5,[True]+[0.]*5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(MappingError):
                    reachable_ik_step(lambda *_:invalid,ROBOT,
                                      [.001,0.,0.,1.,0.,0.,0.],[0.]*6,[1.]*6)
        def validate(joints):
            raise MappingError('Unknown safety validator error')
        with self.assertRaisesRegex(MappingError,'Unknown safety'):
            reachable_ik_step(cartesian_solver,ROBOT,[.001,0.,0.,1.,0.,0.,0.],
                              [0.]*6,[1.]*6,validate)

    def test_invalid_input_and_bounds_fail_before_solver_call(self):
        def unused(*args):
            self.fail('Solver must not receive invalid input')
        for bounds in ([0.]*6,[math.nan]*6,[1.]*5,True,None):
            with self.subTest(bounds=bounds),self.assertRaises(MappingError):
                reachable_ik_step(unused,ROBOT,ROBOT,[0.]*6,bounds)
        with self.assertRaises(MappingError):
            reachable_ik_step(unused,ROBOT,[math.inf]+ROBOT[1:],[0.]*6,[1.]*6)


class CommandRotationSyncTests(unittest.TestCase):
    def mapper(self):
        cfg = load_config(Path(__file__).resolve().parent.parent/'configs/right_full_teleop.json')
        cfg.update(rotation_mapping='neutral_tool_axes')
        mapper = IncrementalMapper(cfg,ROBOT)
        mapper.anchor_robot = ROBOT.copy()
        return mapper

    def test_positive_180_sync_keeps_filter_goal_and_turn_direction(self):
        mapper = self.mapper()
        mapper.target_twist_rad = math.radians(190)
        mapper.filtered_twist_rad = math.radians(190)
        mapper.filtered_goal = ROBOT[:3]+roll(190)
        filtered = mapper.filtered_goal.copy()
        # SDK quaternions may represent +181 degrees with the opposite sign.
        mapper.sync_target_rotation(ROBOT[:3]+[-x for x in roll(181)])
        self.assertAlmostEqual(mapper.target_twist_rad,math.radians(181))
        self.assertAlmostEqual(mapper.filtered_twist_rad,math.radians(190))
        self.assertEqual(mapper.filtered_goal,filtered)
        _,target = mapper.neutral_rotation_step(roll(190),1.,.01)
        self.assertGreater(mapper.target_twist_rad,math.radians(181))
        self.assertLess(angle(multiply(target,inverse(roll(190)))),math.radians(9))

    def test_negative_180_sync_retains_negative_turn(self):
        mapper = self.mapper()
        mapper.target_twist_rad = math.radians(-190)
        mapper.filtered_twist_rad = math.radians(-190)
        mapper.filtered_goal = ROBOT[:3]+roll(-190)
        mapper.sync_target_rotation(ROBOT[:3]+[-x for x in roll(-181)])
        self.assertAlmostEqual(mapper.target_twist_rad,math.radians(-181))
        self.assertAlmostEqual(mapper.filtered_twist_rad,math.radians(-190))
        mapper.neutral_rotation_step(roll(-190),1.,.01)
        self.assertLess(mapper.target_twist_rad,math.radians(-181))


class BoundaryIntentSaturationTests(unittest.TestCase):
    def mapper(self):
        cfg = load_config(Path(__file__).resolve().parent.parent/'configs/right_full_teleop.json')
        cfg.update(rotation_mapping='neutral_tool_axes',position_scale=1.,rotation_scale=1.,
                   translation_reference='tracking_world',max_clutch_rotation_rad=math.radians(190))
        mapper = IncrementalMapper(cfg,ROBOT)
        mapper.basis = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
        mapper.anchor_hand = ROBOT.copy()
        mapper.anchor_robot = ROBOT.copy()
        mapper.previous_hand = ROBOT.copy()
        mapper.neutral_rotation_calibration = {'preserve':'original'}
        return mapper

    def test_only_blocked_translation_axis_drops_pending_intent(self):
        mapper = self.mapper()
        mapper.previous_hand = [.02,.04,0.]+roll(20)
        requested,_ = mapper.requested_goal(mapper.previous_hand)
        mapper.filtered_goal = requested.copy()
        mapper.filtered_twist_rad = math.radians(20)
        original_filter = mapper.filtered_goal.copy()
        original_hand = mapper.anchor_hand.copy()
        original_rotation = mapper.anchor_robot[3:]
        original_tool = [row.copy() for row in mapper.tool_basis]
        accepted = [.01,.03,0.]+roll(10)
        mapper.saturate_boundary_goal(accepted,(1,))
        requested,_ = mapper.requested_goal(mapper.previous_hand)
        self.assertEqual(requested[:3],[.02,.03,0.])
        self.assertEqual(mapper.filtered_goal[:3],[.02,.03,0.])
        self.assertEqual(mapper.filtered_goal[3:],original_filter[3:])
        self.assertEqual(mapper.filtered_twist_rad,math.radians(20))
        self.assertEqual(mapper.anchor_hand,original_hand)
        self.assertEqual(mapper.anchor_robot[3:],original_rotation)
        self.assertEqual(mapper.tool_basis,original_tool)
        self.assertEqual(mapper.neutral_rotation_calibration,{'preserve':'original'})

    def test_rotation_saturation_releases_positive_and_negative_inward_intent(self):
        for sign in (1,-1):
            with self.subTest(sign=sign):
                mapper = self.mapper()
                mapper.previous_hand = ROBOT[:3]+roll(sign*90)
                mapper.target = ROBOT[:3]+roll(sign*31)
                mapper.target_twist_rad = math.radians(sign*31)
                mapper.filtered_goal = ROBOT[:3]+roll(sign*90)
                mapper.filtered_twist_rad = math.radians(sign*90)
                accepted = ROBOT[:3]+roll(sign*30)
                mapper.saturate_boundary_goal(accepted,rotation=True)
                mapper.previous_hand = ROBOT[:3]+roll(sign*89)
                requested,_ = mapper.requested_goal(mapper.previous_hand)
                self.assertLess(angle(multiply(requested[3:],inverse(roll(sign*29)))),1e-7)
                self.assertAlmostEqual(mapper.target_twist_rad,math.radians(sign*30))
                self.assertAlmostEqual(mapper.filtered_twist_rad,math.radians(sign*30))
                self.assertEqual(mapper.anchor_robot,ROBOT)
                self.assertEqual(mapper.anchor_hand,ROBOT)

    def test_180_crossing_correction_keeps_signed_command_branch(self):
        for sign in (1,-1):
            with self.subTest(sign=sign):
                mapper = self.mapper()
                mapper.previous_hand = ROBOT[:3]+roll(sign*190)
                mapper.target = ROBOT[:3]+roll(sign*181)
                mapper.target_twist_rad = math.radians(sign*181)
                mapper.filtered_goal = ROBOT[:3]+roll(sign*190)
                mapper.filtered_twist_rad = math.radians(sign*190)
                # Native solver/feedback can return the opposite sign.
                accepted = ROBOT[:3]+[-value for value in roll(sign*180)]
                mapper.saturate_boundary_goal(accepted,rotation=True)
                self.assertGreater(sum(a*b for a,b in zip(mapper.target[3:],roll(sign*181))),0.)
                self.assertAlmostEqual(mapper.target_twist_rad,math.radians(sign*180))
                mapper.previous_hand = ROBOT[:3]+roll(sign*189)
                requested,_ = mapper.requested_goal(mapper.previous_hand)
                self.assertLess(angle(multiply(requested[3:],inverse(roll(sign*179)))),1e-7)
                mapper.neutral_rotation_step(requested[3:],1.,.01)
                self.assertLess(sign*mapper.target_twist_rad,math.radians(180))

    def test_rotation_offset_stays_in_work_frame_with_nonidentity_anchors(self):
        mapper = self.mapper()
        robot_quaternion = [math.cos(.4),math.sin(.4),0.,0.]
        hand_quaternion = [math.cos(.2),0.,math.sin(.2),0.]
        mapper.anchor_robot[3:] = robot_quaternion
        mapper.anchor_hand[3:] = hand_quaternion
        original_robot = mapper.anchor_robot.copy()
        original_hand = mapper.anchor_hand.copy()
        mapper.previous_hand = ROBOT[:3]+multiply(hand_quaternion,roll(90))
        mapper.target = ROBOT[:3]+multiply(robot_quaternion,roll(31))
        mapper.target_twist_rad = math.radians(31)
        accepted = ROBOT[:3]+multiply(robot_quaternion,roll(30))
        mapper.saturate_boundary_goal(accepted,rotation=True)
        mapper.previous_hand = ROBOT[:3]+multiply(hand_quaternion,roll(89))
        requested,_ = mapper.requested_goal(mapper.previous_hand)
        expected = multiply(robot_quaternion,roll(29))
        self.assertLess(angle(multiply(requested[3:],inverse(expected))),1e-7)
        self.assertEqual(mapper.anchor_robot,original_robot)
        self.assertEqual(mapper.anchor_hand,original_hand)

    def test_repeated_rotation_saturation_accumulates_only_invalid_intent(self):
        mapper = self.mapper()
        mapper.previous_hand = ROBOT[:3]+roll(90)
        mapper.target = ROBOT[:3]+roll(30)
        mapper.target_twist_rad = math.radians(30)
        mapper.saturate_boundary_goal(mapper.target,rotation=True)
        mapper.previous_hand = ROBOT[:3]+roll(100)
        requested,_ = mapper.requested_goal(mapper.previous_hand)
        self.assertLess(angle(multiply(requested[3:],inverse(roll(40)))),1e-7)
        mapper.saturate_boundary_goal(mapper.target,rotation=True)
        mapper.previous_hand = ROBOT[:3]+roll(99)
        requested,_ = mapper.requested_goal(mapper.previous_hand)
        self.assertLess(angle(multiply(requested[3:],inverse(roll(29)))),1e-7)

    def test_new_successful_grip_resets_rotation_offset_without_jump(self):
        mapper = self.mapper()
        mapper.previous_hand = ROBOT[:3]+roll(90)
        mapper.target = ROBOT[:3]+roll(30)
        mapper.target_twist_rad = math.radians(30)
        mapper.saturate_boundary_goal(mapper.target,rotation=True)
        self.assertGreater(angle(mapper.rotation_boundary_offset),.1)
        now = 1_000_000_000
        mapper.update(frame(1,now,0),now,lambda:ROBOT)
        actual = [.1,.2,.3]+roll(45)
        result = mapper.update(frame(2,now+10_000_000,1),now+10_000_000,lambda:actual)
        self.assertEqual(result['state'],'ANCHORED')
        self.assertEqual(mapper.rotation_boundary_offset,[1.,0.,0.,0.])
        self.assertEqual(mapper.target,actual)
        self.assertEqual(mapper.target_twist_rad,0.)


if __name__ == '__main__':
    unittest.main()
