import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import math
from pathlib import Path
import unittest

from incremental import (IncrementalMapper,MappingError,AnchorUnavailable,multiply,angle,limit_ik_step,
                         position_priority_step,wrist_step_safe)
from teleop_config import load_config
from test_incremental import frame,ROBOT
from test_full_teleop import ControllerHarness


class HeadingTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config(Path(__file__).resolve().parent.parent/'configs/right_full_teleop.json')
        # Known work-frame fixture: robot forward=+X, right=-Y, up=+Z.
        # Expected semantic directions below are fixed independently of production calibration.
        self.config.update(headset_to_work_matrix=[[0,0,-1],[-1,0,0],[0,1,0]],
                           rotation_scale=0.)
        self.mapper=IncrementalMapper(self.config,ROBOT)
        self.ns=1_000_000_000;self.seq=0
    def step(self,grip,xyz=None,yaw=math.pi/2,q=None,provider=None):
        self.ns+=10_000_000;self.seq+=1
        s=frame(self.seq,self.ns,grip,xyz,q)
        s['latest']['head']['pose']=[0,0,0,0,math.sin(yaw/2),0,math.cos(yaw/2)]
        return self.mapper.update(s,self.ns,provider or (lambda:ROBOT))
    def assert_direction(self,result,expected):
        self.assertEqual(result['state'],'TRACKING')
        for actual,wanted in zip(result['requested_delta_work_m'],expected):
            self.assertAlmostEqual(actual,wanted,places=7)
    def test_four_operator_headings_keep_robot_right_forward_and_up(self):
        # These room-space trajectories describe the same body motions from four headings.
        headings=[(0.,[.02,0,0],[0,0,-.02]),
                  (math.pi/2,[0,0,-.02],[-.02,0,0]),
                  (math.pi,[-.02,0,0],[0,0,.02]),
                  (-math.pi/2,[0,0,.02],[.02,0,0])]
        for yaw,right,forward in headings:
            with self.subTest(yaw=yaw):
                self.mapper=IncrementalMapper(self.config,ROBOT)
                self.step(0,yaw=yaw);self.step(1,yaw=yaw)
                self.assert_direction(self.step(1,right,yaw=yaw),[0,-.01,0])
                self.assert_direction(self.step(1,forward,yaw=yaw),[.01,0,0])
                self.assert_direction(self.step(1,[0,.02,0],yaw=yaw),[0,0,.01])
    def test_room_heading_does_not_reverse_operator_right(self):
        self.step(0);self.step(1)
        r=self.step(1,[0,0,-.02])
        self.assertAlmostEqual(r['operator_delta_m'][0],.02)
        self.assertAlmostEqual(r['requested_delta_work_m'][1],-.01)
        self.assertAlmostEqual(r['requested_delta_work_m'][0],0.,places=7)
    def test_head_turn_during_grip_does_not_change_translation_axes(self):
        self.step(0);self.step(1)
        first=self.step(1,[0,0,-.02],yaw=math.pi/2)
        second=self.step(1,[0,0,-.02],yaw=-math.pi/2)
        self.assertEqual(first['requested_delta_work_m'],second['requested_delta_work_m'])
    def test_regrip_uses_the_new_operator_heading(self):
        self.step(0);self.step(1)
        self.step(0,yaw=-math.pi/2);self.step(1,yaw=-math.pi/2)
        # After turning 180 degrees, room +Z is the operator's new right.
        self.assert_direction(self.step(1,[0,0,.02],yaw=-math.pi/2),[0,-.01,0])
    def test_unsettled_robot_does_not_commit_heading_and_retry_uses_latest_heading(self):
        def unsettled():raise AnchorUnavailable('Robot still moving')
        self.step(0,yaw=0.)
        r=self.step(1,yaw=0.,provider=unsettled)
        self.assertEqual(r['state'],'WAIT_ROBOT_STATIONARY')
        self.assertIsNone(self.mapper.operator_basis)
        self.assertFalse(self.mapper.active)
        self.step(0,yaw=math.pi/2);self.step(1,yaw=math.pi/2)
        self.assert_direction(self.step(1,[0,0,-.02],yaw=math.pi/2),[0,-.01,0])
    def test_hand_orientation_does_not_change_translation_axes(self):
        self.step(0);self.step(1)
        first=self.step(1,[0,0,-.02])
        # Turn the hand 90 degrees without changing its tracked position.
        # Advance in 15-degree increments to remain within the input jump bound.
        for degrees in range(15,91,15):
            half=math.radians(degrees)/2
            result=self.step(1,[0,0,-.02],q=[math.sin(half),0,0,math.cos(half)])
            self.assertEqual(result['requested_delta_work_m'],first['requested_delta_work_m'])
            self.assertEqual(result['requested_pose_wxyz'][3:],ROBOT[3:])
    def test_head_pitch_and_roll_keep_vertical_up_and_horizontal_axes(self):
        self.step(0)
        self.ns+=10_000_000;self.seq+=1
        s=frame(self.seq,self.ns,1)
        yaw=[math.cos(math.pi/4),0,math.sin(math.pi/4),0]
        pitch=[math.cos(math.pi/12),math.sin(math.pi/12),0,0]
        roll=[math.cos(math.pi/9),0,0,math.sin(math.pi/9)]
        head=multiply(multiply(yaw,pitch),roll)
        s['latest']['head']['pose']=[0,0,0]+head[1:]+head[:1]
        self.assertEqual(self.mapper.update(s,self.ns,lambda:ROBOT)['state'],'ANCHORED')
        self.assert_direction(self.step(1,[0,0,-.02]),[0,-.01,0])
        self.assert_direction(self.step(1,[-.02,0,0]),[.01,0,0])
        self.assert_direction(self.step(1,[0,.02,0]),[0,0,.01])
    def test_tracking_world_remains_independent_of_heading_on_regrip(self):
        self.config['translation_reference']='tracking_world'
        self.mapper=IncrementalMapper(self.config,ROBOT)
        for yaw in (math.pi/2,-math.pi/2):
            self.step(0,yaw=yaw);self.step(1,yaw=yaw)
            self.assertIsNone(self.mapper.operator_basis)
            self.assert_direction(self.step(1,[.02,0,0],yaw=yaw),[0,-.01,0])
    def test_head_pointing_straight_up_cannot_calibrate_heading(self):
        self.step(0)
        self.ns+=10_000_000;s=frame(2,self.ns,1)
        s['latest']['head']['pose']=[0,0,0,math.sin(math.pi/4),0,0,math.cos(math.pi/4)]
        self.assertEqual(self.mapper.update(s,self.ns,lambda:ROBOT)['state'],'WAIT_OPERATOR_HEADING')
        self.assertFalse(self.mapper.active)


class ReservedBoundsTests(ControllerHarness,unittest.TestCase):
    def test_targets_pause_before_90_degree_boundary(self):
        self.assertEqual(self.control.command_joint_bound(),87.)
        self.assertTrue(self.control.joint_target_allowed([0,0,0,87.,0,0]))
        self.assertFalse(self.control.joint_target_allowed([0,0,0,87.01,0,0]))
    def test_small_stop_overshoot_is_tolerated_only_while_paused(self):
        self.feedback.joints[3]=91.4
        self.control.mapper.paused=True
        self.control.validate_feedback(self.feedback.snapshot())
        self.control.mapper.paused=False
        with self.assertRaises(MappingError) as caught:self.control.validate_feedback(self.feedback.snapshot())
        self.assertEqual(caught.exception.details['over_limit_joints'],[4])
    def test_large_stop_overshoot_is_not_silently_allowed(self):
        self.control.mapper.paused=True;self.feedback.joints[3]=92.1
        with self.assertRaises(MappingError):self.control.validate_feedback(self.feedback.snapshot())
    def test_hardware_limit_is_enforced_even_after_soft_pause(self):
        self.control.mapper.paused=True;self.feedback.joints[3]=180.1
        with self.assertRaisesRegex(MappingError,'hardware'):self.control.validate_feedback(self.feedback.snapshot())
    def test_return_path_can_only_move_over_limit_joints_inward(self):
        self.control.seed[3]=91.4;self.control.returning_inside_joints=True
        self.assertTrue(self.control.joint_target_allowed([0,0,0,91.0,0,0]))
        self.assertFalse(self.control.joint_target_allowed([0,0,0,91.5,0,0]))
        self.assertFalse(self.control.joint_target_allowed([88.,0,0,91.,0,0]))


class WristAndPriorityTests(unittest.TestCase):
    def test_wrist_guard_blocks_approaching_zero_but_allows_escape(self):
        a=[0,0,0,0,9.,0];b=a.copy();b[4]=7.9
        self.assertFalse(wrist_step_safe(a,b,8.))
        self.assertTrue(wrist_step_safe(b,a,8.))
    def test_pure_tool_roll_is_allowed_near_wrist_zero(self):
        a=[0,0,0,10.,4.,20.];b=a.copy();b[5]+=.5
        self.assertTrue(wrist_step_safe(a,b,8.))
        b[3]-=.5
        self.assertFalse(wrist_step_safe(a,b,8.))
    def test_rotation_does_not_reduce_valid_translation_step(self):
        def solver(p,_):return [p[0]*1000,0,0,math.degrees(angle(p[3:]))*20,25.,0]
        seed=[0,0,0,0,25.,0]
        goal=[.0006,0,0,math.cos(.05),0,0,math.sin(.05)]
        prior=[0,0,0,1,0,0,0]
        old,_,_=limit_ik_step(solver,prior,goal,seed,[.6]*6)
        target,joints,_,limited=position_priority_step(solver,prior,goal,seed,[.6]*6,lambda _:None)
        self.assertAlmostEqual(target[0],.0006)
        self.assertGreater(target[0],old[0]*20)
        self.assertTrue(limited)
        self.assertTrue(all(abs(a-b)<=.6001 for a,b in zip(joints,seed)))
    def test_unreachable_rotation_preserves_reachable_translation(self):
        def solver(p,_):
            if angle(p[3:])>1e-6:raise MappingError('Unreachable orientation')
            return [p[0]*1000,0,0,0,25.,0]
        target,_,_,limited=position_priority_step(solver,ROBOT,
            [.0005,0,0,math.cos(.05),0,0,math.sin(.05)],[0,0,0,0,25.,0],[.6]*6,lambda _:None)
        self.assertAlmostEqual(target[0],.0005)
        self.assertEqual(target[3:],[1.,0,0,0])
        self.assertTrue(limited)


if __name__=='__main__':unittest.main()
