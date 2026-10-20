"""Recovery must permit local progress without relaxing hardware/range checks."""
import sys
from pathlib import Path
import math
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from incremental import MappingError
from test_full_teleop import ControllerHarness


class LocalControllerRecoveryTests(ControllerHarness,unittest.TestCase):
    def resume_with_held_grip(self, hand):
        for _ in range(40):
            report=self.tick(1,hand)
            if report['mapping']['state']=='RESUMED':
                self.assertFalse(report['motion_command_sent_this_tick'])
                return report
        self.fail('A valid recovery remained paused')

    def reachable_nearby_only(self):
        solve=self.backend.solve
        def local_solver(target,seed):
            if target[0]>.003:
                raise MappingError('IK failed: 1; target is not sent',
                                   {'state':'IK_REFUSED','retryable_ik':True})
            return solve(target,seed)
        self.backend.solve=local_solver

    def test_ik_pause_recovers_via_local_step_with_unreachable_far_goal(self):
        self.active()
        self.feedback.joints=self.control.last_sent_joints.copy()
        actual=self.backend.forward(self.feedback.joints)
        self.control.pause_control('IK_REFUSED',actual,self.feedback.joints)
        self.reachable_nearby_only()
        count=len(self.backend.sent)
        report=self.resume_with_held_grip([0,0,-.012])
        self.assertTrue(report['mapping']['recovery_used_local_step'])
        self.assertEqual(len(self.backend.sent),count)
        before=self.feedback.joints.copy()
        report=self.tick(1,[0,0,-.012])
        self.assertTrue(report['motion_command_sent_this_tick'])
        self.assertLessEqual(abs(self.backend.sent[-1][0]-before[0]),.6001)

    def test_timing_recovery_does_not_require_far_goal_ik(self):
        self.config.update(auto_resume_timing_gaps=True,max_recoverable_control_gap_s=.15)
        self.active()
        self.feedback.joints=self.control.last_sent_joints.copy()
        self.reachable_nearby_only()
        self.ns+=60_000_000
        report=self.tick(1,[0,0,-.012])
        self.assertEqual(report['mapping']['state'],'CONTROL_GAP_PAUSED')
        anchor=self.control.mapper.anchor_hand.copy()
        report=self.resume_with_held_grip([0,0,-.012])
        self.assertTrue(report['mapping']['timing_recovery'])
        self.assertEqual(self.control.mapper.anchor_hand,anchor)
        self.assertTrue(self.tick(1,[0,0,-.012])['motion_command_sent_this_tick'])

    def test_goal_inside_command_bound_recovers_without_extra_margin(self):
        self.config['max_joint_session_deg']=2.
        self.tick(0);self.tick(1)
        self.control.pause_control('JOINT_SESSION_BOUND',self.backend.initial_pose,[0.]*6)
        self.assertAlmostEqual(self.control.command_joint_bound(),1.8)
        self.resume_with_held_grip([0,0,-.0034])  # 1.7 degrees is inside 1.8.

    def test_goal_outside_command_bound_still_cannot_resume(self):
        self.config['max_joint_session_deg']=2.
        self.tick(0);self.tick(1)
        self.control.pause_control('JOINT_SESSION_BOUND',self.backend.initial_pose,[0.]*6)
        for _ in range(40):
            report=self.tick(1,[0,0,-.004])  # 2 degrees remains outside 1.8.
            self.assertEqual(report['mapping']['state'],'JOINT_SESSION_BOUND')
            self.assertFalse(report['motion_command_sent_this_tick'])
        self.assertTrue(self.control.mapper.paused)
        self.assertEqual(self.backend.sent,[])

    def test_invalid_solver_data_is_not_treated_as_recoverable_ik(self):
        self.active()
        count=len(self.backend.sent)
        self.backend.solve=lambda *_:[float('nan')]*6
        with self.assertRaises(MappingError):
            self.tick(1,[0,0,-.012])
        self.assertEqual(len(self.backend.sent),count)
        self.assertFalse(self.control.mapper.active)
        self.assertEqual(self.backend.stops,1)

    def test_slow_gate_feedback_stops_before_send_using_current_time(self):
        self.config.update(auto_resume_timing_gaps=True,max_recoverable_control_gap_s=.15,
                           max_control_gap_s=.03,max_command_gap_s=.03)
        self.active()
        self.feedback.joints=self.control.last_sent_joints.copy()
        count=len(self.backend.sent)
        original_snapshot=self.feedback.snapshot
        calls=[0]
        def delayed_gate_feedback():
            calls[0]+=1
            if calls[0]==2:
                self.ns+=40_000_000
            return original_snapshot()
        self.feedback.snapshot=delayed_gate_feedback
        report=self.tick(1,[0,0,-.002])
        self.assertEqual(report['mapping']['state'],'CONTROL_COMPUTATION_GAP_PAUSED')
        self.assertFalse(report['motion_command_sent_this_tick'])
        self.assertEqual(len(self.backend.sent),count)
        self.assertTrue(self.control.mapper.active)
        self.assertTrue(self.control.mapper.paused)
        self.assertEqual(report['control_phase_ms']['gate_feedback_snapshot'],40.)

    def test_invalid_pose_or_grip_during_gap_still_requires_regrip(self):
        cases=(('pose',[0.]*6,'INVALID_POSE'),
               ('grip',1.1,'INVALID_GRIP'),
               ('grip',math.nan,'INVALID_GRIP'),
               ('grip',True,'INVALID_GRIP'))
        for field,value,state in cases:
            with self.subTest(field=field,value=value):
                self.setUp()
                self.config.update(auto_resume_timing_gaps=True,max_recoverable_control_gap_s=.15,
                                   max_control_gap_s=.03,max_command_gap_s=.03)
                self.active()
                count=len(self.backend.sent)
                self.ns+=60_000_000
                status=self.sample(1,[0,0,-.002])
                status['latest']['right'][field]=value
                report=self.control.tick(status,lambda:status,self.ns)
                self.assertEqual(report['mapping']['state'],state)
                self.assertFalse(report['motion_command_sent_this_tick'])
                self.assertEqual(len(self.backend.sent),count)
                self.assertFalse(self.control.mapper.active)
                self.assertFalse(self.control.mapper.release_seen)
                self.assertEqual(self.tick(1)['mapping']['state'],'RELEASE_GRIP_TO_ARM')


if __name__=='__main__':unittest.main()
