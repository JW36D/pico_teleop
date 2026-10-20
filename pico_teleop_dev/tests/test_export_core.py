"""Training semantics: causal selection, cancelled targets and unwrapped values."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from export_core import arm_values, ArmTimeline, Timeline, DEFAULT_PROFILE


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.ns=1_000_000_000
        self.row={'source_sequence':7,'receive_monotonic_ns':self.ns-1_000_000,
            'robot_feedback_monotonic_ns':self.ns-10_000_000,'robot_joints_deg':[0,1,2,3,4,370],
            'robot_feedback_sequence':12,'robot_feedback_stationary':True,
            'clutch_active':True,'motion_streaming_active':True,'state':'TRACKING',
            'last_actually_sent_command':{'sequence':5,'send_monotonic_ns':self.ns-900_000_000,
                                         'joints_deg':[1,2,3,4,5,380]},
            'gripper':{'feedback':{'valid':True,'receive_monotonic_ns':self.ns-200_000_000,
                                  'opening_controller_units':1003},
                'last_accepted_command':{'outcome':'CONTROLLER_ACCEPTED','ack_monotonic_ns':self.ns-900_000_000,
                                         'target_position_controller_units':750}}}

    def test_latest_past_only_never_future(self):
        timeline=Timeline([{'receive_monotonic_ns':30,'sequence':3},
                           {'receive_monotonic_ns':10,'sequence':1}])
        self.assertIsNone(timeline.at(9))
        self.assertEqual(timeline.at(29)['sequence'],1)
        self.assertEqual(timeline.at(30)['sequence'],3)

    def test_continuous_j6_and_gripper_feedback_outside_target_range_survive(self):
        state,action,trace=arm_values(self.row,self.ns,DEFAULT_PROFILE,'left')
        self.assertAlmostEqual(state[5],math.radians(370))
        self.assertAlmostEqual(action[5],math.radians(380))
        self.assertEqual(state[6],1.003);self.assertEqual(action[6],.75)
        self.assertEqual(trace['action_source'],'actually_sent')
        # Old commands remain valid in an explicitly active streaming mode.
        self.assertEqual(trace['command_send_ns'],self.ns-900_000_000)

    def test_released_arm_uses_measured_hold_and_keeps_gripper_target(self):
        self.row.update(clutch_active=False,motion_streaming_active=False,state='RELEASED')
        state,action,trace=arm_values(self.row,self.ns,DEFAULT_PROFILE,'right')
        self.assertEqual(state[:6],action[:6])
        self.assertEqual(action[6],.75)
        self.assertEqual(trace['action_source'],'derived_hold')
        self.assertIsNone(trace['command_send_ns'])

    def test_braking_or_soft_pause_does_not_reuse_cancelled_target(self):
        self.row.update(clutch_active=False,motion_streaming_active=False,robot_feedback_stationary=False)
        with self.assertRaisesRegex(ValueError,'stop_unconfirmed'):
            arm_values(self.row,self.ns,DEFAULT_PROFILE,'left')
        self.row.update(clutch_active=True,robot_feedback_stationary=True,state='CONTROL_GAP_PAUSED')
        with self.assertRaisesRegex(ValueError,'action_ambiguous'):
            arm_values(self.row,self.ns,DEFAULT_PROFILE,'left')

    def test_future_or_stale_feedback_is_rejected_without_filling(self):
        for delta in (-51_000_000,1):
            self.row['robot_feedback_monotonic_ns']=self.ns+delta
            with self.assertRaisesRegex(ValueError,'feedback:missing_or_stale'):
                arm_values(self.row,self.ns,DEFAULT_PROFILE,'left')

    def test_missing_and_future_gripper_targets_are_rejected(self):
        target=self.row['gripper']['last_accepted_command']
        target['ack_monotonic_ns']=self.ns+1
        with self.assertRaisesRegex(ValueError,'no_causal_accepted'):
            arm_values(self.row,self.ns,DEFAULT_PROFILE,'left')
        self.row['gripper']['last_accepted_command']=None
        with self.assertRaisesRegex(ValueError,'no_causal_accepted'):
            arm_values(self.row,self.ns,DEFAULT_PROFILE,'left')

    def test_release_during_deceleration_uses_observed_hold_after_stop_ack(self):
        self.row.update(clutch_active=False,motion_streaming_active=False,state='RELEASED',
                        robot_feedback_stationary=False,stop_confirmed_monotonic_ns=self.ns-20_000_000)
        state,action,trace=arm_values(self.row,self.ns,DEFAULT_PROFILE,'left')
        self.assertEqual(action[:6],state[:6])
        self.assertNotEqual(action[0],math.radians(self.row['last_actually_sent_command']['joints_deg'][0]))
        self.assertEqual(trace['action_source'],'derived_stop_hold')
        self.assertFalse(trace['feedback_stationary'])
        self.assertIsNone(trace['command_send_ns'])
        self.assertEqual(trace['stop_acknowledged_ns'],self.ns-20_000_000)
        for invalid in (None,self.ns+1,self.ns-950_000_000):
            self.row['stop_confirmed_monotonic_ns']=invalid
            with self.assertRaisesRegex(ValueError,'stop_unconfirmed'):
                arm_values(self.row,self.ns,DEFAULT_PROFILE,'left')

    def test_grip_anchor_holds_observed_position_without_reusing_old_goal(self):
        self.row.update(clutch_active=True,motion_streaming_active=False,state='ANCHORED')
        state,action,trace=arm_values(self.row,self.ns,DEFAULT_PROFILE,'left')
        self.assertEqual(state[:6],action[:6])
        self.assertEqual(trace['action_source'],'derived_anchor_hold')
        self.assertIsNone(trace['command_sequence'])
        self.row['robot_feedback_stationary']=False
        with self.assertRaisesRegex(ValueError,'action_ambiguous'):
            arm_values(self.row,self.ns,DEFAULT_PROFILE,'left')

    def test_release_fix_does_not_accept_paused_or_unknown_control_states(self):
        self.row.update(clutch_active=False,motion_streaming_active=False,
                        robot_feedback_stationary=False,stop_confirmed_monotonic_ns=self.ns-20_000_000)
        for state in ('CONTROL_GAP_PAUSED','IK_REFUSED','UNKNOWN'):
            self.row['state']=state
            with self.assertRaisesRegex(ValueError,'action_ambiguous'):
                arm_values(self.row,self.ns,DEFAULT_PROFILE,'left')

    def test_uncommanded_online_gripper_can_hold_real_feedback(self):
        grip=self.row['gripper']
        grip.update(enabled=True,status='ONLINE',fault=None,last_attempt=None,
                    last_accepted_command=None,accepted_command_count=0,pending_action=None)
        state,action,trace=arm_values(self.row,self.ns,DEFAULT_PROFILE,'left')
        self.assertEqual(action[6],state[6])
        self.assertEqual(action[6],1.003)
        self.assertEqual(trace['gripper_action_source'],'derived_uncommanded_hold')
        self.assertIsNone(trace['gripper_target_ack_ns'])
        for key,value in (('pending_action',500),('fault','rejected'),('last_attempt',{'outcome':'REJECTED'})):
            grip[key]=value
            with self.assertRaisesRegex(ValueError,'no_causal_accepted'):
                arm_values(self.row,self.ns,DEFAULT_PROFILE,'left')
            grip[key]=None

    def test_feedback_selection_uses_response_time_not_telemetry_bundle_time(self):
        old=copy.deepcopy(self.row)
        old['robot_feedback_monotonic_ns']=self.ns-60_000_000
        later=copy.deepcopy(self.row)
        later.update(source_sequence=8,receive_monotonic_ns=self.ns+10_000_000,
                     robot_feedback_monotonic_ns=self.ns-1_000_000,robot_joints_deg=[10]*6)
        later['gripper']['feedback'].update(receive_monotonic_ns=self.ns-2_000_000,opening_controller_units=900)
        state,action,trace=ArmTimeline([old,later]).values(self.ns,DEFAULT_PROFILE,'left')
        self.assertEqual(trace['telemetry_sequence'],7)
        self.assertEqual(trace['feedback_telemetry_sequence'],8)
        self.assertEqual(trace['feedback_age_ms'],1)
        self.assertEqual(trace['gripper_age_ms'],2)
        self.assertAlmostEqual(state[0],math.radians(10))
        self.assertEqual(state[6],.9)
        self.assertEqual(action[0],math.radians(1))

    def test_future_feedback_is_never_selected_from_later_bundle(self):
        old=copy.deepcopy(self.row)
        later=copy.deepcopy(self.row)
        later.update(source_sequence=8,receive_monotonic_ns=self.ns+20_000_000,
                     robot_feedback_monotonic_ns=self.ns+1,robot_joints_deg=[99]*6)
        state,_,trace=ArmTimeline([old,later]).values(self.ns,DEFAULT_PROFILE,'left')
        self.assertEqual(state[0],0)
        self.assertEqual(trace['feedback_telemetry_sequence'],7)

    def test_prior_rejection_policy_remains_available_explicitly(self):
        self.row.update(clutch_active=False,motion_streaming_active=False,state='RELEASED',
                        robot_feedback_stationary=False,stop_confirmed_monotonic_ns=self.ns-20_000_000)
        strict=dict(DEFAULT_PROFILE,release_action='reject_until_stationary')
        with self.assertRaisesRegex(ValueError,'stop_unconfirmed'):
            arm_values(self.row,self.ns,strict,'left')


if __name__=='__main__':unittest.main()
