import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import copy
import json
import math
from pathlib import Path
import unittest

from incremental import IncrementalMapper, MappingError, multiply, inverse, angle, limit_ik_step
from teleop_config import load_config
from teleop_controller import TeleopController
from robot_feedback import FeedbackMonitor, decode_state, frame_signature, query
from test_incremental import frame, ROBOT

ROOT=Path(__file__).resolve().parent.parent


def legacy_config():
    """Exercise supported legacy behavior without constraining the main profile."""
    config=load_config(ROOT/'configs/right_full_teleop.json')
    config.update(joint_session_limits_enabled=True,session_pose_limits_enabled=True,
        clutch_rotation_limits_enabled=True,rotation_path_mode='anchor',
        mapper_control_gap_guard=True,input_jump_mode='step',
        continuous_input_recovery=False,continuous_timing_recovery=False,
        joint_trajectory_enabled=False,tracking_error_grace_s=0.,
        tracking_error_time_alignment=False,require_robot_stationary_to_anchor=True)
    return config


class FakeBackend:
    def __init__(self):
        self.initial_pose=ROBOT.copy(); self.initial_joints=[0.]*6
        self.minimum=[-180.]*6; self.maximum=[180.]*6
        self.sent=[]; self.stops=0; self.fail_send=False; self.fail_ik=False
    def forward(self,joints):
        return [joints[0]/1000,0,0,1,0,0,0]
    def solve(self,target,seed):
        if self.fail_ik:raise MappingError('Unreachable test target')
        return [target[0]*1000,0,0,0,0,0]
    def send(self,joints):
        self.sent.append(joints.copy())
        if self.fail_send:raise MappingError('Send acknowledgement lost')
    def stop(self):self.stops+=1


class FakeFeedback:
    def __init__(self):self.fail=False;self.stationary=True;self.joints=[0.]*6
    def snapshot(self):
        if self.fail:raise MappingError('Feedback timeout')
        return {'joints':self.joints.copy(),'age_ms':1.,'stationary':self.stationary}


class ControllerHarness:
    def setUp(self):
        self.ns=1_000_000_000;self.seq=0
        self.config=legacy_config()
        self.backend=FakeBackend();self.feedback=FakeFeedback()
        self.control=TeleopController(self.config,self.backend,self.feedback,True,lambda:self.ns)
    def sample(self,grip,xyz=None,new=True):
        self.ns+=int(1e9/self.config['control_hz'])
        if new:self.seq+=1
        return frame(self.seq,self.ns,grip,xyz)
    def tick(self,grip,xyz=None,gate=None,new=True):
        status=self.sample(grip,xyz,new)
        return self.control.tick(status,lambda:gate if gate is not None else status,self.ns)
    def active(self):
        self.tick(0);self.tick(1);return self.tick(1,[0,0,-.002])


class SenderGateTests(ControllerHarness,unittest.TestCase):
    def test_preview_never_calls_motion_or_stop(self):
        self.control.execute=False
        self.active();self.tick(0)
        self.assertEqual(self.backend.sent,[]);self.assertEqual(self.backend.stops,0)
    def test_stationary_goal_is_periodically_sent(self):
        self.active()
        self.tick(1,[0,0,-.002]);self.tick(1,[0,0,-.002])
        self.assertEqual(len(self.backend.sent),3)
    def test_no_new_frame_holds_goal_only_while_input_fresh(self):
        self.active();result=self.tick(1,[0,0,-.002],new=False)
        self.assertEqual(result['mapping']['state'],'TRACKING')
        self.assertTrue(result['mapping']['input_frame_reused'])
        self.assertTrue(result['motion_command_sent_this_tick'])
    def test_release_after_ik_before_send_blocks_command(self):
        self.tick(0);self.tick(1)
        gate=frame(self.seq+1,self.ns+int(1e9/self.config['control_hz']),0)
        result=self.tick(1,[0,0,-.002],gate)
        self.assertEqual(result['mapping']['state'],'INPUT_GATE_CLOSED')
        self.assertEqual(self.backend.sent,[])
    def test_b_pressed_after_ik_ends_without_one_more_command(self):
        self.tick(0);self.tick(1)
        gate=frame(self.seq+1,self.ns+int(1e9/self.config['control_hz']),1)
        gate['session_exit_requested']=True
        result=self.tick(1,[0,0,-.002],gate)
        self.assertEqual(result['mapping']['state'],'SESSION_EXIT')
        self.assertEqual(self.backend.sent,[])
    def test_release_stops_and_does_not_repeat_old_goal(self):
        self.active();count=len(self.backend.sent)
        self.tick(0);self.tick(0)
        self.assertEqual(len(self.backend.sent),count);self.assertEqual(self.backend.stops,1)
    def test_feedback_timeout_stops_active_stream(self):
        self.active();self.feedback.fail=True
        with self.assertRaises(MappingError):self.tick(1,[0,0,-.002])
        self.assertEqual(self.backend.stops,1)
    def test_failed_first_send_requests_stop(self):
        self.tick(0);self.tick(1);self.backend.fail_send=True
        with self.assertRaises(MappingError):self.tick(1,[0,0,-.002])
        self.assertEqual(self.backend.stops,1)
    def test_ik_failure_disarms_without_sending_bad_target(self):
        self.config['auto_resume_soft_limits']=False
        self.active();count=len(self.backend.sent);self.backend.fail_ik=True
        result=self.tick(1,[0,0,-.004])
        self.assertEqual(result['mapping']['state'],'IK_REFUSED')
        self.assertEqual(len(self.backend.sent),count);self.assertEqual(self.backend.stops,1)
        self.assertEqual(self.tick(1)['mapping']['state'],'RELEASE_GRIP_TO_ARM')
    def test_stale_input_requires_new_release_press(self):
        self.active()
        status=frame(self.seq+1,self.ns-200_000_000,1)
        self.control.tick(status,lambda:status,self.ns)
        self.assertEqual(self.backend.stops,1)
        self.assertEqual(self.tick(1)['mapping']['state'],'RELEASE_GRIP_TO_ARM')
    def test_unsettled_robot_cannot_reanchor(self):
        self.tick(0);self.feedback.stationary=False
        self.assertEqual(self.tick(1)['mapping']['state'],'WAIT_ROBOT_STATIONARY')
        self.assertEqual(self.backend.sent,[])
    def test_rebooted_input_is_blocked_in_final_gate(self):
        self.tick(0);self.tick(1)
        gate=frame(0,self.ns+int(1e9/self.config['control_hz']),1)
        self.assertEqual(self.tick(1,[0,0,-.002],gate)['mapping']['state'],'INPUT_GATE_CLOSED')


class JointSpeedTests(ControllerHarness,unittest.TestCase):
    def first_joint_step(self, axis, hardware_speeds):
        # A one-axis, reversible fake kinematic model isolates each joint's cap.
        self.ns=1_000_000_000;self.seq=0
        self.config.update(max_joint_speed_deg_s=90.,max_linear_speed_m_s=.25,
                           target_smoothing_tau_s=0.,min_wrist_angle_deg=0.)
        self.backend=FakeBackend();self.feedback=FakeFeedback()
        self.backend.speed_limits=hardware_speeds
        self.backend.forward=lambda joints:[joints[axis]/1000,0,0,1,0,0,0]
        self.backend.solve=lambda target,seed:[target[0]*1000 if i==axis else 0. for i in range(6)]
        self.control=TeleopController(self.config,self.backend,self.feedback,True,lambda:self.ns)
        self.tick(0);self.tick(1)
        result=self.tick(1,[0,0,-.05])
        self.assertEqual(result['mapping']['state'],'TRACKING')
        self.assertTrue(result['motion_command_sent_this_tick'])
        self.assertEqual(len(self.backend.sent),1)
        return self.backend.sent[0][axis]

    def test_lower_hardware_speed_is_enforced_for_each_joint(self):
        hardware_speeds=[35.,45.,55.,65.,75.,85.]
        for axis,speed in enumerate(hardware_speeds):
            with self.subTest(joint=axis+1):
                step=self.first_joint_step(axis,hardware_speeds)
                self.assertGreater(step,0.)
                self.assertLessEqual(step,speed/self.config['control_hz']+1e-4)

    def test_higher_hardware_speed_allows_configured_90_instead_of_60(self):
        for axis in range(6):
            with self.subTest(joint=axis+1):
                step=self.first_joint_step(axis,[180.]*6)
                self.assertGreater(step,60./self.config['control_hz'])
                self.assertLessEqual(step,90./self.config['control_hz']+1e-4)


class RotationTests(unittest.TestCase):
    def setUp(self):
        self.config=legacy_config()
        self.config.update(target_smoothing_tau_s=0.,max_angular_speed_rad_s=100.,max_control_gap_s=.1,
                           rotation_mapping='world')
    def test_all_three_rotation_axes_map_into_work_frame(self):
        for axis in range(3):
            mapper=IncrementalMapper(self.config,ROBOT)
            mapper.update(frame(1,1_000_000_000,0),1_000_000_000,lambda:ROBOT)
            mapper.update(frame(2,1_020_000_000,1),1_020_000_000,lambda:ROBOT)
            q=[0.,0.,0.,math.cos(.1)];q[axis]=math.sin(.1)
            result=mapper.update(frame(3,1_040_000_000,1,q=q),1_040_000_000,lambda:ROBOT)
            target=result['target_pose_wxyz'][3:]
            expected=[math.cos(.1)]+[self.config['headset_to_work_matrix'][i][axis]*math.sin(.1) for i in range(3)]
            self.assertAlmostEqual(angle(multiply(target,inverse(expected))),0.,places=6)
    def test_regrip_at_new_wrist_orientation_has_no_robot_jump(self):
        mapper=IncrementalMapper(self.config,ROBOT)
        q=[math.sin(.3),0.,0.,math.cos(.3)]
        mapper.update(frame(1,1_000_000_000,0,q=q),1_000_000_000,lambda:ROBOT)
        result=mapper.update(frame(2,1_020_000_000,1,q=q),1_020_000_000,lambda:ROBOT)
        self.assertEqual(result['target_pose_wxyz'],ROBOT)
    def test_rotation_wrap_is_refused_before_180_degree_ambiguity(self):
        mapper=IncrementalMapper(self.config,ROBOT)
        mapper.update(frame(1,1_000_000_000,0),1_000_000_000,lambda:ROBOT)
        mapper.update(frame(2,1_020_000_000,1),1_020_000_000,lambda:ROBOT)
        result=None
        for i in range(1,14):
            a=math.radians(i*10)/2;ns=1_020_000_000+i*20_000_000
            result=mapper.update(frame(i+2,ns,1,q=[0,math.sin(a),0,math.cos(a)]),ns,lambda:ROBOT)
        self.assertEqual(result['state'],'CLUTCH_ROTATION_BOUND');self.assertFalse(result['send'])
    def test_low_pass_goal_does_not_exceed_speed_limit(self):
        config=legacy_config()
        config['max_control_gap_s']=.1
        mapper=IncrementalMapper(config,ROBOT)
        mapper.update(frame(1,1_000_000_000,0),1_000_000_000,lambda:ROBOT)
        mapper.update(frame(2,1_020_000_000,1),1_020_000_000,lambda:ROBOT)
        result=mapper.update(frame(3,1_040_000_000,1,[0,0,-.08]),1_040_000_000,lambda:ROBOT)
        self.assertLessEqual(result['target_pose_wxyz'][0],config['max_linear_speed_m_s']*.02+1e-9)

    def test_local_roll_rotates_tool_axis_at_arbitrary_hand_and_robot_anchors(self):
        from incremental import quat
        self.config['rotation_mapping']='controller_local'
        hand_q=quat([.5,.5,.5,.5])
        robot_q=quat([math.cos(.3),0,math.sin(.3),0])
        robot=[.1,.2,.3]+robot_q
        mapper=IncrementalMapper(self.config,robot)
        xyzw=hand_q[1:]+hand_q[:1]
        mapper.update(frame(1,1_000_000_000,0,q=xyzw),1_000_000_000,lambda:robot)
        mapper.update(frame(2,1_020_000_000,1,q=xyzw),1_020_000_000,lambda:robot)
        roll=[math.cos(.1),0,0,math.sin(.1)]
        now_q=multiply(hand_q,roll)
        result=mapper.update(frame(3,1_040_000_000,1,q=now_q[1:]+now_q[:1]),1_040_000_000,lambda:robot)
        expected=multiply(robot_q,roll)
        self.assertAlmostEqual(angle(multiply(result['target_pose_wxyz'][3:],inverse(expected))),0.,places=6)
        self.assertEqual(result['target_pose_wxyz'][:3],robot[:3])

    def test_local_rotation_axis_calibration_can_reverse_roll_without_reflection(self):
        self.config.update(rotation_mapping='controller_local',controller_to_tool_matrix=[[1,0,0],[0,-1,0],[0,0,-1]])
        mapper=IncrementalMapper(self.config,ROBOT)
        mapper.update(frame(1,1_000_000_000,0),1_000_000_000,lambda:ROBOT)
        mapper.update(frame(2,1_020_000_000,1),1_020_000_000,lambda:ROBOT)
        r=mapper.update(frame(3,1_040_000_000,1,q=[0,0,math.sin(.1),math.cos(.1)]),1_040_000_000,lambda:ROBOT)
        self.assertAlmostEqual(r['target_pose_wxyz'][6],-math.sin(.1))

    def test_clutch_pause_does_not_hide_crossing_180_degrees(self):
        self.config['rotation_mapping']='controller_local'
        mapper=IncrementalMapper(self.config,ROBOT)
        mapper.update(frame(1,1_000_000_000,0),1_000_000_000,lambda:ROBOT)
        mapper.update(frame(2,1_020_000_000,1),1_020_000_000,lambda:ROBOT)
        for i in range(1,19):
            ns=1_020_000_000+i*20_000_000;a=math.radians(i*10)/2
            r=mapper.update(frame(i+2,ns,1,q=[0,0,math.sin(a),math.cos(a)]),ns,lambda:ROBOT)
        self.assertEqual(r['state'],'ROTATION_WRAP')
        self.assertFalse(mapper.active)


class RecoveryTests(ControllerHarness,unittest.TestCase):
    def start_pose_pause(self):
        self.config['max_session_translation_m']=.02
        self.active()
        r=self.tick(1,[0,0,-.05])
        self.assertEqual(r['mapping']['state'],'SESSION_BOUND')
        self.assertTrue(self.control.mapper.paused)
        return r
    def test_pose_bound_resumes_with_held_grip_from_actual_pose(self):
        self.start_pose_pause();count=len(self.backend.sent)
        for _ in range(10):self.tick(1,[0,0,-.05])
        self.assertEqual(len(self.backend.sent),count)
        anchor=self.control.mapper.anchor_hand.copy()
        resumed=None
        for _ in range(40):
            r=self.tick(1,[0,0,-.01])
            if r['mapping']['state']=='RESUMED':resumed=r;break
        self.assertIsNotNone(resumed)
        self.assertFalse(resumed['motion_sent_this_tick'])
        self.assertEqual(self.control.mapper.anchor_hand,anchor)
        self.assertEqual(self.control.last_target,ROBOT)
        r=self.tick(1,[0,0,-.01])
        self.assertTrue(r['motion_sent_this_tick'])
        self.assertLessEqual(abs(self.backend.sent[-1][0]),self.config['max_joint_speed_deg_s']/self.config['control_hz']+1e-4)
    def test_focus_loss_while_soft_paused_still_requires_grip_release(self):
        self.start_pose_pause()
        status=self.sample(1,[0,0,0]);status['latest']['app_focus']=False
        self.control.tick(status,lambda:status,self.ns)
        self.assertFalse(self.control.mapper.active)
        self.assertEqual(self.tick(1)['mapping']['state'],'RELEASE_GRIP_TO_ARM')
    def test_ik_failure_can_resume_after_target_becomes_reachable(self):
        self.active();self.backend.fail_ik=True
        self.assertEqual(self.tick(1,[0,0,-.002])['mapping']['state'],'IK_REFUSED')
        for _ in range(40):self.tick(1,[0,0,-.002])
        self.assertTrue(self.control.mapper.paused)
        self.backend.fail_ik=False
        states=[self.tick(1,[0,0,-.002])['mapping']['state'] for _ in range(40)]
        self.assertIn('RESUMED',states)
        self.assertTrue(self.control.streaming)
    def test_recovery_checks_full_joint_goal_before_resuming(self):
        self.config['max_joint_session_deg']=2.
        self.active()
        for _ in range(100):r=self.tick(1,[0,0,-.005])
        self.assertTrue(self.control.mapper.paused)
        count=len(self.backend.sent)
        for _ in range(60):self.tick(1,[0,0,-.005])
        self.assertEqual(len(self.backend.sent),count)
        states=[self.tick(1,[0,0,0])['mapping']['state'] for _ in range(40)]
        self.assertIn('RESUMED',states)
    def test_pause_stop_delay_does_not_cancel_soft_recovery(self):
        self.start_pose_pause();self.ns+=100_000_000
        r=self.tick(1,[0,0,0])
        self.assertEqual(r['mapping']['state'],'RECOVERY_WAIT')
        self.assertTrue(self.control.mapper.active)
    def test_command_gap_is_checked_after_input_and_ik(self):
        self.active()
        status=self.sample(1,[0,0,-.002])
        def gate():
            self.ns+=int((self.config['max_command_gap_s']-1/self.config['control_hz']+.001)*1e9)
            return status
        r=self.control.tick(status,gate,self.ns)
        self.assertEqual(r['mapping']['state'],'COMMAND_GAP')
        self.assertFalse(r['motion_sent_this_tick'])
        self.assertFalse(self.control.mapper.active)
    def test_same_frame_keeps_advancing_rate_limited_goal_without_drift(self):
        self.active()
        status=self.sample(1,[0,0,-.04])
        r=self.control.tick(status,lambda:status,self.ns)
        first=r['mapping']['target_pose_wxyz'][0]
        for _ in range(8):
            self.ns+=5_000_000
            r=self.control.tick(status,lambda:status,self.ns)
        self.assertGreater(r['mapping']['target_pose_wxyz'][0],first)
        self.assertLessEqual(r['mapping']['target_pose_wxyz'][0],.02)
    def test_high_follow_control_gap_disarms(self):
        self.active();self.ns+=int(self.config['max_control_gap_s']*1e9)
        r=self.tick(1,[0,0,-.002])
        self.assertEqual(r['mapping']['state'],'CONTROL_GAP')
        self.assertFalse(r['motion_sent_this_tick'])


class FeedbackTests(unittest.TestCase):
    def test_individual_joint_velocity_limits_are_enforced(self):
        def solver(target,_):return [target[0]*1000,target[0]*500,0,0,0,0]
        target,joints,limited=limit_ik_step(solver,ROBOT,[.002,0,0,1,0,0,0],[0]*6,[.1,.01,.1,.1,.1,.1])
        self.assertTrue(limited)
        self.assertLessEqual(abs(joints[0]),.1)
        self.assertLessEqual(abs(joints[1]),.01)
    def test_old_firmware_units_and_alarm_fields(self):
        r=decode_state({'arm_state':{'joint':[1000]*6,'pose':[100000,0,0,1000,0,0],'arm_err':0,'sys_err':0}})
        self.assertEqual(r['joints'],[1.]*6);self.assertEqual(r['position_m'][0],.1)
    def test_alarm_is_not_silently_ignored(self):
        with self.assertRaises(MappingError):
            decode_state({'arm_state':{'joint':[0]*6,'pose':[0]*6,'arm_err':4,'sys_err':0}})
    def test_monitor_has_no_write_query(self):
        with self.assertRaises(MappingError):query('127.0.0.1','movej')
    def test_raw_frame_change_is_detectable(self):
        work={'frame_name':'Base','pose':[0]*6};tool={'tool_name':'Arm_Tip','pose':[0]*6,'payload':0,'position':[0]*3}
        before=frame_signature(work,tool);work['pose'][4]=-785
        self.assertNotEqual(before,frame_signature(work,tool))
    def test_configuration_rejects_invalid_speed_values(self):
        import tempfile
        data=json.loads((ROOT/'configs/right_full_teleop.json').read_text())
        for value in (float('nan'),float('inf'),0.,-1.,True,'5'):
            data['max_linear_speed_m_s']=value
            with tempfile.NamedTemporaryFile(mode='w',suffix='.json') as f:
                json.dump(data,f);f.flush()
                with self.assertRaises(MappingError):load_config(f.name)
    def test_joint_speed_configuration_has_no_arbitrary_90_degree_maximum(self):
        import tempfile
        data=json.loads((ROOT/'configs/right_full_teleop.json').read_text())
        for value in (90.,90.0001,120.,250.):
            with self.subTest(speed=value):
                data['max_joint_speed_deg_s']=value
                with tempfile.NamedTemporaryFile(mode='w',suffix='.json') as f:
                    json.dump(data,f);f.flush()
                    self.assertEqual(load_config(f.name)['max_joint_speed_deg_s'],value)
    def test_high_follow_cannot_use_slow_or_unchecked_command_periods(self):
        import tempfile
        baseline=json.loads((ROOT/'configs/right_full_teleop.json').read_text())
        baseline.update(follow_mode='high',control_hz=200,max_control_gap_s=.008,max_command_gap_s=.01)
        for key,value in [('control_hz',50),('max_command_gap_s',.011),('advance_on_same_frame',False)]:
            data={**baseline,key:value}
            with tempfile.NamedTemporaryFile(mode='w',suffix='.json') as f:
                json.dump(data,f);f.flush()
                with self.assertRaises(MappingError):load_config(f.name)


if __name__=='__main__':unittest.main()
