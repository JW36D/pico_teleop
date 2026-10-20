"""Left-arm routing and mirrored installation math; every sender is offline."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from incremental import IncrementalMapper,input_health,input_ready,angle,inverse,multiply
from robot_directions import resolve_translation_mapping,euler_quaternion
from teleop_config import load_config
from teleop_controller import TeleopController
from diagnostic_context import DiagnosticContext
from control_sample import control_sample
from robot_backend import Backend
from test_incremental import frame
from test_full_teleop import FakeBackend,FakeFeedback
import test_fault_context_main as main_harness


def left_frame(sequence,ns,grip,xyz=None):
    value=frame(sequence,ns,grip,xyz)
    value['latest']['left']=value['latest'].pop('right')
    value['latest']['right']={'grip':0.,'status':0,'secondaryButton':False,
        'trigger':.9,'pose':[float('nan')]*7}
    return value


class LeftControllerTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config(ROOT/'configs/left_rotation_teleop.json')
        self.config.update(translation_mapping_frame='work',translation_reference='tracking_world',
            rotation_mapping='world',rotation_scale=0.,headset_to_work_matrix=[[1,0,0],[0,1,0],[0,0,1]])
        self.ns=1_000_000_000;self.sequence=0
        self.backend=FakeBackend();self.feedback=FakeFeedback()
        self.controller=TeleopController(self.config,self.backend,self.feedback,True,lambda:self.ns)

    def step(self,grip,xyz=None,gate=None):
        self.ns+=10_000_000;self.sequence+=1
        if self.backend.sent:self.feedback.joints=self.backend.sent[-1].copy()
        status=left_frame(self.sequence,self.ns,grip,xyz)
        return self.controller.tick(status,gate or (lambda:status),self.ns)

    def engage(self):
        self.step(0.);return self.step(1.)

    def test_left_grip_and_motion_ignore_invalid_right_hand(self):
        self.assertEqual(self.engage()['mapping']['state'],'ANCHORED')
        r=self.step(1.,[.002,0,0])
        self.assertTrue(r['motion_command_sent_this_tick'])
        self.assertGreater(self.backend.sent[-1][0],0.)
        self.assertEqual(r['arm'],'left')

    def test_right_grip_alone_cannot_arm_left(self):
        status=left_frame(1,self.ns,0.)
        status['latest']['right']={'grip':1.,'pose':[0,0,0,0,0,0,1]}
        self.controller.tick(status,lambda:status,self.ns)
        self.assertFalse(self.controller.mapper.active)
        self.assertEqual(self.backend.sent,[])

    def test_late_left_release_blocks_send_despite_right_grip(self):
        self.engage()
        def gate():
            status=left_frame(self.sequence,self.ns,0.)
            status['latest']['right']={'grip':1.,'pose':[0,0,0,0,0,0,1]}
            return status
        r=self.step(1.,[.003,0,0],gate)
        self.assertEqual(r['mapping']['state'],'INPUT_GATE_CLOSED')
        self.assertFalse(r['motion_command_sent_this_tick'])
        self.assertEqual(self.backend.sent,[])

    def test_left_y_ends_before_send_while_right_b_is_ignored(self):
        self.engage()
        def gate():
            status=left_frame(self.sequence,self.ns,1.,[.003,0,0])
            status['session_exit_requested']=True
            return status
        r=self.step(1.,[.003,0,0],gate)
        self.assertEqual(r['mapping']['state'],'SESSION_EXIT')
        self.assertEqual(self.backend.sent,[])
        self.setUp();self.engage()
        def right_b():
            status=left_frame(self.sequence,self.ns,1.,[.003,0,0])
            status['latest']['right']['secondaryButton']=True
            return status
        self.assertTrue(self.step(1.,[.003,0,0],right_b)['motion_command_sent_this_tick'])

    def test_left_short_gap_health_and_missing_left_never_fall_back_to_right(self):
        status=left_frame(1,self.ns,1.)
        self.assertTrue(input_ready(status,self.ns,side='left'))
        self.assertTrue(input_health(status,self.ns+120_000_000,100,150,side='left')['recoverable'])
        status['latest']['right']={'grip':1.,'pose':[0,0,0,0,0,0,1]}
        del status['latest']['left']
        self.assertFalse(input_ready(status,self.ns,side='left'))

    def test_left_diagnostics_and_optional_recording_keep_left_identity(self):
        self.engage();r=self.step(1.,[.002,0,0])
        r.update(time_monotonic_ns=self.ns,control_ticks=3)
        status=left_frame(self.sequence,self.ns,1.,[.002,0,0])
        status['latest']['left']['trigger']=.2
        sample=control_sample(status,r,self.controller)
        self.assertIn('left_controller',sample)
        self.assertNotIn('right_controller',sample)
        self.assertEqual(sample['raw_hand_intent']['trigger'],.2)
        cache=DiagnosticContext(controller_side='left')
        cache.capture(status,r,True)
        self.assertEqual(cache.snapshot('test')['samples'][0]['pico']['pose'][0],.002)
        r['mapping']={'state':'SESSION_EXIT'}
        self.assertIsNone(cache.capture(status,r,False))


class LeftGeometryTests(unittest.TestCase):
    def test_left_profile_reuses_confirmed_right_tuning(self):
        left=load_config(ROOT/'configs/left_rotation_teleop.json')
        right=load_config(ROOT/'configs/right_rotation_teleop.json')
        excluded={'arm','robot_body_expected_installation_deg','physical_direction_validation',
            'robot_frame_evidence','rotation_reference'}
        self.assertEqual({k:v for k,v in left.items()if k not in excluded},
            {k:v for k,v in right.items()if k not in excluded})

    def test_opposite_work_pitch_preserves_common_robot_front_right_up(self):
        for side,pitch in [('left',.785),('right',-.785)]:
            config=load_config(ROOT/('configs/'+side+'_rotation_teleop.json'))
            resolved,report=resolve_translation_mapping(config,
                {'rm_get_current_work_frame':{'name':'Base','pose':[0,0,0,0,pitch,0]}},
                [0,45 if side=='left'else -45,0])
            from incremental import rotate_vector
            for direction,expected in [('forward',[1,0,0]),('right',[0,-1,0]),('up',[0,0,1])]:
                rebuilt=rotate_vector(euler_quaternion([0,pitch,0]),report['robot_direction_axes_work'][direction])
                for a,b in zip(rebuilt,expected):self.assertAlmostEqual(a,b)
            self.assertAlmostEqual(resolved['headset_to_work_matrix'][0][1],-math.sin(pitch))

    def test_left_neutral_rotation_uses_actual_tool_and_xy_contract(self):
        config=load_config(ROOT/'configs/left_rotation_teleop.json')
        config,_=resolve_translation_mapping(config,
            {'rm_get_current_work_frame':{'name':'Base','pose':[0,0,0,0,.785,0]}},[0,45,0])
        robot=[.1,-.4,.2]+euler_quaternion([math.pi,.1,.3])
        mapper=IncrementalMapper(config,robot)
        mapper.update(left_frame(1,1_000_000_000,0.),1_000_000_000,lambda:robot)
        r=mapper.update(left_frame(2,1_010_000_000,1.),1_010_000_000,lambda:robot)
        self.assertEqual(r['state'],'ANCHORED')
        self.assertLess(angle(multiply(r['target_pose_wxyz'][3:],inverse(robot[3:]))),1e-6)
        self.assertIn('XY',r['neutral_rotation_calibration']['neutral_pose_contract'])

    def test_left_backend_reads_left_address_without_movement_or_joint_sign_changes(self):
        calls=[];q=[-84.,-65.,-29.,-130.,87.,188.8]
        class SDK:
            def __init__(self,*_):pass
            def rm_create_robot_arm(self,ip,port):calls.append((ip,port));return SimpleNamespace(id=9)
            def rm_get_robot_info(self):return 0,{'arm_dof':6,'arm_model':'RM_65'}
            def rm_get_arm_run_mode(self):return 0,1
            def rm_get_joint_min_pos(self):return 0,[-178,-130,-135,-178,-128,-360]
            def rm_get_joint_max_pos(self):return 0,[178,130,135,178,128,360]
            def rm_get_current_work_frame(self):return 0,{'name':'Base','pose':[0,0,0,0,.785,0]}
            def rm_get_current_tool_frame(self):return 0,{'name':'Arm_Tip','pose':[0.]*6}
            def rm_get_install_pose(self):return {'return_code':0,'x':0,'y':45,'z':0}
            def rm_algo_get_angle(self):return [0,45,0]
            def rm_get_joint_degree(self):return 0,q.copy()
            def rm_get_joint_err_flag(self):return {'return_code':0,'err_flag':[0]*6}
            def rm_get_arm_current_trajectory(self):return {'return_code':0,'trajectory_type':0}
            def rm_algo_forward_kinematics(self,*_):return [0.,0.,0.,1.,0.,0.,0.]
            def rm_delete_robot_arm(self):calls.append('delete')
            @classmethod
            def rm_destroy(cls):calls.append('destroy')
        module=SimpleNamespace(RoboticArm=SDK,rm_thread_mode_e=SimpleNamespace(RM_DUAL_MODE_E=0),
            rm_inverse_kinematics_params_t=lambda *a:a)
        with mock.patch.dict(sys.modules,{'Robotic_Arm.rm_robot_interface':module}),mock.patch('robot_backend.time.sleep'):
            backend=Backend(arm='left')
            self.assertEqual(backend.initial_joints,q)
            backend.close()
        self.assertEqual(calls,[('169.254.128.18',8080),'delete','destroy'])

    def test_left_main_selects_left_backend_feedback_y_and_separate_logs(self):
        outcome,actions,events=main_harness.FaultContextMainTests().run_main(['TRACKING','SESSION_EXIT'],arm='left')
        self.assertEqual(outcome,0)
        self.assertIn(('backend_arm','left'),actions)
        self.assertIn(('feedback_host','169.254.128.18'),actions)
        self.assertEqual(events[-1]['reason'],'B_Y_CHORD')
        self.assertTrue(all(e.get('arm')=='left'for e in events))


if __name__=='__main__':unittest.main()
