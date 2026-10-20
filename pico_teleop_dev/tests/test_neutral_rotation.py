import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import copy
import math
from pathlib import Path
import tempfile
import json
import unittest

from incremental import IncrementalMapper, MappingError, angle, inverse, multiply, rotate_vector
from rotation_axes import neutral_tool_calibration
from teleop_config import load_config
from test_incremental import frame, ROBOT
from test_full_teleop import FakeBackend, FakeFeedback, legacy_config
from teleop_controller import TeleopController
from robot_backend import Backend


ROOT=Path(__file__).resolve().parent.parent


def axis_quaternion(axis, degrees):
    half=math.radians(degrees)/2
    return [math.cos(half)]+[x*math.sin(half) for x in axis]


class NeutralRotationTests(unittest.TestCase):
    def setUp(self):
        self.config=legacy_config()
        self.config.update(rotation_mapping='neutral_tool_axes',max_clutch_rotation_rad=math.radians(190),
            max_tool_tilt_rad=math.pi/2,target_smoothing_tau_s=0.,max_angular_speed_rad_s=100.,
            max_control_gap_s=.1)
        # Realistic downward tool, with a deliberately arbitrary roll and 7-degree tilt.
        self.robot=[.1,.2,.3]+multiply(axis_quaternion([0,0,1],31),
            multiply(axis_quaternion([0,1,0],7),axis_quaternion([1,0,0],180)))
        self.hand=multiply(axis_quaternion([0,1,0],67),axis_quaternion([1,0,0],-38))
        self.ns=1_000_000_000;self.seq=0
        self.mapper=IncrementalMapper(self.config,self.robot)

    def step(self, grip, q=None, yaw=0., xyz=None, robot=None):
        self.ns+=20_000_000;self.seq+=1
        q=q or self.hand
        data=frame(self.seq,self.ns,grip,xyz,q[1:]+q[:1])
        h=axis_quaternion([0,1,0],yaw)
        data['latest']['head']['pose']=[0,0,0]+h[1:]+h[:1]
        return self.mapper.update(data,self.ns,lambda:robot or self.robot)

    def engage(self, yaw=0.):
        self.step(0,yaw=yaw)
        return self.step(1,yaw=yaw)

    def roll_to(self, degrees, yaw=0., flip=False):
        down=rotate_vector(axis_quaternion([0,1,0],yaw),[0,-1,0])
        result=None
        direction=1 if degrees>=0 else -1
        for n in range(5,abs(degrees)+1,5):
            q=multiply(axis_quaternion(down,n*direction),self.hand)
            if flip and n%10==0:q=[-v for v in q]
            result=self.step(1,q,yaw=yaw)
        return result

    def test_neutral_engage_preserves_actual_tool_pose(self):
        result=self.engage()
        self.assertEqual(result['state'],'ANCHORED')
        self.assertEqual(result['target_pose_wxyz'][:3],self.robot[:3])
        self.assertLess(angle(multiply(result['target_pose_wxyz'][3:],inverse(self.robot[3:]))),1e-6)
        self.assertIsNotNone(result['neutral_rotation_calibration'])

    def test_downward_roll_maps_actual_tool_z_at_all_operator_headings(self):
        for yaw in (0,90,180,-90):
            with self.subTest(yaw=yaw):
                self.mapper=IncrementalMapper(self.config,self.robot)
                self.engage(yaw)
                result=self.roll_to(90,yaw)
                expected=multiply(self.robot[3:],axis_quaternion([0,0,1],90))
                self.assertEqual(result['state'],'TRACKING')
                self.assertLess(angle(multiply(result['requested_pose_wxyz'][3:],inverse(expected))),1e-6)
                self.assertEqual(result['requested_pose_wxyz'][:3],self.robot[:3])
                self.assertAlmostEqual(result['tool_twist_deg'],90.)
                self.assertAlmostEqual(result['tool_tilt_deg'],0.,places=5)

    def test_signed_180_roll_is_not_disarmed_by_old_wrap_guard(self):
        for target in (180,-180):
            with self.subTest(target=target):
                self.mapper=IncrementalMapper(self.config,self.robot)
                self.engage()
                result=self.roll_to(target,flip=True)
                self.assertEqual(result['state'],'TRACKING')
                self.assertAlmostEqual(result['tool_twist_deg'],target)
                self.assertTrue(self.mapper.active)

    def test_half_scale_crosses_180_without_orientation_jump(self):
        self.config['rotation_scale']=.5
        self.engage()
        self.roll_to(175)
        poses=[]
        for degree in (179,181):
            q=multiply(axis_quaternion([0,-1,0],degree),self.hand)
            result=self.step(1,q)
            poses.append(result['requested_pose_wxyz'][3:])
        self.assertAlmostEqual(math.degrees(angle(multiply(poses[1],inverse(poses[0])))),1.,places=5)

    def test_waiting_follower_does_not_reverse_signed_190_goal(self):
        self.engage();self.roll_to(190)
        self.config['max_angular_speed_rad_s']=1.
        self.mapper.target=self.robot[:3]+multiply(self.robot[3:],axis_quaternion([0,0,1],5))
        self.mapper.target_twist_rad=math.radians(5)
        result=self.step(1,multiply(axis_quaternion([0,-1,0],190),self.hand))
        local=multiply(inverse(self.robot[3:]),result['target_pose_wxyz'][3:])
        reached=math.degrees(2*math.atan2(local[3],local[0]))
        self.assertGreater(reached,5.)
        self.assertLessEqual(reached,5.+math.degrees(.02)+1e-6)

    def test_negative_180_goal_reverses_positive_command_in_correct_direction(self):
        self.engage();self.roll_to(-180)
        self.config['max_angular_speed_rad_s']=1.
        self.mapper.target=self.robot[:3]+multiply(self.robot[3:],axis_quaternion([0,0,1],5))
        self.mapper.target_twist_rad=math.radians(5)
        result=self.step(1,multiply(axis_quaternion([0,-1,0],-180),self.hand))
        self.assertLess(result['target_tool_twist_deg'],5.)

    def test_out_of_range_half_scale_near_360_disarms_without_bad_quaternion(self):
        self.config['rotation_scale']=.5
        self.engage()
        result=self.roll_to(360)
        self.assertEqual(result['state'],'ROTATION_WRAP')
        self.assertFalse(self.mapper.active)

    def test_beyond_continuous_clutch_range_soft_pauses(self):
        self.engage()
        result=self.roll_to(195)
        self.assertEqual(result['state'],'CLUTCH_ROTATION_BOUND')
        self.assertTrue(self.mapper.paused)
        self.assertTrue(self.mapper.active)

    def test_forward_pitch_has_correct_axis_and_tilt_limit(self):
        self.engage()
        for degree in range(5,96,5):
            result=self.step(1,multiply(axis_quaternion([1,0,0],degree),self.hand))
            if degree==90:
                self.assertEqual(result['state'],'TRACKING')
                self.assertAlmostEqual(result['tool_tilt_deg'],90.)
        self.assertEqual(result['state'],'TOOL_TILT_BOUND')
        self.assertTrue(self.mapper.paused)

    def test_regrip_does_not_relearn_neutral_axes_or_jump(self):
        self.engage()
        basis=copy.deepcopy(self.mapper.tool_basis)
        self.roll_to(45)
        aligned_hand=multiply(axis_quaternion([0,-1,0],45),self.hand)
        moved_robot=self.robot[:3]+multiply(self.robot[3:],axis_quaternion([0,0,1],45))
        self.step(0,aligned_hand)
        result=self.step(1,aligned_hand,robot=moved_robot)
        self.assertEqual(self.mapper.tool_basis,basis)
        self.assertEqual(result['target_pose_wxyz'][:3],moved_robot[:3])
        self.assertLess(angle(multiply(result['target_pose_wxyz'][3:],inverse(moved_robot[3:]))),1e-6)
        now=multiply(aligned_hand,axis_quaternion(self.mapper.neutral_rotation_calibration['hand_long_axis_local'],5))
        result=self.step(1,now,robot=moved_robot)
        expected=multiply(moved_robot[3:],axis_quaternion([0,0,1],5))
        self.assertLess(angle(multiply(result['requested_pose_wxyz'][3:],inverse(expected))),1e-6)

    def test_translation_is_independent_of_hand_rotation(self):
        self.engage()
        first=self.step(1,xyz=[.04,0,0])
        for degree in range(5,91,5):
            result=self.step(1,multiply(axis_quaternion([0,-1,0],degree),self.hand),xyz=[.04,0,0])
        self.assertEqual(first['requested_delta_work_m'],result['requested_delta_work_m'])

    def test_roll_only_ignores_tilt_and_keeps_long_axis_twist(self):
        self.config['rotation_control_mode']='roll_only'
        self.engage()
        self.roll_to(90)
        for degree in range(5,31,5):
            q=multiply(axis_quaternion([1,0,0],degree),
                multiply(axis_quaternion([0,-1,0],90),self.hand))
            result=self.step(1,q)
        expected=multiply(self.robot[3:],axis_quaternion([0,0,1],90))
        self.assertLess(angle(multiply(result['requested_pose_wxyz'][3:],inverse(expected))),1e-6)

    def test_singular_neutral_axis_alignment_is_rejected(self):
        # Tool +Z along robot right: right projection vanishes.
        robot=ROBOT[:3]+axis_quaternion([1,0,0],90)
        with self.assertRaisesRegex(MappingError,'too close'):
            neutral_tool_calibration([0,0,0]+self.hand,robot,
                [[1,0,0],[0,1,0],[0,0,1]],self.config['headset_to_work_matrix'])


class JointSessionBoundsTests(unittest.TestCase):
    def setUp(self):
        self.config=legacy_config()
        self.config['max_joint_session_deg_by_joint']=[90,90,90,110,90,210]
        self.backend=FakeBackend();self.backend.minimum[5]=-360.;self.backend.maximum[5]=360.
        self.feedback=FakeFeedback()
        self.control=TeleopController(self.config,self.backend,self.feedback,True)

    def test_j6_range_does_not_expand_other_joints(self):
        self.assertEqual(self.control.command_joint_bounds(),[87,87,87,107,87,207])
        self.assertTrue(self.control.joint_target_allowed([0,0,0,100,0,180]))
        self.assertFalse(self.control.joint_target_allowed([88,0,0,0,0,180]))
        self.assertFalse(self.control.joint_target_allowed([0,0,0,108,0,180]))

    def test_actual_feedback_uses_each_joint_bound(self):
        self.feedback.joints[5]=180.
        self.control.validate_feedback(self.feedback.snapshot())
        self.feedback.joints[0]=90.1
        with self.assertRaises(MappingError) as caught:self.control.validate_feedback(self.feedback.snapshot())
        self.assertEqual(caught.exception.details['over_limit_joints'],[1])

    def test_settle_tolerance_only_applies_to_known_pause(self):
        self.feedback.joints[5]=211.4
        with self.assertRaises(MappingError):self.control.validate_feedback(self.feedback.snapshot())
        self.control.mapper.paused=True
        self.control.validate_feedback(self.feedback.snapshot())
        self.feedback.joints[5]=212.1
        with self.assertRaises(MappingError):self.control.validate_feedback(self.feedback.snapshot())

    def test_returning_path_is_inward_for_each_over_limit_joint(self):
        self.control.seed[5]=211.4;self.control.returning_inside_joints=True
        self.assertTrue(self.control.joint_target_allowed([0,0,0,0,0,211.]))
        self.assertFalse(self.control.joint_target_allowed([0,0,0,0,0,211.5]))
        self.assertFalse(self.control.joint_target_allowed([88,0,0,0,0,211.]))

    def test_hardware_limit_remains_independent(self):
        self.control.mapper.paused=True;self.feedback.joints[5]=360.1
        with self.assertRaisesRegex(MappingError,'hardware'):
            self.control.validate_feedback(self.feedback.snapshot())

    def test_recovery_checks_entire_j6_goal(self):
        self.control.mapper.pause('IK_REFUSED')
        self.backend.solve=lambda *_:[0,0,0,0,0,208]
        result={'requested_pose_wxyz':ROBOT.copy(),'hand_rotation_deg':5.}
        outcome=self.control.recover(result,self.feedback.snapshot(),ROBOT.copy(),1_000_000_000)
        self.assertEqual(outcome['state'],'JOINT_SESSION_BOUND')

    def test_configuration_ranges_are_finite_and_above_each_joint_reserve(self):
        config=legacy_config()
        config['max_joint_session_deg_by_joint']=[90.,90.,90.,110.,90.,210.]
        for index,value in ((0,3.),(3,0.),(5,float('nan'))):
            bad=copy.deepcopy(config);bad['max_joint_session_deg_by_joint'][index]=value
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'bad.json';path.write_text(json.dumps(bad))
                with self.assertRaises(MappingError):load_config(path)
        # Configured task ranges do not redefine physical hardware limits.
        config['max_joint_session_deg_by_joint']=[210.,180.,250.,300.,220.,1000.]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'legal.json';path.write_text(json.dumps(config))
            self.assertEqual(load_config(path)['max_joint_session_deg_by_joint'],
                             config['max_joint_session_deg_by_joint'])


class J6ContinuityTests(unittest.TestCase):
    def backend(self, minimum=-360., maximum=360.):
        class MathSDK:
            def rm_algo_inverse_kinematics(self,_):return 0,[0,0,0,0,0,-170.]
            def rm_algo_forward_kinematics(self,joints,_):
                return ROBOT[:3]+axis_quaternion([0,0,1],joints[5])
        backend=object.__new__(Backend)
        backend.robot=MathSDK();backend.ik_params=lambda *x:x
        backend.minimum=[-178]*5+[minimum];backend.maximum=[178]*5+[maximum]
        return backend

    def test_principal_sdk_angle_is_unwrapped_near_previous_joint_command(self):
        joints=self.backend().solve(ROBOT,[0,0,0,0,0,185.])
        self.assertAlmostEqual(joints[5],190.)

    def test_equivalent_angle_never_crosses_hardware_limit(self):
        joints=self.backend(-178.,178.).solve(ROBOT,[0,0,0,0,0,170.])
        self.assertAlmostEqual(joints[5],-170.)


if __name__=='__main__':unittest.main()
