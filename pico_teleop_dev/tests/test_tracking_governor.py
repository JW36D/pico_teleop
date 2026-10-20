import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import math
import json
from pathlib import Path
import tempfile
import unittest

from incremental import MappingError, IncrementalMapper
from teleop_config import load_config
from robot_directions import resolve_translation_mapping
from test_full_teleop import ControllerHarness
from test_incremental import frame, ROBOT


class TrackingGovernorTests(ControllerHarness, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.config.update(tracking_governor=True,max_command_lead_m=.002,
                           max_command_lead_angle_rad=.08,tracking_resume_ratio=.4,
                           max_tracking_error_m=.01)

    def wait_for_lag(self):
        self.tick(0);self.tick(1)
        for _ in range(30):
            result=self.tick(1,[0,0,-.02])
            if result['mapping']['state']=='FOLLOWER_WAIT':
                return result
        self.fail('A stationary robot must eventually hold the target')

    def test_lag_holds_target_and_preserves_clutch(self):
        result=self.wait_for_lag()
        held=self.backend.sent[-1].copy()
        for _ in range(20):
            result=self.tick(1,[0,0,-.024])
            self.assertEqual(result['mapping']['state'],'FOLLOWER_WAIT')
            self.assertEqual(self.backend.sent[-1],held)
            self.assertTrue(result['motion_command_sent_this_tick'])
            self.assertTrue(self.control.mapper.active)
            self.assertLessEqual(result['mapping']['command_tracking_error_mm'],2.7)

    def test_feedback_catchup_resumes_without_regrip(self):
        self.wait_for_lag()
        self.feedback.joints=self.control.last_sent_joints.copy()
        result=self.tick(1,[0,0,-.02])
        self.assertEqual(result['mapping']['state'],'TRACKING')
        self.assertFalse(self.control.waiting_for_feedback)
        self.assertTrue(result['motion_command_sent_this_tick'])

    def test_resume_uses_lower_hysteresis_threshold(self):
        self.wait_for_lag()
        command=self.control.last_sent_joints[0]
        self.feedback.joints[0]=command-1.2  # 1.2 mm < 2 mm, but > 0.8 mm resume.
        self.assertEqual(self.tick(1,[0,0,-.02])['mapping']['state'],'FOLLOWER_WAIT')
        self.feedback.joints[0]=command-.4
        self.assertEqual(self.tick(1,[0,0,-.02])['mapping']['state'],'TRACKING')

    def test_release_while_waiting_stops(self):
        self.wait_for_lag()
        count=len(self.backend.sent)
        self.assertEqual(self.tick(0)['mapping']['state'],'RELEASED')
        self.assertEqual(len(self.backend.sent),count)
        self.assertEqual(self.backend.stops,1)

    def test_hard_error_still_stops_and_has_quantitative_details(self):
        self.wait_for_lag()
        self.feedback.joints[0]=-20.
        with self.assertRaisesRegex(MappingError,'tracking error') as caught:
            self.tick(1,[0,0,-.02])
        details=caught.exception.details
        self.assertGreater(details['position_error_mm'],10.)
        self.assertEqual(details['position_bound_mm'],10.)
        self.assertEqual(details['last_sent_pose_wxyz'],self.control.last_sent_target)
        self.assertEqual(details['last_sent_joints_deg'],self.control.last_sent_joints)
        self.assertEqual(self.backend.stops,1)

    def test_feedback_checks_sent_target_not_unemitted_candidate(self):
        self.active()
        self.control.last_target=[.05,0,0,1,0,0,0]
        self.control.last_joints=[50.,0,0,0,0,0]
        self.control.validate_feedback(self.feedback.snapshot())

    def test_reanchoring_after_stop_does_not_wait_for_old_target(self):
        self.wait_for_lag();self.tick(0)
        self.feedback.joints[0]=20.
        self.tick(1,[0,0,-.02])
        result=self.tick(1,[0,0,-.021])
        self.assertEqual(result['mapping']['state'],'TRACKING')
        self.assertTrue(result['motion_command_sent_this_tick'])

    def test_preview_calculates_without_governor_or_motion(self):
        self.control.execute=False
        self.tick(0);self.tick(1)
        for _ in range(30):result=self.tick(1,[0,0,-.02])
        self.assertEqual(result['mapping']['state'],'TRACKING')
        self.assertEqual(self.backend.sent,[])

    def test_delayed_twenty_hz_follower_does_not_accumulate_commands(self):
        self.tick(0);self.tick(1)
        physical=0.; peak=0.; waits=0
        for k in range(1,1001):
            command=self.backend.sent[-1][0] if self.backend.sent else 0.
            physical+=(command-physical)*(1-math.exp(-.01/.6))
            if k%5==0:self.feedback.joints[0]=physical
            result=self.tick(1,[0,0,-.035*math.sin(k*.01*2*math.pi*.5)])
            peak=max(peak,result['mapping'].get('command_tracking_error_mm',0.))
            waits+=result['mapping']['state']=='FOLLOWER_WAIT'
        self.assertGreater(waits,10)
        self.assertLess(peak,3.)
        self.assertEqual(self.backend.stops,0)

    def test_gate_release_while_waiting_cannot_send_one_more_hold(self):
        self.wait_for_lag()
        count=len(self.backend.sent)
        gate=frame(self.seq+1,self.ns+10_000_000,0)
        result=self.tick(1,[0,0,-.02],gate=gate)
        self.assertEqual(result['mapping']['state'],'INPUT_GATE_CLOSED')
        self.assertEqual(len(self.backend.sent),count)

    def test_governor_configuration_cannot_reach_stop_bound(self):
        config=json.loads((Path(__file__).resolve().parent.parent/'configs/right_direction_check.json').read_text())
        config['max_command_lead_m']=.01
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text(json.dumps(config))
            with self.assertRaisesRegex(MappingError,'stop bound'):load_config(p)


class RevisedDirectionTests(unittest.TestCase):
    def test_observation_based_profile_is_a_proper_rotation(self):
        config=load_config(Path(__file__).resolve().parent.parent/'configs/right_direction_check.json')
        mapper=IncrementalMapper(config,ROBOT)
        self.assertEqual(mapper.map_vector([1,0,0]),[-1,0,0])
        self.assertEqual(mapper.map_vector([0,1,0]),[0,0,1])
        self.assertEqual(mapper.map_vector([0,0,-1]),[0,-1,0])
        self.assertEqual(config['rotation_scale'],0.)
        self.assertEqual(config['control_hz'],100)
        self.assertEqual(config['max_tracking_error_m'],.01)

    def test_speed_profile_uses_robot_body_directions_and_normal_teleop_speed(self):
        root=Path(__file__).resolve().parent.parent
        fast=load_config(root/'configs/right_speed_check.json')
        self.assertEqual(fast['translation_mapping_frame'],'robot_body')
        with self.assertRaisesRegex(MappingError,'Resolve robot-body'):
            IncrementalMapper(fast,ROBOT)
        resolved,report=resolve_translation_mapping(fast,{'rm_get_current_work_frame':
            {'name':'Base','pose':[0,0,0,0,-.785,0]}},[0,-45,0])
        mapper=IncrementalMapper(resolved,ROBOT)
        for got,expected in zip(mapper.map_vector([1,0,0]),[0.,-1.,0.]):
            self.assertAlmostEqual(got,expected)
        self.assertGreater(report['robot_direction_axes_work']['forward'][0],.7)
        self.assertLess(report['robot_direction_axes_work']['forward'][2],-.7)
        self.assertEqual(fast['rotation_scale'],0.)
        self.assertEqual(fast['control_hz'],100)
        self.assertEqual(fast['max_session_translation_m'],.3)
        self.assertEqual(fast['min_wrist_angle_deg'],8.)
        self.assertEqual(fast['position_scale'],.5)
        self.assertEqual(fast['max_linear_speed_m_s'],.1)
        self.assertEqual(fast['max_joint_speed_deg_s'],60.)
        self.assertLess(fast['max_command_lead_m']+.1/100,fast['max_tracking_error_m'])

    def test_controller_pointing_direction_does_not_change_position_mapping(self):
        config=load_config(Path(__file__).resolve().parent.parent/'configs/right_speed_check.json')
        config,_=resolve_translation_mapping(config,{'rm_get_current_work_frame':
            {'name':'Base','pose':[0,0,0,0,-.785,0]}},[0,-45,0])
        outputs=[]
        for hand_q in ([0,0,0,1],[math.sin(math.pi/4),0,0,math.cos(math.pi/4)]):
            mapper=IncrementalMapper(config,ROBOT)
            mapper.update(frame(1,1_000_000_000,0,q=hand_q),1_000_000_000,lambda:ROBOT)
            mapper.update(frame(2,1_010_000_000,1,q=hand_q),1_010_000_000,lambda:ROBOT)
            result=mapper.update(frame(3,1_020_000_000,1,[.04,-.03,.01],hand_q),1_020_000_000,lambda:ROBOT)
            outputs.append(result['requested_pose_wxyz'])
        self.assertEqual(outputs[0],outputs[1])


if __name__=='__main__':unittest.main()
