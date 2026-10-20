"""Pure input/path checks; no SDK, sockets or robot commands."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from incremental import (IncrementalMapper, MappingError, angle, input_health,
                         input_ready, inverse, multiply, rotate_vector)
from rotation_axes import ContinuousRotationPath, rotation_vector, vector_rotation
from teleop_config import load_config
from test_incremental import frame

ROOT = Path(__file__).resolve().parent.parent
ROBOT = [0.,0.,0.,0.,1.,0.,0.]


def angular_error(a,b):
    return angle(multiply(a,inverse(b)))


class InputHealthTests(unittest.TestCase):
    def setUp(self):
        self.ns = 1_000_000_000
        self.status = frame(10,self.ns,1.)

    def test_short_missing_frame_is_distinct_from_long_disconnect(self):
        short = input_health(self.status,self.ns+120_000_000,100,250)
        self.assertEqual(short['reason'],'SHORT_GAP')
        self.assertTrue(short['recoverable'])
        self.assertFalse(short['ready'])
        long = input_health(self.status,self.ns+251_000_000,100,250)
        self.assertEqual(long['reason'],'INPUT_TIMEOUT')
        self.assertFalse(long['recoverable'])

    def test_explicit_disconnect_focus_and_tracking_failure_are_not_short_gaps(self):
        for field,value,reason in [('device_connected',False,'DISCONNECTED'),
                                   ('app_focus',False,'FOCUS_LOST'),
                                   ('status',0,'INVALID_TRACKING')]:
            with self.subTest(reason=reason):
                status = copy.deepcopy(self.status)
                if field == 'device_connected': status[field] = value
                elif field == 'app_focus': status['latest'][field] = value
                else: status['latest']['right'][field] = value
                result = input_health(status,self.ns+120_000_000,100,250)
                self.assertEqual(result['reason'],reason)
                self.assertFalse(result['recoverable'])

    def test_missing_controller_status_is_reported_without_inventing_a_failure(self):
        result = input_health(self.status,self.ns)
        self.assertTrue(result['ready'])
        self.assertTrue(result['tracking_status_unknown'])
        self.status['latest']['right']['status'] = 3
        self.assertFalse(input_health(self.status,self.ns)['tracking_status_unknown'])

    def test_head_is_only_required_when_requested(self):
        self.status['latest']['head'] = None
        self.assertTrue(input_ready(self.status,self.ns,require_head=False))
        self.assertEqual(input_health(self.status,self.ns)['reason'],'WAIT_HEAD')
        self.assertTrue(input_health(self.status,self.ns,require_head=False)['ready'])

    def test_invalid_pose_future_time_and_restarted_sequence_are_explicit(self):
        invalid = copy.deepcopy(self.status)
        invalid['latest']['right']['pose'] = [0.]*7
        self.assertEqual(input_health(invalid,self.ns)['reason'],'INVALID_POSE')
        self.assertEqual(input_health(self.status,self.ns-1)['reason'],'INVALID_INPUT')
        self.assertEqual(input_health(self.status,self.ns,last_sequence=11)['reason'],
                         'RECEIVER_RESTARTED')


class RotationPathTests(unittest.TestCase):
    def test_more_than_a_full_turn_keeps_a_nonzero_pending_budget(self):
        path = ContinuousRotationPath([1.,0.,0.,0.])
        path.append([0.,0.,4*math.pi])
        filtered,target = path.advance([1.,0.,0.,0.],1.,.02)
        self.assertAlmostEqual(path.remaining(),4*math.pi)
        self.assertLess(angular_error(filtered,[1.,0.,0.,0.]),1e-6)
        self.assertAlmostEqual(rotation_vector(target)[2],.02)

    def test_only_accepted_fraction_of_rotation_is_consumed(self):
        path = ContinuousRotationPath([1.,0.,0.,0.])
        path.append([0.,0.,1.])
        path.advance([1.,0.,0.,0.],1.,.1)
        path.commit(vector_rotation([0.,0.,.025]))
        self.assertAlmostEqual(path.cursor,.025)
        self.assertAlmostEqual(path.remaining(),.975)
        _,next_target = path.advance(vector_rotation([0.,0.,.025]),1.,.1)
        self.assertAlmostEqual(rotation_vector(next_target)[2],.125)

    def test_longer_sender_interval_consumes_actual_progress_past_preview(self):
        path = ContinuousRotationPath([1.,0.,0.,0.])
        path.append([0.,0.,1.])
        path.advance([1.,0.,0.,0.],1.,.1)
        path.commit(vector_rotation([0.,0.,.15]))
        self.assertAlmostEqual(path.cursor,.15)
        self.assertAlmostEqual(path.remaining(),.85)

    def test_accepted_step_past_true_endpoint_preserves_overshoot_correction(self):
        path = ContinuousRotationPath([1.,0.,0.,0.])
        path.append([0.,0.,.12])
        path.advance([1.,0.,0.,0.],1.,.1)
        path.commit(vector_rotation([0.,0.,.15]))
        self.assertAlmostEqual(path.cursor,.12)
        self.assertAlmostEqual(path.remaining(),.03)

    def test_reverse_or_orthogonal_accepted_rotation_does_not_consume_path(self):
        for accepted in ([0.,0.,-.15],[.15,0.,0.]):
            with self.subTest(accepted=accepted):
                path = ContinuousRotationPath([1.,0.,0.,0.])
                path.append([0.,0.,1.])
                path.advance([1.,0.,0.,0.],1.,.1)
                path.commit(vector_rotation(accepted))
                self.assertAlmostEqual(path.cursor,0.)

    def test_actual_progress_can_cross_a_real_axis_change(self):
        path = ContinuousRotationPath([1.,0.,0.,0.])
        path.append([0.,0.,.1]);path.append([4.,0.,0.])
        path.advance([1.,0.,0.,0.],1.,.05)
        accepted = multiply(vector_rotation([.1,0.,0.]),vector_rotation([0.,0.,.1]))
        path.commit(accepted)
        self.assertAlmostEqual(path.cursor,.2)
        self.assertAlmostEqual(path.remaining(),3.9)

    def test_reverse_cancels_unexecuted_same_axis_intent(self):
        path = ContinuousRotationPath([1.,0.,0.,0.])
        path.append([0.,0.,math.pi/2])
        _,target = path.advance([1.,0.,0.,0.],1.,math.radians(10))
        path.commit(target)
        path.append([0.,0.,-math.pi/2])
        _,target = path.advance(target,1.,math.radians(1))
        self.assertAlmostEqual(rotation_vector(target)[2],math.radians(9))
        self.assertAlmostEqual(path.remaining(),math.radians(10))
        self.assertLess(angular_error(path.goal,[1.,0.,0.,0.]),1e-6)

    def test_same_axis_segments_are_compacted_without_losing_turns(self):
        path = ContinuousRotationPath([1.,0.,0.,0.])
        for _ in range(1000): path.append([0.,0.,.01])
        self.assertEqual(len(path.segments),1)
        self.assertAlmostEqual(path.end,10.)

    def test_small_mixed_axis_lag_targets_latest_orientation(self):
        path = ContinuousRotationPath([1.,0.,0.,0.])
        path.append([.1,0.,0.]);path.append([0.,.1,0.])
        expected = multiply(vector_rotation([0.,.1,0.]),vector_rotation([.1,0.,0.]))
        self.assertEqual(len(path.segments),1)
        _,target = path.advance([1.,0.,0.,0.],1.,10.)
        self.assertLess(angular_error(target,expected),1e-6)

    def test_exact_180_pitch_requires_no_swing_twist_decomposition(self):
        path = ContinuousRotationPath([1.,0.,0.,0.])
        path.append([math.pi,0.,0.])
        _,target = path.advance([1.,0.,0.,0.],1.,math.pi)
        self.assertLess(angular_error(target,[0.,1.,0.,0.]),1e-6)
        path.commit(target)
        self.assertAlmostEqual(path.remaining(),0.)

    def test_braking_overshoot_keeps_endpoint_correction_budget(self):
        path = ContinuousRotationPath([1.,0.,0.,0.])
        path.append([0.,0.,.1])
        path.advance([1.,0.,0.,0.],1.,.1)
        path.commit(vector_rotation([0.,0.,.12]))
        self.assertAlmostEqual(path.remaining(),.02)
        _,target = path.advance(vector_rotation([0.,0.,.12]),1.,.1)
        self.assertAlmostEqual(rotation_vector(target)[2],.1)
        path.commit(target)
        self.assertAlmostEqual(path.remaining(),0.)


class IncrementalRotationMapperTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT/'configs/right_full_teleop.json')
        self.config.update(rotation_mapping='neutral_tool_axes',rotation_path_mode='incremental',
            position_scale=1.,rotation_scale=1.,
            clutch_rotation_limits_enabled=False,session_pose_limits_enabled=False,
            mapper_control_gap_guard=False,input_jump_mode='velocity',
            max_hand_linear_speed_m_s=5.,max_hand_angular_speed_rad_s=20.,
            max_linear_speed_m_s=10.,max_angular_speed_rad_s=100.,
            target_smoothing_tau_s=0.,max_control_gap_s=.03,advance_on_same_frame=True)
        self.ns = 1_000_000_000
        self.seq = 0
        self.mapper = IncrementalMapper(self.config,ROBOT)

    def update(self,grip=1.,q=None,xyz=None,elapsed=.02,head=True,pico_time=None,reuse=None):
        self.ns += int(elapsed*1e9)
        self.seq += 1
        q = q or [1.,0.,0.,0.]
        status = frame(self.seq,self.ns,grip,xyz,q[1:]+q[:1])
        if not head: status['latest']['head'] = None
        if pico_time is not None: status['latest']['pico_timestamp_ns'] = pico_time
        if reuse is not None:
            status = reuse
        result = self.mapper.update(status,self.ns,lambda:ROBOT)
        return result,status

    def engage(self):
        self.update(0.)
        result,_ = self.update(1.)
        self.assertEqual(result['state'],'ANCHORED')

    def test_pitch_can_cross_180_and_continue_without_tilt_or_wrap_pause(self):
        self.engage()
        basis = copy.deepcopy(self.mapper.tool_basis)
        for degree in range(5,366,5):
            q = vector_rotation([math.radians(degree),0.,0.])
            result,_ = self.update(q=q)
            self.assertEqual(result['state'],'TRACKING')
            expected = multiply(ROBOT[3:],self.mapper.map_tool_rotation(q))
            self.assertLess(angular_error(result['requested_pose_wxyz'][3:],expected),1e-6)
        self.assertTrue(self.mapper.active)
        self.assertEqual(self.mapper.tool_basis,basis)

    def test_three_turns_with_backlog_follow_same_direction_until_caught_up(self):
        for sign in (-1.,1.):
            with self.subTest(sign=sign):
                self.mapper = IncrementalMapper(self.config,ROBOT)
                self.engage()
                self.config['max_angular_speed_rad_s'] = 1.4
                previous = ROBOT[3:]
                traveled = 0.
                last_q = None
                for degree in range(5,1081,5):
                    last_q = vector_rotation([0.,-sign*math.radians(degree),0.])
                    if degree % 10 == 0: last_q = [-value for value in last_q]
                    result,_ = self.update(q=last_q)
                    current = result['target_pose_wxyz'][3:]
                    traveled += rotation_vector(multiply(current,inverse(previous)))[2]*(-sign)
                    previous = current
                self.assertGreater(self.mapper.rotation_remaining_rad(),2*math.pi)
                for _ in range(1000):
                    result,_ = self.update(q=last_q)
                    current = result['target_pose_wxyz'][3:]
                    increment = rotation_vector(multiply(current,inverse(previous)))[2]*(-sign)
                    self.assertGreaterEqual(increment,-1e-7)
                    traveled += increment
                    previous = current
                    if self.mapper.rotation_remaining_rad() < 1e-7: break
                self.assertAlmostEqual(traveled,6*math.pi,places=5)
                self.assertTrue(self.mapper.active)
                self.config['max_angular_speed_rad_s'] = 100.

    def test_boundary_discards_blocked_rotation_without_relearning_axes(self):
        self.engage()
        basis = copy.deepcopy(self.mapper.tool_basis)
        for degree in range(5,46,5):
            result,_ = self.update(q=vector_rotation([0.,-math.radians(degree),0.]))
        accepted = ROBOT[:3]+multiply(ROBOT[3:],vector_rotation([0.,0.,math.radians(30)]))
        self.mapper.saturate_boundary_goal(accepted,rotation=True)
        self.assertAlmostEqual(self.mapper.rotation_remaining_rad(),0.)
        result,_ = self.update(q=vector_rotation([0.,-math.radians(44),0.]))
        expected = multiply(ROBOT[3:],vector_rotation([0.,0.,math.radians(29)]))
        self.assertLess(angular_error(result['requested_pose_wxyz'][3:],expected),1e-6)
        self.assertEqual(self.mapper.tool_basis,basis)
        self.assertTrue(self.mapper.active)

    def test_session_sphere_is_disabled_without_changing_translation_axes(self):
        self.engage()
        for distance in (.1,.2,.3,.4,.5):
            result,_ = self.update(xyz=[distance,0.,0.],elapsed=.05)
        self.assertEqual(result['state'],'TRACKING')
        self.assertEqual(result['requested_delta_work_m'],[0.,-.5,0.])

    def test_head_loss_during_grip_is_accepted_but_new_grip_requires_head(self):
        self.engage()
        result,_ = self.update(head=False,xyz=[.01,0.,0.])
        self.assertEqual(result['state'],'TRACKING')
        self.update(0.,head=False)
        result,_ = self.update(head=False)
        self.assertEqual(result['state'],'WAIT_FRESH_INPUT')
        self.assertFalse(self.mapper.active)

    def test_jump_detection_uses_new_input_sample_time_not_control_tick_time(self):
        self.engage()
        result,_ = self.update(xyz=[.12,0.,0.],elapsed=.1)
        self.assertEqual(result['state'],'TRACKING')
        self.assertAlmostEqual(result['hand_linear_speed_m_s'],1.2)
        result,_ = self.update(xyz=[.24,0.,0.],elapsed=.01)
        self.assertEqual(result['state'],'INPUT_JUMP')
        self.assertAlmostEqual(result['hand_linear_speed_m_s'],12.)

    def test_device_time_wins_over_callback_delay_and_reused_frame_does_not_jump(self):
        self.update(0.,pico_time=1_000_000_000)
        self.update(1.,pico_time=1_020_000_000)
        result,status = self.update(xyz=[.08,0.,0.],elapsed=.005,pico_time=1_070_000_000)
        self.assertEqual(result['state'],'TRACKING')
        self.assertAlmostEqual(result['hand_linear_speed_m_s'],1.6)
        self.assertEqual(result['input_time_source'],'pico_timestamp_ns')
        result,_ = self.update(elapsed=.01,reuse=status)
        self.assertEqual(result['state'],'TRACKING')
        self.assertTrue(result['input_frame_reused'])

    def test_release_clears_pending_turns_and_regrip_starts_from_current_pose(self):
        self.engage()
        self.config['max_angular_speed_rad_s'] = 1.
        for degree in range(5,361,5):
            self.update(q=vector_rotation([0.,-math.radians(degree),0.]))
        self.assertGreater(self.mapper.rotation_remaining_rad(),1.)
        self.update(0.)
        self.assertIsNone(self.mapper.rotation_path)
        result,_ = self.update(1.)
        self.assertEqual(result['target_pose_wxyz'],ROBOT)
        self.assertAlmostEqual(self.mapper.rotation_remaining_rad(),0.)

    def test_scale_above_one_is_allowed_but_invalid_scale_is_rejected(self):
        self.config.update(position_scale=1.5,rotation_scale=2.)
        self.mapper = IncrementalMapper(self.config,ROBOT)
        self.engage()
        result,_ = self.update(q=vector_rotation([0.,-math.radians(5),0.]),xyz=[.01,0.,0.])
        self.assertEqual(result['requested_delta_work_m'],[0.,-.015,0.])
        expected = multiply(ROBOT[3:],vector_rotation([0.,0.,math.radians(10)]))
        self.assertLess(angular_error(result['requested_pose_wxyz'][3:],expected),1e-6)
        for value in (-1.,math.inf,math.nan,True):
            self.config['position_scale'] = value
            with self.assertRaises(MappingError): IncrementalMapper(self.config,ROBOT)

    def test_unlimited_old_anchor_algorithm_is_rejected(self):
        self.config['rotation_path_mode'] = 'anchor'
        with self.assertRaises(MappingError): IncrementalMapper(self.config,ROBOT)


if __name__ == '__main__':
    unittest.main()
