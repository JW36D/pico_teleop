import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from legacy_recording import DataRecorder
from control_sample import control_sample
from incremental import MappingError


class RecordingQualityTests(unittest.TestCase):
    def sample(self, controller=None, state='TRACKING', sent=False, feedback_sequence=1, pico_sequence=1):
        controller = controller or types.SimpleNamespace(
            mapper=types.SimpleNamespace(active=True), send_count=3,
            last_target=[.1,0,0,1,0,0,0], last_joints=[5.]*6)
        report = {'time_monotonic_ns':100, 'control_ticks':1, 'mode':'EXECUTE',
            'motion_command_sent_this_tick':sent, 'robot_feedback_age_ms':10.,
            'robot_feedback_monotonic_ns':90, 'robot_feedback_pose_wxyz':[0,0,0,1,0,0,0],
            'robot_feedback_joints_deg':[1.]*6, 'robot_feedback_sequence':feedback_sequence,
            'robot_feedback_metadata':{'sequence':feedback_sequence, 'joint_velocity_deg_s':[2.]*6,
                'velocity_interval_s':.05, 'source':'tcp_query_response',
                'sample_time_semantics':'local_receive_time_robot_clock_unavailable'},
            'mapping':{'state':state, 'requested_pose_wxyz':[.2,0,0,1,0,0,0]}}
        status = {'age_ms':5., 'latest':{'sequence':pico_sequence, 'receive_monotonic_ns':95,
            'right':{'pose':[0,0,0,0,0,0,1], 'grip':1., 'trigger':.5}}}
        return control_sample(status, report, controller), report, status, controller

    def make_recorder(self, directory):
        recorder = DataRecorder(Path(directory)/'session', {'control_hz':100, 'feedback_hz':100})
        self.addCleanup(recorder.close)
        return recorder

    def test_record_keeps_intent_candidate_sent_and_feedback_distinct(self):
        sample, report, status, controller = self.sample(sent=True)
        report['last_actually_sent_command'] = {'sequence':3, 'send_monotonic_ns':99,
            'joints_deg':[4.]*6, 'pose_wxyz':[.08,0,0,1,0,0,0]}
        sample = control_sample(status, report, controller)
        self.assertEqual(sample['requested_pose_wxyz'][0], .2)
        self.assertEqual(sample['command_pose_wxyz'][0], .1)
        self.assertEqual(sample['last_actually_sent_command']['pose_wxyz'][0], .08)
        self.assertEqual(sample['robot_pose_wxyz'][0], 0.)
        self.assertEqual(sample['robot_joints_deg'], [1.]*6)
        self.assertEqual(sample['robot_joint_velocity_deg_s'], [2.]*6)
        self.assertIsNone(sample['robot_sample_timestamp_ns'])

    def test_candidate_without_successful_send_is_not_claimed_as_executed_action(self):
        sample, report, status, controller = self.sample(sent=False)
        controller.last_sent_target = [99.]*7
        controller.last_sent_joints = [99.]*6
        sample = control_sample(status, report, controller)
        self.assertIsNone(sample['last_actually_sent_command'])
        self.assertFalse(sample['quality']['command_sent_this_tick'])

    def test_gripper_intent_accepted_action_and_invalid_feedback_stay_distinct(self):
        _,report,status,controller=self.sample()
        report['gripper']={'raw_trigger':1.,
            'last_accepted_command':{'action':'close','ack_monotonic_ns':99},
            'feedback':{'valid':False,'opening_controller_units':None,'receive_monotonic_ns':98}}
        sample=control_sample(status,report,controller)
        self.assertEqual(sample['gripper']['last_accepted_command']['action'],'close')
        self.assertIsNone(sample['gripper']['feedback']['opening_controller_units'])
        report['gripper']['feedback']['valid']=True
        self.assertFalse(sample['gripper']['feedback']['valid'])

    def test_repeated_feedback_and_input_are_marked_without_smoothing_real_joints(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.make_recorder(directory)
            first, *_ = self.sample()
            second, *_ = self.sample()
            third, *_ = self.sample(feedback_sequence=2, pico_sequence=2)
            first['robot_joints_deg'] = [1.,2.,3.,4.,5.,6.]
            second['robot_joints_deg'] = first['robot_joints_deg'].copy()
            for sample in (first, second, third):
                recorder.append(sample)
            recorder.close()
            rows = [json.loads(line) for line in (recorder.directory/'samples.jsonl').read_text().splitlines()]
            self.assertFalse(rows[0]['quality']['feedback_reused'])
            self.assertTrue(rows[1]['quality']['feedback_reused'])
            self.assertTrue(rows[1]['quality']['pico_frame_reused'])
            self.assertTrue(rows[2]['quality']['new_feedback_response'])
            self.assertEqual(rows[0]['robot_joints_deg'], [1.,2.,3.,4.,5.,6.])
            self.assertEqual([row['record_sequence'] for row in rows], [1,2,3])

    def test_last_successful_command_persists_across_candidate_holds(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.make_recorder(directory)
            first, *_ = self.sample(sent=True)
            first['last_actually_sent_command'] = {'sequence':3, 'joints_deg':[4.]*6, 'send_monotonic_ns':99}
            second, *_ = self.sample(sent=False)
            second['command_joints_deg'] = [20.]*6
            recorder.append(first)
            recorder.append(second)
            recorder.close()
            rows = [json.loads(line) for line in (recorder.directory/'samples.jsonl').read_text().splitlines()]
            self.assertEqual(rows[1]['last_actually_sent_command']['joints_deg'], [4.]*6)
            self.assertEqual(rows[1]['command_joints_deg'], [20.]*6)

    def test_episode_and_segment_boundaries_mark_grip_and_boundary_interventions(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.make_recorder(directory)
            for state, active in [('RELEASED',False), ('ANCHORED',True), ('TRACKING',True),
                                  ('BOUNDARY_LIMITED',True), ('TRACKING',True), ('RELEASED',False),
                                  ('ANCHORED',True), ('INPUT_RECOVERED',True)]:
                sample, report, status, controller = self.sample(state=state)
                sample['clutch_active'] = active
                recorder.append(sample)
            recorder.close()
            rows = [json.loads(line) for line in (recorder.directory/'samples.jsonl').read_text().splitlines()]
            self.assertIsNone(rows[0]['episode_id'])
            self.assertEqual(rows[1]['episode_id'], 1)
            self.assertEqual(rows[4]['episode_id'], 1)
            self.assertEqual(rows[6]['episode_id'], 2)
            self.assertEqual(rows[3]['segment_category'], 'boundary_intervention')
            self.assertTrue(rows[3]['quality']['ik_intervention'])
            self.assertGreater(rows[4]['segment_id'], rows[3]['segment_id'])
            self.assertEqual(rows[7]['segment_category'], 'input_interruption')

    def test_meta_declares_camera_and_robot_sample_clock_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.make_recorder(directory)
            recorder.close()
            meta = json.loads((recorder.directory/'meta.json').read_text())
            self.assertEqual(meta['schema'], 'pico_realman_actions_v2')
            self.assertFalse(meta['feedback_clock']['robot_sample_clock_available'])
            self.assertFalse(meta['camera']['available'])
            self.assertTrue(meta['feedback_clock']['configured_feedback_hz_is_target_not_measured_rate'])

    def test_joint_intervention_bool_arrays_do_not_mark_all_false_as_intervention(self):
        sample, report, status, controller = self.sample()
        report['mapping'].update(joint_acceleration_limited=[False]*6,
                                joint_speed_limited=[False]*6,
                                joint_position_braking=[False]*6)
        self.assertFalse(control_sample(status, report, controller)['quality']['joint_shaping_intervention'])
        report['mapping']['joint_acceleration_limited'][3] = True
        self.assertTrue(control_sample(status, report, controller)['quality']['joint_shaping_intervention'])

    def test_cartesian_optimizer_residual_is_recorded_without_inventing_joint_bound_intervention(self):
        sample, report, status, controller = self.sample()
        report['mapping'].update(joint_acceleration_limited=[False]*6,
            joint_speed_limited=[False]*6, joint_position_braking=[False]*6,
            joint_trajectory_cartesian_optimizer='box_constrained_active_set',
            joint_optimizer_converged=True, joint_optimizer_kkt_residual=1e-12,
            joint_velocity_lower_bounds_deg_s=[-6.]*6,
            joint_velocity_upper_bounds_deg_s=[6.]*6,
            cartesian_velocity_residual_m_s_rad_s=[.01,0.,0.,0.,0.,0.],
            cartesian_linear_velocity_residual_m_s=.01,
            cartesian_angular_velocity_residual_rad_s=0.)
        sample = control_sample(status,report,controller)
        self.assertTrue(sample['quality']['cartesian_optimizer_used'])
        self.assertTrue(sample['quality']['cartesian_optimizer_converged'])
        self.assertTrue(sample['quality']['cartesian_velocity_reduced'])
        self.assertFalse(sample['quality']['joint_shaping_intervention'])
        self.assertEqual(sample['motion_constraint_diagnostics']['cartesian_velocity_residual_m_s_rad_s'],
                         [.01,0.,0.,0.,0.,0.])
        self.assertEqual(sample['motion_constraint_diagnostics']['joint_velocity_upper_bounds_deg_s'],[6.]*6)
        report['mapping']['cartesian_linear_velocity_residual_m_s']=1e-10
        self.assertFalse(control_sample(status,report,controller)['quality']['cartesian_velocity_reduced'])

    def test_writer_failure_invalidates_collection_and_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.make_recorder(directory)
            recorder.append({'invalid':float('nan')})
            deadline = time.monotonic()+.5
            while recorder.error is None and time.monotonic() < deadline:
                time.sleep(.001)
            with self.assertRaisesRegex(MappingError, 'unavailable'):
                recorder.append({'tick':2})
            recorder.close()
            summary = json.loads((recorder.directory/'summary.json').read_text())
            self.assertFalse(summary['complete'])
            self.assertFalse(summary['collection_available'])
            self.assertIsNotNone(summary['error'])

    def test_writer_owns_copy_of_enqueued_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.make_recorder(directory)
            sample = {'joints':[1.]*6}
            recorder.append(sample)
            sample['joints'][0] = 100.
            recorder.close()
            row = json.loads((recorder.directory/'samples.jsonl').read_text())
            self.assertEqual(row['joints'][0], 1.)


if __name__ == '__main__':
    unittest.main()
