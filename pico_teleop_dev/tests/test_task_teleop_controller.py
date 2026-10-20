"""Task teleoperation acceptance behavior with simulated feedback and no hardware."""
import copy
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
from incremental import MappingError
from teleop_config import load_config
from teleop_controller import TeleopController
from test_incremental import frame


class AxisBackend:
    def __init__(self, clock, axis=0, degrees_per_metre=1000.):
        self.clock, self.axis, self.degrees_per_metre = clock, axis, degrees_per_metre
        self.initial_joints = [0.]*6
        self.initial_pose = [0.,0.,0.,1.,0.,0.,0.]
        self.minimum, self.maximum = [-1000.]*6, [1000.]*6
        self.speed_limits = [25.,45.,80.,90.,150.,200.]
        self.acceleration_limits = [300.,400.,500.,600.,700.,800.]
        self.sent, self.sent_ns = [], []
        self.stops = 0
        self.fail_send = False
        self.ik_boundary_m = None

    def forward(self, joints):
        return [joints[self.axis]/self.degrees_per_metre,0.,0.,1.,0.,0.,0.]

    def solve(self, target, seed):
        if (self.ik_boundary_m is not None and target[0] > self.ik_boundary_m+1e-10 and
                target[0] > self.forward(seed)[0]+1e-10):
            raise MappingError('Known IK boundary', {'state':'IK_REFUSED','retryable_ik':True})
        result = [0.]*6
        result[self.axis] = target[0]*self.degrees_per_metre
        return result

    def send(self, joints):
        if self.fail_send:
            raise MappingError('Sender rejected command')
        self.sent.append(joints.copy())
        self.sent_ns.append(self.clock())

    def stop(self):
        self.stops += 1


class FreshFeedback:
    def __init__(self, clock):
        self.clock = clock
        self.joints = [0.]*6
        self.velocity = [0.]*6
        self.received_ns = self.clock()
        self.sequence = 0
        self.stationary = True

    def observe(self, joints=None):
        now = self.clock()
        dt = (now-self.received_ns)/1e9
        next_joints = self.joints.copy() if joints is None else joints.copy()
        if dt > 0:
            self.velocity = [(a-b)/dt for a,b in zip(next_joints,self.joints)]
        self.joints, self.received_ns = next_joints, now
        self.sequence += 1
        self.stationary = max(abs(v) for v in self.velocity) < .001

    def snapshot(self):
        return {'joints':self.joints.copy(), 'age_ms':(self.clock()-self.received_ns)/1e6,
            'stationary':self.stationary, 'sequence':self.sequence,
            'request_monotonic_ns':self.received_ns, 'receive_monotonic_ns':self.received_ns,
            'joint_velocity_deg_s':self.velocity.copy(), 'velocity_interval_s':.01,
            'source':'simulated_feedback', 'sample_time_semantics':'local_receive_time_robot_clock_unavailable'}


class TaskTeleopTests(unittest.TestCase):
    def setUp(self):
        self.ns, self.sequence = 1_000_000_000, 0
        self.config = load_config(ROOT/'configs/right_rotation_teleop.json')
        self.config.update(translation_reference='tracking_world',translation_mapping_frame='work',
            rotation_mapping='world',rotation_scale=0.)
        self.backend = AxisBackend(lambda:self.ns)
        self.feedback = FreshFeedback(lambda:self.ns)
        self.rebuild()
        self.last_status = None

    def rebuild(self, execute=True):
        self.control = TeleopController(self.config, self.backend, self.feedback, execute, lambda:self.ns)

    def step(self, x=0., grip=1., dt=.01, status=None, gate=None, follow=True):
        self.ns += round(dt*1e9)
        self.feedback.observe(self.backend.sent[-1] if follow and self.backend.sent else None)
        if status is None:
            self.sequence += 1
            status = frame(self.sequence,self.ns,grip,[0.,0.,-x])
        self.last_status = copy.deepcopy(status)
        return self.control.tick(status, gate or (lambda:status), self.ns)

    def engage(self, x=0.):
        self.step(x,0.)
        report = self.step(x,1.)
        self.assertEqual(report['mapping']['state'],'ANCHORED')
        return report

    def warmup(self, cycles=30):
        self.engage()
        for tick in range(1,cycles+1):
            self.step(min(.0025*tick,.1))
        self.assertTrue(self.control.mapper.active)
        return min(.0025*cycles,.1)

    def assert_continuous(self):
        self.assertTrue(self.control.mapper.active)
        self.assertFalse(self.control.mapper.paused)
        self.assertEqual(self.backend.stops,0)

    def command_velocities(self):
        result = []
        previous = self.backend.initial_joints
        previous_ns = self.backend.sent_ns[0]-10_000_000 if self.backend.sent_ns else self.ns
        for joints, ns in zip(self.backend.sent,self.backend.sent_ns):
            dt = (ns-previous_ns)/1e9
            self.assertGreater(dt,0.)
            result.append(([(a-b)/dt for a,b in zip(joints,previous)],dt))
            previous, previous_ns = joints, ns
        return result

    def assert_command_limits(self, initial_velocity=None):
        prior = initial_velocity or [0.]*6
        for velocity, dt in self.command_velocities():
            for axis,value in enumerate(velocity):
                self.assertLessEqual(abs(value),min(self.config['max_joint_speed_deg_s'],self.backend.speed_limits[axis])+1e-6)
                self.assertLessEqual(abs(value-prior[axis])/dt,
                    min(self.config['max_joint_accel_deg_s2'],self.backend.acceleration_limits[axis])+1e-5)
            prior = velocity

    def test_each_joint_final_velocity_and_acceleration_obey_task_and_device_limits_on_reversal(self):
        for axis in range(6):
            with self.subTest(axis=axis):
                self.setUp()
                self.backend.axis = axis
                self.rebuild()
                self.engage()
                for tick in range(1,121):
                    self.step(.0025*tick if tick<=60 else .15-.0025*(tick-60))
                self.assert_command_limits()
                self.assert_continuous()
                velocities = [row[0][axis] for row in self.command_velocities()]
                self.assertGreater(max(velocities),0.)
                self.assertLess(min(velocities),0.)

    def test_true_hardware_and_known_ik_boundaries_brake_without_regrip_or_abrupt_velocity_change(self):
        for kind in ('hardware','ik'):
            with self.subTest(kind=kind):
                self.setUp()
                self.backend.axis = 5
                if kind == 'hardware':
                    self.backend.maximum[5] = 35.
                else:
                    self.backend.ik_boundary_m = .035
                self.rebuild()
                self.engage()
                saw_boundary = False
                for tick in range(1,121):
                    report = self.step(min(.0025*tick,.08))
                    saw_boundary |= report['mapping']['state'].startswith('BOUNDARY') or bool(report['mapping'].get('boundary_reason'))
                self.assert_command_limits()
                self.assert_continuous()
                self.assertLess(abs(self.command_velocities()[-1][0][5]),1.)
                if kind == 'hardware':
                    self.assertLessEqual(max(q[5] for q in self.backend.sent),35.+1e-8)
                else:
                    self.assertTrue(saw_boundary)
                before = self.backend.sent[-1][5]
                # A fixed mapping retains the unreachable target instead of
                # rebasing it. Return the hand inside the reachable region.
                for tick in range(1,61):
                    self.step(.08-.001*tick)
                self.assertLess(self.backend.sent[-1][5],before)
                self.assert_continuous()
                self.assert_command_limits()

    def test_head_tracking_loss_during_active_grip_is_allowed_but_new_anchor_requires_head(self):
        self.config['translation_reference'] = 'operator_heading'
        self.rebuild()
        self.engage()
        for tick in range(1,6):
            self.sequence += 1
            status = frame(self.sequence,self.ns+10_000_000,1.,[0.,0.,-.002*tick])
            status['latest']['head']['status'] = 0
            self.step(status=status)
        self.assert_continuous()
        self.assertGreater(len(self.backend.sent),0)
        self.step(.01,0.)
        self.sequence += 1
        status = frame(self.sequence,self.ns+10_000_000,1.)
        status['latest']['head']['status'] = 0
        report = self.step(status=status)
        self.assertFalse(self.control.mapper.active)
        self.assertFalse(report['motion_command_sent_this_tick'])

    def test_short_missing_input_preserves_grip_brakes_then_recovers_without_reanchor(self):
        x = self.warmup()
        stale = copy.deepcopy(self.last_status)
        sent_before = len(self.backend.sent)
        stale_velocities = []
        for tick in range(1,15):
            report = self.step(status=stale)
            self.assert_continuous()
            if tick > 10:
                stale_velocities.append(self.command_velocities()[-1][0][0])
                self.assertTrue(report['mapping'].get('input_gap_hold') or report['mapping']['state']=='INPUT_GAP_HOLD')
        self.assertGreater(len(self.backend.sent),sent_before)
        self.assertTrue(all(abs(b)<=abs(a)+1e-6 for a,b in zip(stale_velocities,stale_velocities[1:])))
        report = self.step(x+.001)
        self.assert_continuous()
        self.assertNotEqual(report['mapping']['state'],'ANCHORED')
        self.assert_command_limits()

    def test_long_gap_and_explicit_invalid_input_stop_even_with_grip_held(self):
        for reason in ('timeout','disconnect','focus','pose','restart'):
            with self.subTest(reason=reason):
                self.setUp()
                x = self.warmup(10)
                status = copy.deepcopy(self.last_status)
                if reason == 'timeout':
                    self.ns += 151_000_000
                elif reason == 'disconnect':
                    status['device_connected'] = False
                elif reason == 'focus':
                    status['latest']['app_focus'] = False
                elif reason == 'pose':
                    status['latest']['right']['pose'][0] = float('nan')
                else:
                    status['latest']['sequence'] = 0
                try:
                    self.step(status=status)
                except MappingError:
                    pass
                self.assertFalse(self.control.mapper.active)
                self.assertGreaterEqual(self.backend.stops,1)
                report = self.step(x)
                self.assertFalse(report['motion_command_sent_this_tick'])
                self.assertFalse(self.control.mapper.active)

    def test_fifty_ms_schedule_gap_continues_without_stop_dwell_or_regrip(self):
        x = self.warmup()
        before = len(self.backend.sent)
        report = self.step(x+.002,dt=.05)
        self.assert_continuous()
        self.assertTrue(report['motion_command_sent_this_tick'])
        self.assertEqual(len(self.backend.sent),before+1)
        for _ in range(5):
            self.step(x+.003)
        self.assert_continuous()
        self.assert_command_limits()

    def test_removed_startup_sphere_and_joint_excursion_do_not_block_task_motion(self):
        self.backend = AxisBackend(lambda:self.ns,axis=5,degrees_per_metre=300.)
        self.rebuild()
        self.engage()
        for tick in range(1,301):
            self.step(min(.002*tick,.5))
        self.assert_continuous()
        self.assertGreater(self.backend.sent[-1][5],120.)
        self.assertGreater(self.backend.forward(self.backend.sent[-1])[0],.4)
        self.assert_command_limits()

    def test_fresh_moving_feedback_can_anchor_without_waiting_to_settle(self):
        self.step(0.,0.)
        self.feedback.joints[0] = .5
        self.feedback.velocity[0] = 15.
        self.feedback.stationary = False
        self.sequence += 1
        self.ns += 10_000_000
        self.feedback.received_ns = self.ns
        status = frame(self.sequence,self.ns,1.)
        report = self.control.tick(status,lambda:status,self.ns)
        self.assertEqual(report['mapping']['state'],'ANCHORED')
        self.assertTrue(self.control.mapper.active)
        self.assertAlmostEqual(report['mapping']['target_pose_wxyz'][0],.0005)
        self.assertFalse(report['motion_command_sent_this_tick'])

    def test_late_gate_release_and_b_prevent_one_more_command(self):
        for reason in ('release','b'):
            with self.subTest(reason=reason):
                self.setUp()
                x = self.warmup(10)
                before = len(self.backend.sent)
                def gate():
                    data = copy.deepcopy(self.last_status)
                    if reason == 'release':
                        data['latest']['right']['grip'] = 0.
                    else:
                        data['session_exit_requested'] = True
                    return data
                report = self.step(x+.001,gate=gate)
                self.assertFalse(report['motion_command_sent_this_tick'])
                self.assertEqual(len(self.backend.sent),before)
                self.assertFalse(self.control.mapper.active)

    def test_late_gate_short_input_gap_retains_control_and_uses_braking_target(self):
        x = self.warmup()
        before = len(self.backend.sent)
        def gate():
            self.ns += 110_000_000
            self.feedback.observe(self.backend.sent[-1])
            return copy.deepcopy(self.last_status)
        report = self.step(x+.002,gate=gate)
        self.assert_continuous()
        self.assertTrue(report['motion_command_sent_this_tick'])
        self.assertEqual(len(self.backend.sent),before+1)
        self.assertTrue(report['mapping'].get('input_gap_hold'))
        for _ in range(3):
            self.step(x+.003)
        self.assert_command_limits()

    def test_actual_command_is_committed_only_after_successful_sender_call(self):
        self.warmup(10)
        prior = copy.deepcopy(self.control.last_actually_sent_command)
        self.assertEqual(prior['joints_deg'],self.backend.sent[-1])
        self.backend.fail_send = True
        with self.assertRaises(MappingError):
            self.step(.026)
        self.assertEqual(self.control.last_actually_sent_command,prior)
        self.setUp()
        self.rebuild(execute=False)
        self.engage()
        report = self.step(.002,follow=False)
        self.assertFalse(report['motion_command_sent_this_tick'])
        self.assertIsNone(report['last_actually_sent_command'])
        self.assertEqual(self.backend.sent,[])

    def test_time_aligned_tracking_ignores_newer_commands_and_one_short_anomaly(self):
        self.warmup(12)
        old_command = copy.deepcopy(self.control.command_history[2])
        snapshot = self.feedback.snapshot()
        snapshot.update(joints=old_command['joints_deg'],
            request_monotonic_ns=old_command['send_monotonic_ns'],
            receive_monotonic_ns=self.ns,age_ms=1.)
        self.control.validate_feedback(snapshot)
        self.assertFalse(self.control.tracking_diagnostic['tracking_error_pending'])
        bad = {**self.feedback.snapshot(),'joints':[-80.,0.,0.,0.,0.,0.]}
        self.control.validate_feedback(bad)
        self.assertTrue(self.control.tracking_diagnostic['tracking_error_pending'])
        self.ns += 100_000_000
        self.control.validate_feedback(bad)
        self.assertTrue(self.control.mapper.active)
        self.control.validate_feedback(self.feedback.snapshot())
        self.assertFalse(self.control.tracking_diagnostic['tracking_error_pending'])
        self.control.validate_feedback(bad)
        self.ns += 160_000_000
        self.feedback.snapshot = lambda:bad
        self.sequence += 1
        status = frame(self.sequence,self.ns,1.)
        with self.assertRaisesRegex(MappingError,'tracking error'):
            self.control.tick(status,lambda:status,self.ns)
        self.assertFalse(self.control.mapper.active)
        self.assertGreaterEqual(self.backend.stops,1)


if __name__ == '__main__':
    unittest.main()
