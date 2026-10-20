"""Final sender/mapper consistency with coupled fake geometry and fake clocks."""
import copy
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))

from incremental import MappingError, angle, inverse, multiply, quat
from teleop_config import load_config
from teleop_controller import TeleopController
from test_incremental import frame
from test_joint_cartesian_trajectory import CoupledBackend, axis_quaternion
from test_task_teleop_controller import FreshFeedback


class CoupledControlBackend(CoupledBackend):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.initial_joints = [0.]*6
        self.initial_pose = self.forward(self.initial_joints)
        self.minimum, self.maximum = [-180.]*5+[-720.], [180.]*5+[720.]
        self.speed_limits = [30.,120.,120.,120.,120.,120.]
        self.acceleration_limits = [600.]*6
        self.sent, self.sent_ns = [], []
        self.stops = 0

    def solve(self, target, seed):
        w, x, y, z = quat(target[3:])
        roll = math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
        pitch = math.degrees(math.asin(max(-1., min(1., 2*(w*y-z*x)))))
        yaw = math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
        yaw = min((yaw+360*k for k in range(-3,4)), key=lambda value:abs(value-seed[5]))
        joints = [target[0]*1000., (target[1]-target[0])*1000., target[2]*1000., roll, pitch, yaw]
        if any(not low <= value <= high for value, low, high in zip(joints, self.minimum, self.maximum)):
            raise MappingError('Fake geometric limit', {'state':'IK_REFUSED','retryable_ik':True})
        return joints

    def send(self, joints):
        self.sent.append(joints.copy())
        self.sent_ns.append(self.clock())

    def stop(self):
        self.stops += 1


class FinalEmissionConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.ns = self.tick_ns = 1_000_000_000
        self.sequence = 0
        self.gate_delay_ns = 0
        self.config = load_config(ROOT/'configs/right_rotation_teleop.json')
        self.config.update(translation_reference='tracking_world', translation_mapping_frame='work',
            rotation_mapping='world', headset_to_work_matrix=[[1,0,0],[0,1,0],[0,0,1]])
        self.backend = CoupledControlBackend(lambda:self.ns)
        self.feedback = FreshFeedback(lambda:self.ns)
        self.control = TeleopController(self.config, self.backend, self.feedback, True, lambda:self.ns)
        self.step(grip=0.)
        self.assertEqual(self.step()['mapping']['state'], 'ANCHORED')
        self.anchor_ns = self.ns

    def step(self, xyz=None, degrees=0., grip=1., late_stop=None):
        self.tick_ns += 10_000_000
        self.ns = self.tick_ns
        self.sequence += 1
        self.feedback.observe(self.backend.sent[-1] if self.backend.sent else None)
        rotation = axis_quaternion(2, degrees)
        status = frame(self.sequence, self.ns, grip, xyz, rotation[1:]+rotation[:1])
        def gate():
            self.ns += self.gate_delay_ns
            latest = copy.deepcopy(status)
            if late_stop == 'release':
                latest['latest']['right']['grip'] = 0.
            elif late_stop == 'B':
                latest['session_exit_requested'] = True
            return latest
        return self.control.tick(status, gate, self.ns)

    def watch_final_sync(self):
        calls = []
        original = self.control.mapper.sync_target_rotation
        def watched(target):
            calls.append(copy.deepcopy(target))
            original(target)
        self.control.mapper.sync_target_rotation = watched
        return calls

    def check_sent_dynamics(self):
        previous, previous_velocity = self.backend.initial_joints.copy(), [0.]*6
        previous_ns = self.backend.sent_ns[0]-10_000_000
        for joints, stamp in zip(self.backend.sent, self.backend.sent_ns):
            dt = (stamp-previous_ns)/1e9
            velocity = [(new-old)/dt for new, old in zip(joints, previous)]
            acceleration = [(new-old)/dt for new, old in zip(velocity, previous_velocity)]
            for index in range(6):
                self.assertLessEqual(abs(velocity[index]), min(120., self.backend.speed_limits[index])+1e-6)
                self.assertLessEqual(abs(acceleration[index]), min(600., self.backend.acceleration_limits[index])+1e-4)
            previous, previous_velocity, previous_ns = joints, velocity, stamp

    def test_coupled_translation_does_not_create_unrequested_sideways_motion(self):
        for index in range(1,151):
            self.gate_delay_ns = (0,2_000_000,5_000_000,1_000_000)[index%4]
            report = self.step([min(.0006*index,.045),0.,0.])
            self.assertTrue(report['motion_command_sent_this_tick'])
            actual = self.backend.forward(self.backend.sent[-1])
            self.assertLess(abs(actual[1]), 1e-6)
            self.assertLess(abs(actual[2]), 1e-9)
            self.assertLess(angle(actual[3:]), 1e-7)
        self.assertGreater(self.backend.forward(self.backend.sent[-1])[0], .025)
        self.check_sent_dynamics()
        self.assertEqual(self.backend.stops, 0)

    def test_two_axis_translation_preserves_zero_third_axis_and_orientation(self):
        for index in range(1,181):
            displacement = .018*math.sin(index*math.pi/180.)
            self.step([displacement, 0., displacement*.8])
            actual = self.backend.forward(self.backend.sent[-1])
            self.assertLess(abs(actual[1]), 1e-6)
            self.assertLess(angle(actual[3:]), 1e-7)
        self.check_sent_dynamics()
        self.assertEqual(self.backend.stops, 0)

    def test_changed_sender_interval_syncs_only_final_fk_pose_once(self):
        calls = self.watch_final_sync()
        for index in range(1,141):
            self.gate_delay_ns = (0,2_000_000,5_000_000,1_000_000)[index%4]
            before = len(calls)
            report = self.step(degrees=min(index*.7,65.))
            self.assertTrue(report['motion_command_sent_this_tick'])
            self.assertEqual(len(calls)-before, 1)
            sent_pose = self.backend.forward(self.backend.sent[-1])
            self.assertLess(angle(multiply(calls[-1][3:],inverse(sent_pose[3:]))), 1e-7)
            self.assertLess(angle(multiply(self.control.mapper.target[3:],inverse(sent_pose[3:]))), 1e-7)
            self.assertLess(angle(multiply(self.control.mapper.rotation_path.actual_orientation,
                                           inverse(sent_pose[3:]))), 1e-7)
        self.check_sent_dynamics()

    def test_consumed_rotation_budget_matches_final_emitted_fk_not_preview(self):
        for index in range(1,101):
            self.gate_delay_ns = (0,2_000_000,5_000_000,1_000_000)[index%4]
            self.step(degrees=index*.9)
            path = self.control.mapper.rotation_path
            actual_roll_rad = math.radians(self.backend.sent[-1][5])
            self.assertAlmostEqual(path.cursor, actual_roll_rad, places=7)

    def test_late_release_and_b_do_not_consume_unsent_rotation_proposal(self):
        for late_stop in ('release','B'):
            with self.subTest(stop=late_stop):
                self.setUp()
                for index in range(1,21):
                    self.step(degrees=index*.5)
                path = self.control.mapper.rotation_path
                before_cursor = path.cursor
                before_sent = len(self.backend.sent)
                calls = self.watch_final_sync()
                report = self.step(degrees=11.,late_stop=late_stop)
                self.assertFalse(report['motion_command_sent_this_tick'])
                self.assertEqual(len(self.backend.sent),before_sent)
                self.assertEqual(calls,[])
                self.assertEqual(path.cursor,before_cursor)
                self.assertFalse(self.control.mapper.active)

    def test_stationary_input_never_moves_and_settled_goal_has_no_drift(self):
        for _ in range(50):
            self.step()
        self.assertEqual(self.backend.sent[-1], [0.]*6)
        for index in range(1,101):
            self.step([min(index*.0002,.01),0.,0.],degrees=min(index*.2,10.))
        for _ in range(400):
            self.step([.01,0.,0.],degrees=10.)
        settled = self.backend.sent[-1].copy()
        for _ in range(50):
            self.step([.01,0.,0.],degrees=10.)
        for before, after in zip(settled, self.backend.sent[-1]):
            self.assertAlmostEqual(before, after, places=7)
        self.assertAlmostEqual(self.backend.forward(settled)[0], .01, places=7)
        self.assertAlmostEqual(settled[5], 10., places=5)
        self.check_sent_dynamics()


if __name__ == '__main__':
    unittest.main()
