"""Signed neutral twists through canonical FK and a real joint angle bound."""
import copy
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from incremental import inverse, multiply, quat
from robot_directions import resolve_translation_mapping
from teleop_config import load_config
from teleop_controller import TeleopController
from test_full_teleop import FakeBackend, FakeFeedback
from test_incremental import frame
from test_neutral_rotation import axis_quaternion


class CanonicalToolBackend(FakeBackend):
    def __init__(self, canonical_component=0, maximum_roll=360., quantization=None,
                 initial_roll=0., clock_ns=None):
        super().__init__()
        self.origin = multiply(axis_quaternion([0, 0, 1], 31),
                               multiply(axis_quaternion([0, 1, 0], 7),
                                        axis_quaternion([1, 0, 0], 180)))
        self.position = [.1, .2, .3]
        self.canonical_component = canonical_component
        self.quantization = quantization
        self.clock_ns = clock_ns
        self.sent_monotonic_ns = []
        self.canonical_signs = []
        self.initial_joints = [0., 0., 0., 0., 60., initial_roll]
        self.minimum = [-178.] * 5 + [-360.]
        self.maximum = [178.] * 5 + [maximum_roll]
        self.initial_pose = self.forward(self.initial_joints)

    def forward(self, joints):
        rotation = quat(multiply(self.origin, axis_quaternion([0, 0, 1], joints[5])))
        sign = -1 if rotation[self.canonical_component] < 0. else 1
        self.canonical_signs.append(sign)
        return self.position + [value*sign for value in rotation]

    def solve(self, target, seed):
        relative = quat(multiply(inverse(self.origin), target[3:]))
        principal = (math.degrees(2*math.atan2(relative[3], relative[0]))+180.) % 360.-180.
        roll = min((principal+360.*k for k in range(-2, 3)),
                   key=lambda candidate: abs(candidate-seed[5]))
        if self.quantization is not None:
            roll = round(roll, self.quantization)
        return [0., 0., 0., 0., 60., roll]

    def send(self,joints):
        super().send(joints)
        if self.clock_ns is not None:
            self.sent_monotonic_ns.append(self.clock_ns())


class ContinuousRotationControllerTests(unittest.TestCase):
    def setup_control(self, canonical_component=0, maximum_roll=360., quantization=None,
                      initial_roll=0., legacy=False):
        self.ns = 1_000_000_000
        self.tick_start_ns = self.ns
        self.gate_delay_ns = 0
        self.sequence = 0
        self.config = load_config(ROOT / 'configs/right_rotation_teleop.json')
        if legacy:
            # Keep regression coverage for the supported prior neutral-twist
            # algorithm rather than accidentally running it on the new profile.
            self.config.update(rotation_path_mode='anchor',clutch_rotation_limits_enabled=True,
                session_pose_limits_enabled=True,mapper_control_gap_guard=True,
                input_jump_mode='step',continuous_input_recovery=False,
                continuous_timing_recovery=False,joint_trajectory_enabled=False,
                position_priority=True,max_joint_speed_deg_s=90.,
                max_clutch_rotation_rad=math.radians(190),max_tool_tilt_rad=math.pi/2,
                max_command_lead_m=.02,max_command_lead_angle_rad=.12,
                tracking_governor_curve='quartic',require_robot_stationary_to_anchor=True)
        self.config, _ = resolve_translation_mapping(self.config,
            {'rm_get_current_work_frame': {'name': 'Base', 'pose': [0, 0, 0, 0, -.785, 0]}},
            [0, -45, 0])
        self.assertFalse(self.config['joint_session_limits_enabled'])
        self.assertEqual(self.config['tracking_governor_mode'], 'continuous')
        self.assertEqual(self.config['boundary_behavior'], 'directional')
        self.backend = CanonicalToolBackend(canonical_component, maximum_roll, quantization,
                                            initial_roll,lambda:self.ns)
        self.feedback = FakeFeedback()
        self.feedback.joints = self.backend.initial_joints.copy()
        self.control = TeleopController(self.config, self.backend, self.feedback,
                                       True, lambda: self.ns)
        self.neutral_hand = multiply(axis_quaternion([0, 1, 0], 67),
                                     axis_quaternion([1, 0, 0], -38))
        self.step(0, 0.)
        self.assertEqual(self.step(1, 0.)['mapping']['state'], 'ANCHORED')
        self.anchor_send_ns = self.ns

    def step(self, grip, degrees):
        self.tick_start_ns += 10_000_000
        self.ns = self.tick_start_ns
        self.sequence += 1
        if self.backend.sent:
            self.feedback.joints = self.backend.sent[-1].copy()
        hand = multiply(axis_quaternion([0, -1, 0], degrees), self.neutral_hand)
        # PICO may use either quaternion sign; physically these poses agree.
        if self.sequence % 7 == 0:
            hand = [-value for value in hand]
        status = frame(self.sequence, self.ns, grip, q=hand[1:]+hand[:1])
        def gate():
            self.ns += self.gate_delay_ns
            return status
        return self.control.tick(status,gate,self.ns)

    def test_signed_185_degree_twists_remain_continuous_across_canonical_fk_signs(self):
        policies_with_flip = set()
        for component in (0, 3):
            for direction in (-1, 1):
                with self.subTest(component=component, direction=direction):
                    self.setup_control(canonical_component=component,legacy=True)
                    calibration = copy.deepcopy(self.control.mapper.neutral_rotation_calibration)
                    commanded = [0.]
                    for magnitude in list(range(1, 186)) + [185] * 240:
                        report = self.step(1, direction*magnitude)
                        self.assertEqual(report['mapping']['state'], 'TRACKING')
                        self.assertTrue(report['motion_command_sent_this_tick'])
                        commanded.append(self.backend.sent[-1][5])
                        self.assertTrue(self.control.mapper.active)
                        self.assertFalse(self.control.mapper.paused)
                    if len(set(self.backend.canonical_signs)) > 1:
                        policies_with_flip.add(direction)
                    steps = [b-a for a, b in zip(commanded, commanded[1:])]
                    self.assertTrue(all(direction*change >= -1e-4 for change in steps))
                    self.assertLessEqual(max(abs(change) for change in steps), .9001)
                    self.assertAlmostEqual(commanded[-1], direction*185., places=3)
                    self.assertGreater(max(direction*value for value in commanded), 180.)
                    self.assertEqual(self.backend.stops, 0)
                    self.assertEqual(self.control.mapper.neutral_rotation_calibration, calibration)
        self.assertEqual(policies_with_flip, {-1, 1})

    def assert_sent_joint_dynamics(self):
        """Check final FakeBackend sender values against its actual timestamps."""
        previous = self.backend.initial_joints.copy()
        previous_stamp = self.anchor_send_ns
        previous_velocity = [0.]*6
        maximum_speed = self.config['max_joint_speed_deg_s']
        maximum_acceleration = self.config['max_joint_accel_deg_s2']
        for joints,stamp in zip(self.backend.sent,self.backend.sent_monotonic_ns):
            dt = (stamp-previous_stamp)/1e9
            self.assertGreater(dt,0.)
            velocity = [(value-old)/dt for value,old in zip(joints,previous)]
            acceleration = [(value-old)/dt for value,old in zip(velocity,previous_velocity)]
            self.assertLessEqual(max(abs(value) for value in velocity),maximum_speed+1e-6)
            self.assertLessEqual(max(abs(value) for value in acceleration),maximum_acceleration+1e-4)
            previous,previous_stamp,previous_velocity = joints,stamp,velocity

    def test_main_profile_full_turns_have_no_branch_reversal_or_360_joint_jump(self):
        policies_with_flip = set()
        for component in (0,3):
            for direction in (-1,1):
                for final_angle in (185,360):
                    with self.subTest(component=component,direction=direction,final_angle=final_angle):
                        self.setup_control(canonical_component=component,maximum_roll=720.)
                        self.backend.minimum[5] = -720.
                        # Reconstruct the trajectory after setting the symmetric
                        # virtual hardware range. No real device is accessed.
                        self.control.joint_trajectory.minimum[5] = -720.
                        calibration = copy.deepcopy(self.control.mapper.neutral_rotation_calibration)
                        previous = 0.
                        crossed_goal = False
                        commanded = []
                        steps = []
                        one_period_budget = math.degrees(self.config['max_angular_speed_rad_s']*.01)
                        for magnitude in list(range(1,final_angle+1))+[final_angle]*400:
                            report = self.step(1,direction*magnitude)
                            self.assertEqual(report['mapping']['state'],'TRACKING')
                            self.assertTrue(report['motion_command_sent_this_tick'])
                            current = self.backend.sent[-1][5]
                            change = direction*(current-previous)
                            # Reversing near 180 before reaching the intended
                            # continuous turn is always an error. Tiny endpoint
                            # correction after acceleration-limited braking is
                            # allowed only after the final goal was reached.
                            if change < -1e-4:
                                self.assertTrue(crossed_goal)
                                self.assertLessEqual(abs(direction*previous-final_angle),
                                                     one_period_budget+1e-6)
                                self.assertLessEqual(abs(change),one_period_budget+1e-6)
                            crossed_goal |= direction*current >= final_angle-1e-3
                            self.assertLessEqual(direction*current-final_angle,one_period_budget+1e-6)
                            self.assertTrue(self.control.mapper.active)
                            self.assertFalse(self.control.mapper.paused)
                            commanded.append(current);steps.append(current-previous)
                            previous = current
                        self.assertGreater(max(direction*value for value in commanded),180.)
                        self.assertAlmostEqual(commanded[-1],direction*final_angle,places=3)
                        self.assertLessEqual(max(abs(value) for value in steps),one_period_budget+1e-6)
                        self.assert_sent_joint_dynamics()
                        self.assertEqual(self.backend.stops,0)
                        self.assertEqual(self.control.mapper.neutral_rotation_calibration,calibration)
                        if len(set(self.backend.canonical_signs)) > 1:
                            policies_with_flip.add(direction)
        self.assertEqual(policies_with_flip,{-1,1})

    def test_main_profile_joint_smoothness_uses_actual_sender_intervals(self):
        # The tick starts every 10 ms; gate/feedback work finishes at varying
        # offsets. Sender intervals therefore differ from nominal control dt.
        for direction in (-1,1):
            with self.subTest(direction=direction):
                self.setup_control(maximum_roll=720.)
                self.backend.minimum[5] = -720.
                self.control.joint_trajectory.minimum[5] = -720.
                for tick,magnitude in enumerate(list(range(1,186))+[185]*500):
                    self.gate_delay_ns = (0,2_000_000,5_000_000,1_000_000)[tick%4]
                    report = self.step(1,direction*magnitude)
                    self.assertTrue(report['motion_command_sent_this_tick'])
                    self.assertTrue(self.control.mapper.active)
                self.assertAlmostEqual(self.backend.sent[-1][5],direction*185.,places=3)
                intervals = [b-a for a,b in zip(self.backend.sent_monotonic_ns,
                                                self.backend.sent_monotonic_ns[1:])]
                self.assertGreater(len(set(intervals)),1)
                self.assert_sent_joint_dynamics()
                self.assertEqual(self.backend.stops,0)

    def test_joint_roll_boundary_keeps_reference_and_recovers_when_hand_is_reachable(self):
        # Native SDK/state joint angles are quantized; use millidegrees here.
        for initial_roll in (0., 30.):
            with self.subTest(initial_roll=initial_roll):
                self.setup_control(maximum_roll=30., quantization=3,
                                   initial_roll=initial_roll)
                calibration = copy.deepcopy(self.control.mapper.neutral_rotation_calibration)
                tool_basis = copy.deepcopy(self.control.mapper.tool_basis)
                anchor_hand = self.control.mapper.anchor_hand.copy()
                anchor_robot = self.control.mapper.anchor_robot.copy()
                states = []
                for degrees in list(range(1, 36)) + [35] * 200:
                    report = self.step(1, degrees)
                    states.append(report['mapping']['state'])
                    self.assertTrue(self.control.mapper.active)
                    self.assertFalse(self.control.mapper.paused)
                    self.assertEqual(self.backend.stops, 0)
                    self.assertLessEqual(self.backend.sent[-1][5], 30.)
                if initial_roll == 0.:
                    # The fixed mapping keeps outward intent, including while
                    # backtracking settles just inside the limit.
                    self.assertIn('BOUNDARY_LIMITED', states)
                else:
                    # Starting exactly at the limit exercises a genuine hold.
                    self.assertIn('BOUNDARY_HOLD', states)
                before = self.backend.sent[-1][5]
                self.assertGreater(before, 29.9)
                self.feedback.stationary = False
                # A small retreat still outside the joint range does not
                # redefine the Grip. Return to a genuinely reachable angle.
                report = self.step(1, 34.)
                self.assertTrue(report['motion_command_sent_this_tick'])
                for degrees in range(33,-6,-1):
                    report=self.step(1,degrees)
                self.assertLess(self.backend.sent[-1][5], before)
                self.assertTrue(self.control.mapper.active)
                self.assertFalse(self.control.mapper.paused)
                self.assertEqual(self.backend.stops, 0)
                self.assertEqual(self.control.mapper.neutral_rotation_calibration, calibration)
                self.assertEqual(self.control.mapper.tool_basis, tool_basis)
                self.assertEqual(self.control.mapper.anchor_hand, anchor_hand)
                self.assertEqual(self.control.mapper.anchor_robot, anchor_robot)


if __name__ == '__main__':
    unittest.main()
