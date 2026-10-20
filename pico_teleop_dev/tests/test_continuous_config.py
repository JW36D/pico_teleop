"""Configuration compatibility and stopping margins for continuous tracking."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from incremental import MappingError
from teleop_config import load_config


class ContinuousConfigTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((ROOT / 'configs/right_rotation_teleop.json').read_text())

    def load_overrides(self, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({**self.raw, **overrides}))
            return load_config(path)

    def test_main_profile_enables_continuous_directional_tracking(self):
        config = load_config(ROOT / 'configs/right_rotation_teleop.json')
        self.assertFalse(config['joint_session_limits_enabled'])
        self.assertEqual(config['tracking_governor_mode'], 'continuous')
        self.assertEqual(config['boundary_behavior'], 'directional')
        self.assertEqual(config['max_linear_accel_m_s2'], 8.)
        self.assertEqual(config['max_angular_accel_rad_s2'], 12.)
        # Main profile is a continuous task controller. Legacy mechanisms are
        # explicitly configured in separate fixtures, not imposed on this one.
        for key, expected in {
            'position_scale': 1., 'rotation_scale': 1.,
            'max_linear_speed_m_s': .45, 'max_angular_speed_rad_s': 2.,
            'max_joint_speed_deg_s': 150., 'max_joint_accel_deg_s2': 600.,
            'control_hz': 100., 'feedback_hz': 100., 'follow_mode': 'low',
            'session_pose_limits_enabled': False,
            'clutch_rotation_limits_enabled': False, 'rotation_path_mode': 'incremental',
            'mapper_control_gap_guard': False, 'input_jump_mode': 'velocity',
            'continuous_input_recovery': True, 'continuous_timing_recovery': True,
            'require_robot_stationary_to_anchor': False, 'joint_trajectory_enabled': True,
            'tracking_governor_curve': 'plateau', 'tracking_error_time_alignment': True,
            'translation_heading_mode':'session', 'boundary_reference_mode':'fixed',
        }.items():
            with self.subTest(parameter=key):
                self.assertEqual(config[key], expected)

    def test_legacy_profiles_keep_pause_and_hold_defaults(self):
        config = load_config(ROOT / 'configs/right_full_teleop.json')
        self.assertTrue(config['joint_session_limits_enabled'])
        self.assertEqual(config['tracking_governor_mode'], 'hold')
        self.assertEqual(config['boundary_behavior'], 'pause')

    def test_high_follow_is_supported_with_its_timing_contract(self):
        # A valid config is not proof that Linux/SDK actually meets the period.
        high=self.load_overrides(follow_mode='high',control_hz=200.,
            max_control_gap_s=.01,max_command_gap_s=.01,
            continuous_timing_recovery=False,auto_resume_timing_gaps=False)
        self.assertEqual(high['follow_mode'],'high')
        with self.assertRaisesRegex(MappingError,'10ms'):
            self.load_overrides(follow_mode='high',control_hz=200.,
                continuous_timing_recovery=False,auto_resume_timing_gaps=False)

    def test_acceleration_must_be_finite_positive_without_arbitrary_caps(self):
        for key, examples in (('max_linear_accel_m_s2', (6., 60.)),
                              ('max_angular_accel_rad_s2', (30., 300.)),
                              ('max_joint_accel_deg_s2', (600., 1200.))):
            for value in (0., -.1, True, '3', math.nan, math.inf):
                with self.subTest(parameter=key, value=value):
                    with self.assertRaises(MappingError):
                        self.load_overrides(**{key: value})
            for value in examples:
                self.assertEqual(self.load_overrides(**{key: value})[key], value)

    def test_continuous_margin_includes_position_and_angular_braking(self):
        for overrides in ({'max_linear_accel_m_s2': 1.},
                          {'max_angular_accel_rad_s2': 5.}):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(MappingError, 'braking.*stop bound'):
                    self.load_overrides(**overrides)

    def test_continuous_margin_must_be_strictly_below_stop_bound(self):
        linear_bound = (self.raw['max_command_lead_m'] +
                        self.raw['max_linear_speed_m_s'] ** 2 /
                        (2 * self.raw['max_linear_accel_m_s2']) +
                        self.raw['max_linear_speed_m_s'] / self.raw['control_hz'])
        angular_bound = (self.raw['max_command_lead_angle_rad'] +
                         self.raw['max_angular_speed_rad_s'] ** 2 /
                         (2 * self.raw['max_angular_accel_rad_s2']) +
                         self.raw['max_angular_speed_rad_s'] / self.raw['control_hz'])
        for overrides in ({'max_tracking_error_m': linear_bound},
                          {'max_tracking_angle_rad': angular_bound}):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(MappingError, 'stop bound'):
                    self.load_overrides(**overrides)
        config = self.load_overrides(max_tracking_error_m=linear_bound + .000001,
                                     max_tracking_angle_rad=angular_bound + .000001)
        self.assertEqual(config['tracking_governor_mode'], 'continuous')

    def test_disabled_governor_allows_smooth_motion_without_lead_margin(self):
        config = self.load_overrides(tracking_governor=False,
                                     max_linear_accel_m_s2=1.,
                                     max_angular_accel_rad_s2=5.)
        self.assertFalse(config['tracking_governor'])
        self.assertEqual(config['tracking_governor_mode'], 'continuous')

    def test_hold_governor_retains_legacy_one_step_margin(self):
        config = self.load_overrides(tracking_governor_mode='hold',
                                     max_linear_accel_m_s2=1.,
                                     max_angular_accel_rad_s2=5.)
        self.assertEqual(config['tracking_governor_mode'], 'hold')
        bound = (self.raw['max_command_lead_m'] +
                 self.raw['max_linear_speed_m_s'] / self.raw['control_hz'])
        with self.assertRaisesRegex(MappingError, 'stop bound'):
            self.load_overrides(tracking_governor_mode='hold', max_tracking_error_m=bound)

    def test_mode_and_joint_limit_flag_types_are_checked(self):
        for overrides in ({'tracking_governor_mode': 'high'},
                          {'boundary_behavior': 'ignore'},
                          {'joint_session_limits_enabled': 0},
                          {'joint_session_limits_enabled': 'false'}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(MappingError):
                    self.load_overrides(**overrides)

    def test_disabled_session_ranges_have_no_arbitrary_numeric_maximum(self):
        for overrides in ({'max_joint_speed_deg_s': 120.},
                          {'max_joint_session_deg': 1000.},
                          {'max_joint_session_deg_by_joint': [1000.] * 6}):
            with self.subTest(overrides=overrides):
                config = self.load_overrides(joint_session_limits_enabled=False, **overrides)
                for key, value in overrides.items():
                    self.assertEqual(config[key], value)


if __name__ == '__main__':
    unittest.main()
