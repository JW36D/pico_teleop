import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import copy
import math
from pathlib import Path
import unittest

from incremental import IncrementalMapper, MappingError
from robot_directions import resolve_translation_mapping, work_delta_to_body
from teleop_config import load_config


ROOT = Path(__file__).resolve().parent.parent


class RobotDirectionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / 'configs/right_speed_check.json')

    def resolve(self, rpy=(0., 0., 0.), config=None, installation=(0., -45., 0.)):
        return resolve_translation_mapping(config or self.config,
            {'rm_get_current_work_frame': {'name': 'Base', 'pose': [0., 0., 0.] + list(rpy)}},
            list(installation))

    def test_level_work_frame_right_front_up_have_robot_semantics(self):
        resolved, report = self.resolve()
        mapper = IncrementalMapper(resolved, [0., 0., 0., 1., 0., 0., 0.])
        self.assertEqual(mapper.map_vector([1., 0., 0.]), [0., -1., 0.])
        self.assertEqual(mapper.map_vector([0., 0., -1.]), [1., 0., 0.])
        self.assertEqual(mapper.map_vector([0., 1., 0.]), [0., 0., 1.])
        self.assertEqual(work_delta_to_body([0., -.05, 0.], report),
                         {'forward': 0., 'right': .05, 'up': 0.})

    def test_pitch_tilt_is_removed_from_requested_physical_up_and_forward(self):
        resolved, report = self.resolve((0., -math.pi / 4, 0.))
        axes = report['robot_direction_axes_work']
        root_half = math.sqrt(.5)
        for got, expected in zip(axes['up'], [root_half, 0., root_half]):
            self.assertAlmostEqual(got, expected)
        for got, expected in zip(axes['forward'], [root_half, 0., -root_half]):
            self.assertAlmostEqual(got, expected)
        self.assertEqual(axes['right'], [0., -1., 0.])
        # The pure translation cannot acquire the work origin offset.
        shifted, shifted_report = resolve_translation_mapping(self.config,
            {'rm_get_current_work_frame': {'name': 'Other', 'pose': [.4, -.2, .8, 0., -math.pi / 4, 0.]}},
            [0., -45., 0.])
        self.assertEqual(resolved['headset_to_work_matrix'], shifted['headset_to_work_matrix'])

    def test_work_yaw_cannot_rotate_the_fixed_robot_direction(self):
        _, report = self.resolve((0., 0., math.pi / 2))
        axes = report['robot_direction_axes_work']
        for got, expected in zip(axes['forward'], [0., -1., 0.]):
            self.assertAlmostEqual(got, expected)
        for got, expected in zip(axes['right'], [-1., 0., 0.]):
            self.assertAlmostEqual(got, expected)
        self.assertEqual(axes['up'], [0., 0., 1.])

    def test_body_reference_rotation_is_distinct_from_work_rotation(self):
        config = copy.deepcopy(self.config)
        config['robot_body_to_install_matrix'] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        _, report = self.resolve(config=config)
        self.assertEqual(report['robot_direction_axes_work']['forward'], [0., 1., 0.])
        self.assertEqual(report['robot_direction_axes_work']['right'], [1., 0., 0.])

    def test_original_config_is_not_mutated_and_resolution_is_repeatable(self):
        original = copy.deepcopy(self.config)
        first, _ = self.resolve((0., -.785, 0.))
        second, _ = self.resolve((0., -.785, 0.), config=first)
        self.assertEqual(self.config, original)
        self.assertEqual(first['headset_to_work_matrix'], second['headset_to_work_matrix'])

    def test_stale_installation_reference_is_rejected(self):
        with self.assertRaisesRegex(MappingError, 'installation differs'):
            self.resolve(installation=(0., 0., 0.))

    def test_reflection_body_reference_is_rejected(self):
        config = copy.deepcopy(self.config)
        config['robot_body_to_install_matrix'][0][0] = -1.
        with self.assertRaises(MappingError):
            self.resolve(config=config)

    def test_legacy_work_profile_keeps_its_mapping(self):
        legacy = load_config(ROOT / 'configs/right_full_teleop.json')
        resolved, report = self.resolve((0., -.785, 0.), config=legacy)
        self.assertEqual(resolved['headset_to_work_matrix'], legacy['headset_to_work_matrix'])
        self.assertEqual(report['translation_mapping_frame'], 'work')


if __name__ == '__main__':
    unittest.main()
