import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import copy
import json
import math
from pathlib import Path
import unittest

from incremental import IncrementalMapper, MappingError, limit_ik_step

CONFIG = json.loads((Path(__file__).resolve().parent.parent / 'configs/right_incremental.json').read_text())
ROBOT = [0., 0., 0., 1., 0., 0., 0.]


def frame(seq, ns, grip, xyz=None, q=None, focus=True):
    return {'server_connected': True, 'device_connected': True,
            'latest': {'sequence': seq, 'receive_monotonic_ns': ns,
                       'input_mode': 1, 'app_focus': focus,
                       'head': {'status':3,'pose':[0,0,0,0,0,0,1]}, 'right': {
                           'grip': grip, 'pose': (xyz or [0., 0., 0.]) + (q or [0., 0., 0., 1.])}}}


class ClutchBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.mapper = IncrementalMapper(CONFIG, ROBOT)
        self.ns = 1_000_000_000

    def update(self, grip, xyz=None, q=None, focus=True):
        self.ns += 20_000_000
        seq = self.ns // 20_000_000
        return self.mapper.update(frame(seq, self.ns, grip, xyz, q, focus),
                                  self.ns, lambda: ROBOT)

    def engage(self):
        self.update(0)
        return self.update(1)

    def test_start_with_held_grip_does_not_arm(self):
        result = self.update(1)
        self.assertEqual(result['state'], 'RELEASE_GRIP_TO_ARM')
        self.assertFalse(result['send'])

    def test_press_has_no_pose_jump(self):
        self.update(0, [1, 2, 3])
        result = self.update(1, [1, 2, 3])
        self.assertEqual(result['target_pose_wxyz'], ROBOT)
        self.assertFalse(result['send'])

    def test_absolute_relative_target_not_repeated_delta(self):
        self.engage()
        result = self.update(1, [0, 0, -.001])
        self.assertAlmostEqual(result['target_pose_wxyz'][0], .0002)
        result = self.update(1, [0, 0, -.001])
        self.assertAlmostEqual(result['target_pose_wxyz'][0], .0002)

    def test_release_and_regrip_reanchors_actual_robot(self):
        self.engage()
        self.update(1, [0, 0, -.001])
        result = self.update(0, [2, 0, 0])
        self.assertFalse(result['send'])
        result = self.update(1, [2, 0, 0])
        self.assertEqual(result['target_pose_wxyz'], ROBOT)
        self.assertFalse(result['send'])

    def test_stale_input_disarms_and_held_grip_cannot_resume(self):
        self.engage()
        data = frame(10, self.ns, 1)
        result = self.mapper.update(data, self.ns + 200_000_000, lambda: ROBOT)
        self.assertEqual(result['state'], 'WAIT_FRESH_INPUT')
        self.assertEqual(self.update(1)['state'], 'RELEASE_GRIP_TO_ARM')

    def test_lost_focus_disarms(self):
        self.engage()
        self.assertEqual(self.update(1, focus=False)['state'], 'WAIT_FRESH_INPUT')
        self.assertEqual(self.update(1)['state'], 'RELEASE_GRIP_TO_ARM')

    def test_input_jump_rejected(self):
        self.engage()
        self.assertEqual(self.update(1, [1, 0, 0])['state'], 'INPUT_JUMP')

    def test_velocity_limited(self):
        self.engage()
        result = self.update(1, [0, 0, -.05])
        self.assertAlmostEqual(result['target_pose_wxyz'][0], .0004)

    def test_session_bound_is_not_bypassed_by_clutch(self):
        self.engage()
        actual = [0.029, 0., 0., 1., 0., 0., 0.]
        self.update(0)
        self.ns += 20_000_000
        self.mapper.update(frame(20, self.ns, 1), self.ns, lambda: actual)
        result = self.update(1, [0, 0, -.02])
        self.assertEqual(result['state'], 'SESSION_BOUND')
        self.assertEqual(result['limit_reason'], ['translation'])
        self.assertAlmostEqual(result['requested_start_distance_mm'], 33.)
        self.assertAlmostEqual(result['allowed_start_distance_mm'], 30.)
        self.assertFalse(result['send'])

    def test_first_motion_bound_reports_request_before_rate_limiting(self):
        config = json.loads((Path(__file__).resolve().parent.parent / 'configs/right_first_motion.json').read_text())
        self.mapper = IncrementalMapper(config, ROBOT)
        self.engage()
        self.update(1, [0, 0, -.05])
        result = self.update(1, [0, 0, -.11])
        self.assertEqual(result['limit_reason'], ['translation'])
        self.assertAlmostEqual(result['requested_start_distance_mm'], 11.)
        self.assertAlmostEqual(result['allowed_start_distance_mm'], 10.)
        self.assertFalse(self.mapper.active)
        self.assertEqual(self.update(1)['state'], 'RELEASE_GRIP_TO_ARM')

    def test_quaternion_sign_change_has_no_rotation(self):
        self.engage()
        result = self.update(1, q=[0, 0, 0, -1])
        self.assertEqual(result['target_pose_wxyz'], ROBOT)
        self.assertFalse(result['send'])

    def test_quaternion_zero_rejected(self):
        self.engage()
        self.assertEqual(self.update(1, q=[0, 0, 0, 0])['state'], 'INVALID_POSE')

    def test_yaw_maps_to_robot_z_and_is_velocity_limited(self):
        self.engage()
        half_angle = .05
        result = self.update(1, q=[0, math.sin(half_angle), 0, math.cos(half_angle)])
        target = result['target_pose_wxyz']
        self.assertAlmostEqual(target[4], 0)
        self.assertAlmostEqual(target[5], 0)
        self.assertAlmostEqual(target[6], math.sin(.001), places=7)

    def test_reflection_mapping_rejected(self):
        config = copy.deepcopy(CONFIG)
        config['headset_to_work_matrix'] = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
        with self.assertRaises(MappingError):
            IncrementalMapper(config, ROBOT)

    def test_ik_backtracking_limits_velocity_without_joint_clipping(self):
        def solve(target, _seed):
            return [target[0] * 1000] + [0] * 5
        requested = [.001, 0, 0, 1, 0, 0, 0]
        bounded, joints, limited = limit_ik_step(solve, ROBOT, requested, [0]*6, .1)
        self.assertTrue(limited)
        self.assertLessEqual(joints[0], .1)
        self.assertAlmostEqual(bounded[0] * 1000, joints[0])

    def test_discontinuous_ik_branch_is_rejected(self):
        with self.assertRaises(MappingError):
            limit_ik_step(lambda *_: [30]*6, ROBOT,
                          [.001, 0, 0, 1, 0, 0, 0], [0]*6, .1)

    def test_position_profile_accepts_target_beyond_first_motion_radius(self):
        config = json.loads((Path(__file__).resolve().parent.parent / 'configs/right_position_teleop.json').read_text())
        self.mapper = IncrementalMapper(config, ROBOT)
        self.engage()
        result = None
        for i in range(1, 5):
            result = self.update(1, [0, 0, -.05*i])
        self.assertEqual(result['state'], 'TRACKING')
        self.assertAlmostEqual(result['requested_delta_work_m'][0], .02)
        self.assertAlmostEqual(result['target_pose_wxyz'][0], .0004)

    def test_position_profile_still_rejects_target_beyond_100_mm(self):
        config = json.loads((Path(__file__).resolve().parent.parent / 'configs/right_position_teleop.json').read_text())
        self.mapper = IncrementalMapper(config, ROBOT)
        self.engage()
        for i in range(1, 22):
            result = self.update(1, [0, 0, -.05*i])
        self.assertEqual(result['state'], 'SESSION_BOUND')
        self.assertEqual(result['allowed_start_distance_mm'], 100.)
        self.assertFalse(result['send'])

    def test_position_profile_keeps_orientation_locked(self):
        config = json.loads((Path(__file__).resolve().parent.parent / 'configs/right_position_teleop.json').read_text())
        self.mapper = IncrementalMapper(config, ROBOT)
        self.engage()
        result = self.update(1, [0, 0, -.02], q=[0, math.sin(.1), 0, math.cos(.1)])
        self.assertEqual(result['target_pose_wxyz'][3:], ROBOT[3:])

    def test_position_profile_still_disarms_stale_data(self):
        config = json.loads((Path(__file__).resolve().parent.parent / 'configs/right_position_teleop.json').read_text())
        self.mapper = IncrementalMapper(config, ROBOT)
        self.engage()
        result = self.mapper.update(frame(100, self.ns, 1), self.ns+200_000_000, lambda: ROBOT)
        self.assertEqual(result['state'], 'WAIT_FRESH_INPUT')
        self.assertFalse(self.mapper.active)


if __name__ == '__main__':
    unittest.main()
