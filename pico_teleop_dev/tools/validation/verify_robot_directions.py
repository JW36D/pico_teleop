"""Read-only SDK verification of robot-body translations at four hand headings.

Synthetic hand input; real FK/IK only. Sender methods are forbidden, and no
physical motion can be enabled from the command line. The declared physical
robot axes remain a candidate until the operator observes them on hardware.
"""
import argparse
from collections import Counter
import fcntl
import json
import math
from pathlib import Path

from incremental import angle, inverse, multiply, norm, rotate_vector, vector
from robot_backend import Backend, check_known_controller_processes, checked_pair
from runtime_paths import LOCK_DIR
TASK_DIR = Path(__file__).resolve().parents[2]
from robot_directions import matrix_vector, resolve_translation_mapping
from teleop_config import load_config
from teleop_controller import TeleopController


def verify_path(native, config, heading_deg, operator_axis, distance_m):
    heading = math.radians(heading_deg)
    head_q = [math.cos(heading/2), 0., math.sin(heading/2), 0.]
    head_xyzw = head_q[1:] + head_q[:1]
    origin = [.4, 1., -.2]
    actual_joints = native.initial_joints.copy()
    now = 1_000_000_000
    states = Counter()

    class MathBackend:
        initial_pose = native.initial_pose
        initial_joints = native.initial_joints
        minimum = native.minimum
        maximum = native.maximum
        speed_limits = native.speed_limits
        solve = native.solve
        forward = native.forward
        def send(self, *_):
            raise AssertionError('Motion sender forbidden in direction verification')
        def stop(self):
            raise AssertionError('Motion stop API forbidden in direction verification')

    class StaticFeedback:
        def snapshot(self):
            return {'joints': actual_joints.copy(), 'stationary': True, 'age_ms': 1.}

    control = TeleopController(config, MathBackend(), StaticFeedback(), False, lambda: now)
    work_quat = native.robot.rm_algo_euler2quaternion(
        native.frames['rm_get_current_work_frame']['pose'][3:])
    hand_displacement = [x * distance_m / config['position_scale'] for x in operator_axis]
    room_displacement = rotate_vector(head_q, hand_displacement)
    body_displacement = [-operator_axis[2]*distance_m, -operator_axis[0]*distance_m,
                         operator_axis[1]*distance_m]
    expected_installation = matrix_vector(config['robot_body_to_install_matrix'], body_displacement)
    expected_work = rotate_vector(inverse(work_quat), expected_installation)
    peak_joint_step = peak_position_error = 0.
    previous_joints = actual_joints.copy()
    for tick in range(502):
        now += int(1e9/config['control_hz'])
        fraction = min(1., max(0., (tick-2)/100))
        xyz = [a+fraction*b for a,b in zip(origin, room_displacement)]
        status = {'server_connected': True, 'device_connected': True, 'latest': {
            'sequence': tick+1, 'receive_monotonic_ns': now, 'input_mode': 1, 'app_focus': True,
            'head': {'status': 3, 'pose': [0., 0., 0.] + head_xyzw},
            'right': {'grip': 0. if tick == 0 else 1., 'pose': xyz + [0., 0., 0., 1.]}}}
        report = control.tick(status, lambda: status, now)
        mapping = report['mapping']
        states[mapping['state']] += 1
        if mapping['state'] not in ('RELEASED', 'ANCHORED', 'TRACKING'):
            raise AssertionError('Native direction trajectory interrupted: '+json.dumps(mapping))
        if mapping['state'] != 'TRACKING':
            continue
        joints = mapping['target_joints_deg']
        peak_joint_step = max(peak_joint_step, max(abs(a-b) for a,b in zip(joints, previous_joints)))
        previous_joints = joints.copy()
        recovered = native.forward(joints)
        error = norm([a-b for a,b in zip(recovered[:3], mapping['target_pose_wxyz'][:3])])
        peak_position_error = max(peak_position_error, error)
        if error > .0001:
            raise AssertionError('SDK IK/FK target mismatch')
    recovered = native.forward(control.last_joints)
    achieved_work = [a-b for a,b in zip(recovered[:3], native.initial_pose[:3])]
    # Independent SDK Euler -> quaternion conversion, then project FK result
    # back into the installation reference; catches incorrect work-frame sign.
    achieved_installation = rotate_vector(work_quat, achieved_work)
    distance_error = norm([a-b for a,b in zip(achieved_work, expected_work)])
    orientation_error = angle(multiply(recovered[3:], inverse(native.initial_pose[3:])))
    if distance_error > .0001 or orientation_error > .001:
        raise AssertionError('Native path did not reach the expected locked-orientation goal')
    if peak_joint_step > min(config['max_joint_speed_deg_s'], min(native.speed_limits))/config['control_hz']+.0002:
        raise AssertionError('Joint velocity bound exceeded')
    return {'heading_deg': heading_deg, 'operator_right_up_back': operator_axis,
            'expected_delta_installation_m': expected_installation,
            'achieved_delta_installation_m': achieved_installation,
            'achieved_delta_work_m': achieved_work,
            'goal_error_mm': distance_error*1000,
            'orientation_error_deg': math.degrees(orientation_error),
            'peak_joint_step_deg': peak_joint_step,
            'peak_ik_fk_error_mm': peak_position_error*1000,
            'target_joint_excursion_deg': [a-b for a,b in zip(control.last_joints, native.initial_joints)],
            'states': dict(states), 'ticks': 502, 'motion_commands_sent': control.send_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=TASK_DIR/'configs/right_speed_check.json')
    parser.add_argument('--distance', type=float, default=.05, help='TCP goal distance, metres')
    args = parser.parse_args()
    if not math.isfinite(args.distance) or not .005 <= args.distance <= .05:
        parser.error('Distance must be 5..50 mm')
    config = load_config(args.config)
    if config['rotation_scale'] != 0 or config['translation_mapping_frame'] != 'robot_body':
        parser.error('A robot-body position-only profile is required')
    lock = (LOCK_DIR/'.single_joint_step.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    native = None
    try:
        check_known_controller_processes()
        native = Backend()
        native.speed_limits = vector(checked_pair(native.robot, 'rm_get_joint_max_speed'), 6)
        def forbidden_sender(*_):
            raise AssertionError('Native motion sender forbidden')
        native.send = forbidden_sender
        config, direction_reference = resolve_translation_mapping(config, native.frames,
            [native.installation[key] for key in ('x', 'y', 'z')])
        cases = [verify_path(native, config, heading, axis, args.distance)
                 for heading in (0, 90, 180, -90)
                 for axis in ([1,0,0], [-1,0,0], [0,0,-1], [0,0,1], [0,1,0], [0,-1,0])]
        actual_change = max(abs(a-b) for a,b in zip(native.joints(require_idle=True), native.initial_joints))
        if native.attempted or actual_change > .05:
            raise AssertionError('Robot moved during read-only verification')
        result = {'synthetic_input': True, 'motion_commands_sent': 0,
                  'physical_direction_validation': 'pending_operator_observation',
                  'config_path': str(args.config), 'distance_m': args.distance,
                  'direction_reference': direction_reference,
                  'initial_joints_deg': native.initial_joints,
                  'actual_joint_change_deg': actual_change, 'cases': cases}
        output = TASK_DIR/'reports'/'robot_directions_math_verified_20261013.json'
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'output': str(output), 'cases': len(cases), 'motion_commands_sent': 0,
              'max_goal_error_mm': max(c['goal_error_mm'] for c in cases),
              'max_joint_step_deg': max(c['peak_joint_step_deg'] for c in cases),
              'actual_joint_change_deg': actual_change,
              'direction_reference': direction_reference}, allow_nan=False), flush=True)
    finally:
        if native:
            native.close()
        lock.close()


if __name__ == '__main__':
    main()
