"""Read-only native SDK checks of mapped translation and wrist rotation paths.

Only queries/FK/IK are used; no send or stop method is called by this script.
"""
import fcntl
import json
import math
from pathlib import Path

from incremental import (IncrementalMapper, MappingError, angle, inverse, limit_ik_step,
                         multiply, norm, pose, quat, scaled_rotation, vector)
from robot_backend import Backend, ARM_IPS, checked_pair, check_known_controller_processes
from runtime_paths import LOCK_DIR
TASK_DIR = Path(__file__).resolve().parents[2]
from robot_feedback import decode_state, query
from teleop_config import load_config


def main():
    config = load_config(TASK_DIR/'configs/right_full_teleop.json')
    lock = (LOCK_DIR/'.single_joint_step.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    backend = None
    try:
        check_known_controller_processes()
        if query(ARM_IPS['right'],'get_arm_power_state')['power_state'] != 1:
            raise MappingError('Arm must already be powered')
        backend = Backend()
        native_limits = vector(checked_pair(backend.robot,'rm_get_joint_max_speed'),6)
        if any(x<=0 for x in native_limits):
            raise MappingError('Invalid hardware joint speed limits')
        observed = decode_state(query(ARM_IPS['right'],'get_current_arm_state'))
        measured_quat = quat(backend.robot.rm_algo_euler2quaternion(observed['orientation_rad']))
        position_error = norm([a-b for a,b in zip(observed['position_m'],backend.initial_pose[:3])])
        orientation_error = angle(multiply(measured_quat,inverse(backend.initial_pose[3:])))
        if position_error > .005 or orientation_error > .005:
            raise MappingError('Native feedback/FK coordinate mismatch')
        mapper = IncrementalMapper(config,backend.initial_pose)
        cases = []
        for kind in ('translation','rotation'):
            for axis in range(3):
                for sign in (-1,1):
                    goal = backend.initial_pose.copy()
                    if kind == 'translation':
                        delta = [0.,0.,0.]; delta[axis] = sign*.02
                        mapped = mapper.map_vector(delta)
                        goal[:3] = [a+d*config['position_scale'] for a,d in zip(goal[:3],mapped)]
                    else:
                        half = sign*math.radians(5)/2
                        q = [math.cos(half),0.,0.,0.];q[axis+1] = math.sin(half)
                        if config['rotation_mapping']=='controller_local':
                            rotation = scaled_rotation(mapper.map_tool_rotation(q),config['rotation_scale'])
                            goal[3:] = quat(multiply(goal[3:],rotation))
                        else:
                            rotation = scaled_rotation(mapper.map_rotation(q),config['rotation_scale'])
                            goal[3:] = quat(multiply(rotation,goal[3:]))
                    current = backend.initial_pose.copy()
                    seed = backend.initial_joints.copy()
                    peak_joint_step = 0.
                    for tick in range(1,401):
                        delta = [b-a for a,b in zip(current[:3],goal[:3])]
                        distance = norm(delta)
                        f = min(1.,config['max_linear_speed_m_s']/config['control_hz']/distance) if distance else 1.
                        requested_pos = [a+d*f for a,d in zip(current[:3],delta)]
                        rotation = multiply(goal[3:],inverse(current[3:]))
                        rotation_angle = angle(rotation)
                        f = min(1.,config['max_angular_speed_rad_s']/config['control_hz']/rotation_angle) if rotation_angle else 1.
                        requested = requested_pos+quat(multiply(scaled_rotation(rotation,f),current[3:]))
                        bounded,joints,_ = limit_ik_step(backend.solve,current,requested,seed,
                            [min(config['max_joint_speed_deg_s'],v)/config['control_hz'] for v in native_limits])
                        step = max(abs(a-b) for a,b in zip(joints,seed))
                        peak_joint_step = max(peak_joint_step,step)
                        recovered = backend.forward(joints)
                        if norm([a-b for a,b in zip(bounded[:3],recovered[:3])]) > .0001 or angle(multiply(bounded[3:],inverse(recovered[3:]))) > .001:
                            raise MappingError('Native inverse/forward pose mismatch')
                        current,seed = recovered,joints
                        distance = norm([a-b for a,b in zip(current[:3],goal[:3])])
                        rotation_error = angle(multiply(current[3:],inverse(goal[3:])))
                        if distance <= .00001 and rotation_error <= .0002:
                            break
                    else:
                        raise MappingError('Native test path did not converge')
                    cases.append({'kind':kind,'pico_axis':axis,'sign':sign,'simulated_ticks':tick,
                        'goal_pose_wxyz':goal,'position_error_mm':distance*1000,
                        'rotation_error_deg':math.degrees(rotation_error),
                        'peak_joint_step_deg':peak_joint_step,
                        'max_joint_excursion_deg':max(abs(a-b) for a,b in zip(seed,backend.initial_joints))})
        after = backend.joints(require_idle=True)
        actual_change = max(abs(a-b) for a,b in zip(after,backend.initial_joints))
        if actual_change > .05:
            raise MappingError('Robot moved during read-only verification')
        result = {'motion_commands_sent':0,'config':config,'native_joint_speed_limits_deg_s':native_limits,
            'initial_joints_deg':backend.initial_joints,'initial_pose_wxyz':backend.initial_pose,
            'controller_fk_position_error_mm':position_error*1000,
            'controller_fk_rotation_error_deg':math.degrees(orientation_error),
            'actual_joint_change_deg':actual_change,'cases':cases}
        (TASK_DIR/'reports'/'follow_fix_math_verified_20261012.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result,allow_nan=False),flush=True)
    finally:
        if backend:
            backend.close()  # Its stop method is a no-op unless motion was attempted.
        lock.close()


if __name__=='__main__':
    main()
