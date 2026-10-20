"""Real SDK FK/IK, synthetic rotation input, and forbidden motion senders."""
import argparse
from collections import Counter
from pathlib import Path
import fcntl
import json
import math

from incremental import (angle, controller_pose, inverse, multiply, norm, operator_yaw_basis,
                         rotate_vector, vector)
from robot_backend import Backend, checked_pair, check_known_controller_processes, read_pico
from runtime_paths import LOCK_DIR
TASK_DIR = Path(__file__).resolve().parents[2]
from robot_directions import resolve_translation_mapping
from rotation_axes import twist_and_tilt
from teleop_config import load_config
from teleop_controller import TeleopController


def rotation(axis, degrees):
    half=math.radians(degrees)/2
    return [math.cos(half)]+[x*math.sin(half) for x in axis]


def run_case(native, config, hand_body, yaw_deg, kind, degrees):
    yaw=rotation([0,1,0],yaw_deg)
    anchor_hand=multiply(yaw,hand_body)
    axis=rotate_vector(yaw,[0,-1,0] if kind=='roll' else [1,0,0])
    now=1_000_000_000
    states=Counter()
    first_limit=None
    goal_limit_reason=None
    peak_step=peak_position=peak_roll_other=0.
    peak_tilt=peak_twist=0.
    last_joints=native.initial_joints.copy()
    final_pose=native.initial_pose.copy()

    class MathBackend:
        initial_joints=native.initial_joints
        initial_pose=native.initial_pose
        minimum=native.minimum
        maximum=native.maximum
        speed_limits=native.speed_limits
        solve=native.solve
        forward=native.forward
        def send(self,*_):raise AssertionError('Motion sender forbidden')
        def stop(self):raise AssertionError('Motion stop API forbidden')

    class StaticFeedback:
        def snapshot(self):return {'joints':native.initial_joints.copy(),'age_ms':1.,'stationary':True}

    control=TeleopController(config,MathBackend(),StaticFeedback(),False,lambda:now)
    ramp=max(2.,abs(degrees)/45.)
    ticks=int((ramp+3.)*config['control_hz'])+3
    for tick in range(ticks):
        now+=int(1e9/config['control_hz'])
        commanded=degrees*min(1.,max(0.,(tick-2)/(ramp*config['control_hz'])))
        hand=multiply(rotation(axis,commanded),anchor_hand)
        # Alternating quaternion signs exercise the actual PICO double cover.
        if tick%11==0:hand=[-x for x in hand]
        sample={'server_connected':True,'device_connected':True,'latest':{
            'sequence':tick+1,'receive_monotonic_ns':now,'input_mode':1,'app_focus':True,
            'head':{'status':3,'pose':[0,0,0]+yaw[1:]+yaw[:1]},
            'right':{'grip':0. if tick==0 else 1.,'pose':[.3,1.,-.2]+hand[1:]+hand[:1]}}}
        report=control.tick(sample,lambda:sample,now)
        mapping=report['mapping'];states[mapping['state']]+=1
        if mapping.get('rotation_goal_reachable') is False:
            goal_limit_reason=mapping.get('rotation_goal_limit_reason')
        if mapping['state'] not in ('RELEASED','ANCHORED','TRACKING') and first_limit is None:
            first_limit={'hand_angle_deg':commanded,'state':mapping['state'],'reason':mapping.get('reason')}
        if mapping['state']!='TRACKING':continue
        joints=mapping['target_joints_deg']
        peak_step=max(peak_step,max(abs(a-b) for a,b in zip(joints,last_joints)))
        last_joints=joints.copy();final_pose=native.forward(joints)
        peak_position=max(peak_position,norm([a-b for a,b in zip(final_pose[:3],native.initial_pose[:3])]))
        local=multiply(inverse(native.initial_pose[3:]),final_pose[3:])
        twist,tilt=twist_and_tilt(local)
        peak_tilt=max(peak_tilt,math.degrees(tilt))
        # Quaternion can change sign at 180; signed joint excursion is authoritative for pure roll.
        if kind=='roll':
            peak_twist=max(peak_twist,abs(joints[5]-native.initial_joints[5]))
            peak_roll_other=max(peak_roll_other,max(abs(a-b) for a,b in zip(joints[:5],native.initial_joints[:5])))
    requested=control.mapper.requested_goal(controller_pose(sample['latest']['right']['pose']))[0]
    goal_error=math.degrees(angle(multiply(final_pose[3:],inverse(requested[3:]))))
    reached=goal_error<.1 and first_limit is None
    if kind=='roll' and (not reached or peak_roll_other>.03 or peak_position>.0001):
        raise AssertionError('Pure tool roll failed: '+json.dumps({'degrees':degrees,'yaw':yaw_deg,
            'goal_error_deg':goal_error,'other_joint_change_deg':peak_roll_other,
            'position_error_mm':peak_position*1000,'first_limit':first_limit,'states':dict(states)}))
    if peak_step>config['max_joint_speed_deg_s']/config['control_hz']+.0002:
        raise AssertionError('Joint speed bound exceeded')
    return {'kind':kind,'operator_yaw_deg':yaw_deg,'requested_angle_deg':degrees,
            'reached':reached,'goal_orientation_error_deg':goal_error,
            'peak_tool_tilt_deg':peak_tilt,'peak_twist_joint6_deg':peak_twist,
            'peak_other_joints_deg':peak_roll_other,'peak_tcp_position_error_mm':peak_position*1000,
            'peak_joint_step_deg':peak_step,
            'joint_excursion_deg':[a-b for a,b in zip(last_joints,native.initial_joints)],
            'first_limit':first_limit,'goal_limit_reason':goal_limit_reason,
            'states':dict(states),'ticks':ticks,'motion_commands_sent':control.send_count}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/right_rotation_teleop.json')
    args=parser.parse_args()
    config=load_config(TASK_DIR/args.config)
    lock=(LOCK_DIR/'.single_joint_step.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    native=None
    try:
        check_known_controller_processes();native=Backend()
        native.speed_limits=vector(checked_pair(native.robot,'rm_get_joint_max_speed'),6)
        def forbidden(*_):raise AssertionError('Native motion sender forbidden')
        native.send=forbidden
        config,directions=resolve_translation_mapping(config,native.frames,
            [native.installation[key] for key in ('x','y','z')])
        hand_body=rotation([1,0,0],-38)
        reference_source='synthetic_neutral_hand'
        try:
            pico=read_pico();sample=pico.get('latest') or {}
            if pico.get('fresh') and sample.get('app_focus'):
                basis=operator_yaw_basis(sample['head']['pose'])
                yaw=math.degrees(math.atan2(-basis[0][2],basis[0][0]))
                hand_body=multiply(inverse(rotation([0,1,0],yaw)),controller_pose(sample['right']['pose'])[3:])
                reference_source='current_real_PICO_neutral_pose'
        except (KeyError,MappingError,OSError):pass
        cases=[run_case(native,config,hand_body,yaw,kind,degree)
            for yaw in (0,90,180,-90)
            for kind,degree in (('roll',180),('roll',-180),('pitch',90),('pitch',-90))]
        actual_change=max(abs(a-b) for a,b in zip(native.joints(require_idle=True),native.initial_joints))
        if native.attempted or actual_change>.05:raise AssertionError('Robot changed during no-motion verification')
        result={'motion_commands_sent':0,'reference_source':reference_source,'hand_body_quaternion_wxyz':hand_body,
            'initial_joints_deg':native.initial_joints,'initial_pose_wxyz':native.initial_pose,
            'actual_joint_change_deg':actual_change,'directions':directions,
            'control_point':'current Arm_Tip; no measured gripper tip offset','cases':cases}
        path=TASK_DIR/'reports'/'neutral_rotation_native_verified_20261013.json'
        path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'output':str(path),'reference_source':reference_source,'motion_commands_sent':0,
            'cases':[{'kind':c['kind'],'yaw':c['operator_yaw_deg'],'angle':c['requested_angle_deg'],
                'reached':c['reached'],'error_deg':c['goal_orientation_error_deg'],
                'max_tilt':c['peak_tool_tilt_deg'],'limit':c['first_limit']} for c in cases]}),flush=True)
    finally:
        if native:native.close()
        lock.close()


if __name__=='__main__':main()
