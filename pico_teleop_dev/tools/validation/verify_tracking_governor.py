"""真实 SDK 正/逆解 + 模拟延迟反馈验证；没有真实运动发送路径。"""
from pathlib import Path
import fcntl
import argparse
import json
import math
from collections import Counter

from incremental import angle, inverse, multiply, norm
from teleop_config import load_config
from teleop_controller import TeleopController
from robot_directions import resolve_translation_mapping


def simulate(native, config, tau, enabled):
    q0=native.initial_joints.copy()
    physical=q0.copy()
    command=q0.copy()
    observed=q0.copy()
    previous=q0.copy()
    now=1_000_000_000
    states=Counter()
    peak=0.
    stop_count=0
    step=1/config['control_hz']
    config={**config,'tracking_governor':enabled}

    class MathOnlyBackend:
        initial_joints=q0
        initial_pose=native.initial_pose
        minimum=native.minimum
        maximum=native.maximum
        speed_limits=[180.,180.,225.,225.,225.,225.]
        solve=native.solve
        forward=native.forward
        def send(self,joints):
            nonlocal command
            command=joints.copy()  # 只更新模拟目标，绝不调用 SDK 发送。
        def stop(self):
            nonlocal stop_count
            stop_count+=1

    class SimulatedFeedback:
        def snapshot(self):
            return {'joints':observed.copy(),'age_ms':(tick%5)*step*1000,
                    'stationary':max(abs(a-b) for a,b in zip(observed,previous))<.03}

    control=TeleopController(config,MathOnlyBackend(),SimulatedFeedback(),True,lambda:now)
    error=None
    for tick in range(1800):
        now+=int(step*1e9)
        physical=[a+(b-a)*(1-math.exp(-step/tau)) for a,b in zip(physical,command)]
        if tick%5==0:
            previous=observed.copy();observed=physical.copy()
        t=tick*step
        # 快速往复的绝对手柄目标，比等待到位后松 Grip 更容易累积跟随误差。
        hand_range=min(.13,.7*config['max_session_translation_m']/config['position_scale'])
        x=hand_range*math.sin(max(0.,t-.2)*2*math.pi*.3)
        y=.015*math.sin(max(0.,t-.2)*2*math.pi*.2)
        sample={'server_connected':True,'device_connected':True,'latest':{
            'sequence':tick+1,'receive_monotonic_ns':now,'app_focus':True,'input_mode':1,
            'head':{'status':3,'pose':[0,0,0,0,0,0,1]},
            'right':{'grip':0. if tick<2 else 1.,'pose':[x,y,0,0,0,0,1]}}}
        try:
            report=control.tick(sample,lambda:sample,now)
        except Exception as exc:
            error={'message':str(exc),'details':getattr(exc,'details',{})}
            break
        states[report['mapping']['state']]+=1
        peak=max(peak,report['mapping'].get('command_tracking_error_mm',0.))
    return {'synthetic_feedback_tau_s':tau,'governor':enabled,'ticks':tick+1,
            'state_counts':dict(states),'peak_command_error_mm':peak,
            'virtual_stop_count':stop_count,'error':error,'real_motion_commands':0}


def main():
    # 导入硬件仅用于只读查询/本地数学，Native Backend.send 从不调用。
    from robot_backend import Backend,check_known_controller_processes
    from runtime_paths import LOCK_DIR
    TASK_DIR = Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/right_direction_check.json')
    args=parser.parse_args()
    lock=(LOCK_DIR/'.single_joint_step.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    native=None
    try:
        check_known_controller_processes()
        native=Backend()
        def forbidden_sender(*_):
            raise AssertionError('No real motion sender is allowed in this verification')
        native.send=forbidden_sender
        config=load_config(TASK_DIR/args.config)
        config,direction_reference=resolve_translation_mapping(config,native.frames,
            [native.installation[key] for key in ('x','y','z')])
        cases=[simulate(native,config,tau,enabled) for tau in (.2,.5,1.2) for enabled in (False,True)]
        enabled=[c for c in cases if c['governor']]
        if any(c['error'] for c in enabled):
            raise AssertionError('Governor failed delayed feedback simulation: '+json.dumps(enabled))
        if not any(c['state_counts'].get('FOLLOWER_WAIT',0)>0 for c in enabled):
            raise AssertionError('Governor was not exercised')
        after=native.joints(require_idle=True)
        actual_change=max(abs(a-b) for a,b in zip(after,native.initial_joints))
        if native.attempted or actual_change>.05:
            raise AssertionError('Arm changed during read-only verification')
        # 新映射只能验证工作坐标数学，真实机器人左右仍需要现场观察。
        requested=native.initial_pose.copy()
        right_axis=[row[0] for row in config['headset_to_work_matrix']]
        requested[:3]=[a+.01*b for a,b in zip(requested[:3],right_axis)]
        mapped=native.solve(requested,native.initial_joints)
        recovered=native.forward(mapped)
        result={'motion_commands_sent':0,'config_path':args.config,'initial_joints_deg':native.initial_joints,
                'actual_joint_change_deg':actual_change,'cases':cases,
                'direction_reference':direction_reference,
                'candidate_right_work_axis':right_axis,
                'mapped_right_10mm_fk_error_mm':norm([a-b for a,b in zip(requested[:3],recovered[:3])])*1000,
                'mapped_right_10mm_rotation_error_deg':math.degrees(angle(multiply(requested[3:],inverse(recovered[3:])))),
                'mapped_right_joint_delta_deg':[a-b for a,b in zip(mapped,native.initial_joints)]}
        filename=('robot_body_governor_native_verified_20261013.json'
                  if config['translation_mapping_frame']=='robot_body' else
                  ('speed_check_native_verified_20261013.json' if args.config=='configs/right_speed_check.json'
                   else 'tracking_governor_native_verified_20261013.json'))
        (TASK_DIR/'reports'/filename).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally:
        if native:native.close()
        lock.close()


if __name__=='__main__':main()
