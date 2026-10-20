"""Native SDK integration check with synthetic hands; ALWAYS NO MOTION.

Reads the real robot and PICO socket, calculates local roll / translation / soft
bound recovery at the configured period. It cannot enable execute mode.
"""
import collections
from pathlib import Path
import fcntl
import json
import math
import time

from incremental import angle, inverse, multiply, norm, quat, vector
from robot_backend import Backend, ARM_IPS, checked_pair, read_pico, check_known_controller_processes
from runtime_paths import LOCK_DIR
TASK_DIR = Path(__file__).resolve().parents[2]
from robot_feedback import FeedbackMonitor
from teleop_config import load_config
from teleop_controller import TeleopController
from full_teleop import percentile


def main():
    config = load_config(TASK_DIR/'configs/right_full_teleop.json')
    lock = (LOCK_DIR/'.single_joint_step.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    backend = monitor = None
    try:
        check_known_controller_processes()
        backend = Backend()
        backend.speed_limits = vector(checked_pair(backend.robot,'rm_get_joint_max_speed'),6)
        def forbidden_sender(_):
            raise AssertionError('This verification must never send motion')
        backend.send = forbidden_sender
        monitor = FeedbackMonitor(ARM_IPS['right'],config,process_check=check_known_controller_processes)
        monitor.start()
        control = TeleopController(config,backend,monitor,execute=False)
        hand_anchor = quat(backend.robot.rm_algo_euler2quaternion([.8,-.4,1.2]))
        origin = [.4,.1,.2]
        start = time.monotonic()
        next_tick = start
        last_tick = last_input = None
        status = None
        seq = 0
        intervals = []
        states = collections.Counter()
        trace = []
        peak_roll_coupling = peak_roll_error = peak_roll_translation = 0.
        actual_pico_rates = []
        while time.monotonic()-start < 17.5:
            wait = next_tick-time.monotonic()
            if wait>0:time.sleep(wait)
            now = time.monotonic(); elapsed = now-start
            next_tick += 1/config['control_hz']
            if now-next_tick > 1/config['control_hz']:
                next_tick = now+1/config['control_hz']
            if last_tick is not None:intervals.append((now-last_tick)*1000)
            last_tick = now
            # 包含真实执行循环中两次 Unix socket 查询的开销；返回内容只作通信指标。
            observed = read_pico();read_pico()
            actual_pico_rates.append(observed['receive_hz_2s'])
            if last_input is None or now-last_input >= 1/72:
                last_input=now;seq+=1
                xyz=origin.copy();roll=0.
                if .2 <= elapsed < 8.2:
                    roll=math.radians(15)*math.sin(2*math.pi*(elapsed-.2)/4)
                elif 8.2 <= elapsed < 10.2:
                    a=math.sin(2*math.pi*(elapsed-8.2)/2)
                    xyz=[origin[0]+.01*a,origin[1]+.015*a,origin[2]+.01*a]
                elif 10.2 <= elapsed < 13.2:
                    xyz[1]+=.64*(elapsed-10.2)/3
                elif 13.2 <= elapsed < 16.2:
                    xyz[1]+=.64*(16.2-elapsed)/3
                q=multiply(hand_anchor,[math.cos(roll/2),0,0,math.sin(roll/2)])
                status={'server_connected':True,'device_connected':True,'latest':{
                    'sequence':seq,'receive_monotonic_ns':time.monotonic_ns(),
                    'app_focus':True,'input_mode':1,'head':{'status':3,'pose':[0,0,0,0,0,0,1]},'right':{
                    'grip':0. if elapsed<.2 else 1.,'pose':xyz+q[1:]+q[:1]}}}
            report=control.tick(status,lambda:status,time.monotonic_ns())
            state=report['mapping']['state'];states[state]+=1
            if state in ('CONTROL_GAP','INPUT_JUMP','ROTATION_WRAP','WAIT_FRESH_INPUT','RELEASE_GRIP_TO_ARM'):
                raise AssertionError('Unexpected interruption: '+state)
            if .2 <= elapsed < 8.2 and state=='TRACKING':
                mapping=report['mapping'];joints=mapping['target_joints_deg']
                coupling=max(abs(a-b) for a,b in zip(joints[:5],backend.initial_joints[:5]))
                peak_roll_coupling=max(peak_roll_coupling,coupling)
                body=quat(multiply(inverse(control.mapper.anchor_robot[3:]),mapping['target_pose_wxyz'][3:]))
                expected=math.degrees(2*math.atan2(body[3],body[0]))
                peak_roll_error=max(peak_roll_error,abs(joints[5]-backend.initial_joints[5]-expected))
                peak_roll_translation=max(peak_roll_translation,norm([
                    a-b for a,b in zip(mapping['target_pose_wxyz'][:3],control.mapper.anchor_robot[:3])])*1000)
            if not trace or elapsed-trace[-1]['elapsed_s']>=.1 or state!=trace[-1]['mapping']['state']:
                trace.append({'elapsed_s':elapsed,**report})
        if not states['SESSION_BOUND'] or not states['RESUMED']:
            raise AssertionError('Soft limit/recovery was not exercised')
        if peak_roll_coupling > .02 or peak_roll_error > .02 or peak_roll_translation > .05:
            raise AssertionError('Pure roll incorrectly moved other joints or TCP position')
        if backend.attempted or control.send_count:
            raise AssertionError('Verification unexpectedly attempted motion')
        actual_change=max(abs(a-b) for a,b in zip(backend.joints(require_idle=True),backend.initial_joints))
        if actual_change>.05:raise AssertionError('Arm moved during preview')
        result={'synthetic_input':True,'motion_commands_sent':0,'config':config,
            'ticks':sum(states.values()),'states':dict(states),'actual_joint_change_deg':actual_change,
            'control_interval_ms':{'p50':percentile(intervals,.5),'p95':percentile(intervals,.95),
                'p99':percentile(intervals,.99),'max':max(intervals)},
            'pure_roll_other_joints_peak_deg':peak_roll_coupling,
            'pure_roll_joint6_error_deg':peak_roll_error,'pure_roll_tcp_translation_mm':peak_roll_translation,
            'observed_real_pico_hz_range':[min(actual_pico_rates),max(actual_pico_rates)]}
        (TASK_DIR/'reports'/('control_%d_native_verified_20261012.json'%config['control_hz'])).write_text(json.dumps(result,indent=2)+'\n')
        (TASK_DIR/'state/follow-fix-synthetic-preview.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in trace))
        print(json.dumps(result),flush=True)
    finally:
        if monitor:monitor.close()
        if backend:backend.close()
        lock.close()


if __name__=='__main__':main()
