"""左右臂恢复各自调试前的关节姿态；默认离线，--plan 只读，--execute 实机。

不读取 PICO，不启动遥操，不镜像另一臂；一次只恢复指定臂，使用低速分段 MoveJ。
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

from runtime_paths import LOCK_DIR, PROJECT_DIR
from incremental import MappingError, angle, inverse, multiply, norm, pose, vector
from recovery_guard import (SPEED_PERCENT, check_limits, check_start, execute_recovery,
                            trajectory_type, waypoints, FEEDBACK_LIMIT_TOLERANCE_DEG)

ROOT = Path(__file__).resolve().parents[2]
INITIAL = ROOT/'configs/right_initial_pose.json'
PLAN = ROOT/'state/right_recovery_plan.json'
INITIALS = {arm: ROOT/('configs/%s_initial_pose.json' % arm) for arm in ('right', 'left')}


def load_initial(path=None, *, arm='right'):
    if arm not in INITIALS:
        raise MappingError('Choose recovery arm left or right')
    path = INITIALS[arm] if path is None else Path(path)
    document = json.loads(path.read_text())
    if document.get('arm') != arm or document.get('name') != arm+'_before_pico_teleop':
        raise MappingError('Expected the recorded pre-teleop '+arm+'-arm pose')
    document['joints_deg'] = vector(document['joints_deg'], 6)
    document['pose_wxyz'] = pose(document['pose_wxyz'])
    document['installation_deg'] = vector(document['installation_deg'], 3)
    document['file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return document


def validate_reviewed(plan, initial):
    if plan.get('schema') != 1 or plan.get('arm') != initial['arm']:
        raise MappingError('Invalid '+initial['arm']+'-arm recovery plan')
    if plan.get('initial_file_sha256') != initial['file_sha256']:
        raise MappingError('Initial pose file changed; run --plan again')
    if vector(plan.get('target_joints_deg'), 6) != initial['joints_deg']:
        raise MappingError('Recovery target differs from the recorded initial pose')
    if type(plan.get('speed_percent')) is not int or plan['speed_percent'] != SPEED_PERCENT:
        raise MappingError('Recovery speed must remain 1 percent')
    points = waypoints(plan['start_joints_deg'], plan['target_joints_deg'],
                       plan['minimum_deg'], plan['maximum_deg'])
    if plan.get('waypoints_deg') != points:
        raise MappingError('Recovery waypoints changed; run --plan again')


def check_context(backend, initial, feedback):
    if backend.frames != initial['frames']:
        raise MappingError('Work/tool frames differ from the saved initial pose')
    actual_installation = [backend.installation[k] for k in ('x', 'y', 'z')]
    if max(abs(a-b) for a, b in zip(actual_installation, initial['installation_deg'])) > .1:
        raise MappingError('Installation differs from the recorded model')
    current_pose = backend.forward(feedback['joints'])
    if norm([a-b for a, b in zip(current_pose[:3], feedback['position_m'])]) > .005:
        raise MappingError('Controller and SDK position disagree')
    # 欧拉角转换是本地 SDK 数学接口，没有设置机器人状态。
    actual_q = backend.robot.rm_algo_euler2quaternion(feedback['orientation_rad'])
    if angle(multiply(actual_q, inverse(current_pose[3:]))) > .005:
        raise MappingError('Controller and SDK orientation disagree')
    saved_pose = backend.forward(initial['joints_deg'])
    if norm([a-b for a, b in zip(saved_pose[:3], initial['pose_wxyz'][:3])]) > .001:
        raise MappingError('Saved initial position no longer matches SDK model')
    if angle(multiply(saved_pose[3:], inverse(initial['pose_wxyz'][3:]))) > .005:
        raise MappingError('Saved initial orientation no longer matches SDK model')
    return current_pose, saved_pose


def build_plan(backend, monitor, initial, other_reference=None):
    feedback = monitor.snapshot()
    backend.joints(require_idle=True)
    current_pose, saved_pose = check_context(backend, initial, feedback)
    start = feedback['joints']
    points = waypoints(start, initial['joints_deg'], backend.minimum, backend.maximum)
    # 以下只是关节空间等分的正解采样，并不具备机器人整机碰撞模型。
    samples = []
    for k in range(101):
        joint = [a+(b-a)*k/100 for a, b in zip(start, initial['joints_deg'])]
        samples.append({'fraction': k/100, 'pose_wxyz': backend.forward(joint)})
    result = {'schema': 1, 'arm': initial['arm'], 'created_unix_ns': time.time_ns(),
              'initial_file_sha256': initial['file_sha256'], 'source': initial['source'],
              'start_joints_deg': start, 'target_joints_deg': initial['joints_deg'],
              'minimum_deg': backend.minimum, 'maximum_deg': backend.maximum,
              'waypoints_deg': points, 'speed_percent': SPEED_PERCENT,
              'frames': backend.frames, 'installation_deg': initial['installation_deg'],
              'start_pose_wxyz': current_pose, 'target_pose_wxyz': saved_pose,
              'TCP_delta_work_mm': [(b-a)*1000 for a, b in zip(current_pose[:3], saved_pose[:3])],
              'orientation_change_deg': math.degrees(
                  angle(multiply(saved_pose[3:], inverse(current_pose[3:])))),
              'nominal_FK_samples': samples, 'collision_check_performed': False,
              'other_arm_reference_read_only': other_reference, 'motion_sent': False,
              'feedback_limit_tolerance_deg': FEEDBACK_LIMIT_TOLERANCE_DEG,
              'feedback_limit_overshoot_deg': [max(low-q, q-high, 0.) for q, low, high
                                             in zip(start, backend.minimum, backend.maximum)],
              'max_joint_change_deg': max(abs(a-b) for a,b in zip(start, initial['joints_deg'])),
              'path_type': 'joint_linear_segments_no_angle_wrapping',
              'total_excursion_cap_deg': None}
    check_start(backend.joints(require_idle=True), start)
    return result


def connect_backend(arm):
    """Keep ordinary teleop checks unchanged; tolerate only recovery feedback noise."""
    from robot_backend import Backend

    class RecoveryNativeBackend(Backend):
        def validate_joint_feedback(self, joints):
            return check_limits(joints, self.minimum, self.maximum, feedback=True)

    return RecoveryNativeBackend(arm=arm)


class RecoveryBackend:
    def __init__(self, native, monitor):
        self.native, self.monitor = native, monitor

    def sample(self):
        # 使用独立只读连接的编码器反馈，不拿命令角度当实际到位。
        trajectory = self.native.robot.rm_get_arm_current_trajectory()
        if trajectory.get('return_code') != 0:
            raise MappingError('Cannot read recovery trajectory state')
        feedback = self.monitor.snapshot()
        trajectory_type(trajectory.get('trajectory_type'))
        return {'joints': feedback['joints'], 'trajectory': trajectory['trajectory_type']}

    def movej(self, target, speed):
        self.native.attempted = True
        return self.native.robot.rm_movej(target, speed, 0, 0, 0)

    def stop(self):
        self.native.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('right', 'left'), default='right')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--offline', action='store_true', help='Default: print recorded target without connecting')
    mode.add_argument('--plan', action='store_true', help='Read actual arm state and save a no-motion plan')
    mode.add_argument('--execute', action='store_true', help='Execute an unchanged plan from an operator SSH terminal')
    parser.add_argument('--plan-file', type=Path, help='Default: state/<arm>_recovery_plan.json')
    args = parser.parse_args(argv)
    args.plan_file = args.plan_file or ROOT/('state/%s_recovery_plan.json' % args.arm)
    native = monitor = lock = log = None
    cancelled = False
    old_handlers = {}

    def record(event):
        event = {**event, 'arm': args.arm, 'time_monotonic_ns': time.monotonic_ns()}
        encoded = json.dumps(event, ensure_ascii=False, allow_nan=False)
        print(encoded, flush=True)
        if log:
            log.write(encoded+'\n')

    def interrupted(*_):
        nonlocal cancelled
        cancelled = True

    try:
        initial = load_initial(arm=args.arm)
        if not args.plan and not args.execute:
            record({'event': 'OFFLINE', 'connected': False, 'motion_sent': False,
                    'target_joints_deg': initial['joints_deg'], 'source': initial['source'],
                    'speed_percent': SPEED_PERCENT,
                    'next_step': '--plan after power-on; --execute uses that unchanged plan'})
            return 0
        if sys.version_info < (3, 9):
            raise MappingError('Use Python 3.9 or newer; set PICO_CONTROL_PYTHON to choose the interpreter')
        if args.execute and not sys.stdin.isatty():
            raise MappingError('Execute requires an operator interactive SSH terminal')
        reviewed = None
        if args.execute:
            reviewed = json.loads(args.plan_file.read_text())
            validate_reviewed(reviewed, initial)
        # 本地默认离线及单元测试均不会导入这些硬件依赖。
        import fcntl
        from robot_backend import ARM_IPS, check_known_controller_processes
        from robot_feedback import FeedbackMonitor, query
        lock = (LOCK_DIR/'.single_joint_step.lock').open('a')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_known_controller_processes()
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            old_handlers[sig] = signal.signal(sig, interrupted)
        # 开机/报警/坐标系变化全部拒绝；本脚本不会使能、清错或设置坐标系。
        monitor = FeedbackMonitor(ARM_IPS[args.arm], {
            'feedback_hz': 10., 'feedback_timeout_s': .25,
            'max_feedback_age_ms': 400., 'stationary_tolerance_deg': .03},
            process_check=check_known_controller_processes)
        monitor.start()
        native = connect_backend(args.arm)
        if cancelled:
            raise KeyboardInterrupt('Recovery cancelled during initialization')
        if args.plan:
            # 另一臂仅作现场记录，不能复制它的关节角或把它当碰撞检查。
            other_arm = 'left' if args.arm == 'right' else 'right'
            other = None
            try:
                other = query(ARM_IPS[other_arm], 'get_current_arm_state', .25)
            except (MappingError, OSError, ValueError) as exc:
                other = {'unavailable': str(exc)}
            result = build_plan(native, monitor, initial, other)
            args.plan_file.parent.mkdir(parents=True, exist_ok=True)
            temp = args.plan_file.with_name(args.plan_file.name+'.tmp')
            temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
            temp.replace(args.plan_file)
            record({k: v for k, v in result.items() if k != 'nominal_FK_samples'})
            print('READ-ONLY PLAN: no motion. %s arm, %d segments, largest joint change %.3f deg.' %
                  (args.arm, len(result['waypoints_deg']), result['max_joint_change_deg']), flush=True)
            print('Check the full arm/gripper sweep and keep the other arm clear before --execute.', flush=True)
            return 0
        feedback = monitor.snapshot()
        check_context(native, initial, feedback)
        check_start(native.joints(require_idle=True), reviewed['start_joints_deg'])
        if native.minimum != reviewed['minimum_deg'] or native.maximum != reviewed['maximum_deg']:
            raise MappingError('Hardware limits changed since plan; run --plan again')
        if native.frames != reviewed['frames'] or initial['installation_deg'] != reviewed['installation_deg']:
            raise MappingError('Recovery context changed since plan')
        state = ROOT/'state'
        state.mkdir(exist_ok=True)
        log_path = state/('%s-recovery-%d.jsonl' % (args.arm, time.time_ns()))
        log = log_path.open('a', buffering=1)
        (state/('latest_%s_recovery.txt' % args.arm)).write_text(str(log_path)+'\n')
        record({'event': 'RECOVERY_START', 'mode': 'EXECUTE', 'plan_file': str(args.plan_file),
                'log_file': str(log_path), 'target_joints_deg': initial['joints_deg'],
                'segments': len(reviewed['waypoints_deg']), 'speed_percent': SPEED_PERCENT})
        execute_recovery(RecoveryBackend(native, monitor), reviewed['start_joints_deg'],
                         initial['joints_deg'], native.minimum, native.maximum,
                         record=record, cancelled=lambda: cancelled)
        # 整段完成后仍发送一次受控停止，清理本进程的运动状态；不会关闭电源。
        native.stop()
        return 0
    except KeyboardInterrupt:
        record({'event': 'CANCELLED', 'motion_active': bool(native and native.attempted)})
        return 130
    except (MappingError, OSError, ValueError, KeyError, ImportError) as exc:
        record({'event': 'ERROR', 'error': str(exc)})
        return 1
    finally:
        # 即使执行中断/反馈报错也要尝试停机，再释放 SDK 和共享锁。
        if native:
            try:
                native.close()
            except BaseException as exc:
                record({'event': 'CLEANUP_ERROR', 'error': str(exc)})
        if monitor:
            monitor.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if log:
            log.close()
        if lock:
            lock.close()


if __name__ == '__main__':
    sys.exit(main())
