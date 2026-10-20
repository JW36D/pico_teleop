"""Shared single-arm 6DoF incremental teleoperation; default is right preview."""
import argparse
import collections
import fcntl
import math
from pathlib import Path
import signal
import sys
import time

from runtime_paths import LOCK_DIR, CONFIG_DIR
from incremental import MappingError, angle, inverse, multiply, norm, quat, vector
from robot_backend import Backend, read_pico, TASK_DIR, ARM_IPS, check_known_controller_processes, checked_pair
from robot_feedback import FeedbackMonitor
from robot_directions import resolve_translation_mapping, work_delta_to_body
from teleop_config import load_config
from teleop_controller import TeleopController
from control_sample import control_sample
from diagnostic_logger import AsyncDiagnosticLogger
from diagnostic_context import DiagnosticContext
from operator_reference import SharedOperatorReference
from gripper_control import GripperWorker
from button_events import ExitGuard
from session_stop import SessionStopFlag
from collection_ipc import Publisher


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[int((len(ordered)-1)*p)], 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_DIR/'right_rotation_teleop.json')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preview', action='store_true')
    modes.add_argument('--execute', action='store_true', help='Real motion, operator interactive terminal required')
    parser.add_argument('--duration', type=float, default=0., help='0 continuous; otherwise 1..3600 seconds')
    parser.add_argument('--operator-reference', type=Path, help='Session heading shared by dual-arm workers')
    parser.add_argument('--session-stop', type=Path, help='Shared exit latch, owned by the supervisor')
    parser.add_argument('--collection-socket', type=Path)
    parser.add_argument('--session-id')
    args = parser.parse_args()
    if sys.version_info < (3,9):
        parser.error('Python 3.9 or newer is required; set PICO_CONTROL_PYTHON to choose the interpreter')
    if not math.isfinite(args.duration) or not (args.duration==0 or 1<=args.duration<=3600):
        parser.error('Duration must be 0, or 1..3600 seconds')
    try:
        config = load_config(args.config)
    except (MappingError, ValueError, OSError) as exc:
        parser.error(str(exc))
    arm=config.get('arm','right')
    prefix='' if arm=='right' else 'left_'
    state_dir = TASK_DIR/'state'
    state_dir.mkdir(exist_ok=True)
    lock_file = (LOCK_DIR/'.single_joint_step.lock').open('a')
    arm_lock_file = (LOCK_DIR/('.pico_'+arm+'.lock')).open('a')
    try:
        # The two PICO workers share the global lock. Older motion/recovery
        # tools retain an exclusive lock and exclude both of these workers.
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH|fcntl.LOCK_NB)
        fcntl.flock(arm_lock_file.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        arm_lock_file.close()
        lock_file.close()
        print('Another '+arm+' control session or exclusive arm tool is running.', file=sys.stderr)
        return 1
    log_path = state_dir/(('full-session-' if arm=='right' else 'left-full-session-')+str(time.time_ns())+'.jsonl')
    diagnostic = AsyncDiagnosticLogger(log_path, state_dir/(prefix+'full_status.json'))
    (state_dir/('latest_'+prefix+'full_session.txt')).write_text(str(log_path)+'\n')
    stopped = False
    backend = monitor = controller = None
    gripper = None
    diagnostic_failed = False
    diagnostic_context = DiagnosticContext(window_s=2.,controller_side=arm)
    context_available = True
    outcome = 0
    collection_publisher = shared_stop = None
    exit_guard = ExitGuard()

    def exit_requested():
        return stopped or exit_guard.requested or bool(shared_stop and shared_stop.is_set())

    def session_input():
        status = read_pico()
        if exit_guard.observe(status) and shared_stop:
            shared_stop.set()
        status['session_exit_requested'] = exit_requested()
        return status

    def interrupted(*_):
        nonlocal stopped
        stopped = True
        if shared_stop:
            shared_stop.set()

    def record(event, critical=True, loop_start_ns=None):
        nonlocal diagnostic_failed
        event.setdefault('time_monotonic_ns',time.monotonic_ns())
        event.setdefault('mode','EXECUTE' if args.execute else 'PREVIEW_NO_MOTION')
        event.setdefault('arm',arm)
        event['motion_streaming_active'] = bool(controller and controller.streaming)
        event['diagnostic_output'] = diagnostic.stats()
        if diagnostic_failed:return
        try:
            diagnostic.append(event, critical=critical, loop_start_ns=loop_start_ns)
        except Exception as exc:
            diagnostic_failed=True
            diagnostic.console_message('诊断日志不可用，遥操继续：'+str(exc))

    def record_error(event):
        # A diagnostic filesystem error must not bypass motion cleanup merely
        # because the attempt to log that original error also fails.
        try:
            record(event)
        except Exception as exc:
            diagnostic.console_message('ERROR: %s; diagnostic output: %s' % (event.get('error'), exc))

    def record_context(reason, error=None):
        # Observing a fault must not create a new motion-stop condition or
        # perform filesystem work in the control loop.
        if not context_available or diagnostic_failed:
            return
        try:
            event=diagnostic_context.snapshot(reason,error=error)
            if event['sample_count']:
                record(event)
        except Exception as exc:
            diagnostic.console_message('故障诊断上下文不可用：'+str(exc))

    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        if args.session_stop:
            shared_stop = SessionStopFlag(args.session_stop)
        startup_input = session_input()  # event baseline before SDK startup
        if not startup_input.get('button_stream'):
            raise MappingError('Receiver lacks reliable button events; update receiver before teleoperation')
        check_known_controller_processes()
        backend = Backend() if arm=='right' else Backend(arm=arm)
        config, direction_reference = resolve_translation_mapping(config, backend.frames,
            [backend.installation[key] for key in ('x','y','z')])
        backend.stream_high_follow = config['follow_mode']=='high'
        backend.stream_trajectory_mode = config['stream_trajectory_mode']
        backend.stream_radio = config['stream_radio']
        backend.speed_limits = vector(checked_pair(backend.robot,'rm_get_joint_max_speed'),6)
        if any(v<=0 for v in backend.speed_limits):
            raise MappingError('Invalid hardware joint speed limits')
        monitor = FeedbackMonitor(ARM_IPS[arm], config,process_check=check_known_controller_processes)
        initial_feedback = monitor.start()
        if norm([a-b for a,b in zip(initial_feedback['position_m'],backend.initial_pose[:3])]) > .005:
            raise MappingError('Controller state and SDK FK differ by more than 5 mm')
        feedback_quat = quat(backend.robot.rm_algo_euler2quaternion(initial_feedback['orientation_rad']))
        if angle(multiply(feedback_quat,inverse(backend.initial_pose[3:]))) > .005:
            raise MappingError('Controller orientation and SDK FK differ by more than 0.005 rad')
        startup_ik = backend.validate_startup_ik()
        controller = TeleopController(config, backend, monitor, args.execute)
        controller.stop_requested = exit_requested
        if args.operator_reference is not None:
            controller.mapper.operator_reference = SharedOperatorReference(args.operator_reference).resolve
        if config.get('gripper', {}).get('enabled'):
            gripper = GripperWorker(ARM_IPS[arm], arm, config['gripper'], args.execute,
                config['max_input_age_ms'], notify=diagnostic.console_message)
            gripper.stop_requested = exit_requested
            gripper.start()
        if args.collection_socket:
            collection_publisher = Publisher(args.collection_socket, arm, args.session_id,
                metadata={'config':config, 'frames':backend.frames, 'installation_deg':backend.installation,
                    'motion_capabilities':getattr(backend,'motion_capabilities',None),
                    'initial_feedback':initial_feedback, 'initial_joints_deg':backend.initial_joints,
                    'tool_tcp_calibrated':None, 'tool_tcp_unknown_reason':'grasp TCP not measured'})
        record({'event':'START','mode':'EXECUTE' if args.execute else 'PREVIEW_NO_MOTION',
                'pid':__import__('os').getpid(),'config_path':str(args.config),'config':config,
                'duration_seconds':args.duration,'robot_joints_deg':backend.initial_joints,
                'hardware_joint_speed_limits_deg_s':backend.speed_limits,
                'robot_pose_wxyz':backend.initial_pose,'frames':backend.frames,
                'installation_deg':backend.installation,'algorithm_installation_deg':backend.algorithm_installation,
                'log_path':str(log_path),'follow_mode':config['follow_mode'],
                'rotation_mapping':config['rotation_mapping'],
                'operator_reference_file':str(args.operator_reference) if args.operator_reference else None,
                'diagnostic_context':{'window_s':2.,'storage':'bounded_ram',
                    'write_trigger':'passive_clutch_loss_or_exception',
                    'training_recording':False},
                'initial_feedback_metadata':{k:v for k,v in initial_feedback.items()
                    if k not in ('joints','position_m','orientation_rad')},
                'direction_reference':direction_reference,
                'startup_ik_validation':startup_ik,
                'motion_capabilities':getattr(backend,'motion_capabilities',None),
                'constraint_status':{
                    'startup_pose_limits_enabled':config.get('session_pose_limits_enabled',True),
                    'startup_joint_limits_enabled':config.get('joint_session_limits_enabled',True),
                    'fixed_rotation_limits_enabled':config.get('clutch_rotation_limits_enabled',True),
                    'task_workspace':config.get('task_workspace'),
                    'joint_trajectory_enabled':config.get('joint_trajectory_enabled',False),
                    'joint_solver':'coupled_cartesian_box_qp' if config.get('joint_trajectory_enabled') else 'continuous_ik',
                    'cartesian_rotation_weight_m':config.get('cartesian_rotation_weight_m'),
                    'max_hand_linear_speed_m_s':config.get('max_hand_linear_speed_m_s'),
                    'max_hand_angular_speed_rad_s':config.get('max_hand_angular_speed_rad_s'),
                    'fresh_input_ms':config.get('max_input_age_ms'),
                    'input_stop_ms':config.get('max_input_gap_ms') if config.get('continuous_input_recovery') else config.get('max_input_age_ms'),
                    'timing_stop_s':config.get('max_recoverable_control_gap_s') if config.get('continuous_timing_recovery') else config.get('max_control_gap_s'),
                    'feedback_stop_ms':config.get('max_feedback_age_ms'),
                    'tracking_error_m':config.get('max_tracking_error_m'),
                    'tracking_error_rad':config.get('max_tracking_angle_rad'),
                    'tracking_error_grace_s':config.get('tracking_error_grace_s',0.),
                    'stationary_required_for_grip':config.get('require_robot_stationary_to_anchor',True)}})
        collision=(getattr(backend,'motion_capabilities',{}) or {}).get('self_collision',{})
        if collision.get('enabled') is False:
            diagnostic.console_message('设备状态：本机自碰撞检测关闭；当前任务工作区未标定。')
        start = time.monotonic()
        period = 1/config['control_hz']
        next_tick = start
        next_log = start
        intervals = collections.deque(maxlen=200)
        send_intervals = collections.deque(maxlen=200)
        previous_send_ns = None
        previous_tick = None
        previous_state = None
        previous_stage_timing = None
        ticks = skipped_slots = 0
        end_reason = 'DURATION_ELAPSED'
        while not stopped and (args.duration==0 or time.monotonic()-start<args.duration):
            # 固定周期读取最新输入、推进连续目标；不按 PICO 回调频率直接触发运动。
            wait = next_tick-time.monotonic()
            if wait>0:
                time.sleep(wait)
            tick_start = time.monotonic()
            next_tick += period
            if tick_start-next_tick>period:
                skipped_slots += int((tick_start-next_tick)/period)+1
                next_tick = tick_start+period  # No catch-up burst of stale poses.
            if previous_tick is not None:
                intervals.append((tick_start-previous_tick)*1000)
            previous_tick = tick_start
            if stopped:
                break
            loop_start_ns = time.monotonic_ns()
            if not diagnostic_failed:
                try:
                    diagnostic.check_error()
                except MappingError as exc:
                    diagnostic_failed=True
                    diagnostic.console_message('诊断日志不可用，遥操继续：'+str(exc))
            input_start_ns = time.monotonic_ns()
            status = session_input()
            latest_control_input = status
            input_end_ns = time.monotonic_ns()
            # The callback reads stop state again immediately before every sender invocation.
            def gated_input():
                nonlocal latest_control_input
                latest_control_input = {'session_exit_requested':True} if exit_requested() else session_input()
                return latest_control_input
            report = controller.tick(status, gated_input, time.monotonic_ns())
            if gripper:
                report['gripper'] = gripper.update(latest_control_input,
                    bool(controller.mapper.active) and not exit_requested(), time.monotonic_ns())
            controller_end_ns = time.monotonic_ns()
            if config['translation_mapping_frame']=='robot_body':
                mapping = report['mapping']
                requested = mapping.get('requested_delta_work_m')
                if requested is not None:
                    mapping['requested_delta_robot_m'] = work_delta_to_body(requested,direction_reference)
                report['robot_feedback_delta_robot_m'] = work_delta_to_body(
                    [a-b for a,b in zip(report['robot_feedback_pose_wxyz'][:3],backend.initial_pose[:3])],
                    direction_reference)
            ticks += 1
            report.update(pico_receive_hz=status['receive_hz_2s'],pico_age_ms=status['age_ms'],
                mode='EXECUTE' if args.execute else 'PREVIEW_NO_MOTION',arm=arm,
                time_monotonic_ns=time.monotonic_ns(),control_ticks=ticks,
                elapsed_s=round(tick_start-start,3),skipped_schedule_slots=skipped_slots,
                control_interval_ms={'p50':percentile(intervals,.5),'p95':percentile(intervals,.95),
                                     'p99':percentile(intervals,.99)})
            current_state = report['mapping']['state']
            if report.get('motion_command_sent_this_tick'):
                sent_ns=(report.get('last_actually_sent_command') or {}).get('send_monotonic_ns')
                if type(sent_ns) is int and previous_send_ns is not None and sent_ns>previous_send_ns:
                    send_intervals.append((sent_ns-previous_send_ns)/1e6)
                previous_send_ns=sent_ns if type(sent_ns) is int else None
            elif not getattr(controller,'streaming',False) or not bool(
                    getattr(getattr(controller,'mapper',None),'active',False)):
                previous_send_ns=None
            if collection_publisher:
                sample = control_sample(status,report,controller)
                sample.update(kind='telemetry', receive_monotonic_ns=sample['control_monotonic_ns'],
                    control_interval_ms=report['control_interval_ms'])
                collection_publisher.submit(sample)
            if context_available:
                try:
                    passive_loss=diagnostic_context.capture(status,report,
                        bool(getattr(getattr(controller,'mapper',None),'active',False)))
                    if passive_loss and not stopped:
                        record_context(passive_loss)
                except Exception as exc:
                    context_available=False
                    diagnostic.console_message('故障诊断缓存不可用，遥操继续：'+str(exc))
            report_end_ns = time.monotonic_ns()
            report['stage_timing_ms'] = {
                'read_input':round((input_end_ns-input_start_ns)/1e6,3),
                'controller':round((controller_end_ns-input_end_ns)/1e6,3),
                'report':round((report_end_ns-controller_end_ns)/1e6,3),
                'diagnostic_enqueue':0.,
                'loop':round((report_end_ns-loop_start_ns)/1e6,3)}
            report['previous_stage_timing_ms'] = previous_stage_timing
            if tick_start>=next_log or current_state!=previous_state:
                report['actual_send_interval_ms']={
                    'p50':percentile(send_intervals,.5),'p95':percentile(send_intervals,.95),
                    'p99':percentile(send_intervals,.99),
                    'max':max(send_intervals) if send_intervals else None,
                    'sample_count':len(send_intervals),
                    'semantics':'successful SDK call starts; continuous streaming only; last 200 intervals'}
                critical = current_state!=previous_state and current_state not in ('TRACKING','FOLLOWER_WAIT')
                record(report,critical=critical,loop_start_ns=loop_start_ns)
                next_log = tick_start+config['log_interval_s']
            previous_state = current_state
            previous_stage_timing = report['stage_timing_ms']
            if current_state == 'SESSION_EXIT' or exit_requested():
                end_reason = 'B_Y_CHORD' if not stopped else 'SIGNAL'; break
        if stopped:
            end_reason = 'SIGNAL'
        if gripper:
            gripper.cancel()
        controller.stop()
        record({'event':'SESSION_END','reason':end_reason,
                'mode':'EXECUTE' if args.execute else 'PREVIEW_NO_MOTION',
                'motion_command_count':controller.send_count,'control_ticks':ticks,
                'actual_send_interval_ms':{'p50':percentile(send_intervals,.5),
                    'p95':percentile(send_intervals,.95),'p99':percentile(send_intervals,.99),
                    'max':max(send_intervals) if send_intervals else None,
                    'sample_count':len(send_intervals),
                    'semantics':'successful SDK call starts; continuous streaming only; last 200 intervals'},
                'control_interval_ms':{'p50':percentile(intervals,.5),'p95':percentile(intervals,.95),
                                       'p99':percentile(intervals,.99)}})
    except Exception as exc:
        outcome = 1
        # Stop before any shutdown waits or attempts to report a fatal error.
        if gripper:
            gripper.cancel()
        if controller:
            try:
                controller.stop()
            except Exception as stop_exc:
                diagnostic.console_message('Stop failed: ' + str(stop_exc))
        record_context('CONTROL_EXCEPTION',error={'message':str(exc),
            'details':getattr(exc,'details',{})})
        record_error({'event':'ERROR','error':str(exc),'details':getattr(exc,'details',{}),
                'mode':'EXECUTE' if args.execute else 'PREVIEW_NO_MOTION'})
    finally:
        if gripper:
            gripper.cancel()
        # Stop commands are sent only if this process actually attempted motion.
        try:
            try:
                if controller:
                    controller.stop()
            finally:
                if backend:
                    try:
                        backend.close()
                    finally:
                        if controller and not backend.attempted:
                            controller.streaming = False
        except Exception as exc:
            outcome = 1
            record_error({'event':'STOP_NOT_CONFIRMED' if backend and backend.attempted else 'ERROR',
                    'phase':'cleanup','error':str(exc)})
        finally:
            try:
                for resource, name in ((gripper,'gripper'),(monitor,'feedback'),
                        (collection_publisher,'collection_publisher'),(shared_stop,'session_stop')):
                    if resource:
                        try:
                            resource.close()
                        except Exception as exc:
                            outcome = 1
                            record_error({'event':'ERROR','phase':name+'_cleanup','error':str(exc)})
            finally:
                try:
                    diagnostic_result = diagnostic.close(timeout=5.)
                    if not diagnostic_result['complete']:
                        outcome = 1
                finally:
                    arm_lock_file.close()
                    lock_file.close()
    return outcome


if __name__=='__main__':
    raise SystemExit(main())
