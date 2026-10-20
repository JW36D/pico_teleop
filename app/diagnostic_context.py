"""A bounded RAM history for fault diagnosis, independent of motion decisions.

Capture is deliberately separate from logging. Nothing is written until the
caller asks for a snapshot and submits it to its diagnostic logger. These are
control-cycle observations, not a training trajectory or independent encoder
samples at the control frequency.
"""
from collections import deque
import copy
import math
import time


_MOTION_KEYS = (
    'input_health', 'gate_details', 'boundary_reason', 'rotation_goal_reachable',
    'rotation_goal_limit_reason', 'angular_remaining_rad',
    'rotation_path_pending_segments', 'input_frame_reused', 'input_sample_dt_s',
    'linear_governor_ratio', 'angular_governor_ratio', 'continuous_slowdown',
    'continuous_linear_lead_m', 'continuous_angular_lead_rad',
    'aligned_tracking_error_mm', 'aligned_tracking_error_deg',
    'tracking_comparison_time_ns', 'tracking_error_pending',
    'tracking_error_duration_s', 'command_tracking_error_mm',
    'command_tracking_error_deg', 'requested_orientation_error_deg',
    'linear_velocity_m_s', 'angular_velocity_rad_s',
    'emitted_linear_velocity_m_s', 'emitted_angular_velocity_rad_s',
    'emitted_linear_accel_m_s2', 'emitted_angular_accel_rad_s2',
    'joint_command_dt_s', 'joint_trajectory_sample_monotonic_ns',
    'joint_speed_limited', 'joint_acceleration_limited',
    'joint_position_braking', 'joint_rate_limited',
    'cartesian_requested_velocity_m_s_rad_s',
    'cartesian_achieved_velocity_m_s_rad_s',
    'cartesian_velocity_residual_m_s_rad_s',
    'cartesian_linear_velocity_residual_m_s',
    'cartesian_angular_velocity_residual_rad_s',
    'cartesian_weighted_velocity_residual_m_s',
    'joint_optimizer_converged', 'joint_optimizer_kkt_residual',
    'joint_optimizer_iterations', 'final_joint_calculation_ms', 'send_call_ms',
)


def _identity(sequence, stamp):
    return None if sequence is None and stamp is None else (sequence, stamp)


class DiagnosticContext:
    """Keep at most ``window_s`` seconds and ``max_samples`` control ticks.

    ``capture`` returns an informational passive-disarm event, or None. The
    caller alone decides whether to log it; this class has no device access,
    control callbacks, motion configuration, or motion-side effects.
    """

    def __init__(self, window_s=2.0, max_samples=512, controller_side='right'):
        if controller_side not in ('left','right'):
            raise ValueError('controller_side must be left or right')
        self.controller_side=controller_side
        if isinstance(window_s, bool) or not isinstance(window_s, (int, float)) or not math.isfinite(window_s) or window_s <= 0:
            raise ValueError('window_s must be finite and positive')
        if type(max_samples) is not int or max_samples < 1:
            raise ValueError('max_samples must be a positive integer')
        self.window_s = float(window_s)
        self.max_samples = max_samples
        self._window_ns = round(window_s*1e9)
        self._samples = deque()
        self._previous_active = False
        self._previous_state = None
        self._capacity_evictions = 0
        self._out_of_order_captures = 0

    def capture(self, status, report, controller_active):
        """Copy selected diagnostic fields; never retain caller-owned lists.

        Report time is the host control/report timestamp. Repeated feedback and
        PICO identities are retained and labelled instead of being interpolated
        or presented as new measurements. A duplicate control tick replaces its
        prior observation; an out-of-order host timestamp is ignored.
        """
        status = status if isinstance(status, dict) else {}
        report = report if isinstance(report, dict) else {}
        mapping = report.get('mapping') or {}
        latest = status.get('latest') or {}
        right = latest.get(self.controller_side) or {}
        metadata = report.get('robot_feedback_metadata') or {}
        state = mapping.get('state')
        active = bool(controller_active)
        stamp = report.get('time_monotonic_ns')
        if type(stamp) is not int:
            stamp = time.monotonic_ns()
        tick = report.get('control_ticks', report.get('control_tick'))
        if self._samples and stamp < self._samples[-1]['control_monotonic_ns']:
            self._out_of_order_captures += 1
            return None
        same_tick = bool(self._samples and tick is not None and self._samples[-1]['control_tick'] == tick)
        previous = self._samples[-2] if same_tick and len(self._samples) > 1 else (
            None if same_tick or not self._samples else self._samples[-1])
        feedback_sequence = report.get('robot_feedback_sequence', metadata.get('sequence'))
        feedback_receive = metadata.get('receive_monotonic_ns', report.get('robot_feedback_monotonic_ns'))
        feedback_id = _identity(feedback_sequence, feedback_receive)
        pico_id = _identity(latest.get('sequence'), latest.get('receive_monotonic_ns'))
        old_feedback_id = _identity(previous['feedback']['sequence'], previous['feedback']['receive_monotonic_ns']) if previous else None
        old_pico_id = _identity(previous['pico']['sequence'], previous['pico']['receive_monotonic_ns']) if previous else None
        sample = {
            'control_tick': tick, 'control_monotonic_ns': stamp,
            'state': state, 'controller_active': active,
            'arm':self.controller_side,'controller_side':self.controller_side,
            'episode_id': report.get('episode_id', report.get('episode')),
            'segment_id': report.get('segment_id', report.get('segment')),
            'motion_sent_this_tick': report.get('motion_command_sent_this_tick'),
            'send_count': report.get('send_count'),
            'feedback': {
                'sequence': feedback_sequence,
                'request_monotonic_ns': metadata.get('request_monotonic_ns', report.get('robot_feedback_request_monotonic_ns')),
                'receive_monotonic_ns': feedback_receive,
                'robot_sample_timestamp_ns': metadata.get('robot_sample_timestamp_ns'),
                'source': metadata.get('source', report.get('robot_feedback_source')),
                'sample_time_semantics': metadata.get('sample_time_semantics', 'local_receive_time_robot_clock_unavailable'),
                'age_ms': report.get('robot_feedback_age_ms'),
                'joints_deg': report.get('robot_feedback_joints_deg'),
                'pose_wxyz': report.get('robot_feedback_pose_wxyz'),
                'joint_velocity_deg_s': metadata.get('joint_velocity_deg_s', report.get('robot_joint_velocity_deg_s')),
                'velocity_interval_s': metadata.get('velocity_interval_s', report.get('robot_velocity_interval_s')),
                'reused_from_previous_control_tick': None if feedback_id is None or old_feedback_id is None else feedback_id == old_feedback_id,
            },
            'pico': {
                'sequence': latest.get('sequence'),
                'receive_monotonic_ns': latest.get('receive_monotonic_ns'),
                'pico_timestamp_ns': latest.get('pico_timestamp_ns'),
                'age_ms_at_tick_input': status.get('age_ms'),
                'grip': right.get('grip'), 'secondary_button': right.get('secondaryButton'),
                'pose': right.get('pose'), 'controller_status': right.get('status'),
                'app_focus': latest.get('app_focus'),
                'reused_from_previous_control_tick': None if pico_id is None or old_pico_id is None else pico_id == old_pico_id,
            },
            'last_actually_sent_command': report.get('last_actually_sent_command'),
            'candidate': {
                'joints_deg': mapping.get('target_joints_deg'),
                'pose_wxyz': mapping.get('target_pose_wxyz'),
                'joint_velocity_deg_s': mapping.get('joint_velocity_deg_s'),
                'joint_acceleration_deg_s2': mapping.get('joint_acceleration_deg_s2'),
            },
            'requested_pose_wxyz': mapping.get('requested_pose_wxyz'),
            'diagnostics': {key: mapping[key] for key in _MOTION_KEYS if key in mapping},
        }
        # Only the selected small structure is copied, never the entire report.
        sample = copy.deepcopy(sample)
        if same_tick:
            self._samples[-1] = sample
        else:
            self._samples.append(sample)
        cutoff = stamp-self._window_ns
        while self._samples and self._samples[0]['control_monotonic_ns'] < cutoff:
            self._samples.popleft()
        while len(self._samples) > self.max_samples:
            self._samples.popleft()
            self._capacity_evictions += 1

        event = None
        if self._previous_active and not active:
            gate = mapping.get('gate_details') or {}
            grip = gate.get('grip')
            grip_released = state == 'INPUT_GATE_CLOSED' and type(grip) in (int, float) and grip <= .2
            if state not in ('RELEASED', 'SESSION_EXIT') and not grip_released:
                event = {'event': 'PASSIVE_CONTROL_DISARM', 'control_tick': tick,
                    'time_monotonic_ns': stamp, 'previous_state': self._previous_state,
                    'state': state, 'gate_details': copy.deepcopy(gate)}
        self._previous_active, self._previous_state = active, state
        return event

    def snapshot(self, reason, *, error=None):
        """Return a stable JSON-compatible copy, intended only for fault logs."""
        samples = list(self._samples)
        feedback_ids = {_identity(x['feedback']['sequence'], x['feedback']['receive_monotonic_ns']) for x in samples}
        pico_ids = {_identity(x['pico']['sequence'], x['pico']['receive_monotonic_ns']) for x in samples}
        feedback_ids.discard(None)
        pico_ids.discard(None)
        if isinstance(error, BaseException):
            error = {'error': str(error), 'details': getattr(error, 'details', None)}
        return copy.deepcopy({
            'event': 'FAULT_CONTEXT', 'schema': 'pico_realman_fault_context_v1',
            'arm':self.controller_side,'controller_side':self.controller_side,
            'diagnostic_only': True, 'training_recording': False,
            'reason': reason, 'error': error,
            'window_s': self.window_s, 'max_samples': self.max_samples,
            'sample_count': len(samples),
            'first_control_monotonic_ns': samples[0]['control_monotonic_ns'] if samples else None,
            'last_control_monotonic_ns': samples[-1]['control_monotonic_ns'] if samples else None,
            'distinct_feedback_observations': len(feedback_ids),
            'distinct_pico_frames': len(pico_ids),
            'capacity_evictions_total': self._capacity_evictions,
            'out_of_order_captures_ignored_total': self._out_of_order_captures,
            'time_semantics': {
                'control_monotonic_ns': 'host report timestamp; one row per control tick, not independent robot samples',
                'feedback': 'host TCP request/receive time; robot internal sample clock unavailable unless explicitly supplied',
                'feedback_velocity': 'derived from successive feedback receive times using velocity_interval_s',
                'trajectory_sample_monotonic_ns': 'host time used for command trajectory integration',
                'send_monotonic_ns': 'host timestamp immediately before SDK send call; not robot execution time',
                'candidate': 'current calculated candidate, which may not have been sent',
                'last_actually_sent_command': 'last command whose SDK send call returned successfully',
                'pico_pose': 'original protocol pose; quaternion xyzw, robot pose quaternion wxyz',
            },
            'samples': samples,
        })
