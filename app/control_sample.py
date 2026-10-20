"""Build collection telemetry with explicit feedback and command provenance."""
import copy
import time

# Annotation tolerances distinguish tiny arithmetic residuals from meaningful
# optimizer mismatch. They never change a motion command or stop condition.
CARTESIAN_RESIDUAL_ANNOTATION_TOLERANCES = {'linear_m_s':1e-6, 'angular_rad_s':1e-5}


def control_sample(status, report, controller):
    def intervention(value):
        return any(value) if isinstance(value, (list,tuple)) else bool(value)
    latest = status.get('latest') or {}
    side=report.get('arm',getattr(controller,'config',{}).get('arm','right'))
    right = latest.get(side) or {}
    mapping = report['mapping']
    state = mapping['state']
    feedback = report.get('robot_feedback_metadata') or {}
    actual_command = report.get('last_actually_sent_command')
    boundary = state in ('BOUNDARY_HOLD', 'BOUNDARY_LIMITED') or bool(mapping.get('boundary_reason'))
    linear_residual = mapping.get('cartesian_linear_velocity_residual_m_s')
    angular_residual = mapping.get('cartesian_angular_velocity_residual_rad_s')
    cartesian_reduced = any(isinstance(value,(int,float)) and not isinstance(value,bool) and value > tolerance
        for value,tolerance in ((linear_residual,CARTESIAN_RESIDUAL_ANNOTATION_TOLERANCES['linear_m_s']),
                               (angular_residual,CARTESIAN_RESIDUAL_ANNOTATION_TOLERANCES['angular_rad_s'])))
    input_gap = (('INPUT' in state and any(token in state for token in ('HOLD', 'GAP', 'RECOVER', 'TIMEOUT'))) or
                 any(token in state for token in ('INPUT_HOLD', 'INPUT_GAP', 'INPUT_RECOVER',
                                                'WAIT_FRESH_INPUT', 'RESUMED', 'PAUSED')) or
                 bool(mapping.get('input_hold')) or bool(mapping.get('input_recovered')))
    sample = {'control_monotonic_ns':report['time_monotonic_ns'],
        'wall_time_ns':time.time_ns(), 'control_tick':report['control_ticks'],
        'mode':report['mode'], 'state':state, 'clutch_active':controller.mapper.active,
        'arm':side,'controller_side':side,
        'motion_sent':report['motion_command_sent_this_tick'], 'send_count':controller.send_count,
        'pico_sequence':latest.get('sequence'), 'pico_monotonic_ns':latest.get('receive_monotonic_ns'),
        'pico_timestamp_ns':latest.get('pico_timestamp_ns'), 'pico_age_ms':status.get('age_ms'),
        'head':latest.get('head'), side+'_controller':right, 'input_mode':latest.get('input_mode'),
        'app_focus':latest.get('app_focus'),
        'robot_feedback_sequence':report.get('robot_feedback_sequence', feedback.get('sequence')),
        'robot_feedback_monotonic_ns':report.get('robot_feedback_monotonic_ns'),
        'robot_feedback_request_monotonic_ns':report.get('robot_feedback_request_monotonic_ns', feedback.get('request_monotonic_ns')),
        'robot_sample_timestamp_ns':feedback.get('robot_sample_timestamp_ns'),
        'robot_feedback_time_semantics':report.get('robot_feedback_sample_time_semantics',
            feedback.get('sample_time_semantics', 'local_receive_time_robot_clock_unavailable')),
        'robot_feedback_source':report.get('robot_feedback_source', feedback.get('source')),
        'robot_feedback_age_ms':report['robot_feedback_age_ms'],
        'robot_feedback_transport':feedback.get('transport'),
        'robot_configuration_checks':{
            'checked_monotonic_ns':feedback.get('configuration_checked_monotonic_ns'),
            'age_ms':feedback.get('configuration_check_age_ms'),
            'deadline_ms':feedback.get('configuration_check_deadline_ms'),
            'transport':feedback.get('configuration_transport')},
        'robot_joint_velocity_deg_s':report.get('robot_joint_velocity_deg_s', feedback.get('joint_velocity_deg_s')),
        'robot_velocity_interval_s':report.get('robot_velocity_interval_s', feedback.get('velocity_interval_s')),
        'robot_pose_wxyz':report['robot_feedback_pose_wxyz'],
        'robot_joints_deg':report['robot_feedback_joints_deg'],
        'requested_pose_wxyz':mapping.get('requested_pose_wxyz'),
        'raw_hand_intent':{'pose':right.get('pose'), 'grip':right.get('grip'), 'trigger':right.get('trigger')},
        'command_pose_wxyz':controller.last_target.copy(),
        'command_joints_deg':controller.last_joints.copy(),
        'candidate_joint_velocity_deg_s':mapping.get('joint_velocity_deg_s'),
        'candidate_joint_acceleration_deg_s2':mapping.get('joint_acceleration_deg_s2'),
        'motion_constraint_diagnostics':{key:copy.deepcopy(value) for key,value in mapping.items()
            if key.startswith(('joint_','cartesian_','effective_joint_','max_emitted_joint_')) or
                key in ('ik_acceleration_override','rotation_step_reduced','continuous_slowdown',
                        'linear_governor_ratio','angular_governor_ratio')},
        'last_actually_sent_command':copy.deepcopy(actual_command),
        'motion_streaming_active':bool(getattr(controller, 'streaming', False)),
        'stop_confirmed_monotonic_ns':getattr(controller, 'stop_confirmed_ns', None),
        'robot_feedback_stationary':feedback.get('stationary'),
        'stop_confirmation_semantics':'software stop acknowledged; stationary feedback required for derived hold',
        'rotation_goal_reachable':mapping.get('rotation_goal_reachable'),
        'boundary_reason':mapping.get('boundary_reason'),
        'quality':{'boundary_intervention':boundary, 'input_gap_or_recovery':input_gap,
            'ik_intervention':boundary or bool(mapping.get('joint_rate_limited')) or
                bool(mapping.get('rotation_step_reduced')) or bool(mapping.get('ik_acceleration_override')),
            'joint_shaping_intervention':intervention(mapping.get('joint_acceleration_limited')) or
                intervention(mapping.get('joint_speed_limited')) or
                intervention(mapping.get('joint_position_braking')) or
                intervention(mapping.get('joint_trajectory_limited')),
            'cartesian_optimizer_used':bool(mapping.get('joint_trajectory_cartesian_optimizer')),
            'cartesian_optimizer_converged':mapping.get('joint_optimizer_converged'),
            'cartesian_velocity_reduced':cartesian_reduced,
            'feedback_transport_transient':bool((feedback.get('transport') or {}).get('transient_error')),
            'feedback_transport_recovered':bool(feedback.get('transport_recovered_this_sample')),
            'command_sent_this_tick':bool(report['motion_command_sent_this_tick'])},
        'neutral_calibration':mapping.get('neutral_rotation_calibration') if state == 'ANCHORED' else None}
    if 'gripper' in report:
        sample['gripper'] = copy.deepcopy(report['gripper'])
    for key in ('episode_id', 'segment_id'):
        if report.get(key) is not None:
            sample[key] = report[key]
    return sample
