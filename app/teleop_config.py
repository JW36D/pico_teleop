"""Shared tuning parameters: validate legal values and device relationships.

Only value validity and parameter relationships are checked here.
Real device motion limits are read and applied at runtime.
"""
import json
import math
from pathlib import Path
from incremental import MappingError
from gripper_control import gripper_config

DEFAULTS = {'target_smoothing_tau_s': 0., 'max_clutch_rotation_rad': 3.12,
            'feedback_hz': 20., 'max_feedback_age_ms': 250.,
            'feedback_timeout_s': .15, 'max_control_gap_s': .1,
            'stationary_tolerance_deg': .05, 'log_interval_s': .5,
            'rotation_mapping':'world','controller_to_tool_matrix':[[1,0,0],[0,1,0],[0,0,1]],
            'auto_resume_soft_limits':False,'advance_on_same_frame':False,'recovery_dwell_s':.15,
            'follow_mode':'low','stream_trajectory_mode':0,'stream_radio':0,'max_command_gap_s':.1,
            'translation_reference':'tracking_world','joint_session_reserve_deg':0.,
            'joint_settle_tolerance_deg':0.,'min_wrist_angle_deg':0.,'position_priority':False,
            'tracking_governor':False,'max_command_lead_m':.004,
            'max_command_lead_angle_rad':.08,'tracking_resume_ratio':.4,
            'translation_mapping_frame':'work','robot_body_to_install_matrix':[[1,0,0],[0,1,0],[0,0,1]],
            'robot_body_expected_installation_deg':None,
            'max_joint_session_deg_by_joint':None,'max_tool_tilt_rad':math.pi/2,
            'rotation_control_mode':'full',
            'auto_resume_timing_gaps':False,'max_recoverable_control_gap_s':.15,
            'joint_session_limits_enabled':True,'tracking_governor_mode':'hold',
            'boundary_behavior':'pause','max_linear_accel_m_s2':3.,
            'max_angular_accel_rad_s2':12.,
            'max_session_translation_m':.3,'max_session_rotation_rad':math.pi,
            'max_joint_session_deg':90.,'session_pose_limits_enabled':True,
            'clutch_rotation_limits_enabled':True,'rotation_path_mode':'anchor',
            'mapper_control_gap_guard':True,'input_jump_mode':'step',
            'max_hand_linear_speed_m_s':5.,'max_hand_angular_speed_rad_s':20.,
            'max_input_step_m':.1,'max_input_step_rad':math.pi/6,
            'max_input_gap_ms':150.,'continuous_input_recovery':False,
            'continuous_timing_recovery':False,'joint_trajectory_enabled':False,
            'max_joint_accel_deg_s2':600.,'tracking_governor_curve':'quartic',
            'command_lead_full_speed_ratio':.7,'tracking_error_grace_s':0.,
            'tracking_error_time_alignment':False,'task_workspace':None,
            'feedback_checks_interval_s':1.,'require_robot_stationary_to_anchor':True,
            'translation_heading_mode':'grip','boundary_reference_mode':'rebase'}
NUMERIC_PARAMETERS = (
    'position_scale', 'rotation_scale', 'max_session_translation_m', 'max_session_rotation_rad',
    'max_clutch_rotation_rad', 'max_linear_speed_m_s', 'max_angular_speed_rad_s', 'max_joint_speed_deg_s',
    'max_joint_session_deg', 'max_tracking_error_m', 'max_tracking_angle_rad', 'max_input_age_ms',
    'max_input_step_m', 'max_input_step_rad', 'target_smoothing_tau_s', 'control_hz',
    'feedback_hz', 'max_feedback_age_ms', 'feedback_timeout_s', 'max_control_gap_s',
    'stationary_tolerance_deg', 'log_interval_s', 'max_command_gap_s', 'recovery_dwell_s',
    'joint_session_reserve_deg', 'joint_settle_tolerance_deg', 'min_wrist_angle_deg', 'max_command_lead_m',
    'max_command_lead_angle_rad', 'tracking_resume_ratio', 'max_tool_tilt_rad', 'max_recoverable_control_gap_s',
    'max_linear_accel_m_s2', 'max_angular_accel_rad_s2', 'max_joint_accel_deg_s2', 'max_hand_linear_speed_m_s',
    'max_hand_angular_speed_rad_s', 'max_input_gap_ms', 'command_lead_full_speed_ratio', 'tracking_error_grace_s',
    'feedback_checks_interval_s',
)


def load_config(path):
    config = {**DEFAULTS, **json.loads(Path(path).read_text())}
    config['gripper'] = gripper_config(config.get('gripper'))
    if config.get('arm') not in ('left','right'):
        raise MappingError('Choose arm left or right')
    for key in NUMERIC_PARAMETERS:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise MappingError('Invalid finite numeric parameter: ' + key)
        nonnegative = key in ('position_scale','rotation_scale','target_smoothing_tau_s',
                               'joint_session_reserve_deg','joint_settle_tolerance_deg','min_wrist_angle_deg',
                               'tracking_error_grace_s','recovery_dwell_s')
        if value < 0 or (value==0 and not nonnegative):
            raise MappingError(key+' must be '+('nonnegative' if nonnegative else 'positive'))
    weight=config.setdefault('cartesian_rotation_weight_m',
        config['max_linear_speed_m_s']/config['max_angular_speed_rad_s'])
    if type(weight) not in (int,float) or not math.isfinite(weight) or weight<=0:
        raise MappingError('cartesian_rotation_weight_m must be finite and positive')
    if not 0 < config['tracking_resume_ratio'] <= 1 or not 0 < config['command_lead_full_speed_ratio'] < 1:
        raise MappingError('Tracking ratios must be within the unit interval')
    if config['max_control_gap_s'] < 1/config['control_hz']:
        raise MappingError('Control deadline must be at least one nominal period')
    if config['max_feedback_age_ms']/1000 < 1/config['feedback_hz']:
        raise MappingError('Feedback age limit is smaller than its nominal period')
    if config['rotation_mapping'] not in ('world','controller_local','neutral_tool_axes'):
        raise MappingError('Unknown rotation_mapping')
    if config['rotation_mapping']=='neutral_tool_axes' and config['translation_reference']!='operator_heading':
        raise MappingError('Neutral rotation requires operator_heading')
    if config['rotation_control_mode'] not in ('full','roll_only'):
        raise MappingError('Unknown rotation_control_mode')
    if config['rotation_control_mode']=='roll_only' and config['rotation_mapping']!='neutral_tool_axes':
        raise MappingError('Roll-only control requires neutral_tool_axes')
    if config['rotation_path_mode']=='anchor' and config['rotation_mapping']!='neutral_tool_axes' and config['max_clutch_rotation_rad']>3.12:
        raise MappingError('Rotation beyond the legacy wrap bound requires neutral_tool_axes')
    if config['translation_reference'] not in ('tracking_world','operator_heading'):
        raise MappingError('Unknown translation_reference')
    if config['translation_mapping_frame'] not in ('work','robot_body'):
        raise MappingError('Unknown translation_mapping_frame')
    if config['translation_mapping_frame']=='robot_body':
        from incremental import IncrementalMapper, vector
        if config['translation_reference']!='operator_heading':
            raise MappingError('Robot-body mapping requires operator_heading input')
        IncrementalMapper.checked_basis(config['robot_body_to_install_matrix'])
        vector(config['robot_body_expected_installation_deg'],3)
    if config['joint_session_limits_enabled'] and config['joint_session_reserve_deg']>=config['max_joint_session_deg']:
        raise MappingError('Joint reserve must be smaller than session range')
    if config['joint_session_limits_enabled'] and config['max_joint_session_deg_by_joint'] is not None:
        from incremental import vector
        limits=vector(config['max_joint_session_deg_by_joint'],6)
        for i,value in enumerate(limits):
            if not config['joint_session_reserve_deg'] < value:
                raise MappingError('Invalid joint %d session limit'%(i+1))
    for key in ('auto_resume_soft_limits','advance_on_same_frame','position_priority','tracking_governor','auto_resume_timing_gaps','joint_session_limits_enabled',
                'session_pose_limits_enabled','clutch_rotation_limits_enabled','mapper_control_gap_guard',
                'continuous_input_recovery','continuous_timing_recovery','joint_trajectory_enabled',
                'tracking_error_time_alignment','require_robot_stationary_to_anchor'):
        if type(config[key]) is not bool:
            raise MappingError(key+' must be a boolean')
    if config['tracking_governor_mode'] not in ('hold','continuous'):
        raise MappingError('tracking_governor_mode must be hold or continuous')
    if config['boundary_behavior'] not in ('pause','directional'):
        raise MappingError('boundary_behavior must be pause or directional')
    for key,values in {
        'rotation_path_mode':('anchor','incremental'),
        'input_jump_mode':('step','velocity'),
        'tracking_governor_curve':('quartic','plateau'),
        'translation_heading_mode':('grip','session'),
        'boundary_reference_mode':('rebase','fixed'),
    }.items():
        if config[key] not in values:raise MappingError('Unknown '+key)
    if not config['clutch_rotation_limits_enabled'] and config['rotation_path_mode']!='incremental':
        raise MappingError('Removing rotation ranges requires continuous incremental rotation representation')
    if config['continuous_input_recovery'] and config['max_input_gap_ms']<=config['max_input_age_ms']:
        raise MappingError('Input gap deadline must exceed the fresh-input window')
    if config['continuous_timing_recovery'] and config['follow_mode']!='low':
        raise MappingError('Continuous timing recovery requires low follow')
    workspace=config['task_workspace']
    if workspace is not None:
        from incremental import vector
        if not isinstance(workspace,dict) or set(workspace)!={'min_m','max_m'}:
            raise MappingError('Task workspace requires min_m/max_m in the SDK work frame')
        if any(a>=b for a,b in zip(vector(workspace['min_m'],3),vector(workspace['max_m'],3))):
            raise MappingError('Task workspace minimum must be below maximum')
    if config['tracking_governor']:
        position_braking = rotation_braking = 0.
        if config['tracking_governor_mode']=='continuous':
            position_braking = config['max_linear_speed_m_s']**2/(2*config['max_linear_accel_m_s2'])
            rotation_braking = config['max_angular_speed_rad_s']**2/(2*config['max_angular_accel_rad_s2'])
        if config['max_command_lead_m']+position_braking+config['max_linear_speed_m_s']/config['control_hz'] >= config['max_tracking_error_m']:
            raise MappingError('Command lead plus braking distance and one position step must remain below the tracking stop bound')
        if config['max_command_lead_angle_rad']+rotation_braking+config['max_angular_speed_rad_s']/config['control_hz'] >= config['max_tracking_angle_rad']:
            raise MappingError('Command lead plus braking angle and one rotation step must remain below the tracking stop bound')
    if config['follow_mode'] not in ('low','high'):
        raise MappingError('follow_mode must be low or high')
    mode,radio = config['stream_trajectory_mode'],config['stream_radio']
    if type(mode) is not int or mode not in (0,1,2) or type(radio) is not int or not 0<=radio<=(100 if mode==1 else 999):
        raise MappingError('Invalid stream trajectory mode/radio')
    if config['max_command_gap_s'] < 1/config['control_hz']:
        raise MappingError('Command deadline is smaller than its nominal period')
    if config['auto_resume_timing_gaps']:
        if config['follow_mode']!='low' or not config['auto_resume_soft_limits']:
            raise MappingError('Timing recovery requires low follow and soft-limit recovery')
        if config['max_recoverable_control_gap_s'] <= max(config['max_control_gap_s'],config['max_command_gap_s']):
            raise MappingError('Recoverable timing bound must exceed the normal deadlines')
    if config['follow_mode']=='high' and (config['control_hz']<100 or
        config['max_control_gap_s']>.01 or config['max_command_gap_s']>.01 or
        not config['advance_on_same_frame']):
        raise MappingError('High follow requires stable <=10ms control and command periods with frame resampling')
    return config
