"""Testable control step; only execute mode can invoke a motion sender."""
import time
import math
import collections
from incremental import (AnchorUnavailable, IncrementalMapper, MappingError,
                         angle, controller_pose, input_ready, input_health, inverse, limit_ik_step, multiply,
                         norm, pose, vector, wrist_step_safe,position_priority_step,
                         _recoverable_ik_error,reachable_ik_step)
from continuous_motion import ContinuousMotionShaper, _rotation_vector
from joint_trajectory import JointTrajectory


class TeleopController:
    def __init__(self, config, backend, feedback, execute=False, clock_ns=time.monotonic_ns):
        self.config, self.backend, self.feedback = config, backend, feedback
        self.arm=config.get('arm','right')
        self.stop_requested = lambda: False
        self.stop_confirmed_ns = None
        self.execute, self.clock_ns = execute, clock_ns
        self.mapper = IncrementalMapper(config, backend.initial_pose)
        self.seed = backend.initial_joints.copy()
        self.last_target = backend.initial_pose.copy()
        self.last_joints = backend.initial_joints.copy()
        self.streaming = False
        self.last_tick_ns = None
        self.send_count = 0
        self.last_send_ns = None
        self.recovery_since_ns = None
        self.returning_inside_joints = False
        self.last_sent_target = backend.initial_pose.copy()
        self.last_sent_joints = backend.initial_joints.copy()
        self.waiting_for_feedback = False
        self.phase_ms = {}
        self.motion_shaper = (ContinuousMotionShaper(config)
            if config.get('tracking_governor_mode','hold')=='continuous' else None)
        self.joint_trajectory = (JointTrajectory(config,backend)
            if config.get('joint_trajectory_enabled',False) else None)
        self.pending_joint_candidate = None
        self.pending_joint_goal = None
        self.pending_emission = None
        self.joint_send_reference_ns = None
        self.input_gap_active = False
        self.last_actually_sent_command = None
        self.command_history = collections.deque(maxlen=1000)
        self.tracking_error_since_ns = None
        self.tracking_diagnostic = {}
        self.episode = self.segment = 0

    def timed(self, name, operation):
        started = self.clock_ns()
        try:
            return operation()
        finally:
            self.phase_ms[name] = (self.clock_ns()-started)/1e6

    def timing_pause(self, reason, duration_s, actual, measured):
        """Only bounded timing jitter in low follow preserves the active clutch."""
        recoverable = (self.config.get('auto_resume_timing_gaps',False) and
            self.config.get('follow_mode','low')=='low' and
            self.config.get('auto_resume_soft_limits',False) and
            duration_s <= self.config.get('max_recoverable_control_gap_s',.15))
        if recoverable:
            self.pause_control(reason,actual,measured['joints'])
            return {'state':reason+'_PAUSED','send':False,'soft_paused':True,
                    'pause_reason':reason,'gap_s':duration_s,'needs_regrip':False}
        self.stop()
        self.mapper.disarm()
        return {'state':reason,'send':False,'gap_s':duration_s,'needs_regrip':True}

    def joint_session_limits(self):
        limits = self.config.get('max_joint_session_deg_by_joint')
        return vector(limits,6) if limits is not None else [self.config['max_joint_session_deg']]*6

    def command_joint_bounds(self):
        # 软件会话边界与实际硬件限位不同；提前 3° 拒绝目标，为停止和反馈延迟留出空间。
        return [limit-min(self.config.get('joint_session_reserve_deg',0.),limit*.1)
                for limit in self.joint_session_limits()]

    def command_joint_bound(self):
        return min(self.command_joint_bounds())

    def joint_target_allowed(self, candidate):
        if not self.config.get('joint_session_limits_enabled',True):
            return True
        limits = self.command_joint_bounds()
        before = [abs(a-b) for a,b in zip(self.seed,self.backend.initial_joints)]
        after = [abs(a-b) for a,b in zip(candidate,self.backend.initial_joints)]
        if all(x<=limit for x,limit in zip(after,limits)):
            return True
        # 停止时的小幅越界允许沿已验证的返回目标逐步向内撤回，不能继续向外运动。
        return self.returning_inside_joints and all(
            a<=b+1e-5 if b>limit else a<=limit for a,b,limit in zip(after,before,limits))

    def validate_math_target(self, joints):
        if any(not low<=q<=high for q,low,high in zip(joints,self.backend.minimum,self.backend.maximum)):
            raise MappingError('IK result crosses joint limits',
                               {'state':'HARDWARE_BOUND','retryable_ik':True})
        if not self.joint_target_allowed(joints):
            raise MappingError('Joint target reaches reserved session boundary',{'state':'JOINT_SESSION_BOUND'})
        if not wrist_step_safe(self.seed,joints,self.config.get('min_wrist_angle_deg',0.)):
            raise MappingError('Target enters RM65 wrist singularity region',{'state':'WRIST_SINGULARITY'})

    def stop(self):
        if self.execute and self.streaming:
            self.backend.stop()
        self.streaming = False
        self.stop_confirmed_ns = self.clock_ns()
        self.last_send_ns = None
        self.waiting_for_feedback = False
        if self.motion_shaper:
            self.motion_shaper.reset()
        self.pending_joint_candidate = None
        self.pending_joint_goal = None
        self.pending_emission = None
        self.joint_send_reference_ns = None
        self.input_gap_active = False
        self.tracking_error_since_ns = None

    def follower_wait(self, actual):
        """低跟随时目标不能无视实际滞后一直积分；保留 Grip 锚点，等编码器跟上。

        在真正发送运动的流程中启用；预览仍可计算完整数学轨迹。
        两级阈值避免在边界反复切换，硬跟踪停止阈值始终独立检查。
        """
        if not self.execute or not self.config.get('tracking_governor',False):
            return False
        distance = norm([a-b for a,b in zip(actual[:3],self.last_sent_target[:3])])
        rotation = angle(multiply(actual[3:],inverse(self.last_sent_target[3:])))
        low = self.config['tracking_resume_ratio'] if self.waiting_for_feedback else 1.
        self.waiting_for_feedback = (distance > self.config['max_command_lead_m']*low or
                                     rotation > self.config['max_command_lead_angle_rad']*low)
        return self.waiting_for_feedback

    def pause_control(self, reason, actual, joints):
        """软暂停只停止运动、保留操作锚点；硬故障仍解除 Grip 控制。"""
        self.stop()
        self.recovery_since_ns = None
        self.returning_inside_joints = False
        if self.config.get('auto_resume_soft_limits',False):
            self.mapper.pause(reason)
            self.seed = vector(joints,6)
            self.last_joints = self.seed.copy()
            self.last_target = actual.copy()
            self.last_sent_target = actual.copy()
            self.last_sent_joints = self.seed.copy()
            self.mapper.target = actual.copy()
            self.mapper.filtered_goal = actual.copy()
            self.mapper.rebase_rotation(actual)
        else:
            self.mapper.disarm()

    def recover(self, result, measured, actual, now_ns):
        """回到范围内且目标可达、反馈停稳后，保留原锚点，从实际姿态重新起步。"""
        requested = result['requested_pose_wxyz']
        distance = norm([a-b for a,b in zip(requested[:3],self.backend.initial_pose[:3])])
        rotation = angle(multiply(requested[3:],inverse(self.backend.initial_pose[3:])))
        hand_rotation = result.get('hand_unwrapped_rotation_rad',math.radians(result['hand_rotation_deg']))
        session_rotation = self.config['max_session_rotation_rad']
        rotation_outside = session_rotation < math.pi-1e-6 and rotation > session_rotation-.01
        if (distance > self.config['max_session_translation_m']-.003 or
            rotation_outside or
            hand_rotation > self.config['max_clutch_rotation_rad']-math.radians(2) or
            (result.get('tool_tilt_deg',0.) > math.degrees(self.config.get('max_tool_tilt_rad',math.pi))-2
             and self.mapper.pause_reason=='TOOL_TILT_BOUND')):
            self.recovery_since_ns = None
            return {**result,'state':'RECOVERY_WAIT','recovery_reason':'Return farther inside pose bounds'}
        timing_recovery = self.mapper.pause_reason in (
            'CONTROL_GAP','COMMAND_GAP','CONTROL_COMPUTATION_GAP')
        local_recovery = False
        if not timing_recovery:
            # A software-range pause needs a goal back inside the existing range.
            # An IK pause may recover via a continuous local step when a distant
            # full goal is still unreachable. No command is emitted here.
            try:
                solved = self.timed('recovery_ik',lambda:
                    self.backend.solve(requested,measured['joints']))
            except MappingError as exc:
                if not _recoverable_ik_error(exc):
                    raise
                if self.mapper.pause_reason!='IK_REFUSED':
                    self.recovery_since_ns = None
                    return {**result,'state':'IK_REFUSED','reason':str(exc)}
                bounds = [min(self.config['max_joint_speed_deg_s'],v)/self.config['control_hz']
                    for v in getattr(self.backend,'speed_limits',[self.config['max_joint_speed_deg_s']]*6)]
                try:
                    target,candidate,_ = self.timed('recovery_local_ik',lambda:
                        limit_ik_step(self.backend.solve,actual,requested,measured['joints'],
                                      bounds,self.validate_math_target))
                except MappingError as local_error:
                    if not _recoverable_ik_error(local_error) and local_error.details.get('state') not in (
                            'JOINT_SESSION_BOUND','WRIST_SINGULARITY') and not str(local_error).startswith('Cannot find a continuous IK step'):
                        raise
                    self.recovery_since_ns = None
                    return {**result,'state':local_error.details.get('state','IK_REFUSED'),
                            'reason':str(local_error)}
                if (norm([a-b for a,b in zip(target[:3],actual[:3])])<1e-7 and
                        angle(multiply(target[3:],inverse(actual[3:])))<1e-6):
                    self.recovery_since_ns = None
                    return {**result,'state':'RECOVERY_WAIT','recovery_reason':'No reachable local motion'}
                local_recovery = True
            else:
                candidate = vector(solved,6)
            if any(not low<=q<=high for q,low,high in zip(candidate,self.backend.minimum,self.backend.maximum)):
                self.recovery_since_ns = None
                return {**result,'state':'IK_REFUSED','reason':'Recovered IK goal crosses hardware limit'}
            if self.config.get('joint_session_limits_enabled',True) and any(abs(a-b)>limit for a,b,limit in
                   zip(candidate,self.backend.initial_joints,self.command_joint_bounds())):
                self.recovery_since_ns = None
                return {**result,'state':'JOINT_SESSION_BOUND','rejected_target_joints_deg':candidate}
            minimum = self.config.get('min_wrist_angle_deg',0.)
            if (not wrist_step_safe(measured['joints'],candidate,minimum) or
                (self.mapper.pause_reason=='WRIST_SINGULARITY' and abs(candidate[4])<minimum+2)):
                self.recovery_since_ns = None
                return {**result,'state':'WRIST_SINGULARITY','recovery_reason':'Return away from q5=0',
                        'candidate_wrist_angle_deg':candidate[4]}
        if not measured['stationary']:
            self.recovery_since_ns = None
            return {**result,'state':'RECOVERY_WAIT','recovery_reason':'Robot is still settling'}
        if self.recovery_since_ns is None:
            self.recovery_since_ns = now_ns
        if (now_ns-self.recovery_since_ns)/1e9 < self.config.get('recovery_dwell_s',.15):
            return {**result,'state':'RECOVERY_WAIT','recovery_reason':'Valid target dwell'}
        self.seed = vector(measured['joints'],6)
        self.last_joints = self.seed.copy()
        self.last_target = actual.copy()
        self.last_sent_target = actual.copy()
        self.last_sent_joints = self.seed.copy()
        self.returning_inside_joints = self.config.get('joint_session_limits_enabled',True) and any(abs(a-b)>limit
            for a,b,limit in zip(self.seed,self.backend.initial_joints,self.command_joint_bounds()))
        self.mapper.target = actual.copy()
        self.mapper.filtered_goal = actual.copy()
        self.mapper.rebase_rotation(actual)
        self.mapper.paused = False
        self.mapper.pause_reason = None
        self.recovery_since_ns = None
        # 本周期不发送；下个周期从反馈姿态经过平滑/限速追向原锚点生成的目标，避免恢复跳跃。
        return {**result,'state':'RESUMED','send':False,'soft_paused':False,
                'target_pose_wxyz':actual.copy(),'target_joints_deg':self.seed.copy(),
                'recovery_used_local_step':local_recovery,'timing_recovery':timing_recovery}

    def anchor(self):
        measured = self.feedback.snapshot()
        if self.config.get('require_robot_stationary_to_anchor',True) and not measured['stationary']:
            raise AnchorUnavailable('Robot has not settled after release')
        self.seed = vector(measured['joints'], 6)
        if self.joint_trajectory:
            self.joint_trajectory.reset(self.seed,measured.get('joint_velocity_deg_s'))
            self.joint_send_reference_ns=None
        return pose(self.backend.forward(self.seed))

    def validate_feedback(self, measured):
        joints = vector(measured['joints'], 6)
        for q, low, high in zip(joints, self.backend.minimum, self.backend.maximum):
            if not low <= q <= high:
                raise MappingError('Actual joint feedback crosses hardware limit',
                    {'joints_deg':joints,'hardware_min_deg':self.backend.minimum,'hardware_max_deg':self.backend.maximum})
        limited=self.config.get('joint_session_limits_enabled',True)
        excursion = [abs(a-b) for a,b in zip(joints,self.backend.initial_joints)]
        limits = self.joint_session_limits() if limited else []
        if limited and all(x<=limit for x,limit in zip(excursion,self.command_joint_bounds())):
            self.returning_inside_joints = False
        if limited and any(x>limit for x,limit in zip(excursion,limits)):
            stopped_near_boundary = ((self.mapper.paused and not self.streaming) or self.returning_inside_joints)
            tolerated = stopped_near_boundary and all(x<=limit+self.config.get('joint_settle_tolerance_deg',0.)
                for x,limit in zip(excursion,limits))
            if not tolerated:
                raise MappingError('Actual joints exceeded session bound',{
                    'joints_deg':joints,'initial_joints_deg':self.backend.initial_joints,
                    'joint_excursion_deg':excursion,'session_bound_deg':self.config['max_joint_session_deg'],
                    'session_bounds_by_joint_deg':limits,
                    'over_limit_joints':[i+1 for i,(x,limit) in enumerate(zip(excursion,limits)) if x>limit]})
        actual = pose(self.backend.forward(joints))
        if self.config.get('session_pose_limits_enabled',True) and (norm([a-b for a,b in zip(actual[:3],self.backend.initial_pose[:3])]) > self.config['max_session_translation_m'] or angle(multiply(actual[3:],inverse(self.backend.initial_pose[3:]))) > self.config['max_session_rotation_rad']):
            raise MappingError('Actual pose exceeded session bound')
        workspace=self.config.get('task_workspace')
        if workspace and any(not a<=v<=b for a,v,b in zip(workspace['min_m'],actual[:3],workspace['max_m'])):
            raise MappingError('Actual tool pose crosses the configured task workspace')
        if not self.execute and max(abs(a-b) for a,b in zip(joints,self.backend.initial_joints)) > .05:
            raise MappingError('Arm moved during preview; possible competing controller')
        if self.execute and self.streaming:
            # 此处只核对已尝试发送的目标；还没发出的计算结果不能作为跟踪依据。
            reference=self.last_sent_target
            aligned_stamp=None
            if self.config.get('tracking_error_time_alignment',False) and self.command_history:
                aligned_stamp=measured.get('request_monotonic_ns',measured.get('receive_monotonic_ns'))
                if type(aligned_stamp) is int:
                    reference=next((c['pose_wxyz'] for c in reversed(self.command_history)
                        if c['send_monotonic_ns']<=aligned_stamp),self.backend.initial_pose)
            distance = norm([a-b for a,b in zip(actual[:3],reference[:3])])
            rotation = angle(multiply(actual[3:],inverse(reference[3:])))
            self.tracking_diagnostic={'tracking_comparison_time_ns':aligned_stamp,
                'aligned_tracking_error_mm':distance*1000,
                'aligned_tracking_error_deg':math.degrees(rotation)}
            if distance > self.config['max_tracking_error_m'] or rotation > self.config['max_tracking_angle_rad']:
                now=self.clock_ns()
                if self.tracking_error_since_ns is None:self.tracking_error_since_ns=now
                duration=(now-self.tracking_error_since_ns)/1e9
                self.tracking_diagnostic.update(tracking_error_pending=True,tracking_error_duration_s=duration)
                if duration<self.config.get('tracking_error_grace_s',0.):return actual
                raise MappingError('Robot tracking error exceeded bound',{
                    'position_error_mm':distance*1000,'position_bound_mm':self.config['max_tracking_error_m']*1000,
                    'orientation_error_deg':math.degrees(rotation),'orientation_bound_deg':math.degrees(self.config['max_tracking_angle_rad']),
                    'actual_pose_wxyz':actual,'last_sent_pose_wxyz':self.last_sent_target.copy(),
                    'actual_joints_deg':joints,'last_sent_joints_deg':self.last_sent_joints.copy(),
                    'feedback_age_ms':measured['age_ms'],'send_count':self.send_count,
                    'follow_mode':self.config.get('follow_mode','low'),
                    'waiting_for_feedback':self.waiting_for_feedback})
            self.tracking_error_since_ns=None
            self.tracking_diagnostic['tracking_error_pending']=False
        return actual

    def tick(self, status, latest_input, now_ns=None):
        now_ns = self.clock_ns() if now_ns is None else now_ns
        self.phase_ms = {}
        try:
            exiting = status.get('session_exit_requested', False) or self.stop_requested()
            if exiting:
                self.stop(); self.mapper.disarm()
            measured = self.timed('feedback_snapshot',self.feedback.snapshot)
            actual = self.timed('feedback_validation',lambda:self.validate_feedback(measured))
            if exiting:
                return self.report({'state':'SESSION_EXIT', 'send':False}, actual, measured, False)
            gap = (now_ns-self.last_tick_ns)/1e9 if self.last_tick_ns is not None else 1/self.config['control_hz']
            self.last_tick_ns = now_ns
            if gap > (self.config['max_recoverable_control_gap_s'] if self.config.get('continuous_timing_recovery') else self.config['max_control_gap_s']) and self.mapper.active and not self.mapper.paused:
                ctrl = (status.get('latest') or {}).get(self.arm) or {}
                # Let release and invalid input follow their normal stop paths.
                timing_input_valid = (input_ready(status,now_ns,self.config['max_input_age_ms'],require_head=not self.mapper.active if self.config.get('rotation_path_mode')=='incremental' else True,side=self.arm) and
                    type(ctrl.get('grip')) in (int,float) and .2<ctrl['grip']<=1)
                if timing_input_valid:
                    try:
                        controller_pose(ctrl.get('pose'))
                    except MappingError:
                        timing_input_valid = False
                if timing_input_valid:
                    result = self.timing_pause('CONTROL_GAP',gap,actual,measured)
                    return self.report(result, actual, measured, False)
            self.pending_joint_candidate=None
            self.pending_joint_goal=None
            self.pending_emission=None
            if self.config.get('continuous_input_recovery') and self.mapper.active:
                health=input_health(status,now_ns,self.config['max_input_age_ms'],
                    self.config['max_input_gap_ms'],require_head=False,last_sequence=self.mapper.last_sequence,side=self.arm)
                if health['recoverable']:
                    if not self.input_gap_active:self.segment+=1
                    self.input_gap_active=True
                    return self.controlled_input_hold(status,actual,measured,now_ns,gap,health,latest_input)
                if self.input_gap_active and health['ready']:
                    self.input_gap_active=False
                    self.segment+=1
                    self.mapper.last_ns=now_ns-int(1e9/self.config['control_hz'])
            was_active = self.mapper.active
            was_paused = self.mapper.paused
            previous_twist = self.mapper.target_twist_rad
            result = self.timed('mapper',lambda:self.mapper.update(status, now_ns, self.anchor))
            if was_active and not self.mapper.active:
                self.stop()
                self.seed = vector(measured['joints'], 6)
                self.recovery_since_ns = None
                self.returning_inside_joints = False
            if self.mapper.paused:
                if not was_paused:
                    self.pause_control(result['state'],actual,measured['joints'])
                if result['state']=='RECOVERY_CHECK':
                    result = self.timed('recovery',lambda:self.recover(result,measured,actual,now_ns))
                else:
                    self.recovery_since_ns = None
                return self.report(result,actual,measured,False)
            if result['state'] == 'ANCHORED':
                self.last_target = result['target_pose_wxyz']
                self.last_joints = self.seed.copy()
                self.last_sent_target = self.last_target.copy()
                self.last_sent_joints = self.seed.copy()
                self.waiting_for_feedback = False
                self.episode+=1
                self.segment+=1
                if self.motion_shaper:
                    self.motion_shaper.reset()
            if result['state']=='TRACKING' and not self.motion_shaper and self.follower_wait(actual):
                # update() 仍读取最新手柄、松手/超限等状态，操作意图不丢失。
                # 仅回退本周期目标推进；继续发送原来的同一目标，不叠加位移。
                self.mapper.target = self.last_target.copy()
                result.update(state='FOLLOWER_WAIT',send=False,waiting_for_feedback=True,
                    target_pose_wxyz=self.last_target.copy(),target_joints_deg=self.last_joints.copy(),
                    command_tracking_error_mm=norm([a-b for a,b in zip(actual[:3],self.last_sent_target[:3])])*1000,
                    command_tracking_error_deg=math.degrees(angle(multiply(actual[3:],inverse(self.last_sent_target[3:])))))
            if result['state'] == 'TRACKING':
                # 目标先经过笛卡尔限速，再进行连续 IK 回退；每关节都受配置和硬件速度约束。
                prior = pose(self.backend.forward(self.seed))
                proposed = result['target_pose_wxyz']
                step_dt = min(gap,result.get('dt_s',gap),1/self.config['control_hz'])
                if self.motion_shaper:
                    angular_remaining = None
                    if self.config.get('rotation_path_mode')=='incremental':
                        angular_remaining=self.mapper.rotation_remaining_rad()
                    elif self.mapper.rotation_mapping=='neutral_tool_axes':
                        from rotation_axes import split_swing_twist
                        origin = self.mapper.anchor_robot[3:]
                        prior_swing,prior_twist = split_swing_twist(
                            multiply(inverse(origin),prior[3:]),previous_twist)
                        goal_swing,_ = split_swing_twist(
                            multiply(inverse(origin),self.mapper.filtered_goal[3:]),
                            self.mapper.filtered_twist_rad)
                        angular_remaining = abs(self.mapper.filtered_twist_rad-prior_twist)+angle(
                            multiply(goal_swing,inverse(prior_swing)))
                    proposed,shaping = self.timed('continuous_shaping',lambda:
                        self.motion_shaper.advance(prior,self.mapper.filtered_goal,proposed,
                            actual,step_dt,governed=self.execute and self.config.get('tracking_governor',False),
                            angular_remaining_rad=angular_remaining))
                    result.update(shaping)
                    self.waiting_for_feedback = False
                boundary_reason = None
                try:
                    bounds = [min(self.config['max_joint_speed_deg_s'],v)*min(gap,1/self.config['control_hz'])
                              for v in getattr(self.backend,'speed_limits',[self.config['max_joint_speed_deg_s']]*6)]
                    if self.config.get('boundary_behavior','pause')=='directional':
                        target,joints,limited,rotation_limited,boundary_reason = self.timed('ik_step',lambda:
                            reachable_ik_step(self.backend.solve,prior,proposed,self.seed,bounds,
                                self.validate_math_target,self.config.get('position_priority',False)))
                        result['rotation_step_reduced'] = rotation_limited
                    elif self.config.get('position_priority',False):
                        target,joints,limited,rotation_limited = self.timed('ik_step',lambda:
                            position_priority_step(self.backend.solve,prior,
                            proposed,self.seed,bounds,self.validate_math_target))
                        result['rotation_step_reduced'] = rotation_limited
                    else:
                        target,joints,limited = self.timed('ik_step',lambda:
                            limit_ik_step(self.backend.solve,prior,
                            proposed,self.seed,bounds,self.validate_math_target))
                except MappingError as exc:
                    if (not _recoverable_ik_error(exc) and exc.details.get('state') not in (
                            'JOINT_SESSION_BOUND','WRIST_SINGULARITY') and
                            not str(exc).startswith('Cannot find a continuous IK step')):
                        raise
                    state = exc.details.get('state','IK_REFUSED')
                    self.pause_control(state,actual,measured['joints'])
                    return self.report({**result,'state':state,'send':False,
                        'soft_paused':self.mapper.paused,'reason':str(exc)}, actual, measured, False)
                delta = [q-q0 for q,q0 in zip(joints,self.backend.initial_joints)]
                if not self.joint_target_allowed(joints):
                    self.pause_control('JOINT_SESSION_BOUND',actual,measured['joints'])
                    return self.report({**result,'state':'JOINT_SESSION_BOUND','send':False,
                        'soft_paused':self.mapper.paused,
                        'rejected_target_joints_deg':joints,'joint_delta_deg':delta}, actual, measured, False)
                if not wrist_step_safe(self.seed,joints,self.config.get('min_wrist_angle_deg',0.)):
                    self.pause_control('WRIST_SINGULARITY',actual,measured['joints'])
                    return self.report({**result,'state':'WRIST_SINGULARITY','send':False,
                        'soft_paused':self.mapper.paused,'candidate_wrist_angle_deg':joints[4]},actual,measured,False)
                for q, low, high in zip(joints,self.backend.minimum,self.backend.maximum):
                    if not low <= q <= high:
                        raise MappingError('Target crosses hardware joint limit')
                if self.joint_trajectory:
                    # Preserve the accepted local Cartesian direction. Clipping
                    # inverse-kinematics joint rates independently can break the
                    # coordinated translation/rotation of the end effector.
                    cartesian_goal=proposed if boundary_reason is None else target
                    desired=[(a-b)/step_dt for a,b in zip(cartesian_goal[:3],prior[:3])]+[
                        v/step_dt for v in _rotation_vector(multiply(cartesian_goal[3:],inverse(prior[3:])))]
                    self.pending_joint_goal=(cartesian_goal.copy(),desired.copy())
                    self.pending_emission=(prior.copy(),proposed.copy(),target.copy(),boundary_reason)
                    joints,joint_diagnostic=self.timed('joint_trajectory',lambda:
                        self.joint_trajectory.propose_cartesian(cartesian_goal,step_dt,self.backend,
                            target_cartesian_velocity=desired))
                    target=pose(self.backend.forward(joints))
                    result.update(joint_diagnostic)
                    self.pending_joint_candidate=(joints.copy(),step_dt)
                # Backtracking may follow a different rotational arc; check the emitted pose too.
                if self.config.get('session_pose_limits_enabled',True) and (norm([a-b for a,b in zip(target[:3],self.backend.initial_pose[:3])]) > self.config['max_session_translation_m'] or angle(multiply(target[3:],inverse(self.backend.initial_pose[3:]))) > self.config['max_session_rotation_rad']):
                    self.pause_control('SESSION_BOUND',actual,measured['joints'])
                    return self.report({**result,'state':'SESSION_BOUND','send':False,
                        'soft_paused':self.mapper.paused,'limit_reason':['emitted_pose']}, actual, measured, False)
                # Keep the command quaternion's sign continuous even if SDK FK
                # returned the equivalent opposite sign across a large rotation.
                if sum(a*b for a,b in zip(target[3:],self.last_target[3:]))<0:
                    target[3:]=[-v for v in target[3:]]
                if not self.joint_trajectory:
                    self.mapper.sync_target_rotation(target)
                if self.motion_shaper and not self.joint_trajectory:
                    old_linear=self.motion_shaper.linear_velocity.copy()
                    old_angular=self.motion_shaper.angular_velocity.copy()
                    self.motion_shaper.sync(prior,target,step_dt)
                    emitted_linear_accel=norm([a-b for a,b in zip(
                        self.motion_shaper.linear_velocity,old_linear)])/step_dt
                    emitted_angular_accel=norm([a-b for a,b in zip(
                        self.motion_shaper.angular_velocity,old_angular)])/step_dt
                    result.update(emitted_linear_velocity_m_s=self.motion_shaper.linear_velocity.copy(),
                        emitted_angular_velocity_rad_s=self.motion_shaper.angular_velocity.copy(),
                        emitted_linear_accel_m_s2=emitted_linear_accel,
                        emitted_angular_accel_rad_s2=emitted_angular_accel,
                        ik_acceleration_override=(emitted_linear_accel>self.config['max_linear_accel_m_s2']+1e-6 or
                            emitted_angular_accel>self.config['max_angular_accel_rad_s2']+1e-6))
                if boundary_reason is not None and not self.joint_trajectory:
                    moved = (norm([a-b for a,b in zip(target[:3],prior[:3])])>1e-7 or
                             angle(multiply(target[3:],inverse(prior[3:])))>1e-6)
                    result.update(state='BOUNDARY_LIMITED' if moved else 'BOUNDARY_HOLD',
                        boundary_reason=boundary_reason,send=moved,soft_paused=False,needs_regrip=False)
                    wanted_delta = [b-a for a,b in zip(prior[:3],proposed[:3])]
                    accepted_delta = [b-a for a,b in zip(prior[:3],target[:3])]
                    blocked_axes = [i for i,(a,b) in enumerate(zip(wanted_delta,accepted_delta))
                                    if abs(a)>1e-8 and abs(b)<1e-8]
                    wanted_size,accepted_size = norm(wanted_delta),norm(accepted_delta)
                    if not blocked_axes and wanted_size>1e-8 and accepted_size<wanted_size-1e-8:
                        # A same-direction boundary backtrack loses the outward
                        # portion; velocity-only reductions never enter this path.
                        cross = [wanted_delta[1]*accepted_delta[2]-wanted_delta[2]*accepted_delta[1],
                                 wanted_delta[2]*accepted_delta[0]-wanted_delta[0]*accepted_delta[2],
                                 wanted_delta[0]*accepted_delta[1]-wanted_delta[1]*accepted_delta[0]]
                        if norm(cross)<1e-6*wanted_size*max(accepted_size,1e-8):
                            blocked_axes=[i for i,a in enumerate(wanted_delta) if abs(a)>1e-8]
                    blocked_rotation = angle(multiply(target[3:],inverse(proposed[3:])))>1e-6
                    fixed_reference=self.config.get('boundary_reference_mode','rebase')=='fixed'
                    if fixed_reference:
                        blocked_axes,blocked_rotation=[],False
                    if blocked_axes or blocked_rotation:
                        self.mapper.saturate_boundary_goal(target,blocked_axes,blocked_rotation)
                    result.update(boundary_saturated_translation_axes=blocked_axes,
                                  boundary_saturated_rotation=blocked_rotation,
                                  boundary_requested_goal_preserved=fixed_reference)
                workspace=self.config.get('task_workspace')
                if workspace and any(not a<=v<=b for a,v,b in zip(workspace['min_m'],target[:3],workspace['max_m'])):
                    raise MappingError('Final joint trajectory crosses the configured task workspace')
                self.seed = joints
                self.last_target, self.last_joints = target, joints
                result.update(target_pose_wxyz=target, target_joints_deg=joints, joint_rate_limited=limited,
                    continuous_timing_recovery=self.config.get('continuous_timing_recovery') and gap>self.config['max_control_gap_s'])
                result['command_tracking_error_mm'] = norm([a-b for a,b in zip(actual[:3],target[:3])])*1000
                result['command_tracking_error_deg'] = math.degrees(angle(multiply(target[3:],inverse(actual[3:]))))
                result['requested_rotation_from_anchor_deg'] = math.degrees(angle(multiply(
                    result['requested_pose_wxyz'][3:],inverse(self.mapper.anchor_robot[3:]))))
                result['waiting_for_feedback'] = False
                if self.mapper.rotation_mapping=='neutral_tool_axes':
                    result['requested_orientation_error_deg'] = math.degrees(angle(multiply(
                        result['requested_pose_wxyz'][3:],inverse(actual[3:]))))
                    try:
                        requested_joints=self.timed('goal_ik',lambda:
                            self.backend.solve(result['requested_pose_wxyz'],self.seed))
                        self.validate_math_target(requested_joints)
                        result['rotation_goal_reachable'] = True
                    except MappingError as exc:
                        if not _recoverable_ik_error(exc) and exc.details.get('state') not in (
                                'JOINT_SESSION_BOUND','WRIST_SINGULARITY'):
                            raise
                        result['rotation_goal_reachable'] = False
                        result['rotation_goal_limit_reason'] = str(exc)
            return self.send_result(result,status,actual,measured,now_ns,latest_input)
        except BaseException:
            self.stop(); self.mapper.disarm()
            raise

    def controlled_input_hold(self, status, actual, measured, now_ns, gap, health, latest_input):
        # Do not advance stale hand intent. Brake the last committed joint
        # velocity, retain Grip/calibration, and resume from the next valid input.
        dt=min(gap,1/self.config['control_hz'])
        prior=pose(self.backend.forward(self.joint_trajectory.joints))
        joints,diagnostic=self.joint_trajectory.propose_cartesian(prior,dt,self.backend,
            target_cartesian_velocity=[0.]*6)
        target=pose(self.backend.forward(joints))
        self.seed=joints.copy()
        self.last_target,self.last_joints=target,joints.copy()
        self.pending_joint_candidate=(joints.copy(),dt)
        self.pending_joint_goal=(prior.copy(),[0.]*6)
        self.pending_emission=(prior.copy(),prior.copy(),prior.copy(),None)
        result={'state':'INPUT_GAP_HOLD','send':True,'soft_paused':False,'needs_regrip':False,
            'input_gap_hold':True,'input_health':health,'target_pose_wxyz':target,
            'target_joints_deg':joints,**diagnostic}
        return self.send_result(result,status,actual,measured,now_ns,latest_input)

    def send_result(self, result, status, actual, measured, now_ns, latest_input):
        should_stream=self.mapper.active and result['state'] in (
            'TRACKING','WAIT_NEW_FRAME','FOLLOWER_WAIT','BOUNDARY_HOLD','BOUNDARY_LIMITED','INPUT_GAP_HOLD') and (
            result['state'] in ('TRACKING','BOUNDARY_HOLD','BOUNDARY_LIMITED','INPUT_GAP_HOLD') or self.streaming)
        sent=False
        if self.execute and should_stream:
            gate=self.timed('read_gate',latest_input)
            measured_gate=self.timed('gate_feedback_snapshot',self.feedback.snapshot)
            self.timed('gate_feedback_validation',lambda:self.validate_feedback(measured_gate))
            now=self.clock_ns()
            ctrl=(gate.get('latest') or {}).get(self.arm) or {}
            require_head=(not self.mapper.active if self.config.get('rotation_path_mode')=='incremental' else True)
            health=input_health(gate,now,self.config['max_input_age_ms'],
                self.config.get('max_input_gap_ms') if self.config.get('continuous_input_recovery') else None,
                require_head=require_head,last_sequence=self.mapper.last_sequence,side=self.arm)
            gate_valid=health['ready'] or (health['recoverable'] and self.mapper.active)
            result['input_health']=health
            if health['recoverable']:result['input_gap_hold']=True
            if gate_valid:
                grip=ctrl.get('grip')
                gate_valid=(type(grip) in (int,float) and 0<=grip<=1 and
                    gate['latest'].get('sequence',-1)>=status['latest']['sequence'])
            if gate.get('session_exit_requested', False) or self.stop_requested():
                self.stop();self.mapper.disarm()
                result={'state':'SESSION_EXIT','send':False}
            elif not gate_valid or ctrl.get('grip',0)<=.2:
                self.stop();self.mapper.disarm()
                result={'state':'INPUT_GATE_CLOSED','send':False,'gate_details':{
                    'valid':gate_valid,'grip':ctrl.get('grip'),'reason':health['reason'],
                    'app_focus':(gate.get('latest') or {}).get('app_focus'),
                    'head_status':((gate.get('latest') or {}).get('head') or {}).get('status'),
                    'controller_status':ctrl.get('status'),'input_age_ms':health['input_age_ms']}}
            else:
                control_deadline=(self.config['max_recoverable_control_gap_s']
                    if self.config.get('continuous_timing_recovery') else self.config['max_control_gap_s'])
                command_deadline=(self.config['max_recoverable_control_gap_s']
                    if self.config.get('continuous_timing_recovery') else self.config.get('max_command_gap_s',self.config['max_control_gap_s']))
                if (now-now_ns)/1e9>control_deadline:
                    if self.config.get('auto_resume_timing_gaps'):
                        result=self.timing_pause('CONTROL_COMPUTATION_GAP',(now-now_ns)/1e9,
                            pose(self.backend.forward(measured_gate['joints'])),measured_gate)
                    else:raise MappingError('Control computation exceeded deadline')
                elif self.streaming and self.last_send_ns is not None and (now-self.last_send_ns)/1e9>command_deadline:
                    result=self.timing_pause('COMMAND_GAP',(now-self.last_send_ns)/1e9,
                        pose(self.backend.forward(measured_gate['joints'])),measured_gate)
                else:
                    if self.pending_joint_goal:
                        self.final_joint_step(result,now)
                    # Candidate computation may take time. Check the shared exit
                    # latch again at the actual emission point.
                    if self.stop_requested():
                        self.stop(); self.mapper.disarm()
                        return self.report({'state':'SESSION_EXIT','send':False}, actual, measured, False)
                    result['command_tracking_error_mm']=norm([a-b for a,b in zip(actual[:3],self.last_target[:3])])*1000
                    result['command_tracking_error_deg']=math.degrees(angle(multiply(self.last_target[3:],inverse(actual[3:]))))
                    send_started=self.clock_ns()
                    self.streaming=True
                    self.last_send_ns=send_started
                    self.last_sent_target=self.last_target.copy()
                    self.last_sent_joints=self.last_joints.copy()
                    self.timed('send',lambda:self.backend.send(self.last_joints))
                    result['final_joint_calculation_ms']=(send_started-now)/1e6
                    result['send_call_ms']=(self.clock_ns()-send_started)/1e6
                    self.send_count+=1
                    self.last_actually_sent_command={'sequence':self.send_count,'send_monotonic_ns':send_started,
                        'trajectory_sample_monotonic_ns':now,
                        'joints_deg':self.last_sent_joints.copy(),'pose_wxyz':self.last_sent_target.copy()}
                    self.command_history.append(self.last_actually_sent_command.copy())
                    if self.pending_joint_candidate:
                        self.joint_trajectory.commit(*self.pending_joint_candidate)
                        self.pending_joint_candidate=None
                        self.pending_joint_goal=None
                        self.joint_send_reference_ns=now
                    sent=True
        if not self.execute and self.pending_joint_candidate:
            self.sync_final_emission(result,self.last_target,self.pending_joint_candidate[1])
            self.joint_trajectory.commit(*self.pending_joint_candidate)
            self.pending_joint_candidate=None
            self.pending_joint_goal=None
        return self.report(result,actual,measured,sent)

    def final_joint_step(self, result, command_time_ns):
        """Shape at the command-trajectory sample time after all gates.

        A 50ms scheduling hiccup cannot be treated as a 10ms joint step. The
        local Cartesian velocity supplies intent, while final speed/acceleration and
        hardware braking are recomputed at the final command sampling time.
        SDK launch time is recorded separately: mathematical sampling bounds
        do not claim to bound measured motion or network-arrival acceleration.
        """
        dt=((command_time_ns-self.joint_send_reference_ns)/1e9
            if self.joint_send_reference_ns is not None else 1/self.config['control_hz'])
        goal,desired=self.pending_joint_goal
        prior=pose(self.backend.forward(self.joint_trajectory.joints))
        joints,diagnostic=self.joint_trajectory.propose_cartesian(goal,dt,self.backend,
            target_cartesian_velocity=desired)
        target=pose(self.backend.forward(joints))
        workspace=self.config.get('task_workspace')
        if workspace and any(not a<=v<=b for a,v,b in zip(workspace['min_m'],target[:3],workspace['max_m'])):
            raise MappingError('Final joint trajectory crosses the configured task workspace')
        if sum(a*b for a,b in zip(target[3:],self.last_target[3:]))<0:
            target[3:]=[-v for v in target[3:]]
        self.sync_final_emission(result,target,dt)
        self.seed=joints.copy()
        self.last_target,self.last_joints=target,joints.copy()
        self.pending_joint_candidate=(joints.copy(),dt)
        result.update(diagnostic,target_pose_wxyz=target,target_joints_deg=joints,
            joint_command_dt_s=dt,joint_trajectory_sample_monotonic_ns=command_time_ns)

    def sync_final_emission(self, result, target, dt):
        """Consume hand rotation and boundary intent only at the final waypoint.

        A preview candidate can differ from the sender-gated command. In
        particular, consuming its rotation path early loses signed multi-turn
        intent when the actual sender interval changes.
        """
        prior,proposed,reachable,boundary_reason=self.pending_emission
        self.mapper.sync_target_rotation(target)
        if self.motion_shaper:
            old_linear=self.motion_shaper.linear_velocity.copy()
            old_angular=self.motion_shaper.angular_velocity.copy()
            self.motion_shaper.sync(prior,target,dt)
            linear_accel=norm([a-b for a,b in zip(self.motion_shaper.linear_velocity,old_linear)])/dt
            angular_accel=norm([a-b for a,b in zip(self.motion_shaper.angular_velocity,old_angular)])/dt
            result.update(emitted_linear_velocity_m_s=self.motion_shaper.linear_velocity.copy(),
                emitted_angular_velocity_rad_s=self.motion_shaper.angular_velocity.copy(),
                emitted_linear_accel_m_s2=linear_accel,emitted_angular_accel_rad_s2=angular_accel,
                ik_acceleration_override=(linear_accel>self.config['max_linear_accel_m_s2']+1e-6 or
                    angular_accel>self.config['max_angular_accel_rad_s2']+1e-6))
        if boundary_reason is None:return
        moved=(norm([a-b for a,b in zip(target[:3],prior[:3])])>1e-7 or
            angle(multiply(target[3:],inverse(prior[3:])))>1e-6)
        result.update(state='BOUNDARY_LIMITED' if moved else 'BOUNDARY_HOLD',
            boundary_reason=boundary_reason,send=moved,soft_paused=False,needs_regrip=False)
        # Only a real IK/geometry boundary can discard outward intent. Ordinary
        # velocity/acceleration shaping must retain the still-reachable goal.
        wanted=[b-a for a,b in zip(prior[:3],proposed[:3])]
        accepted=[b-a for a,b in zip(prior[:3],reachable[:3])]
        blocked=[i for i,(a,b) in enumerate(zip(wanted,accepted)) if abs(a)>1e-8 and abs(b)<1e-8]
        size,accepted_size=norm(wanted),norm(accepted)
        if not blocked and size>1e-8 and accepted_size<size-1e-8:
            cross=[wanted[1]*accepted[2]-wanted[2]*accepted[1],
                wanted[2]*accepted[0]-wanted[0]*accepted[2],
                wanted[0]*accepted[1]-wanted[1]*accepted[0]]
            if norm(cross)<1e-6*size*max(accepted_size,1e-8):
                blocked=[i for i,a in enumerate(wanted) if abs(a)>1e-8]
        blocked_rotation=angle(multiply(reachable[3:],inverse(proposed[3:])))>1e-6
        fixed_reference=self.config.get('boundary_reference_mode','rebase')=='fixed'
        if fixed_reference:
            blocked,blocked_rotation=[],False
        if blocked or blocked_rotation:self.mapper.saturate_boundary_goal(target,blocked,blocked_rotation)
        result.update(boundary_saturated_translation_axes=blocked,boundary_saturated_rotation=blocked_rotation,
            boundary_requested_goal_preserved=fixed_reference)

    def report(self, result, actual, measured, sent):
        limited = self.config.get('joint_session_limits_enabled',True)
        result.update(self.tracking_diagnostic)
        return {'mode':'EXECUTE' if self.execute else 'PREVIEW_NO_MOTION',
                'arm':self.arm,'controller_side':self.arm,
                'control_phase_ms':self.phase_ms.copy(),
                'motion_sent_this_tick':sent,'motion_command_sent_this_tick':sent,
                'send_count':self.send_count,'robot_feedback_age_ms':measured['age_ms'],
                'episode':self.episode,'segment':self.segment,
                'episode_id':self.episode,'segment_id':self.segment,
                'last_actually_sent_command':self.last_actually_sent_command,
                'robot_feedback_metadata':{k:v for k,v in measured.items() if k not in ('joints','position_m','orientation_rad')},
                'joint_trajectory_enabled':bool(self.joint_trajectory),
                'software_bound_status':{
                    'startup_pose':self.config.get('session_pose_limits_enabled',True),
                    'startup_joint':limited,
                    'fixed_rotation_range':self.config.get('clutch_rotation_limits_enabled',True),
                    'task_workspace_configured':self.config.get('task_workspace') is not None},
                'robot_feedback_monotonic_ns':measured.get('receive_monotonic_ns'),
                'robot_feedback_joints_deg':vector(measured['joints'],6),
                'joint_session_limits_enabled':limited,
                'joint_command_bound_deg':self.command_joint_bound() if limited else None,
                'joint_command_bounds_deg':self.command_joint_bounds() if limited else None,
                'joint_session_bounds_deg':self.joint_session_limits() if limited else None,
                'hardware_joint_limits_deg':{'min':self.backend.minimum,'max':self.backend.maximum},
                'wrist_angle_deg':measured['joints'][4],
                'robot_feedback_pose_wxyz':actual,'mapping':result}
