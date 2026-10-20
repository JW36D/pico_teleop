"""Pure clutch-based relative pose mapper. No sockets or robot APIs."""
import math


class MappingError(RuntimeError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


class AnchorUnavailable(MappingError):
    """The robot is still settling; retry anchoring while Grip remains held."""
    pass


def vector(values, size):
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise MappingError('Missing or incorrect vector size')
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) for x in values):
        raise MappingError('Nonfinite or invalid vector')
    return list(values)


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def quat(q):
    q = vector(q, 4)
    magnitude = norm(q)
    if abs(magnitude - 1) > .02:
        raise MappingError('Invalid quaternion norm')
    return [x / magnitude for x in q]


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return [w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v]


def inverse(q):
    return [q[0], -q[1], -q[2], -q[3]]


def rotate_vector(q, values):
    return multiply(multiply(quat(q),[0.]+vector(values,3)),inverse(quat(q)))[1:]


def operator_yaw_basis(head_pose):
    """PICO 房间坐标→操作者水平坐标；只取朝向，忽略头部俯仰/滚转。"""
    head = controller_pose(head_pose)
    forward = rotate_vector(head[3:],[0.,0.,-1.])
    horizontal = math.hypot(forward[0],forward[2])
    if horizontal < .3:
        raise MappingError('Look horizontally forward to establish operator heading')
    forward = [forward[0]/horizontal,0.,forward[2]/horizontal]
    right = [-forward[2],0.,forward[0]]
    return [right,[0.,1.,0.],[-x for x in forward]]


def wrist_step_safe(previous, candidate, minimum_deg):
    """RM65 q5≈0 时 J4/J6 轴近乎平行；允许局部滚转及离开奇异区，拒绝继续进入。"""
    if minimum_deg<=0:
        return True
    before,after = previous[4],candidate[4]
    if before*after<0:
        return False
    if abs(after) < minimum_deg and abs(after)+.001 < abs(before):
        return False
    d4,d6 = candidate[3]-previous[3],candidate[5]-previous[5]
    if min(abs(before),abs(after))<minimum_deg and d4*d6<0 and min(abs(d4),abs(d6))>.1 and abs(after)<=abs(before)+.001:
        return False
    return True


def angle(q):
    return 2 * math.acos(min(1., abs(quat(q)[0])))


def scaled_rotation(q, fraction, shortest=True):
    q = quat(q)
    if shortest and q[0] < 0:
        q = [-x for x in q]
    half = math.acos(min(1., q[0]))
    if half < 1e-8:
        return [1., 0., 0., 0.]
    ratio = math.sin(half * fraction) / math.sin(half)
    return [math.cos(half * fraction)] + [x * ratio for x in q[1:]]


def pose(values):
    values = vector(values, 7)
    return values[:3] + quat(values[3:])


def controller_pose(values):
    # 位姿格式统一：PICO 是 xyz+xyzw，后续全部使用 xyz+wxyz，单位米/弧度。
    values = vector(values, 7)
    return values[:3] + quat([values[6], *values[3:6]])


def _recoverable_ik_error(error):
    """Only a solver's declared reachability refusal may trigger backtracking.

    Keep the two existing native-solver messages compatible. Validation and
    malformed data must propagate rather than becoming an apparently valid
    smaller step.
    """
    return (error.details.get('retryable_ik') is True or
            error.details.get('state') in ('IK_REFUSED', 'IK_UNREACHABLE') or
            str(error).startswith(('IK failed:', 'IK result crosses joint limits', 'Unreachable ')))


def limit_ik_step(solve, previous_pose, requested_pose, seed, max_step_deg, validate=None):
    """Reduce the Cartesian step until IK stays within speed and optional bounds.

    The solve callback is math only; this function never sends a command.
    Backtracking retains a Cartesian path, rather than clipping each joint.
    Solver reachability refusals and reserved session boundaries get at most
    eight attempts; bad vectors, unrelated errors and cancellation propagate.
    """
    previous_pose, candidate = pose(previous_pose), pose(requested_pose)
    seed = vector(seed, 6)
    bounds = vector(max_step_deg, 6) if isinstance(max_step_deg, (list,tuple)) else [float(max_step_deg)]*6
    if any(not math.isfinite(x) or x<=0 for x in bounds):
        raise MappingError('Joint step bounds must be finite and positive')
    limited = False
    last_error = None
    for _ in range(8):
        fraction = .5
        try:
            solved = solve(candidate, seed)
        except MappingError as exc:
            if not _recoverable_ik_error(exc):
                raise
            last_error = exc
        else:
            # Validate returned data outside the recoverable solver catch.
            joints = vector(solved, 6)
            try:
                if validate is not None:
                    validate(joints)
            except MappingError as exc:
                if exc.details.get('state') != 'JOINT_SESSION_BOUND':
                    raise
                last_error = exc
            else:
                last_error = None
                differences = [abs(a-b) for a,b in zip(joints,seed)]
                if all(delta<=bound+1e-4 for delta,bound in zip(differences,bounds)):
                    return candidate, joints, limited
                maximum_ratio = max(delta/bound for delta,bound in zip(differences,bounds))
                fraction = min(.5, .8 / maximum_ratio)
        translation = [a + (b-a)*fraction for a, b in zip(previous_pose[:3], candidate[:3])]
        rotation = multiply(candidate[3:], inverse(previous_pose[3:]))
        candidate = translation + quat(multiply(scaled_rotation(rotation, fraction), previous_pose[3:]))
        limited = True
    if last_error is not None:
        raise last_error
    raise MappingError('Cannot find a continuous IK step within joint velocity bound')


def position_priority_step(solve, previous_pose, requested_pose, seed, max_step_deg, validate):
    """先分配关节速度给平移，再分配剩余能力给旋转；不能因姿态难求而一起缩小平移。"""
    previous_pose,requested_pose = pose(previous_pose),pose(requested_pose)
    bounds = vector(max_step_deg,6)
    translation_goal = requested_pose[:3]+previous_pose[3:]
    translated,joints,limited = limit_ik_step(solve,previous_pose,translation_goal,seed,bounds,validate)
    rotation = multiply(requested_pose[3:],inverse(previous_pose[3:]))
    if angle(rotation)<1e-7:
        return translated,joints,limited,False
    fraction = 1.
    for _ in range(12):
        target = translated[:3]+quat(multiply(scaled_rotation(rotation,fraction),previous_pose[3:]))
        try:
            solved = solve(target,joints)
        except MappingError as exc:
            if not _recoverable_ik_error(exc):
                raise
        else:
            candidate = vector(solved,6)
            try:
                validate(candidate)
            except MappingError as exc:
                if exc.details.get('state') not in ('JOINT_SESSION_BOUND', 'WRIST_SINGULARITY'):
                    raise
            else:
                if all(abs(a-b)<=v+1e-4 for a,b,v in zip(candidate,seed,bounds)):
                    return target,candidate,limited or fraction<1.,fraction<1.
        # 这里只缩小旋转，已验证的平移位置保持不变；末端位置不用为难跟的姿态让路。
        fraction *= .5
    return translated,joints,True,True


def reachable_ik_step(solve, previous_pose, requested_pose, seed, max_step_deg,
                      validate=None, position_priority=True):
    """Keep valid Cartesian directions available at a reachability boundary.

    Return ``target, joints, limited, rotation_limited, boundary_reason``.
    A declared IK/wrist/session boundary is recoverable here; invalid data,
    unknown errors and cancellation still propagate. Every accepted pose is
    solved and validated, and every joint stays within the supplied step bound.
    No joint is clipped independently of its Cartesian pose.

    Normally translation receives the joint-speed budget before rotation. If
    that path cannot move, try the coupled pose, then its translation planes,
    individual axes and rotation alone. This permits tangential/inward motion
    without clearing Grip or waiting for the robot to settle.
    """
    previous_pose, requested_pose = pose(previous_pose), pose(requested_pose)
    seed = vector(seed, 6)
    if isinstance(max_step_deg, (list, tuple)):
        bounds = vector(max_step_deg, 6)
    elif (isinstance(max_step_deg, (int, float)) and
          not isinstance(max_step_deg, bool)):
        bounds = [max_step_deg] * 6
    else:
        raise MappingError('Joint step bounds must be finite and positive')
    if any(not math.isfinite(value) or value <= 0 for value in bounds):
        raise MappingError('Joint step bounds must be finite and positive')

    def boundary_reason(error, validator=False):
        if (error.details.get('hard_fault') is True or
            error.details.get('state') in ('ROBOT_FAULT','HARD_FAULT','INVALID_INPUT')):
            return None
        state = error.details.get('state')
        if state in ('JOINT_SESSION_BOUND', 'WRIST_SINGULARITY', 'HARDWARE_BOUND'):
            return state
        if (not validator or error.details.get('retryable_ik') is True) and _recoverable_ik_error(error):
            return 'IK_REFUSED'
        return None

    def interpolate(origin, goal, fraction):
        translation = [a+(b-a)*fraction for a,b in zip(origin[:3],goal[:3])]
        rotation = multiply(goal[3:], inverse(origin[3:]))
        return translation+quat(multiply(scaled_rotation(rotation,fraction),origin[3:]))

    def attempt(goal, origin=previous_pose, solve_seed=seed):
        candidate = goal
        reason = None
        limited = False
        for _ in range(8):
            fraction = .5
            try:
                solved = solve(candidate, solve_seed)
            except MappingError as error:
                refused = boundary_reason(error)
                if refused is None:
                    raise
                reason = refused
            else:
                # Malformed results are not reachability refusals.
                joints = vector(solved, 6)
                try:
                    if validate is not None:
                        validate(joints)
                except MappingError as error:
                    refused = boundary_reason(error, validator=True)
                    if refused is None:
                        raise
                    reason = refused
                else:
                    ratios = [abs(a-b)/bound for a,b,bound in zip(joints,seed,bounds)]
                    if all(ratio <= 1.+1e-4/bound for ratio,bound in zip(ratios,bounds)):
                        return (candidate, joints, limited, reason), reason
                    fraction = min(.5, .8/max(ratios))
            candidate = interpolate(origin, candidate, fraction)
            limited = True
        return None, reason

    has_rotation = angle(multiply(requested_pose[3:],inverse(previous_pose[3:]))) > 1e-7
    translation_goal = requested_pose[:3]+previous_pose[3:]

    def finish_translation(solution, reason, force_limited=False):
        target,joints,limited,_ = solution
        if angle(multiply(requested_pose[3:],inverse(target[3:]))) < 1e-7:
            return target,joints,limited or force_limited,False,reason
        rotated,rotation_reason = attempt(target[:3]+requested_pose[3:],target,joints)
        if rotated is not None:
            target,joints,rotation_limited,_ = rotated
            return target,joints,limited or force_limited or rotation_limited,rotation_limited,reason or rotation_reason
        # Translation remains useful even when rotation is completely blocked.
        return target,joints,True,True,reason or rotation_reason

    first_reason = None
    translated = None
    if position_priority:
        translated, first_reason = attempt(translation_goal)
        if translated is not None and first_reason is None:
            return finish_translation(translated,None)
        if first_reason is None:
            # Exhausting a speed budget is not evidence of a reachability
            # boundary, and must not authorize changing the requested path.
            return previous_pose,seed,True,has_rotation,None

    if position_priority and not has_rotation:
        coupled,coupled_reason = translated,first_reason
    else:
        coupled,coupled_reason = attempt(requested_pose)
    reason = first_reason or coupled_reason
    if coupled is not None:
        target,joints,limited,coupled_reason = coupled
        if not limited or reason is None:
            return target,joints,limited,has_rotation and limited,reason
    if reason is None:
        return previous_pose,seed,True,has_rotation,None

    # Keep each projected direction in the same fixed work frame. Test larger
    # translation subsets first; candidates never reverse a requested axis.
    active_axes = [i for i in range(3) if abs(requested_pose[i]-previous_pose[i]) > 1e-10]
    masks = ([tuple(active_axes)] if active_axes else [])
    masks += [(i,j) for n,i in enumerate(active_axes) for j in active_axes[n+1:]]
    masks += [(i,) for i in active_axes]
    # The translation-priority route has already tested the full translation
    # subset; likewise the coupled route duplicates it when no rotation exists.
    seen = {tuple(active_axes)} if position_priority or not has_rotation else set()
    candidates = [candidate for candidate in (translated,coupled) if candidate is not None]
    def progress(candidate):
        return norm([a-b for a,b in zip(candidate[0][:3],previous_pose[:3])])
    best = max(candidates,key=progress) if candidates else None
    best_distance = progress(best) if best is not None else -1.
    for axes in masks:
        if axes in seen:
            continue
        seen.add(axes)
        goal = previous_pose.copy()
        for axis in axes:
            goal[axis] = requested_pose[axis]
        candidate, candidate_reason = attempt(goal)
        reason = reason or candidate_reason
        if candidate is not None:
            distance = norm([a-b for a,b in zip(candidate[0][:3],previous_pose[:3])])
            if distance > best_distance:
                best, best_distance = candidate, distance
    if best is not None:
        return finish_translation(best,reason or best[3],True)
    if has_rotation:
        rotated,rotation_reason = attempt(previous_pose[:3]+requested_pose[3:])
        reason = reason or rotation_reason
        if rotated is not None:
            target,joints,rotation_limited,_ = rotated
            return target,joints,True,rotation_limited,reason
    # Holding the last accepted command needs no new IK solution. Its seed is
    # already known valid; a later inward/tangential request is tested afresh.
    return previous_pose,seed,True,has_rotation,reason


def input_health(status, now_ns, max_age_ms=100, max_gap_ms=None,
                 require_head=True, last_sequence=None, check_pose=True, side='right'):
    """Classify transient missing frames separately from invalid tracking.

    The current PICO controller protocol omits controller ``status``. Its
    absence is reported, not invented as a tracking failure. A status supplied
    by a newer protocol must explicitly indicate valid tracking (3).
    """
    result = {'ready':False, 'recoverable':False, 'reason':'INVALID_INPUT',
              'input_age_ms':None, 'tracking_status_unknown':False}
    if not isinstance(status, dict):
        return result
    if not status.get('server_connected') or not status.get('device_connected'):
        return {**result, 'reason':'DISCONNECTED'}
    sample = status.get('latest')
    if not isinstance(sample, dict):
        return result
    if sample.get('app_focus') is not True:
        return {**result, 'reason':'FOCUS_LOST'}
    right = sample.get(side)
    if sample.get('input_mode') != 1 or not isinstance(right, dict):
        return result
    result['tracking_status_unknown'] = right.get('status') is None
    if right.get('status') is not None and right.get('status') != 3:
        return {**result, 'reason':'INVALID_TRACKING'}
    sequence = sample.get('sequence')
    if type(sequence) is not int:
        return result
    if last_sequence is not None and sequence < last_sequence:
        return {**result, 'reason':'RECEIVER_RESTARTED'}
    stamp = sample.get('receive_monotonic_ns')
    if type(stamp) is not int or type(now_ns) is not int or now_ns < stamp:
        return result
    result['input_age_ms'] = (now_ns-stamp)/1e6
    if check_pose:
        try:
            controller_pose(right.get('pose'))
        except MappingError:
            return {**result, 'reason':'INVALID_POSE'}
    if require_head:
        head = sample.get('head')
        if not isinstance(head, dict) or head.get('status') != 3:
            return {**result, 'reason':'WAIT_HEAD'}
        if check_pose:
            try:
                controller_pose(head.get('pose'))
            except MappingError:
                return {**result, 'reason':'WAIT_HEAD'}
    if result['input_age_ms'] > max_age_ms:
        short = max_gap_ms is not None and result['input_age_ms'] <= max_gap_ms
        return {**result, 'reason':'SHORT_GAP' if short else 'INPUT_TIMEOUT',
                'recoverable':short}
    return {**result, 'ready':True, 'reason':'FRESH'}


def input_ready(status, now_ns, max_age_ms=100, require_head=True, side='right'):
    # Pose/grip errors retain their distinct mapper states instead of being
    # collapsed into WAIT_FRESH_INPUT by the inexpensive admission check.
    return input_health(status,now_ns,max_age_ms,require_head=require_head,
                        check_pose=False,side=side)['ready']


class IncrementalMapper:
    def __init__(self, config, initial_robot_pose):
        self.config = config
        self.side=config.get('arm','right')
        if self.side not in ('left','right'):
            raise MappingError('Choose controller side left or right')
        if config.get('translation_mapping_frame','work')=='robot_body' and not config.get('robot_body_mapping_resolved'):
            raise MappingError('Resolve robot-body directions with actual work frame before mapping input')
        self.initial = pose(initial_robot_pose)
        self.target = self.initial.copy()
        self.anchor_hand = None
        self.anchor_robot = None
        self.previous_hand = None
        self.release_seen = False
        self.active = False
        self.last_sequence = None
        self.last_ns = None
        self.last_input_receive_ns = None
        self.last_input_pico_ns = None
        self.filtered_goal = self.initial.copy()
        self.paused = False
        self.pause_reason = None
        self.previous_relative = [1., 0., 0., 0.]
        self.rotation_mapping = config.get('rotation_mapping', 'world')
        self.operator_basis = None
        self.session_operator_basis = None
        self.operator_reference = None
        self.neutral_rotation_calibration = None
        self.filtered_twist_rad = 0.
        self.target_twist_rad = 0.
        self.rotation_boundary_offset = [1.,0.,0.,0.]
        self.rotation_path_mode = config.get('rotation_path_mode','anchor')
        self.rotation_path = None
        self.rotation_path_last_hand = None
        self.rotation_input_path_rad = 0.
        if self.rotation_path_mode not in ('anchor','incremental'):
            raise MappingError('Unknown rotation_path_mode')
        if not config.get('clutch_rotation_limits_enabled',True) and self.rotation_path_mode != 'incremental':
            raise MappingError('Unlimited rotation requires incremental rotation_path_mode')
        if self.rotation_mapping not in ('world', 'controller_local', 'neutral_tool_axes'):
            raise MappingError('Unknown rotation_mapping')
        self.tool_basis = config.get('controller_to_tool_matrix', [[1,0,0],[0,1,0],[0,0,1]])
        self.basis = config['headset_to_work_matrix']
        self.basis = self.checked_basis(self.basis)
        self.tool_basis = self.checked_basis(self.tool_basis)
        for key in ('position_scale', 'rotation_scale'):
            if (isinstance(config[key],bool) or not isinstance(config[key], (int, float))
                or not math.isfinite(config[key]) or config[key] < 0):
                raise MappingError('Scale must be finite and nonnegative')
        for key in ('max_session_translation_m', 'max_session_rotation_rad',
                    'max_linear_speed_m_s', 'max_angular_speed_rad_s',
                    'max_input_step_m', 'max_input_step_rad', 'max_input_age_ms'):
            if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
                raise MappingError('Invalid bound: ' + key)
        if config.get('input_jump_mode','step') not in ('step','velocity'):
            raise MappingError('Unknown input_jump_mode')
        if config.get('input_jump_mode','step') == 'velocity':
            for key in ('max_hand_linear_speed_m_s','max_hand_angular_speed_rad_s'):
                value = config.get(key)
                if (isinstance(value,bool) or not isinstance(value,(int,float))
                    or not math.isfinite(value) or value <= 0):
                    raise MappingError('Invalid hand speed bound: '+key)

    @staticmethod
    def checked_basis(basis):
        # 两个映射矩阵都必须是旋转矩阵；镜像/反射不能用于三维姿态换算。
        if not isinstance(basis, (list,tuple)) or len(basis) != 3:
            raise MappingError('Matrix must be 3x3')
        basis = [vector(row, 3) for row in basis]
        for i in range(3):
            for j in range(3):
                dot = sum(basis[i][k] * basis[j][k] for k in range(3))
                if abs(dot - (1 if i == j else 0)) > 1e-6:
                    raise MappingError('Mapping matrix is not orthonormal')
        a, b, c = basis
        determinant = (a[0]*(b[1]*c[2]-b[2]*c[1])
                       - a[1]*(b[0]*c[2]-b[2]*c[0])
                       + a[2]*(b[0]*c[1]-b[1]*c[0]))
        if abs(determinant - 1) > 1e-6:
            raise MappingError('Mapping matrix must have determinant +1')
        return basis

    def map_vector(self, v):
        return [sum(row[i] * v[i] for i in range(3)) for row in self.basis]

    def map_rotation(self, q):
        # 旧模式：手柄世界坐标中的转轴，转换为机械臂工作坐标中的转轴。
        return [q[0]] + self.map_vector(q[1:])

    def map_tool_rotation(self, q):
        # 新模式：手柄自身 XYZ 轴对应工具自身 XYZ 轴，默认 Z→Z（拧手腕→夹爪滚转）。
        return [q[0]] + [sum(row[i]*q[i+1] for i in range(3)) for row in self.tool_basis]

    def incremental_requested_rotation(self, hand):
        """Accumulate adjacent physical rotations in the fixed calibrated axes."""
        from rotation_axes import rotation_vector
        previous = self.rotation_path_last_hand
        if previous is None or self.rotation_path is None:
            raise MappingError('Incremental rotation requires a Grip anchor')
        room_step = multiply(hand[3:],inverse(previous))
        hand_step = rotation_vector(room_step)
        self.rotation_input_path_rad += norm(hand_step)
        if self.rotation_mapping in ('controller_local','neutral_tool_axes'):
            local_step = multiply(multiply(inverse(self.anchor_hand[3:]),room_step),
                                  self.anchor_hand[3:])
            mapped = rotation_vector(self.map_tool_rotation(local_step))
            if self.config.get('rotation_control_mode','full') == 'roll_only':
                mapped = [0.,0.,mapped[2]]
            world_step = rotate_vector(self.anchor_robot[3:],mapped)
        else:
            world_step = self.map_vector(hand_step)
        self.rotation_path.append([value*self.config['rotation_scale'] for value in world_step])
        self.rotation_path_last_hand = hand[3:].copy()
        return self.rotation_path.goal.copy()

    def rotation_remaining_rad(self):
        """Filtered angular path budget, including complete outstanding turns."""
        if self.rotation_path is not None:
            return self.rotation_path.remaining()
        return angle(multiply(self.filtered_goal[3:],inverse(self.target[3:])))

    def requested_goal(self, hand):
        """计算未平滑、未限速的操作意图；原始输入和目标均进入诊断日志。"""
        room_delta = [b-a for a,b in zip(self.anchor_hand[:3],hand[:3])]
        # 每次 Grip 以操作者身体正前方建立水平参考；握持期间固定，转头/转手柄不改平移轴。
        operator_delta = ([sum(a*b for a,b in zip(row,room_delta)) for row in self.operator_basis]
                          if self.operator_basis is not None else room_delta)
        delta = self.map_vector(operator_delta)
        requested_pos = [a+d*self.config['position_scale'] for a,d in zip(self.anchor_robot[:3],delta)]
        # q_anchor^-1 * q_now：在按下 Grip 时的手柄局部坐标里描述相对转动。
        local_delta = quat(multiply(inverse(self.anchor_hand[3:]),hand[3:]))
        if sum(a*b for a,b in zip(local_delta,self.previous_relative)) < 0:
            local_delta = [-x for x in local_delta]
        self.previous_relative = local_delta
        if self.rotation_path_mode == 'incremental':
            requested_quat = self.incremental_requested_rotation(hand)
            mapped_local = quat(multiply(inverse(self.anchor_robot[3:]),requested_quat))
        elif self.rotation_mapping in ('controller_local','neutral_tool_axes'):
            mapped_local = quat(self.map_tool_rotation(local_delta))
            out_of_range = (self.rotation_mapping=='neutral_tool_axes' and
                2*math.acos(max(-1.,min(1.,local_delta[0])))>self.config['max_clutch_rotation_rad'])
            # A rejected quaternion near -identity has no well-defined fractional
            # axis. Keep a finite diagnostic pose; this request is never emitted.
            delta_rotation = (multiply(inverse(self.anchor_robot[3:]),self.target[3:]) if out_of_range
                else scaled_rotation(mapped_local,self.config['rotation_scale'],
                    shortest=self.rotation_mapping!='neutral_tool_axes'))
            if self.rotation_mapping=='neutral_tool_axes' and self.config.get('rotation_control_mode')=='roll_only' and not out_of_range:
                from rotation_axes import twist_and_tilt
                twist,_=twist_and_tilt(mapped_local)
                if twist is None:
                    delta_rotation=multiply(inverse(self.anchor_robot[3:]),self.target[3:])
                else:
                    twist*=self.config['rotation_scale']
                    delta_rotation=[math.cos(twist/2),0.,0.,math.sin(twist/2)]
            # 必须右乘：围绕机器人起始工具自身轴转动，绝对手柄朝向不参与对齐。
            requested_quat = quat(multiply(self.anchor_robot[3:],delta_rotation))
        else:
            world_delta = multiply(hand[3:],inverse(self.anchor_hand[3:]))
            delta_rotation = scaled_rotation(self.map_rotation(world_delta),self.config['rotation_scale'])
            requested_quat = quat(multiply(delta_rotation,self.anchor_robot[3:]))
        # The boundary correction drops only an unreachable orientation intent.
        # It leaves the original Grip/tool/head calibration frames unchanged.
        if self.rotation_path_mode != 'incremental':
            requested_quat = quat(multiply(self.rotation_boundary_offset,requested_quat))
        diagnostic = {
            'right_hand_pose_wxyz':hand.copy(), 'anchor_hand_pose_wxyz':self.anchor_hand.copy(),
            'anchor_robot_pose_wxyz':self.anchor_robot.copy(),
            'hand_local_delta_wxyz':local_delta,
            'hand_rotation_deg':math.degrees(angle(local_delta)),
            'hand_unwrapped_rotation_rad':(self.rotation_input_path_rad if self.rotation_path_mode == 'incremental'
                else 2*math.acos(max(-1.,min(1.,local_delta[0])))),
            'rotation_mapping':self.rotation_mapping,
            'rotation_path_mode':self.rotation_path_mode,
            'raw_delta_pico_m':room_delta,'operator_delta_m':operator_delta,
            'operator_yaw_basis':self.operator_basis,
            'translation_heading_mode':self.config.get('translation_heading_mode','grip'),
            'boundary_reference_mode':self.config.get('boundary_reference_mode','rebase'),
            'requested_delta_work_m':[d*self.config['position_scale'] for d in delta],
            'requested_pose_wxyz':requested_pos+requested_quat}
        diagnostic['rotation_boundary_offset_wxyz'] = self.rotation_boundary_offset.copy()
        if self.rotation_mapping=='neutral_tool_axes':
            from rotation_axes import twist_and_tilt
            twist,tilt = twist_and_tilt(mapped_local)
            diagnostic.update(tool_twist_deg=None if twist is None else math.degrees(twist),
                tool_tilt_deg=math.degrees(tilt),
                rotation_control_mode=self.config.get('rotation_control_mode','full'),
                neutral_rotation_calibration=self.neutral_rotation_calibration)
        return requested_pos+requested_quat,diagnostic

    def rebase_rotation(self, actual):
        if self.rotation_path_mode == 'incremental':
            return
        if self.rotation_mapping=='neutral_tool_axes' and self.anchor_robot is not None:
            from rotation_axes import split_swing_twist
            relative=multiply(inverse(self.anchor_robot[3:]),actual[3:])
            _,twist=split_swing_twist(relative,self.target_twist_rad)
            self.target_twist_rad=self.filtered_twist_rad=twist

    def sync_target_rotation(self, target):
        """Commit an accepted IK pose without erasing the filtered hand intent.

        Keep the continuous twist cache near its previous turn count. This is
        necessary when IK or a follower scales a target after it has crossed
        180 degrees: resetting both caches would lose intent or reverse motion.
        """
        target = pose(target)
        if self.rotation_path_mode == 'incremental' and self.rotation_path is not None:
            self.rotation_path.commit(target[3:])
        elif self.rotation_mapping == 'neutral_tool_axes' and self.anchor_robot is not None:
            from rotation_axes import split_swing_twist
            relative = multiply(inverse(self.anchor_robot[3:]),target[3:])
            _,self.target_twist_rad = split_swing_twist(relative,self.target_twist_rad)
        self.target = target

    def saturate_boundary_goal(self, accepted_pose, translation_axes=(), rotation=False):
        """Discard blocked intent while preserving other pending directions.

        Call only after a declared geometric/joint/wrist refusal, never merely
        after a speed or follower limit. The correction makes the next inward
        hand movement immediately reachable instead of unwinding an invisible
        backlog. Translation axes are fixed work-frame components. Rotation
        uses a separate work-frame offset; original neutral/tool/hand frames
        and the 190-degree hand check remain unchanged.
        """
        accepted = pose(accepted_pose)
        axes = tuple(translation_axes)
        if any(type(axis) is not int or not 0 <= axis < 3 for axis in axes):
            raise MappingError('Boundary translation axes must be 0, 1 or 2')
        axes = tuple(dict.fromkeys(axes))
        if self.previous_hand is None or self.anchor_robot is None:
            raise MappingError('Boundary saturation requires an active Grip anchor')
        requested,_ = self.requested_goal(self.previous_hand)
        # Use the command's branch, rather than the distant hand goal, when
        # matching a quaternion sign across +180/-180 degrees.
        if sum(a*b for a,b in zip(accepted[3:],self.target[3:])) < 0:
            accepted[3:] = [-value for value in accepted[3:]]
        self.sync_target_rotation(accepted)
        for axis in axes:
            self.anchor_robot[axis] += accepted[axis]-requested[axis]
            self.filtered_goal[axis] = accepted[axis]
        if rotation:
            self.rotation_boundary_offset = quat(multiply(
                multiply(accepted[3:],inverse(requested[3:])),self.rotation_boundary_offset))
            self.filtered_goal[3:] = accepted[3:]
            self.filtered_twist_rad = self.target_twist_rad
            if self.rotation_path_mode == 'incremental':
                self.rotation_path.reset(accepted[3:])
                self.rotation_path_last_hand = self.previous_hand[3:].copy()

    def neutral_rotation_step(self, requested_quat, alpha, dt):
        from rotation_axes import split_swing_twist
        anchor=self.anchor_robot[3:]
        goal_swing,goal_twist=split_swing_twist(multiply(inverse(anchor),requested_quat))
        filter_swing,_=split_swing_twist(multiply(inverse(anchor),self.filtered_goal[3:]),
                                        self.filtered_twist_rad)
        filtered_twist=self.filtered_twist_rad+(goal_twist-self.filtered_twist_rad)*alpha
        swing_delta=multiply(goal_swing,inverse(filter_swing))
        filtered_swing=quat(multiply(scaled_rotation(swing_delta,alpha),filter_swing))
        filtered_local=quat(multiply(filtered_swing,
            [math.cos(filtered_twist/2),0.,0.,math.sin(filtered_twist/2)]))
        filtered_quat=quat(multiply(anchor,filtered_local))
        target_swing,target_twist=split_swing_twist(multiply(inverse(anchor),self.target[3:]),
                                                  self.target_twist_rad)
        swing_delta=multiply(filtered_swing,inverse(target_swing))
        twist_delta=filtered_twist-target_twist
        budget=abs(twist_delta)+angle(swing_delta)
        fraction=min(1.,self.config['max_angular_speed_rad_s']*dt/budget) if budget else 1.
        target_twist+=twist_delta*fraction
        target_swing=quat(multiply(scaled_rotation(swing_delta,fraction),target_swing))
        local=quat(multiply(target_swing,[math.cos(target_twist/2),0.,0.,math.sin(target_twist/2)]))
        self.filtered_twist_rad=filtered_twist
        self.target_twist_rad=target_twist
        return filtered_quat,quat(multiply(anchor,local))

    def pause(self, reason):
        # 软超限保留手柄/机器人锚点；回到原来的可用目标范围后，控制器再平滑恢复。
        self.paused = True
        self.pause_reason = reason

    def reject_bound(self, result):
        if self.config.get('auto_resume_soft_limits', False):
            self.pause(result['state'])
            result['soft_paused'] = True
        else:
            self.disarm()
        return result

    def disarm(self, reset_heading=False):
        # 断流、失焦、输入跳变等硬故障必须重新松开/按下 Grip，不能自动恢复。
        self.active = False
        self.release_seen = False
        self.anchor_hand = None
        self.last_sequence = None
        self.last_ns = None
        self.last_input_receive_ns = None
        self.last_input_pico_ns = None
        self.rotation_path = None
        self.rotation_path_last_hand = None
        self.paused = False
        self.pause_reason = None
        if reset_heading:
            self.operator_basis = None

    def update(self, status, now_ns, robot_pose_provider):
        require_head = not self.active or self.rotation_path_mode != 'incremental'
        if not input_ready(status, now_ns, self.config['max_input_age_ms'],require_head=require_head,side=self.side):
            self.disarm(reset_heading=True)
            return {'state': 'WAIT_FRESH_INPUT', 'send': False}
        sample = status['latest']
        ctrl = sample[self.side]
        grip = ctrl.get('grip')
        if isinstance(grip, bool) or not isinstance(grip, (int, float)) or not math.isfinite(grip) or not 0 <= grip <= 1:
            self.disarm()
            return {'state': 'INVALID_GRIP', 'send': False}
        try:
            hand = controller_pose(ctrl['pose'])
        except (KeyError, MappingError):
            self.disarm()
            return {'state': 'INVALID_POSE', 'send': False}
        if grip <= .2:
            self.active = False
            self.release_seen = True
            self.anchor_hand = None
            self.last_sequence = None
            self.paused = False
            self.pause_reason = None
            self.rotation_path = None
            self.rotation_path_last_hand = None
            return {'state': 'RELEASED', 'send': False}
        if not self.release_seen:
            return {'state': 'RELEASE_GRIP_TO_ARM', 'send': False}
        if not self.active:
            if grip < .7:
                return {'state': 'RELEASED', 'send': False}
            next_operator_basis = None
            if self.config.get('translation_reference','tracking_world')=='operator_heading':
                if self.config.get('translation_heading_mode','grip')=='session' and self.session_operator_basis is not None:
                    next_operator_basis = [row.copy() for row in self.session_operator_basis]
                else:
                    try:
                        next_operator_basis = operator_yaw_basis(sample['head'].get('pose'))
                    except MappingError as exc:
                        return {'state':'WAIT_OPERATOR_HEADING','send':False,'reason':str(exc)}
            try:
                robot_pose = pose(robot_pose_provider())
            except AnchorUnavailable:
                # No anchor has been committed. Retry with the latest hand,
                # head and measured robot pose; holding Grip must not replay
                # the hand movement made while the robot was settling.
                return {'state': 'WAIT_ROBOT_STATIONARY', 'send': False}
            if self.operator_reference is not None and next_operator_basis is not None:
                next_operator_basis = self.operator_reference(next_operator_basis)
            if self.rotation_mapping=='neutral_tool_axes' and self.neutral_rotation_calibration is None:
                from rotation_axes import neutral_tool_calibration
                try:
                    tool_basis,calibration = neutral_tool_calibration(hand,robot_pose,
                        next_operator_basis,self.basis)
                    if self.side=='left':
                        calibration['controller_side']='left'
                        calibration['neutral_pose_contract']='hand down, XY side toward operator'
                except MappingError as exc:
                    return {'state':'WAIT_NEUTRAL_ROTATION','send':False,'reason':str(exc)}
                self.tool_basis = tool_basis
                self.neutral_rotation_calibration = calibration
            # 只有机器人锚点成功建立才提交新朝向；接管失败时不保留一次未完成的标定。
            self.operator_basis = next_operator_basis
            if self.config.get('translation_heading_mode','grip')=='session' and next_operator_basis is not None:
                self.session_operator_basis = [row.copy() for row in next_operator_basis]
            self.anchor_hand = hand
            self.previous_hand = hand
            self.anchor_robot = robot_pose
            self.target = robot_pose.copy()
            self.filtered_goal = robot_pose.copy()
            self.filtered_twist_rad = self.target_twist_rad = 0.
            self.rotation_boundary_offset = [1.,0.,0.,0.]
            if self.rotation_path_mode == 'incremental':
                from rotation_axes import ContinuousRotationPath
                self.rotation_path = ContinuousRotationPath(robot_pose[3:])
                self.rotation_path_last_hand = hand[3:].copy()
                self.rotation_input_path_rad = 0.
            self.active = True
            self.previous_relative = [1.,0.,0.,0.]
            self.last_sequence = sample['sequence']
            self.last_ns = now_ns
            self.last_input_receive_ns = sample['receive_monotonic_ns']
            self.last_input_pico_ns = sample.get('pico_timestamp_ns')
            return {'state': 'ANCHORED', 'send': False, 'target_pose_wxyz': self.target,
                    'anchor_hand_pose_wxyz':hand.copy(),'anchor_robot_pose_wxyz':robot_pose.copy(),
                    'rotation_mapping':self.rotation_mapping,'operator_yaw_basis':self.operator_basis,
                    'neutral_rotation_calibration':self.neutral_rotation_calibration}
        reused = sample['sequence'] == self.last_sequence
        if reused and not self.config.get('advance_on_same_frame', False):
            return {'state': 'WAIT_NEW_FRAME', 'send': False}
        if sample['sequence'] < self.last_sequence:
            self.disarm(reset_heading=True)
            return {'state': 'RECEIVER_RESTARTED', 'send': False}
        step = [b-a for a, b in zip(self.previous_hand[:3], hand[:3])]
        step_rotation = multiply(hand[3:], inverse(self.previous_hand[3:]))
        input_diagnostic = {}
        if self.config.get('input_jump_mode','step') == 'velocity':
            # Repeated control ticks are not new tracking samples. Use device
            # sample time when available, otherwise receiver monotonic time.
            if not reused:
                input_dt = (sample['receive_monotonic_ns']-self.last_input_receive_ns)/1e9
                time_source = 'receive_monotonic_ns'
                pico_now,pico_before = sample.get('pico_timestamp_ns'),self.last_input_pico_ns
                if (isinstance(pico_now,(int,float)) and not isinstance(pico_now,bool)
                    and isinstance(pico_before,(int,float)) and not isinstance(pico_before,bool)
                    and math.isfinite(pico_now) and math.isfinite(pico_before)):
                    input_dt = (pico_now-pico_before)/1e9
                    time_source = 'pico_timestamp_ns'
                if input_dt <= 0:
                    self.disarm(reset_heading=True)
                    return {'state':'INPUT_TIMESTAMP_RESET','send':False,
                            'input_sample_dt_s':input_dt,'input_time_source':time_source}
                linear = norm(step)/input_dt
                angular = angle(step_rotation)/input_dt
                input_diagnostic.update(input_sample_dt_s=input_dt,input_time_source=time_source,
                    hand_linear_speed_m_s=linear,hand_angular_speed_rad_s=angular)
                if (linear > self.config['max_hand_linear_speed_m_s']
                    or angular > self.config['max_hand_angular_speed_rad_s']):
                    self.disarm(reset_heading=True)
                    return {'state':'INPUT_JUMP','send':False,**input_diagnostic}
        elif norm(step) > self.config['max_input_step_m'] or angle(step_rotation) > self.config['max_input_step_rad']:
            self.disarm(reset_heading=True)
            return {'state': 'INPUT_JUMP', 'send': False}
        dt = (now_ns - self.last_ns) / 1e9
        if dt <= 0 or (self.config.get('mapper_control_gap_guard',True)
                      and dt > self.config.get('max_control_gap_s', .1) and not self.paused):
            self.disarm()
            return {'state': 'CONTROL_GAP', 'send': False}
        requested, diagnostic = self.requested_goal(hand)
        requested_pos, requested_quat = requested[:3],requested[3:]
        # 记住每个有效输入，包括软暂停期间的输入；返回范围内不能误判为一次巨大跳变。
        self.previous_hand = hand
        self.last_sequence = sample['sequence']
        self.last_ns = now_ns
        if not reused:
            self.last_input_receive_ns = sample['receive_monotonic_ns']
            self.last_input_pico_ns = sample.get('pico_timestamp_ns')
        diagnostic.update(dt_s=dt,sequence=sample['sequence'],input_frame_reused=reused,**input_diagnostic)
        if self.config['rotation_scale'] > 0 and self.config.get('clutch_rotation_limits_enabled',True):
            calibrated = self.rotation_mapping=='neutral_tool_axes'
            wrap = (2*math.pi-.05 if calibrated else math.pi-.05)
            if diagnostic['hand_unwrapped_rotation_rad'] >= wrap:
                self.disarm()
                return {'state':'ROTATION_WRAP','send':False,**diagnostic}
            checked_rotation = (diagnostic['hand_unwrapped_rotation_rad'] if calibrated
                                else math.radians(diagnostic['hand_rotation_deg']))
            if checked_rotation > self.config.get('max_clutch_rotation_rad',3.12):
                return self.reject_bound({'state':'CLUTCH_ROTATION_BOUND','send':False,
                    'allowed_hand_rotation_deg':math.degrees(self.config.get('max_clutch_rotation_rad',3.12)),
                    **diagnostic})
            if calibrated and self.config.get('rotation_control_mode','full')=='full' and diagnostic['tool_tilt_deg'] > math.degrees(self.config['max_tool_tilt_rad'])+1e-6:
                return self.reject_bound({'state':'TOOL_TILT_BOUND','send':False,
                    'allowed_tool_tilt_deg':math.degrees(self.config['max_tool_tilt_rad']),**diagnostic})
        start_distance = norm([b-a for a, b in zip(self.initial[:3], requested_pos)])
        start_angle = angle(multiply(requested_quat, inverse(self.initial[3:])))
        if self.config.get('session_pose_limits_enabled',True) and (start_distance > self.config['max_session_translation_m'] or start_angle > self.config['max_session_rotation_rad']):
            reasons = []
            if start_distance > self.config['max_session_translation_m']:
                reasons.append('translation')
            if start_angle > self.config['max_session_rotation_rad']:
                reasons.append('rotation')
            rejected = {'state': 'SESSION_BOUND', 'send': False,
                        'limit_reason': reasons,
                        'requested_start_distance_mm': start_distance * 1000,
                        'allowed_start_distance_mm': self.config['max_session_translation_m'] * 1000,
                        'requested_start_angle_deg': math.degrees(start_angle),
                        'allowed_start_angle_deg': math.degrees(self.config['max_session_rotation_rad']),
                        **diagnostic,
                        'startup_pose_wxyz': self.initial.copy()}
            return self.reject_bound(rejected)
        if self.paused:
            return {'state':'RECOVERY_CHECK','send':False,'soft_paused':True,
                    'pause_reason':self.pause_reason,**diagnostic}
        # 每个控制周期推进绝对目标，即使没有新 PICO 帧也继续走向最新有效目标，避免重复帧降速。
        # 这里不累加手柄增量，因此同一帧重复计算不会使机器人漂移。
        tau = self.config.get('target_smoothing_tau_s', 0.)
        alpha = 1. if tau <= 0 else 1-math.exp(-dt/tau)
        filtered_pos = [a+(b-a)*alpha for a,b in zip(self.filtered_goal[:3], requested_pos)]
        if self.rotation_path_mode == 'incremental':
            filtered_quat,target_quat = self.rotation_path.advance(self.target[3:],alpha,
                self.config['max_angular_speed_rad_s']*dt)
            diagnostic['angular_remaining_rad'] = self.rotation_path.remaining()
            diagnostic['rotation_path_pending_segments'] = len(self.rotation_path.segments)
        elif self.rotation_mapping=='neutral_tool_axes':
            filtered_quat,target_quat=self.neutral_rotation_step(requested_quat,alpha,dt)
            diagnostic['filtered_tool_twist_deg']=math.degrees(self.filtered_twist_rad)
            diagnostic['target_tool_twist_deg']=math.degrees(self.target_twist_rad)
        else:
            filter_rotation = multiply(requested_quat, inverse(self.filtered_goal[3:]))
            filtered_quat = quat(multiply(scaled_rotation(filter_rotation, alpha), self.filtered_goal[3:]))
        self.filtered_goal = filtered_pos + filtered_quat
        translation = [b-a for a, b in zip(self.target[:3], filtered_pos)]
        distance = norm(translation)
        fraction = min(1., self.config['max_linear_speed_m_s'] * dt / distance) if distance else 1.
        target_pos = [a + d*fraction for a, d in zip(self.target[:3], translation)]
        if self.rotation_path_mode != 'incremental' and self.rotation_mapping!='neutral_tool_axes':
            rotation = multiply(filtered_quat, inverse(self.target[3:]))
            angular_distance = angle(rotation)
            fraction = min(1., self.config['max_angular_speed_rad_s'] * dt / angular_distance) if angular_distance else 1.
            target_quat = quat(multiply(scaled_rotation(rotation, fraction), self.target[3:]))
        old_target = self.target
        self.target = target_pos + target_quat
        changed = norm([a-b for a, b in zip(old_target[:3], target_pos)]) > 1e-7 or angle(multiply(target_quat, inverse(old_target[3:]))) > 1e-6
        return {'state':'TRACKING','send':changed,'target_pose_wxyz':self.target,
                'filtered_goal_pose_wxyz':self.filtered_goal.copy(),**diagnostic}
