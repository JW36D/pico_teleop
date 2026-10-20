"""Shape the final joint command; pure arithmetic, no robot APIs or I/O.

Joint angles retain their continuous IK representation (J6 is never reduced
modulo 360). Limits apply to the final, finite-difference command velocity and
acceleration, after IK. A changing goal may be passed briefly while braking;
hardware position limits instead have a stopping-distance budget.

``propose`` does not change state. Commit the accepted step only after command
gates and FK checks pass. ``advance`` is the convenient propose/commit form.
Local IK steps should supply ``target_velocity_deg_s`` so a second braking
stage does not mistake a nearby intermediate waypoint for the final goal.
"""
import math

from incremental import MappingError, inverse, multiply, norm, pose, quat, vector


def _positive(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value <= 0):
        raise MappingError('Invalid joint trajectory parameter: ' + name)
    return float(value)


def _braking_speed(clearance, acceleration, dt):
    """Reserve the next step and continuous stopping distance after that step.

    This invariant also covers changes in control period; it does not depend
    on the next period being identical to the current one.
    """
    if clearance <= 0:
        return 0.
    increment = acceleration*dt
    root = math.sqrt(increment*increment+2*acceleration*clearance)
    return (2*acceleration*clearance)/(root+increment)


def _capability(backend, names, size, fallback, label):
    values = next((getattr(backend, name) for name in names
                   if getattr(backend, name, None) is not None), None)
    if values is None:
        return [fallback]*size, 'configured_only_device_limit_unavailable'
    values = vector(values, size)
    if any(value <= 0 for value in values):
        raise MappingError('Invalid device joint capability: ' + label)
    return values, 'controller_query'


def _rotation_log(rotation):
    """World-axis log of a local rotation; invariant to quaternion sign."""
    rotation = quat(rotation)
    if rotation[0] < 0:
        rotation = [-value for value in rotation]
    sine = norm(rotation[1:])
    if sine < 1e-14:
        return [0., 0., 0.]
    factor = 2*math.atan2(sine, rotation[0])/sine
    return [value*factor for value in rotation[1:]]


def _solve_positive_definite(matrix, rhs):
    """Small Cholesky solve; damping makes every reduced QP system positive."""
    size = len(rhs)
    lower = [[0.]*size for _ in range(size)]
    for row in range(size):
        for column in range(row+1):
            value = matrix[row][column]-sum(lower[row][k]*lower[column][k]
                                          for k in range(column))
            if row == column:
                if value <= 0 or not math.isfinite(value):
                    raise MappingError('Joint Cartesian optimizer matrix is not positive definite')
                lower[row][column] = math.sqrt(value)
            else:
                lower[row][column] = value/lower[column][column]
    intermediate = [0.]*size
    for row in range(size):
        intermediate[row] = (rhs[row]-sum(lower[row][k]*intermediate[k]
                                        for k in range(row)))/lower[row][row]
    solution = [0.]*size
    for row in range(size-1, -1, -1):
        solution[row] = (intermediate[row]-sum(lower[k][row]*solution[k]
                                             for k in range(row+1, size)))/lower[row][row]
    return solution


def _box_qp(matrix, rhs, lower, upper, initial):
    """Solve a positive-definite quadratic over a box, including KKT release.

    Free axes are solved jointly. An infeasible free optimum is approached
    until the first bound is hit; bounds are released when their gradient has
    the wrong KKT sign. This avoids the incorrect 'solve then independently
    clamp each axis' approximation that distorts a Cartesian path.
    """
    size = len(rhs)
    value = [max(low, min(high, start)) for low, high, start in
             zip(lower, upper, initial)]
    active = [(-2 if low == high else -1 if x == low else 1 if x == high else 0)
              for x, low, high in zip(value, lower, upper)]
    scale = max([abs(entry) for entry in rhs]+[1e-16])
    tolerance = scale*1e-9
    releases = 0
    for iteration in range(128):
        free = [index for index in range(size) if active[index] == 0]
        fixed = [index for index in range(size) if active[index] != 0]
        if free:
            reduced = [[matrix[row][column] for column in free] for row in free]
            reduced_rhs = [rhs[row]-sum(matrix[row][column]*value[column]
                                       for column in fixed) for row in free]
            optimum = _solve_positive_definite(reduced, reduced_rhs)
            direction = [optimum[k]-value[index] for k, index in enumerate(free)]
            fraction, hit, hit_side = 1., None, 0
            for index, delta in zip(free, direction):
                if delta > 0 and value[index]+delta > upper[index]:
                    candidate = (upper[index]-value[index])/delta
                    if candidate < fraction:
                        fraction, hit, hit_side = candidate, index, 1
                elif delta < 0 and value[index]+delta < lower[index]:
                    candidate = (lower[index]-value[index])/delta
                    if candidate < fraction:
                        fraction, hit, hit_side = candidate, index, -1
            for index, delta in zip(free, direction):
                value[index] += fraction*delta
            if hit is not None:
                # This is an exact active-set intersection, not post-solve
                # clipping of a robot command or its acceleration.
                value[hit] = upper[hit] if hit_side == 1 else lower[hit]
                active[hit] = hit_side
                continue
        gradient = [sum(matrix[row][column]*value[column] for column in range(size))-rhs[row]
                    for row in range(size)]
        violations = [(max(0., -gradient[index]) if side == -1 else
                       max(0., gradient[index]) if side == 1 else 0., index)
                      for index, side in enumerate(active)]
        violation, release = max(violations)
        if violation <= tolerance:
            kkt = max([abs(gradient[index]) for index in range(size) if active[index] == 0]+
                      [violation, 0.])
            return value, {'iterations': iteration+1, 'bound_releases': releases,
                           'kkt_residual': kkt, 'converged': True}
        active[release] = 0
        releases += 1
    raise MappingError('Joint Cartesian optimizer did not converge numerically',
                       {'state': 'JOINT_TRAJECTORY_OPTIMIZER_ERROR'})


def _cartesian_jacobian(backend, joints, minimum, maximum):
    """Central FK differences in m/degree and world radians/degree.

    A 0.01 degree difference avoids excess relative error from the native SDK's
    float32 outputs. At an actual hardware boundary use a one-sided difference
    entirely within legal angles. The backend FK callback must be local math.
    """
    prior = pose(backend.forward(joints.copy()))
    jacobian = [[0.]*len(joints) for _ in range(6)]
    difference_deg = .01
    for index, (joint, low, high) in enumerate(zip(joints, minimum, maximum)):
        positive, negative = min(difference_deg, high-joint), min(difference_deg, joint-low)
        plus, minus = prior, prior
        if positive:
            candidate = joints.copy()
            candidate[index] += positive
            plus = pose(backend.forward(candidate))
        if negative:
            candidate = joints.copy()
            candidate[index] -= negative
            minus = pose(backend.forward(candidate))
        denominator = positive+negative
        if denominator <= 0:
            raise MappingError('No legal FK finite difference inside hardware limits')
        difference = [(a-b)/denominator for a, b in zip(plus[:3], minus[:3])]
        difference += [value/denominator for value in
                       _rotation_log(multiply(plus[3:], inverse(minus[3:])))]
        for row, value in enumerate(difference):
            jacobian[row][index] = value
    return prior, jacobian, difference_deg


class JointTrajectory:
    def __init__(self, config, backend):
        self.minimum = vector(backend.minimum, len(backend.minimum))
        self.maximum = vector(backend.maximum, len(self.minimum))
        self.size = len(self.minimum)
        if not self.size or any(low >= high for low, high in
                                zip(self.minimum, self.maximum)):
            raise MappingError('Invalid hardware joint position limits')
        speed = _positive(config['max_joint_speed_deg_s'], 'max_joint_speed_deg_s')
        acceleration = _positive(config['max_joint_accel_deg_s2'], 'max_joint_accel_deg_s2')
        self.hardware_speeds, self.speed_source = _capability(
            backend, ('speed_limits', 'joint_max_speed_deg_s'), self.size, speed, 'speed')
        self.hardware_accelerations, self.acceleration_source = _capability(
            backend, ('acceleration_limits', 'joint_max_accel_deg_s2'),
            self.size, acceleration, 'acceleration')
        self.speed_limits = [min(speed, value) for value in self.hardware_speeds]
        self.acceleration_limits = [min(acceleration, value)
                                    for value in self.hardware_accelerations]
        self.cartesian_rotation_weight = _positive(config.get('cartesian_rotation_weight_m',
            config.get('max_linear_speed_m_s', .25)/config.get('max_angular_speed_rad_s', 1.4)),
            'cartesian_rotation_weight_m')
        self.joints = None
        self.velocity = [0.]*self.size
        if getattr(backend, 'initial_joints', None) is not None:
            self.reset(backend.initial_joints)

    def _positions(self, joints):
        joints = vector(joints, self.size)
        if any(not low <= value <= high for value, low, high in
               zip(joints, self.minimum, self.maximum)):
            raise MappingError('Joint trajectory position crosses hardware limit',
                               {'state': 'HARDWARE_BOUND', 'joints_deg': joints})
        return joints

    def reset(self, joints, velocity=None):
        """Start from fresh measured position/velocity, without creating a step.

        Measured velocity above the selected task speed is decelerated smoothly
        on subsequent steps. Velocity above a real device speed limit is an
        invalid state rather than silently clipped data.
        """
        joints = self._positions(joints)
        velocity = vector(velocity if velocity is not None else [0.]*self.size,
                          self.size)
        if any(abs(value) > maximum+1e-8 for value, maximum in
               zip(velocity, self.hardware_speeds)):
            raise MappingError('Measured joint velocity exceeds device capability',
                               {'state': 'JOINT_TRAJECTORY_INFEASIBLE',
                                'joint_velocity_deg_s': velocity})
        self.joints, self.velocity = joints, velocity

    def propose(self, goal_joints, dt, target_velocity_deg_s=None, speed_scale=1.):
        """Return a final command candidate without modifying the trajectory.

        With no velocity input, ``goal_joints`` is the full continuous target.
        With velocity input it is a local IK waypoint whose upstream Cartesian
        trajectory already handles target approach and braking. ``speed_scale``
        scales desired speed only; an abrupt slowdown still brakes continuously.
        """
        if self.joints is None:
            raise MappingError('Joint trajectory must be reset before proposing a step')
        goal = self._positions(goal_joints)
        dt = _positive(dt, 'dt')
        if (isinstance(speed_scale, bool) or not isinstance(speed_scale, (int, float)) or
                not math.isfinite(speed_scale) or not 0 <= speed_scale <= 1):
            raise MappingError('Joint trajectory speed_scale must be between zero and one')
        feedforward = (vector(target_velocity_deg_s, self.size)
                       if target_velocity_deg_s is not None else None)
        next_joints, velocities, accelerations = [], [], []
        position_braking, speed_limited, acceleration_limited, startup_recovery = [], [], [], []
        desired_velocities = []
        for index, (previous, old_velocity, target, low, high, speed, acceleration) in enumerate(
                zip(self.joints, self.velocity, goal, self.minimum, self.maximum,
                    self.speed_limits, self.acceleration_limits)):
            change = acceleration*dt
            distance = target-previous
            if feedforward is None:
                requested_speed = _braking_speed(abs(distance), acceleration, dt)
                desired = math.copysign(min(speed, requested_speed), distance) if distance else 0.
            else:
                desired = max(-speed, min(speed, feedforward[index]))
            desired *= speed_scale
            desired_velocities.append(desired)
            # Preserve a valid moving handover above the selected task speed,
            # decelerating to that speed within the normal acceleration bound.
            effective_speed = max(speed, abs(old_velocity)-change)
            speed_low, speed_high = -effective_speed, effective_speed
            accel_low, accel_high = old_velocity-change, old_velocity+change
            boundary_low = -_braking_speed(previous-low, acceleration, dt)
            boundary_high = _braking_speed(high-previous, acceleration, dt)
            allowed_low = max(speed_low, accel_low, boundary_low)
            allowed_high = min(speed_high, accel_high, boundary_high)
            if allowed_low > allowed_high+1e-10:
                raise MappingError('Cannot brake joint before its hardware position limit',
                    {'state': 'JOINT_TRAJECTORY_INFEASIBLE', 'joint_index': index,
                     'joint_position_deg': previous, 'joint_velocity_deg_s': old_velocity,
                     'hardware_min_deg': low, 'hardware_max_deg': high,
                     'joint_acceleration_limit_deg_s2': acceleration, 'dt_s': dt})
            velocity = max(allowed_low, min(allowed_high, desired))
            emitted = previous+velocity*dt
            # The stored derivative is computed from the actual floating-point
            # command, which is exactly what data/diagnostics must describe.
            emitted_velocity = (emitted-previous)/dt
            emitted_acceleration = (emitted_velocity-old_velocity)/dt
            next_joints.append(emitted)
            velocities.append(emitted_velocity)
            accelerations.append(emitted_acceleration)
            position_braking.append(desired < boundary_low-1e-10 or desired > boundary_high+1e-10)
            speed_limited.append(feedforward is not None and abs(feedforward[index])*speed_scale > speed+1e-10)
            acceleration_limited.append(desired < accel_low-1e-10 or desired > accel_high+1e-10)
            startup_recovery.append(abs(old_velocity) > speed+1e-8)
        self._validate_step(next_joints, dt)
        diagnostics = {
            'joint_trajectory_shaping': True,
            'joint_trajectory_velocity_feedforward': feedforward is not None,
            'joint_velocity_deg_s': velocities,
            'joint_acceleration_deg_s2': accelerations,
            'joint_desired_velocity_deg_s': desired_velocities,
            'max_emitted_joint_speed_deg_s': max(abs(value) for value in velocities),
            'max_emitted_joint_acceleration_deg_s2': max(abs(value) for value in accelerations),
            'effective_joint_speed_limits_deg_s': self.speed_limits.copy(),
            'effective_joint_acceleration_limits_deg_s2': self.acceleration_limits.copy(),
            'joint_speed_limit_source': self.speed_source,
            'joint_acceleration_limit_source': self.acceleration_source,
            'joint_position_braking': position_braking,
            'joint_speed_limited': speed_limited,
            'joint_acceleration_limited': acceleration_limited,
            'joint_startup_speed_recovery': startup_recovery,
            'joint_desired_speed_scale': float(speed_scale),
        }
        return next_joints, diagnostics

    # Explicit name for callers preparing a command before sender validation.
    preview = propose

    def _velocity_bounds(self, dt):
        lower, upper, boundary_lower, boundary_upper = [], [], [], []
        for index, (joint, previous_velocity, low, high, speed, acceleration) in enumerate(
                zip(self.joints, self.velocity, self.minimum, self.maximum,
                    self.speed_limits, self.acceleration_limits)):
            change = acceleration*dt
            effective_speed = max(speed, abs(previous_velocity)-change)
            boundary_low = -_braking_speed(joint-low, acceleration, dt)
            boundary_high = _braking_speed(high-joint, acceleration, dt)
            allowed_low = max(-effective_speed, previous_velocity-change, boundary_low)
            allowed_high = min(effective_speed, previous_velocity+change, boundary_high)
            if allowed_low > allowed_high:
                raise MappingError('Cannot brake joint before its hardware position limit',
                    {'state': 'JOINT_TRAJECTORY_INFEASIBLE', 'joint_index': index,
                     'joint_position_deg': joint, 'joint_velocity_deg_s': previous_velocity,
                     'hardware_min_deg': low, 'hardware_max_deg': high,
                     'joint_acceleration_limit_deg_s2': acceleration, 'dt_s': dt})
            lower.append(allowed_low)
            upper.append(allowed_high)
            boundary_lower.append(boundary_low)
            boundary_upper.append(boundary_high)
        return lower, upper, boundary_lower, boundary_upper

    def propose_cartesian(self, target_pose, dt, backend,
                          target_cartesian_velocity=None, desired_velocity=None):
        """Jointly minimize 6D Cartesian rate error inside final joint limits.

        ``target_pose`` is an already-shaped local step, including the expanded
        signed rotation direction. Optional velocity is [vx,vy,vz,wx,wy,wz] in
        m/s and world rad/s; it avoids reinterpreting that waypoint after a
        sender-period change. ``desired_velocity`` is its compatibility alias.
        The tiny numerical tie-breaker favors continuous and small null-space
        motion. No wrist-angle exclusion, arbitrary speed cap, or I/O is added.
        """
        if self.joints is None:
            raise MappingError('Joint trajectory must be reset before proposing a step')
        dt = _positive(dt, 'dt')
        target = pose(target_pose)
        if target_cartesian_velocity is not None and desired_velocity is not None:
            raise MappingError('Specify only one Cartesian velocity input')
        if target_cartesian_velocity is None:
            target_cartesian_velocity = desired_velocity
        prior, jacobian, finite_difference = _cartesian_jacobian(
            backend, self.joints, self.minimum, self.maximum)
        desired = (vector(target_cartesian_velocity, 6)
                   if target_cartesian_velocity is not None else
                   [(a-b)/dt for a, b in zip(target[:3], prior[:3])]+
                   [value/dt for value in _rotation_log(multiply(target[3:], inverse(prior[3:])))])
        lower, upper, boundary_lower, boundary_upper = self._velocity_bounds(dt)
        weights = [1., 1., 1.]+[self.cartesian_rotation_weight]*3
        weighted = [[entry*weights[row] for entry in jacobian[row]] for row in range(6)]
        request = [entry*weight for entry, weight in zip(desired, weights)]
        matrix = [[sum(weighted[row][i]*weighted[row][j] for row in range(6))
                   for j in range(self.size)] for i in range(self.size)]
        diagonal_scale = max(matrix[index][index] for index in range(self.size))
        regularization = (diagonal_scale if diagonal_scale > 0 else 1.)*1e-8
        rhs = [sum(weighted[row][index]*request[row] for row in range(6))+
               regularization*self.velocity[index] for index in range(self.size)]
        for index in range(self.size):
            matrix[index][index] += 2*regularization
        velocity, solver = _box_qp(matrix, rhs, lower, upper, self.velocity)
        accepted = [joint+speed*dt for joint, speed in zip(self.joints, velocity)]
        accepted, velocity = self._validate_step(accepted, dt)
        acceleration = [(speed-prior_speed)/dt for speed, prior_speed in zip(velocity, self.velocity)]
        achieved = [sum(jacobian[row][index]*velocity[index] for index in range(self.size))
                    for row in range(6)]
        residual = [value-requested for value, requested in zip(achieved, desired)]
        diagnostics = {
            'joint_trajectory_shaping': True,
            'joint_trajectory_velocity_feedforward': target_cartesian_velocity is not None,
            'joint_trajectory_cartesian_optimizer': 'box_constrained_active_set',
            'joint_velocity_deg_s': velocity,
            'joint_acceleration_deg_s2': acceleration,
            'max_emitted_joint_speed_deg_s': max(abs(value) for value in velocity),
            'max_emitted_joint_acceleration_deg_s2': max(abs(value) for value in acceleration),
            'effective_joint_speed_limits_deg_s': self.speed_limits.copy(),
            'effective_joint_acceleration_limits_deg_s2': self.acceleration_limits.copy(),
            'joint_speed_limit_source': self.speed_source,
            'joint_acceleration_limit_source': self.acceleration_source,
            'joint_position_braking': [
                (abs(speed-bound_low) < 1e-7 and bound_low >= low) or
                (abs(speed-bound_high) < 1e-7 and bound_high <= high)
                for speed, bound_low, bound_high, low, high in
                zip(velocity, boundary_lower, boundary_upper, lower, upper)],
            'joint_acceleration_limited': [abs(speed-prior_speed) >= maximum*dt-1e-7
                for speed, prior_speed, maximum in zip(velocity, self.velocity, self.acceleration_limits)],
            'joint_speed_limited': [abs(speed) >= maximum-1e-7
                                    for speed, maximum in zip(velocity, self.speed_limits)],
            'joint_startup_speed_recovery': [abs(speed) > maximum+1e-8
                                           for speed, maximum in zip(self.velocity, self.speed_limits)],
            'joint_velocity_lower_bounds_deg_s': lower,
            'joint_velocity_upper_bounds_deg_s': upper,
            'cartesian_rotation_weight_m': self.cartesian_rotation_weight,
            'cartesian_requested_velocity_m_s_rad_s': desired,
            'cartesian_achieved_velocity_m_s_rad_s': achieved,
            'cartesian_velocity_residual_m_s_rad_s': residual,
            'cartesian_linear_velocity_residual_m_s': norm(residual[:3]),
            'cartesian_angular_velocity_residual_rad_s': norm(residual[3:]),
            'cartesian_weighted_velocity_residual_m_s': norm([value*weight for value, weight in zip(residual, weights)]),
            'cartesian_jacobian_difference_deg': finite_difference,
            'joint_optimizer_regularization': regularization,
            'joint_optimizer_iterations': solver['iterations'],
            'joint_optimizer_bound_releases': solver['bound_releases'],
            'joint_optimizer_kkt_residual': solver['kkt_residual'],
            'joint_optimizer_converged': solver['converged'],
        }
        return accepted, diagnostics

    def _validate_step(self, joints, dt):
        joints = self._positions(joints)
        velocities = [(value-previous)/dt for value, previous in zip(joints, self.joints)]
        accelerations = [(value-previous)/dt for value, previous in zip(velocities, self.velocity)]
        for index, (velocity, acceleration, old_velocity, speed, acceleration_limit, hardware_speed) in enumerate(
                zip(velocities, accelerations, self.velocity, self.speed_limits,
                    self.acceleration_limits, self.hardware_speeds)):
            effective_speed = max(speed, abs(old_velocity)-acceleration_limit*dt)
            if (abs(velocity) > min(effective_speed, hardware_speed)+1e-7 or
                    abs(acceleration) > acceleration_limit+1e-6):
                raise MappingError('Final joint command violates trajectory bounds',
                    {'state': 'JOINT_TRAJECTORY_INFEASIBLE', 'joint_index': index,
                     'joint_velocity_deg_s': velocity,
                     'joint_acceleration_deg_s2': acceleration})
        return joints, velocities

    def commit(self, accepted_joints, dt):
        """Commit precisely the checked final command; reject external clipping."""
        if self.joints is None:
            raise MappingError('Joint trajectory must be reset before committing a step')
        dt = _positive(dt, 'dt')
        joints, velocities = self._validate_step(accepted_joints, dt)
        self.joints, self.velocity = joints, velocities

    def advance(self, goal_joints, dt, target_velocity_deg_s=None, speed_scale=1.):
        joints, diagnostics = self.propose(goal_joints, dt,
            target_velocity_deg_s=target_velocity_deg_s, speed_scale=speed_scale)
        self.commit(joints, dt)
        return joints, diagnostics
