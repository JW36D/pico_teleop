"""Continuous Cartesian command shaping without robot APIs or I/O.

The tracking governor reduces only velocity pointing away from measured robot
feedback. It does not freeze the target or discard inward/tangential commands.
The configured lead is a soft threshold: acceleration-limited braking may use
the separately validated stopping margin before the controller's hard stop.

Rotation direction comes from the mapper's small proposed increment, not the
shortest path to its full goal; this preserves signed twists beyond 180 degrees.
Call ``sync`` after IK accepts a step so subsequent acceleration starts from
the velocity actually emitted rather than accumulating an unexecuted command.
"""
import math

from incremental import MappingError, inverse, multiply, norm, pose, quat, vector


def _positive(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value <= 0):
        raise MappingError('Invalid continuous motion bound: ' + name)
    return float(value)


def _dot(left, right):
    return sum(a*b for a, b in zip(left, right))


def _rotation_vector(rotation):
    """World-axis logarithm of a small rotation, invariant to quaternion sign."""
    rotation = quat(rotation)
    if rotation[0] < 0:
        rotation = [-v for v in rotation]
    sine = norm(rotation[1:])
    if sine < 1e-12:
        return [0., 0., 0.]
    scale = 2*math.atan2(sine, rotation[0])/sine
    return [v*scale for v in rotation[1:]]


def _rotation_from_vector(rotation):
    magnitude = norm(rotation)
    if magnitude < 1e-12:
        return [1., 0., 0., 0.]
    scale = math.sin(magnitude/2)/magnitude
    return [math.cos(magnitude/2)]+[v*scale for v in rotation]


def _toward(previous, desired, maximum_change):
    delta = [b-a for a, b in zip(previous, desired)]
    size = norm(delta)
    fraction = min(1., maximum_change/size) if size else 1.
    return [a+d*fraction for a, d in zip(previous, delta)]


def _braking_speed(distance, acceleration, dt):
    # Reserve this discrete step as well as the subsequent stopping distance.
    # The equivalent form avoids catastrophic cancellation near zero distance.
    initial = acceleration*dt
    root = math.sqrt(initial*initial+2*acceleration*distance)
    return (2*acceleration*distance)/(root+initial) if distance else 0.


def _govern(desired, lead, maximum, curve='quartic', full_speed_ratio=.7):
    """Slow outward velocity continuously; preserve inward and tangent parts."""
    distance = norm(lead)
    if distance < 1e-12:
        return desired.copy(), 1.
    direction = [v/distance for v in lead]
    outward = _dot(desired, direction)
    if outward <= 0:
        return desired.copy(), 1.
    ratio = min(1., distance/maximum)
    if curve=='plateau':
        progress=max(0.,min(1.,(ratio-full_speed_ratio)/(1-full_speed_ratio)))
        factor=1-progress*progress*(3-2*progress)
    else:
        factor = max(0., 1-ratio**4)
    return [v-outward*(1-factor)*d for v, d in zip(desired, direction)], factor


class ContinuousMotionShaper:
    def __init__(self, config):
        self.linear_speed = _positive(config['max_linear_speed_m_s'], 'max_linear_speed_m_s')
        self.angular_speed = _positive(config['max_angular_speed_rad_s'], 'max_angular_speed_rad_s')
        self.linear_accel = _positive(config.get('max_linear_accel_m_s2', 3.), 'max_linear_accel_m_s2')
        self.angular_accel = _positive(config.get('max_angular_accel_rad_s2', 12.), 'max_angular_accel_rad_s2')
        self.linear_lead = _positive(config.get('max_command_lead_m', .02), 'max_command_lead_m')
        self.angular_lead = _positive(config.get('max_command_lead_angle_rad', .12), 'max_command_lead_angle_rad')
        self.governor_curve = config.get('tracking_governor_curve','quartic')
        self.full_speed_ratio = config.get('command_lead_full_speed_ratio',.7)
        self.reset()

    def reset(self):
        self.linear_velocity = [0., 0., 0.]
        self.angular_velocity = [0., 0., 0.]

    def sync(self, prior_pose, emitted_pose, dt):
        """Commit the velocity of the step that IK/controller actually accepted."""
        prior, emitted = pose(prior_pose), pose(emitted_pose)
        dt = _positive(dt, 'dt')
        linear_velocity = vector([(b-a)/dt for a, b in zip(prior[:3], emitted[:3])], 3)
        rotation = _rotation_vector(multiply(emitted[3:], inverse(prior[3:])))
        angular_velocity = vector([v/dt for v in rotation], 3)
        self.linear_velocity, self.angular_velocity = linear_velocity, angular_velocity

    def advance(self, prior_pose, filtered_goal_pose, proposed_pose, actual_pose,
                dt, governed=True, angular_remaining_rad=None):
        """Return a candidate pose and diagnostics, without committing velocity.

        ``angular_remaining_rad`` is a nonnegative path-length braking budget.
        For a signed neutral twist the caller should supply its unwrapped twist
        difference plus swing difference. The fallback quaternion distance is
        only a budget; it never chooses rotation direction.

        A sudden target change can briefly carry the command past the new goal
        while braking; freezing exactly at that point would violate acceleration.
        A speed correction after an external oversized sync can override the
        acceleration limit and is explicitly diagnosed. IK or hard-stop
        intervention also remains outside this module's acceleration guarantee.
        """
        prior, goal = pose(prior_pose), pose(filtered_goal_pose)
        proposed, actual = pose(proposed_pose), pose(actual_pose)
        dt = _positive(dt, 'dt')
        if not isinstance(governed, bool):
            raise MappingError('governed must be boolean')
        if angular_remaining_rad is not None and (
                isinstance(angular_remaining_rad, bool) or
                not isinstance(angular_remaining_rad, (int, float)) or
                not math.isfinite(angular_remaining_rad) or angular_remaining_rad < 0):
            raise MappingError('Invalid angular remaining path length')

        translation = [b-a for a, b in zip(prior[:3], goal[:3])]
        distance = norm(translation)
        if not math.isfinite(distance):
            raise MappingError('Nonfinite continuous motion distance')
        speed = min(self.linear_speed, _braking_speed(distance, self.linear_accel, dt))
        desired_linear = [v*speed/distance for v in translation] if distance else [0., 0., 0.]

        proposal_rotation = _rotation_vector(multiply(proposed[3:], inverse(prior[3:])))
        proposal_angle = norm(proposal_rotation)
        if angular_remaining_rad is None:
            angular_remaining_rad = norm(_rotation_vector(multiply(goal[3:], inverse(prior[3:]))))
        angular_speed = min(self.angular_speed, proposal_angle/dt,
                            _braking_speed(angular_remaining_rad, self.angular_accel, dt))
        desired_angular = ([v*angular_speed/proposal_angle for v in proposal_rotation]
                           if proposal_angle else [0., 0., 0.])

        linear_lead = [a-b for a, b in zip(prior[:3], actual[:3])]
        angular_lead = _rotation_vector(multiply(prior[3:], inverse(actual[3:])))
        if not math.isfinite(norm(linear_lead)):
            raise MappingError('Nonfinite continuous motion lead')
        linear_factor = angular_factor = 1.
        if governed:
            desired_linear, linear_factor = _govern(desired_linear, linear_lead, self.linear_lead,
                self.governor_curve,self.full_speed_ratio)
            desired_angular, angular_factor = _govern(desired_angular, angular_lead, self.angular_lead,
                self.governor_curve,self.full_speed_ratio)
        linear = _toward(self.linear_velocity, desired_linear, self.linear_accel*dt)
        angular = _toward(self.angular_velocity, desired_angular, self.angular_accel*dt)

        overrides = []
        for velocity, maximum, label in ((linear, self.linear_speed, 'linear'),
                                         (angular, self.angular_speed, 'angular')):
            magnitude = norm(velocity)
            if magnitude > maximum+1e-12:
                velocity[:] = [v*maximum/magnitude for v in velocity]
                overrides.append(label+'_speed_clip')

        target = [a+v*dt for a, v in zip(prior[:3], linear)]
        target += quat(multiply(_rotation_from_vector([v*dt for v in angular]), prior[3:]))
        target = pose(target)
        linear_accel = norm([b-a for a, b in zip(self.linear_velocity, linear)])/dt
        angular_accel = norm([b-a for a, b in zip(self.angular_velocity, angular)])/dt
        diagnostics = {
            'continuous_shaping': True,
            'tracking_governor_curve':self.governor_curve,
            'linear_velocity_m_s': linear,
            'angular_velocity_rad_s': angular,
            'linear_speed_m_s': norm(linear),
            'angular_speed_rad_s': norm(angular),
            'linear_accel_m_s2': linear_accel,
            'angular_accel_rad_s2': angular_accel,
            'linear_governor_ratio': linear_factor,
            'angular_governor_ratio': angular_factor,
            'continuous_slowdown': linear_factor < 1-1e-9 or angular_factor < 1-1e-9,
            'continuous_linear_lead_m': norm(linear_lead),
            'continuous_angular_lead_rad': norm(angular_lead),
            'angular_remaining_rad': angular_remaining_rad,
            'acceleration_limit_overridden': bool(overrides),
            'acceleration_override_reasons': overrides,
        }
        return target, diagnostics
