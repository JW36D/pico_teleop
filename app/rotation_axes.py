"""Neutral grip calibration: hand down/buttons toward operator -> current tool.

No hardware calls. Logical neutral hand axes are right/back/down; the tool's
long axis is its actual local +Z. Calibration remains fixed across regrips.
"""
import collections
import math

from incremental import IncrementalMapper, MappingError, inverse, multiply, norm, quat, rotate_vector


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def matmul(left, right):
    return [[sum(a*b for a,b in zip(row,column)) for column in zip(*right)] for row in left]


def rotation_matrix(q):
    return transpose([rotate_vector(q, axis) for axis in ([1.,0.,0.],[0.,1.,0.],[0.,0.,1.])])


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def neutral_tool_calibration(hand_pose, robot_pose, operator_basis, operator_to_work):
    """Derive the fixed hand-local -> tool-local rotation without guessing SDK axes."""
    room_to_operator = IncrementalMapper.checked_basis(operator_basis)
    operator_to_work = IncrementalMapper.checked_basis(operator_to_work)
    hand = rotation_matrix(hand_pose[3:])
    tool = rotation_matrix(robot_pose[3:])
    # In operator [right,up,back] coordinates, canonical hand XYZ is right/back/down.
    canonical_to_operator = [[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]]
    canonical_hand_room = matmul(transpose(room_to_operator), canonical_to_operator)
    tool_z = [row[2] for row in tool]
    robot_right = [row[0] for row in operator_to_work]
    projection = sum(a*b for a,b in zip(robot_right,tool_z))
    tool_x = [a-projection*b for a,b in zip(robot_right,tool_z)]
    length = norm(tool_x)
    if length < .2:
        raise MappingError('Tool long axis is too close to robot right for neutral calibration')
    tool_x = [x/length for x in tool_x]
    tool_y = cross(tool_z,tool_x)
    canonical_tool_work = transpose([tool_x,tool_y,tool_z])
    basis = matmul(matmul(matmul(transpose(tool),canonical_tool_work),
                          transpose(canonical_hand_room)),hand)
    basis = IncrementalMapper.checked_basis(basis)
    logical_hand_local = matmul(transpose(hand),canonical_hand_room)
    return basis, {'controller_to_tool_matrix':basis,
        'neutral_hand_pose_wxyz':list(hand_pose),'neutral_robot_pose_wxyz':list(robot_pose),
        'hand_long_axis_local':[row[2] for row in logical_hand_local],
        'hand_pitch_axis_local':[row[0] for row in logical_hand_local],
        'tool_roll_axis_work':tool_z,'tool_pitch_axis_work':tool_x,
        'neutral_pose_contract':'hand down, AB side toward operator',
        'calibration_reference':'first_successful_grip_then_fixed'}


def twist_and_tilt(q):
    """Signed tool-Z twist and tool-axis tilt; q retains its continuous sign."""
    q = quat(q)
    approach = rotate_vector(q,[0.,0.,1.])
    tilt = math.acos(max(-1.,min(1.,approach[2])))
    magnitude = math.hypot(q[0],q[3])
    twist = None if magnitude < 1e-8 else 2*math.atan2(q[3],q[0])
    return twist,tilt


def split_swing_twist(q, near_twist=None):
    """A signed scalar twist keeps rotation direction even with >180° command lag."""
    q=quat(q)
    twist,tilt=twist_and_tilt(q)
    if twist is None:
        raise MappingError('Tool tilt makes long-axis twist ambiguous')
    if near_twist is not None:
        twist=min([twist+2*math.pi*k for k in range(-2,3)],key=lambda x:abs(x-near_twist))
    twist_q=[math.cos(twist/2),0.,0.,math.sin(twist/2)]
    swing=quat(multiply(q,inverse(twist_q)))
    if swing[0]<0:swing=[-x for x in swing]
    return swing,twist


def rotation_vector(q):
    """Shortest adjacent quaternion increment, retaining its axis direction."""
    q = quat(q)
    if q[0] < 0:
        q = [-value for value in q]
    length = norm(q[1:])
    if length < 1e-12:
        return [0.,0.,0.]
    radians = 2*math.atan2(length,q[0])
    return [value*radians/length for value in q[1:]]


def vector_rotation(value):
    radians = norm(value)
    if radians < 1e-12:
        return [1.,0.,0.,0.]
    return [math.cos(radians/2)]+[v*math.sin(radians/2)/radians for v in value]


class ContinuousRotationPath:
    """Ordered rotation increments without an anchor-angle or tilt singularity.

    The receiver still supplies only its latest sample. This class retains the
    small rotation increments needed to distinguish one turn from several
    turns while the robot is catching up. Translation never replays packets.
    Filter and target cursors advance along that path; a rejected/limited IK
    proposal consumes only its accepted angular movement.
    """
    def __init__(self, initial):
        self.reset(initial)

    def reset(self, orientation):
        self.base = self.goal = quat(orientation)
        self.actual_orientation = self.goal.copy()
        self.segments = collections.deque()
        self.end = self.filtered = self.cursor = 0.
        self.proposal = None

    def append(self, world_increment):
        radians = norm(world_increment)
        if radians < 1e-12:
            return
        if self.proposal is not None:
            self.commit(self.proposal[3])
        axis = [value/radians for value in world_increment]
        # A reversed hand movement cancels the unexecuted part of the same
        # rotation. It must not replay an old +90/-90 excursion merely because
        # the robot was still at +10. Completed turns remain continuous.
        while self.segments and radians > 1e-12:
            start,end,orientation,previous_axis = self.segments[-1]
            if sum(a*b for a,b in zip(axis,previous_axis)) > -1.+1e-7:
                break
            available = end-max(start,self.cursor)
            if available <= 1e-12:
                break
            cancel = min(radians,available)
            new_end = end-cancel
            self.segments.pop()
            self.end = max(self.cursor,new_end)
            self.goal = quat(multiply(vector_rotation(
                [value*(new_end-start) for value in previous_axis]),orientation))
            if new_end > start+1e-12:
                self.segments.append((start,new_end,orientation,previous_axis))
            radians -= cancel
        self.filtered = min(self.filtered,self.end)
        if radians < 1e-12:
            return
        if self.segments and sum(a*b for a,b in zip(axis,self.segments[-1][3])) > 1.-1e-7:
            start,end,orientation,previous_axis = self.segments.pop()
            self.segments.append((start,self.end+radians,orientation,previous_axis))
        else:
            self.segments.append((self.end,self.end+radians,self.goal.copy(),axis))
        self.end += radians
        self.goal = quat(multiply(vector_rotation([value*radians for value in axis]),self.goal))
        # For ordinary small lag, retain the latest orientation rather than
        # old axis wiggles. Only a pending path of >= pi can require a branch
        # history to distinguish continuing a turn from reversing it.
        pending = self.end-self.cursor
        if len(self.segments) > 1 and pending < math.pi-1e-6:
            current = self.sample(self.cursor)
            direct = rotation_vector(multiply(self.goal,inverse(current)))
            distance = norm(direct)
            filtered_fraction = max(0.,min(1.,(self.filtered-self.cursor)/pending))
            self.segments.clear()
            self.end = self.cursor+distance
            self.filtered = self.cursor+distance*filtered_fraction
            if distance > 1e-12:
                self.segments.append((self.cursor,self.end,current,
                                      [value/distance for value in direct]))

    def sample(self, cursor):
        if not self.segments:
            return self.goal.copy()
        cursor = max(self.segments[0][0],min(self.end,cursor))
        for start,end,orientation,axis in self.segments:
            if cursor <= end+1e-12:
                step = [value*max(0.,cursor-start) for value in axis]
                return quat(multiply(vector_rotation(step),orientation))
        return self.goal.copy()

    def commit(self, accepted):
        accepted = quat(accepted)
        self.actual_orientation = accepted.copy()
        if self.proposal is None:
            return
        start,end,origin,proposed = self.proposal
        actual = rotation_vector(multiply(accepted,inverse(origin)))
        cursor = start
        # The sender's actual interval may exceed the mapper's nominal period.
        # Its valid accepted step can therefore pass ``proposal.end``. Consume
        # accepted progress on the real path, not a fraction capped to that
        # preview budget; never consume the orthogonal or reverse component.
        for segment_start,segment_end,orientation,axis in self.segments:
            if segment_end <= cursor+1e-12:
                continue
            if sum(value*direction for value,direction in zip(actual,axis)) <= 1e-12:
                break
            relative = quat(multiply(accepted,inverse(self.sample(cursor))))
            if relative[0] < 0:
                relative = [-value for value in relative]
            projection = sum(value*direction for value,direction in zip(relative[1:],axis))
            # This is the signed twist about the current segment's fixed world
            # axis. Orthogonal orientation error remains a correction budget.
            progress = 2*math.atan2(projection,relative[0])
            available = segment_end-cursor
            consumed = max(0.,min(available,progress))
            cursor += consumed
            if consumed < available-1e-12:
                break
        self.cursor = min(self.end,cursor)
        self.filtered = max(self.filtered,self.cursor)
        self.proposal = None
        # Retain the current segment so its unexecuted remainder is exact.
        while self.segments and self.segments[0][1] <= self.cursor+1e-12:
            segment = self.segments.popleft()
            self.base = quat(multiply(vector_rotation(
                [value*(segment[1]-segment[0]) for value in segment[3]]),segment[2]))

    def advance(self, actual_target, alpha, max_radians):
        # Pure mapper clients accept its proposal directly. A controller calls
        # commit first with the IK/FK pose and therefore overrides this fallback.
        if self.proposal is not None:
            self.commit(self.proposal[3])
        self.filtered += (self.end-self.filtered)*alpha
        start = self.cursor
        end = start+min(max_radians,max(0.,self.filtered-start))
        proposed = self.sample(end)
        actual_target = quat(actual_target)
        self.actual_orientation = actual_target.copy()
        if sum(a*b for a,b in zip(proposed,actual_target)) < 0:
            proposed = [-value for value in proposed]
        self.proposal = (start,end,actual_target,proposed)
        return self.sample(self.filtered),proposed

    def remaining(self):
        # A dynamics limiter may briefly overshoot the path endpoint while
        # braking. Keep that correction budget visible even when every hand
        # increment has already been consumed.
        correction = norm(rotation_vector(multiply(
            self.sample(self.cursor),inverse(self.actual_orientation))))
        return max(0.,self.filtered-self.cursor)+correction
