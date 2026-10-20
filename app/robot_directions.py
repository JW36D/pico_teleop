"""Convert fixed robot-body directions to the active SDK work frame.

Pure mathematics: no socket, SDK setter, or movement API. Robot-body axes are
front/left/up. Operator axes are right/up/back, matching operator_yaw_basis.
The configured body-to-installation rotation is an explicit physical-frame
assumption; the startup report makes it visible for the operator's axis check.
"""
import math

from incremental import IncrementalMapper, MappingError, inverse, multiply, quat, rotate_vector, vector


OPERATOR_TO_BODY = [[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]]


def euler_quaternion(rpy):
    roll, pitch, yaw = vector(rpy, 3)
    rotations = []
    for axis, value in enumerate((roll, pitch, yaw)):
        q = [math.cos(value / 2), 0., 0., 0.]
        q[axis + 1] = math.sin(value / 2)
        rotations.append(q)
    return quat(multiply(rotations[2], multiply(rotations[1], rotations[0])))


def matrix_vector(matrix, value):
    return [sum(a * b for a, b in zip(row, value)) for row in matrix]


def resolve_translation_mapping(config, frames, installation_deg):
    """Return a new config and an auditable direction report.

    SDK FK already reports current-work coordinates. Only requested *vectors*
    are transformed from the fixed robot reference to that work frame here;
    installation rotation must not be reapplied to an FK pose.
    """
    result = dict(config)
    mode = config.get('translation_mapping_frame', 'work')
    if mode == 'work':
        result['headset_to_work_matrix'] = IncrementalMapper.checked_basis(
            config['headset_to_work_matrix'])
        return result, {'translation_mapping_frame': mode,
                        'operator_to_work_matrix': result['headset_to_work_matrix']}
    if mode != 'robot_body':
        raise MappingError('Unknown translation_mapping_frame')
    if config.get('translation_reference') != 'operator_heading':
        raise MappingError('Robot-body mapping requires operator_heading input')

    expected = vector(config.get('robot_body_expected_installation_deg'), 3)
    actual_installation = vector(installation_deg, 3)
    if max(abs(a - b) for a, b in zip(expected, actual_installation)) > .1:
        raise MappingError('Robot installation differs from direction reference',
                           {'expected_installation_deg': expected,
                            'actual_installation_deg': actual_installation})
    body_to_install = IncrementalMapper.checked_basis(config['robot_body_to_install_matrix'])
    work = frames.get('rm_get_current_work_frame')
    if not isinstance(work, dict) or not isinstance(work.get('name'), str):
        raise MappingError('Missing SDK work frame for robot directions')
    work_pose = vector(work.get('pose'), 6)
    install_to_work = inverse(euler_quaternion(work_pose[3:]))

    # Columns of this rotation are robot front/left/up expressed in work.
    body_columns = [rotate_vector(install_to_work,
        [body_to_install[row][column] for row in range(3)]) for column in range(3)]
    front, left, up = body_columns
    right = [-x for x in left]
    # Operator [right, up, back] -> robot [right, up, -front].
    basis = [[right[row], up[row], -front[row]] for row in range(3)]
    result['headset_to_work_matrix'] = IncrementalMapper.checked_basis(basis)
    result['robot_body_mapping_resolved'] = True
    result['robot_direction_axes_work'] = {'forward': front, 'right': right, 'up': up}
    report = {'translation_mapping_frame': mode,
              'operator_heading_reference': ('first_grip_session' if
                  config.get('translation_heading_mode','grip')=='session' else 'each_grip_then_fixed'),
              'robot_body_to_install_matrix': body_to_install,
              'robot_direction_axes_work': result['robot_direction_axes_work'],
              'operator_to_work_matrix': result['headset_to_work_matrix'],
              'work_frame': {'name': work['name'], 'pose': work_pose},
              'installation_deg': actual_installation,
              'physical_direction_validation': config.get('physical_direction_validation', 'pending')}
    return result, report


def work_delta_to_body(delta, direction_report):
    """Project a work displacement to robot front/right/up for diagnostics."""
    value = vector(delta, 3)
    axes = direction_report['robot_direction_axes_work']
    return {name: sum(a * b for a, b in zip(value, axis)) for name, axis in axes.items()}
