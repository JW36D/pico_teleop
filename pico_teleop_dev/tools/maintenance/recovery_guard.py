"""任一机械臂已记录姿态的分段恢复与反馈检查；此文件没有机器人接口。"""
import math
import time

from incremental import MappingError, vector

SEGMENT_DEG = 5.0
START_TOLERANCE_DEG = .05
ARRIVAL_TOLERANCE_DEG = .05
CORRIDOR_MARGIN_DEG = .15
SPEED_PERCENT = 1
# 只容忍硬限位附近的编码器/浮点读数误差，不扩大任何发送目标的范围。
FEEDBACK_LIMIT_TOLERANCE_DEG = .01


def check_limits(joints, minimum, maximum, *, feedback=False):
    joints, minimum, maximum = (vector(v, 6) for v in (joints, minimum, maximum))
    tolerance = FEEDBACK_LIMIT_TOLERANCE_DEG if feedback else 0.
    for i, (q, low, high) in enumerate(zip(joints, minimum, maximum)):
        if not -720 <= low < high <= 720 or not low-tolerance <= q <= high+tolerance:
            raise MappingError('Recovery J%d is outside valid hardware limits' % (i+1))
    return joints


def waypoints(start, target, minimum, maximum):
    """按最大关节角差等分，六轴协调回位；没有镜像、逆解或角度绕回。"""
    actual_start = check_limits(start, minimum, maximum, feedback=True)
    target = check_limits(target, minimum, maximum)
    if max(abs(a-b) for a, b in zip(actual_start, target)) <= ARRIVAL_TOLERANCE_DEG:
        return []
    # 保留真实起点供复核；发送点从合法范围内开始，绝不发送 128.001° 等读数。
    start = [max(low, min(high, q)) for q, low, high in zip(actual_start, minimum, maximum)]
    excursion = max(abs(a-b) for a, b in zip(start, target))
    count = math.ceil(excursion/SEGMENT_DEG)
    points = [[a+(b-a)*k/count for a, b in zip(start, target)]
              for k in range(1, count+1)]
    points[-1] = target.copy()  # 最终点精确使用记录，避免插值浮点误差。
    for point in points:
        check_limits(point, minimum, maximum)
    return points


def trajectory_type(value, require_idle=False):
    if type(value) is not int or value not in (0, 1):
        raise MappingError('Unexpected recovery trajectory type')
    if require_idle and value != 0:
        raise MappingError('An existing trajectory is active; recovery refused')


def check_start(actual, reviewed):
    actual, reviewed = vector(actual, 6), vector(reviewed, 6)
    if max(abs(a-b) for a, b in zip(actual, reviewed)) > START_TOLERANCE_DEG:
        raise MappingError('Joints changed since recovery plan; run --plan again')


def check_corridor(actual, start, target, minimum, maximum):
    actual = check_limits(actual, minimum, maximum, feedback=True)
    for i, (q, a, b) in enumerate(zip(actual, start, target)):
        if not min(a, b)-CORRIDOR_MARGIN_DEG <= q <= max(a, b)+CORRIDOR_MARGIN_DEG:
            raise MappingError('Recovery J%d left this segment corridor' % (i+1))
    return actual


def execute_recovery(backend, start, target, minimum, maximum, *,
                     record=lambda event: None, cancelled=lambda: False,
                     clock=time.monotonic, sleep=time.sleep, segment_timeout_s=15.):
    """每段只发送一次非阻塞 MoveJ；反馈到位并停稳后才能发送下一段。

    backend.sample() 提供已验证报警/新鲜度的实际关节和轨迹类型。
    通信返回成功不表示到位；取消、错误或超时后停机且不重试动作。
    恢复特意使用关节规划跨越 q5=0，不使用遥操 IK 的腕部保护。
    """
    points = waypoints(start, target, minimum, maximum)
    attempted = False
    try:
        sample = backend.sample()
        trajectory_type(sample['trajectory'], require_idle=True)
        check_start(check_limits(sample['joints'], minimum, maximum, feedback=True), start)
        previous = list(start)
        for number, endpoint in enumerate(points, 1):
            if cancelled():
                raise KeyboardInterrupt('Recovery cancelled before sending')
            sample = backend.sample()
            trajectory_type(sample['trajectory'], require_idle=True)
            check_start(sample['joints'], previous)
            # 边界从最新编码器角度算起，容忍上一段到位的微小误差。
            segment_start = check_limits(sample['joints'], minimum, maximum, feedback=True)
            record({'event': 'SEGMENT_START', 'segment': number, 'segments': len(points),
                    'start_joints_deg': segment_start, 'target_joints_deg': endpoint,
                    'speed_percent': SPEED_PERCENT})
            # 打印/状态检查期间也可能收到 Ctrl+C，最后一道检查紧邻发送。
            if cancelled():
                raise KeyboardInterrupt('Recovery cancelled before sending')
            attempted = True  # 丢失应答仍可能已经开始，必须尝试停止。
            rc = backend.movej(endpoint, SPEED_PERCENT)
            if rc != 0:
                raise MappingError('Recovery MoveJ failed: %s; no retry' % rc)
            deadline, stable, next_log = clock()+segment_timeout_s, 0, clock()
            while clock() < deadline:
                if cancelled():
                    raise KeyboardInterrupt('Recovery cancelled')
                sample = backend.sample()
                trajectory_type(sample['trajectory'])
                actual = check_corridor(sample['joints'], segment_start, endpoint,
                                        minimum, maximum)
                error = max(abs(a-b) for a, b in zip(actual, endpoint))
                stable = stable+1 if error <= ARRIVAL_TOLERANCE_DEG and sample['trajectory'] == 0 else 0
                if clock() >= next_log:
                    record({'event': 'RECOVERING', 'segment': number,
                            'actual_joints_deg': actual, 'max_arrival_error_deg': error,
                            'trajectory_type': sample['trajectory']})
                    next_log = clock()+.5
                if stable >= 3:
                    record({'event': 'SEGMENT_DONE', 'segment': number,
                            'actual_joints_deg': actual})
                    previous = actual
                    break
                sleep(.1)
            else:
                raise MappingError('Recovery segment did not settle within timeout')
        sample = backend.sample()
        trajectory_type(sample['trajectory'], require_idle=True)
        check_start(check_limits(sample['joints'], minimum, maximum, feedback=True), target)
        record({'event': 'RECOVERY_DONE', 'actual_joints_deg': sample['joints'],
                'segments': len(points), 'max_arrival_error_deg':
                max(abs(a-b) for a, b in zip(sample['joints'], target))})
        return sample['joints']
    except BaseException:
        if attempted:
            try:
                backend.stop()
                record({'event': 'SOFTWARE_STOP', 'confirmed': True})
            except BaseException as exc:
                record({'event': 'SOFTWARE_STOP', 'confirmed': False, 'error': str(exc)})
        raise
