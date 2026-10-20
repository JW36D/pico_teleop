"""Shared RM65 backend and PICO input; each worker owns exactly one arm."""
import json
import math
import socket
import time
from runtime_paths import APP_DIR, RECEIVER_SOCKET
from incremental import MappingError, angle, inverse, multiply, norm, pose, vector
from readonly_client import ARM_IPS, ReadOnlyError, check_known_controller_processes, query_one, summarize_arm_state

TASK_DIR = APP_DIR

def read_pico():
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(.06)
        client.connect(str(RECEIVER_SOCKET))
        client.sendall(b'status\n')
        with client.makefile('rb') as f:
            return json.loads(f.readline(65536))


def checked_pair(robot, name):
    rc, data = getattr(robot, name)()
    if rc != 0:
        raise MappingError(f'{name} failed: {rc}')
    return data


class Backend:
    def __init__(self, arm='right'):
        if arm not in ARM_IPS:
            raise MappingError('Unknown arm: '+str(arm))
        self.arm=arm
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e, rm_inverse_kinematics_params_t
        self.runtime = RoboticArm
        self.ik_params = rm_inverse_kinematics_params_t
        self.robot = RoboticArm(rm_thread_mode_e.RM_DUAL_MODE_E)
        self.connected = False
        self.attempted = False
        try:
            handle = self.robot.rm_create_robot_arm(ARM_IPS[arm], 8080)
            if handle is None or handle.id <= 0:
                raise MappingError('Cannot connect '+arm+' arm')
            self.connected = True
            info = checked_pair(self.robot, 'rm_get_robot_info')
            if info.get('arm_dof') != 6 or info.get('arm_model') not in ('RM_65', 'RM65'):
                raise MappingError('Expected '+arm+' six-axis RM65')
            if checked_pair(self.robot, 'rm_get_arm_run_mode') != 1:
                raise MappingError('Arm must already be in real mode; mode is not changed')
            self.minimum = vector(checked_pair(self.robot, 'rm_get_joint_min_pos'), 6)
            self.maximum = vector(checked_pair(self.robot, 'rm_get_joint_max_pos'), 6)
            self.motion_capabilities = self.read_motion_capabilities()
            self.frames = {name: checked_pair(self.robot, name) for name in
                           ('rm_get_current_work_frame', 'rm_get_current_tool_frame')}
            # 安装角度只读取，用来核对侧装模型；不能根据照片直接写回控制器。
            self.installation = self.robot.rm_get_install_pose()
            if self.installation.get('return_code')!=0:
                raise MappingError('Cannot read arm installation parameters')
            installed = vector([self.installation[key] for key in ('x','y','z')],3)
            self.algorithm_installation = vector(self.robot.rm_algo_get_angle(),3)
            if max(abs(a-b) for a,b in zip(installed,self.algorithm_installation))>.1:
                raise MappingError('SDK and controller installation angles disagree')
            self.initial_joints = self.joints(require_idle=True)
            self.initial_pose = pose(self.robot.rm_algo_forward_kinematics(self.initial_joints, 0))
            time.sleep(.15)
            second = self.joints(require_idle=True)
            if max(abs(a-b) for a, b in zip(self.initial_joints, second)) > .03:
                raise MappingError(arm.capitalize()+' arm is not stationary')
        except BaseException:
            self.close()
            raise

    def read_motion_capabilities(self):
        """Read controller capabilities/state without changing any setting.

        SDK values are degrees/second and degrees/second squared. These units
        were checked against raw JSON (0.001 RPM and 0.001 RPM/s, respectively;
        multiply both raw integer arrays by 0.006), not inferred from the
        getter documentation's erroneous acceleration unit.

        Optional/unsupported getters are reported explicitly. An unknown
        collision state must not be represented as enabled collision safety.
        """
        result = {'query_only': True,
                  'unit_source': 'https://develop.realman-robotics.com/robot/apic/classes/jointConfigSettings/'}
        for key, method, units, attribute in (
                ('joint_speed', 'rm_get_joint_max_speed', 'deg/s', 'speed_limits'),
                ('joint_acceleration', 'rm_get_joint_max_acc', 'deg/s^2', 'acceleration_limits'),
                ('self_collision', 'rm_get_self_collision_enable', None, None)):
            entry = {'method': method, 'status': 'unavailable'}
            if units is not None:
                entry['units'] = units
            try:
                getter = getattr(self.robot, method, None)
                if getter is None:
                    entry.update(status='unsupported', reason='SDK getter is absent')
                else:
                    rc, data = getter()
                    entry['return_code'] = rc
                    if rc != 0:
                        entry['reason'] = 'Controller did not return a valid capability/state'
                    elif attribute is None:
                        if not isinstance(data, bool):
                            raise MappingError('Self-collision query returned invalid data')
                        entry.update(status='available', enabled=data)
                    else:
                        values = vector(data, 6)
                        if any(value <= 0 for value in values):
                            raise MappingError('Joint capability must be positive')
                        setattr(self, attribute, values)
                        entry.update(status='available', values=values.copy())
            except Exception as exc:
                entry.update(status='unavailable', reason=str(exc))
            result[key] = entry
        return result

    def joints(self, require_idle=False):
        flags = self.robot.rm_get_joint_err_flag()
        if flags.get('return_code') != 0 or len(flags.get('err_flag', [])) != 6 or any(flags['err_flag']):
            raise MappingError('Invalid or nonzero joint alarm')
        if require_idle:
            trajectory = self.robot.rm_get_arm_current_trajectory()
            if trajectory.get('return_code') != 0 or trajectory.get('trajectory_type') != 0:
                raise MappingError('An existing arm trajectory is active')
        joints = vector(checked_pair(self.robot, 'rm_get_joint_degree'), 6)
        return self.validate_joint_feedback(joints)

    def validate_joint_feedback(self, joints):
        """Teleop feedback stays strict; recovery overrides only this validation."""
        for q, low, high in zip(joints, self.minimum, self.maximum):
            if not math.isfinite(low) or not math.isfinite(high) or not low < high or not low <= q <= high:
                raise MappingError('Invalid limits or joint feedback outside limits')
        return joints

    def anchor(self):
        actual = self.joints(require_idle=True)
        return pose(self.robot.rm_algo_forward_kinematics(actual, 0))

    def solve(self, target, seed):
        # SDK 输入为 xyz+wxyz、关节单位为度；seed 使用上个命令，避免逆解切换到另一臂形。
        rc, joints = self.robot.rm_algo_inverse_kinematics(self.ik_params(seed, target, 0))
        if rc != 0:
            raise MappingError(f'IK failed: {rc}; target is not sent',
                               {'state':'IK_REFUSED','retryable_ik':True})
        joints = vector(joints, 6)
        # J6 can rotate beyond 180°. Keep an equivalent angle nearest the previous
        # command so SDK normalization cannot introduce a spurious 360° jump.
        choices=[joints[5]+360*k for k in range(-2,3)
                 if self.minimum[5]<=joints[5]+360*k<=self.maximum[5]]
        if choices:
            near=min(choices,key=lambda q:abs(q-seed[5]))
            if abs(near-joints[5])>1e-5:
                equivalent=joints.copy();equivalent[5]=near
                before=self.forward(joints);after=self.forward(equivalent)
                if norm([a-b for a,b in zip(before[:3],after[:3])])>.0001 or angle(multiply(after[3:],inverse(before[3:])))>.001:
                    raise MappingError('Equivalent J6 angle failed FK verification')
                joints=equivalent
        for q, low, high in zip(joints, self.minimum, self.maximum):
            if not low <= q <= high:
                raise MappingError('IK result crosses joint limits',
                                   {'state':'IK_REFUSED','retryable_ik':True})
        return joints

    def forward(self, joints):
        # 用实际编码器关节角算末端姿态；这与“已经发出目标”是两回事。
        return pose(self.robot.rm_algo_forward_kinematics(joints, 0))

    def validate_startup_ik(self):
        """Check IK geometry without using its diagnostic solution as a command.

        Near a straight elbow, tiny native FK rounding errors can redistribute
        joint angles by more than 0.03 degrees. Joint equality is therefore not
        a valid FK/IK consistency test. The controller keeps the actual encoder
        seed; subsequent command steps still enforce joint continuity and speed.
        """
        target = pose(self.initial_pose)
        seed = vector(self.initial_joints, 6)
        solved = vector(self.solve(target.copy(), seed.copy()), 6)
        details = {'initial_joints_deg':seed, 'ik_joints_deg':solved,
                   'joint_delta_deg':[a-b for a,b in zip(solved,seed)],
                   'max_joint_delta_deg':max(abs(a-b) for a,b in zip(solved,seed)),
                   'command_seed_source':'actual_joint_feedback',
                   'fk_position_tolerance_m':.0001, 'fk_orientation_tolerance_rad':.001}
        if any(not low<=q<=high for q,low,high in zip(solved,self.minimum,self.maximum)):
            raise MappingError('Startup IK result crosses hardware joint limits', details)
        reconstructed = pose(self.forward(solved.copy()))
        details.update(fk_position_error_m=norm([a-b for a,b in zip(reconstructed[:3],target[:3])]),
                       fk_orientation_error_rad=angle(multiply(reconstructed[3:],inverse(target[3:]))))
        if details['fk_position_error_m']>.0001 or details['fk_orientation_error_rad']>.001:
            raise MappingError('Startup IK FK check failed', details)
        return details

    def send(self, joints):
        self.attempted = True
        # 完整版可使用高跟随；控制器必须先保证 5ms 周期和 <=10ms 发送间隔。
        # 原位置版本未配置这些属性，继续使用原来的低跟随。
        rc = self.robot.rm_movej_canfd(joints,getattr(self,'stream_high_follow',False),0,
            getattr(self,'stream_trajectory_mode',0),getattr(self,'stream_radio',0))
        if rc != 0:
            raise MappingError(f'Joint streaming failed: {rc}; no retry')

    def stop(self):
        if self.attempted:
            rc = self.robot.rm_set_arm_stop()
            if rc != 0:
                raise MappingError(f'Software stop not confirmed: {rc}')
            self.attempted = False

    def close(self):
        try:
            self.stop()
        finally:
            if self.connected:
                self.robot.rm_delete_robot_arm()
                self.connected = False
            self.runtime.rm_destroy()
