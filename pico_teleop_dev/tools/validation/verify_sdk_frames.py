"""核对本机 SDK 正解的坐标含义；只读连接一次，后续仅本进程离线 Algo。

离线 rm_algo_set_angle/workframe 只设置此进程的数学库，绝不发送控制器设置。
"""
from pathlib import Path
import fcntl
import json
import math

from incremental import angle, inverse, multiply, norm, rotate_vector


def main():
    from robot_backend import Backend,check_known_controller_processes
    from runtime_paths import LOCK_DIR
    TASK_DIR = Path(__file__).resolve().parents[2]
    from Robotic_Arm.rm_robot_interface import (Algo,rm_robot_arm_model_e,rm_force_type_e,
        rm_frame_t,rm_pose_t,rm_position_t,rm_euler_t,rm_algo_base2workframe)
    lock=(LOCK_DIR/'.single_joint_step.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    native=None
    try:
        check_known_controller_processes()
        native=Backend()
        q=native.initial_joints.copy()
        connected_fk=native.initial_pose.copy()
        frames=native.frames
        install=[native.installation[k] for k in ('x','y','z')]
        # 实际连接仅查询，关闭后不会在同一进程再次初始化机器人通信。
        native.close();native=None
        algo=Algo(rm_robot_arm_model_e.RM_MODEL_RM_65_E,rm_force_type_e.RM_MODEL_RM_B_E)
        work=frames['rm_get_current_work_frame']['pose']
        tool=frames['rm_get_current_tool_frame']['pose']
        algo.rm_algo_set_toolframe(rm_frame_t('',tool))
        cases=[]
        for mounting in ([0,0,0],install):
            algo.rm_algo_set_angle(*mounting)
            for frame in ([0,0,0,0,0,0],work,[.1,.2,.3,0,0,math.pi/2]):
                algo.rm_algo_set_workframe(rm_frame_t('',frame))
                cases.append({'installation_deg':mounting,'work_frame_pose':frame,
                              'fk_wxyz':algo.rm_algo_forward_kinematics(q,0)})
        # 安装参考下、工作坐标为单位阵的 FK，再显式转到实际工作坐标。
        algo.rm_algo_set_angle(*install)
        algo.rm_algo_set_workframe(rm_frame_t('',[0,0,0,0,0,0]))
        base_euler=algo.rm_algo_forward_kinematics(q,1)
        p=rm_pose_t();p.position=rm_position_t(*base_euler[:3]);p.euler=rm_euler_t(*base_euler[3:])
        matrix=algo.rm_algo_pos2matrix(work)
        wrapper_result=algo.rm_algo_base2workframe(matrix,p,0)
        # SDK 1.1.6 的 Python 方法误调用 matrix2pos，只返回工作坐标本身。
        # 用签名已核对的原生数学符号及独立公式校验，均没有通信或设置控制器。
        work_q=algo.rm_algo_euler2quaternion(work[3:])
        # 本机原生转换函数直接应用传入变换，因此传 Base→Work 的逆矩阵。
        inverse_origin=[-x for x in rotate_vector(inverse(work_q),work[:3])]
        inverse_work=inverse_origin+algo.rm_algo_quaternion2euler(inverse(work_q))
        inverse_matrix=algo.rm_algo_pos2matrix(inverse_work)
        converted=rm_algo_base2workframe(inverse_matrix,p)
        explicit=[converted.position.x,converted.position.y,converted.position.z,
                  *algo.rm_algo_euler2quaternion([converted.euler.rx,converted.euler.ry,converted.euler.rz])]
        manual_pos=rotate_vector(inverse(work_q),[a-b for a,b in zip(base_euler[:3],work[:3])])
        algo.rm_algo_set_workframe(rm_frame_t('',work))
        rebuilt=algo.rm_algo_forward_kinematics(q,0)
        rebuilt_error=norm([a-b for a,b in zip(connected_fk[:3],rebuilt[:3])])*1000
        conversion_error=norm([a-b for a,b in zip(rebuilt[:3],explicit[:3])])*1000
        orientation_error=math.degrees(angle(multiply(rebuilt[3:],inverse(connected_fk[3:]))))
        manual_error=norm([a-b for a,b in zip(rebuilt[:3],manual_pos)])*1000
        if rebuilt_error>.1 or conversion_error>.1 or manual_error>.1 or orientation_error>.02:
            raise AssertionError('Frame verification detail: '+json.dumps({
                'connected_fk':connected_fk,'rebuilt_fk':rebuilt,'explicit_conversion':explicit,
                'rebuilt_error_mm':rebuilt_error,'conversion_error_mm':conversion_error,
                'manual_error_mm':manual_error,'orientation_error_deg':orientation_error}))
        result={'robot_motion_commands':0,'controller_frame_settings_sent':0,
                'offline_math_parameters_only':True,'joints_deg':q,
                'connected_frames':frames,'connected_installation_deg':install,
                'connected_fk_wxyz':connected_fk,'offline_cases':cases,
                'rebuilt_current_fk_wxyz':rebuilt,'explicit_base_to_work_wxyz':explicit,
                'connected_vs_rebuilt_position_error_mm':rebuilt_error,
                'explicit_conversion_error_mm':conversion_error,
                'manual_transform_error_mm':manual_error,
                'native_conversion_matrix_convention':'Inverse Work-in-base matrix (Base-to-Work transform)',
                'python_wrapper_result_wxyz':wrapper_result,
                'python_wrapper_position_error_mm':norm([a-b for a,b in zip(rebuilt[:3],wrapper_result[:3])])*1000,
                'python_wrapper_bug':'Algo.rm_algo_base2workframe calls rm_algo_matrix2pos in SDK 1.1.6; current teleop does not use that helper',
                'connected_vs_rebuilt_rotation_error_deg':orientation_error,
                'finding':'SDK 1.1.6 connected FK matches installed-reference FK transformed into the current work frame; Base is a name, not a level-axis guarantee',
                'no_mapping_change_made':True}
        (TASK_DIR/'reports'/'sdk_frame_semantics_verified_20261013.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally:
        if native:native.close()
        lock.close()


if __name__=='__main__':main()
