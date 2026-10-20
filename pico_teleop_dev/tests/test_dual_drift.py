"""Unequal arm boundaries must not silently change the hand/robot correspondence."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))

from incremental import IncrementalMapper, MappingError, angle, inverse, multiply
from operator_reference import SharedOperatorReference
from teleop_config import load_config
from teleop_controller import TeleopController
from continuous_motion import ContinuousMotionShaper
from test_incremental import frame
from test_task_teleop_controller import AxisBackend, FreshFeedback
from test_final_emission_consistency import CoupledControlBackend
from test_joint_cartesian_trajectory import axis_quaternion

REST=[0.,0.,0.,1.,0.,0.,0.]


def config(arm='right'):
    c=load_config(ROOT/('configs/%s_rotation_teleop.json'%arm))
    c.update(translation_mapping_frame='work',rotation_mapping='world',
             headset_to_work_matrix=[[1,0,0],[0,1,0],[0,0,1]])
    return c


def sample(side,sequence,ns,grip,xyz=None,degrees=0.,head_yaw=0.):
    q=axis_quaternion(2,degrees)
    value=frame(sequence,ns,grip,xyz,q[1:]+q[:1])
    value['latest']['head']['pose']=[0,0,0,0,math.sin(head_yaw/2),0,math.cos(head_yaw/2)]
    if side=='left':value['latest']['left']=value['latest'].pop('right')
    return value


class SharedHeadingTests(unittest.TestCase):
    def test_different_grip_times_and_head_turns_keep_one_horizontal_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.json'
            mappers={side:IncrementalMapper(config(side),REST) for side in ('right','left')}
            for side,m in mappers.items():
                m.operator_reference=SharedOperatorReference(path).resolve
                yaw=0. if side=='right' else math.radians(35)
                m.update(sample(side,1,1_000_000_000,0.,head_yaw=yaw),1_000_000_000,lambda:REST)
                result=m.update(sample(side,2,1_010_000_000,1.,head_yaw=yaw),1_010_000_000,lambda:REST)
                self.assertEqual(result['state'],'ANCHORED')
            self.assertEqual(mappers['right'].operator_basis,mappers['left'].operator_basis)
            for side,m in mappers.items():
                basis=copy.deepcopy(m.operator_basis)
                # A release and subsequent regrip must not rotate this side's axes.
                m.update(sample(side,3,1_020_000_000,0.),1_020_000_000,lambda:REST)
                m.update(sample(side,4,1_030_000_000,1.,head_yaw=math.pi/2),1_030_000_000,lambda:REST)
                result=m.update(sample(side,5,1_040_000_000,1.,[.01,0,0],head_yaw=math.pi/2),1_040_000_000,lambda:REST)
                self.assertEqual(m.operator_basis,basis)
                self.assertAlmostEqual(result['requested_pose_wxyz'][0],.01)
                self.assertAlmostEqual(result['requested_pose_wxyz'][2],0.)

    def test_new_supervised_session_can_establish_a_new_heading(self):
        with tempfile.TemporaryDirectory() as directory:
            first=SharedOperatorReference(Path(directory)/'first.json')
            second=SharedOperatorReference(Path(directory)/'second.json')
            a=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
            b=[[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
            self.assertEqual(first.resolve(a),a)
            self.assertEqual(first.resolve(b),a)
            self.assertEqual(second.resolve(b),b)

    def test_simultaneous_first_grips_cannot_write_different_references(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.json';barrier=threading.Barrier(2)
            candidates=[[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
                        [[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]]
            def resolve(candidate):
                reference=SharedOperatorReference(path)
                barrier.wait(timeout=2.)
                return reference.resolve(candidate)
            with ThreadPoolExecutor(max_workers=2) as pool:
                values=list(pool.map(resolve,candidates))
            self.assertEqual(values[0],values[1])
            self.assertIn(values[0],candidates)


class FixedBoundaryTests(unittest.TestCase):
    def run_two_arms(self,joint_trajectory):
        ns=1_000_000_000
        controls={};backends={};feedbacks={};states=[]
        for side in ('right','left'):
            c=config(side);c.update(translation_reference='tracking_world',rotation_scale=0.,
                joint_trajectory_enabled=joint_trajectory)
            b=AxisBackend(lambda:ns,degrees_per_metre=100.)
            b.ik_boundary_m=.02 if side=='right' else .10
            f=FreshFeedback(lambda:ns)
            backends[side],feedbacks[side]=b,f
            controls[side]=TeleopController(c,b,f,True,lambda:ns)
        def step(sequence,grip,x):
            nonlocal ns
            ns+=10_000_000
            reports={}
            for side,control in controls.items():
                b=backends[side];feedbacks[side].observe(b.sent[-1] if b.sent else None)
                value=sample(side,sequence,ns,grip,[x,0,0])
                reports[side]=control.tick(value,lambda:value,ns)
            return reports
        step(1,0.,0.);step(2,1.,0.)
        anchors={side:copy.deepcopy(c.mapper.anchor_robot) for side,c in controls.items()}
        seq=2
        for tick in range(160):
            seq+=1;reports=step(seq,1.,min(.001*tick,.04))
            states.append(reports['right']['mapping']['state'])
            for side,c in controls.items():
                self.assertEqual(c.mapper.anchor_robot,anchors[side])
        for side,r in reports.items():
            self.assertAlmostEqual(r['mapping']['requested_pose_wxyz'][0],.04)
        self.assertTrue(any(s.startswith('BOUNDARY') for s in states))
        # Return both hands to the same starting point without regripping.
        for tick in range(240):
            seq+=1;reports=step(seq,1.,max(0.,.04-.001*tick))
        for side,c in controls.items():
            self.assertEqual(c.mapper.anchor_robot,anchors[side])
            self.assertAlmostEqual(backends[side].sent[-1][0],0.,places=5)
            self.assertAlmostEqual(reports[side]['mapping']['requested_pose_wxyz'][0],0.)
            self.assertTrue(c.mapper.active)
            self.assertEqual(backends[side].stops,0)

    def test_joint_shaped_arms_return_without_permanent_relative_offset(self):
        self.run_two_arms(True)

    def test_non_joint_shaped_boundary_also_preserves_reference(self):
        self.run_two_arms(False)

    def test_orientation_boundary_does_not_redefine_the_grip_orientation(self):
        ns=1_000_000_000
        b=CoupledControlBackend(lambda:ns);f=FreshFeedback(lambda:ns)
        # Use an actual joint boundary so FK and IK describe the same geometry.
        b.maximum[5]=15.
        b.minimum[5]=-15.
        c=config();c.update(translation_reference='tracking_world')
        control=TeleopController(c,b,f,True,lambda:ns)
        sequence=0;boundary_seen=False
        def step(grip,degrees):
            nonlocal ns,sequence,boundary_seen
            ns+=10_000_000;sequence+=1
            f.observe(b.sent[-1] if b.sent else None)
            value=sample('right',sequence,ns,grip,degrees=degrees)
            report=control.tick(value,lambda:value,ns)
            boundary_seen|=report['mapping']['state'].startswith('BOUNDARY')
            return report
        step(0.,0.);step(1.,0.)
        initial=control.mapper.anchor_robot.copy()
        for tick in range(160):report=step(1.,min(tick*.5,45.))
        self.assertTrue(boundary_seen)
        self.assertAlmostEqual(math.degrees(angle(report['mapping']['requested_pose_wxyz'][3:])),45.,places=4)
        self.assertEqual(control.mapper.anchor_robot,initial)
        for tick in range(300):report=step(1.,max(0.,45.-tick*.5))
        self.assertLess(angle(b.forward(b.sent[-1])[3:]),1e-6)
        self.assertEqual(control.mapper.anchor_robot,initial)
        self.assertTrue(control.mapper.active)


class SpeedResponseTests(unittest.TestCase):
    def test_speed_tuning_improves_a_delayed_follower_without_erasing_intent(self):
        new=config()
        old={**new,'max_linear_speed_m_s':.25,'max_angular_speed_rad_s':1.4,
             'max_linear_accel_m_s2':5.,'max_angular_accel_rad_s2':12.,
             'max_command_lead_m':.03,'max_command_lead_angle_rad':.15,
             'command_lead_full_speed_ratio':.7}
        def travel(c):
            shaper=ContinuousMotionShaper(c)
            command=REST.copy();actual=REST.copy();goal=[.30,0,0]+REST[3:]
            for tick in range(1000):
                next_command,diag=shaper.advance(command,goal,goal,actual,.01,governed=True)
                self.assertLessEqual(diag['linear_speed_m_s'],c['max_linear_speed_m_s']+1e-9)
                self.assertLessEqual(diag['linear_accel_m_s2'],c['max_linear_accel_m_s2']+1e-8)
                shaper.sync(command,next_command,.01);command=next_command
                # Illustrative first-order follower; this is not a hardware model.
                actual[0]+=(command[0]-actual[0])*.01/.15
                if actual[0]>=.285:return (tick+1)*.01
            self.fail('Follower did not arrive')
        before,after=travel(old),travel(new)
        self.assertLess(after,before*.8)


if __name__=='__main__':unittest.main()
