"""Main-loop fault observation with every device interface replaced offline."""
import copy
import signal
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import full_teleop
from incremental import MappingError


class FaultContextMainTests(unittest.TestCase):
    def run_main(self, scenario, fail_capture=False, arm='right', gripper_enabled=False):
        actions,events=[],[]
        signal_handlers={}
        states=iter(scenario)
        config={'arm':arm,'follow_mode':'low','stream_trajectory_mode':0,'stream_radio':0,
            'record_data':False,'control_hz':100,'log_interval_s':.5,
            'rotation_mapping':'neutral_tool_axes','translation_mapping_frame':'work',
            'max_input_age_ms':100., 'gripper':{'enabled':gripper_enabled}}

        class Backend:
            def __init__(self,arm='right'):
                actions.append(('backend_arm',arm))
                self.frames={};self.installation={'x':0.,'y':0.,'z':0.}
                self.algorithm_installation=[0.]*3;self.initial_pose=[0.,0.,0.,1.,0.,0.,0.]
                self.initial_joints=[0.]*6;self.attempted=False
                self.robot=mock.Mock();self.robot.rm_algo_euler2quaternion.return_value=[1.,0.,0.,0.]
            def validate_startup_ik(self):return {'valid':True}
            def close(self):actions.append('backend_close')

        class Feedback:
            def __init__(self,host,*_,**__):actions.append(('feedback_host',host))
            def start(self):return {'position_m':[0.]*3,'orientation_rad':[0.]*3}
            def close(self):pass

        class Controller:
            def __init__(self,*_,**__):
                self.mapper=SimpleNamespace(active=False);self.streaming=False;self.send_count=0
            def tick(self,*_,**__):
                state=next(states)
                if state=='ERROR':raise MappingError('test tracking fault')
                if state=='SIGNAL_RELEASE':
                    signal_handlers[signal.SIGINT](signal.SIGINT,None)
                    state='WAIT_FRESH_INPUT'
                self.mapper.active=state=='TRACKING';self.streaming=self.mapper.active
                return {'mapping':{'state':state},'robot_feedback_metadata':{'sequence':7},
                    'robot_feedback_joints_deg':[0.]*6,'robot_feedback_pose_wxyz':[0.,0.,0.,1.,0.,0.,0.]}
            def stop(self):self.streaming=False;actions.append('stop')

        class Logger:
            def __init__(self,*_,**__):pass
            def append(self,event,**__):
                actions.append(event.get('event','report'));events.append(copy.deepcopy(event))
            def stats(self):return {}
            def check_error(self):pass
            def console_message(self,message):actions.append('console')
            def close(self,**__):return {'complete':True}

        class Gripper:
            def __init__(self,host,side,*_,**__):actions.append(('gripper_route',host,side))
            def start(self):actions.append('gripper_start')
            def update(self,status,active,now):
                actions.append(('gripper_active',active))
                return {'status':'FAULT','fault':'test offline gripper'}
            def cancel(self):actions.append('gripper_cancel')
            def close(self):actions.append('gripper_close')

        class BrokenContext(full_teleop.DiagnosticContext):
            def capture(self,*_,**__):raise RuntimeError('diagnostic unavailable')

        pico={'button_stream':{'receiver_id':'fake','exit_sequence':0},'receive_hz_2s':72.,'age_ms':1.,'latest':{'sequence':3,'receive_monotonic_ns':1,
            arm:{'grip':1.}}}
        with tempfile.TemporaryDirectory() as directory:
            task_dir=Path(directory)/'pico_xr';task_dir.mkdir()
            with mock.patch.multiple(full_teleop,TASK_DIR=task_dir,Backend=Backend,
                    FeedbackMonitor=Feedback,TeleopController=Controller,AsyncDiagnosticLogger=Logger,GripperWorker=Gripper,
                    DiagnosticContext=BrokenContext if fail_capture else full_teleop.DiagnosticContext), \
                    mock.patch.object(full_teleop,'load_config',return_value=config), \
                    mock.patch.object(full_teleop,'resolve_translation_mapping',return_value=(config,{})), \
                    mock.patch.object(full_teleop,'checked_pair',return_value=[180.]*6), \
                    mock.patch.object(full_teleop,'check_known_controller_processes'), \
                    mock.patch.object(full_teleop,'read_pico',return_value=pico), \
                    mock.patch.object(full_teleop.signal,'signal',side_effect=lambda sig,fn:signal_handlers.update({sig:fn})), \
                    mock.patch.object(sys,'argv',['full_teleop.py','--execute']):
                outcome=full_teleop.main()
            self.assertFalse((task_dir/'state'/'recordings').exists())
            if arm=='left':
                self.assertTrue((task_dir/'state'/'latest_left_full_session.txt').exists())
                self.assertFalse((task_dir/'state'/'latest_full_session.txt').exists())
        return outcome,actions,events

    def test_exception_stops_before_fault_snapshot_and_preserves_error(self):
        outcome,actions,events=self.run_main(['TRACKING','ERROR'])
        self.assertEqual(outcome,1)
        self.assertLess(actions.index('stop'),actions.index('FAULT_CONTEXT'))
        self.assertLess(actions.index('FAULT_CONTEXT'),actions.index('ERROR'))
        context=next(e for e in events if e.get('event')=='FAULT_CONTEXT')
        self.assertEqual(context['sample_count'],1)
        self.assertEqual(context['error']['message'],'test tracking fault')
        self.assertFalse(context['training_recording'])

    def test_passive_disarm_dumps_context_while_normal_release_does_not(self):
        for state,expected in [('WAIT_FRESH_INPUT',1),('RELEASED',0)]:
            with self.subTest(state=state):
                outcome,_,events=self.run_main(['TRACKING',state,'SESSION_EXIT'])
                self.assertEqual(outcome,0)
                contexts=[e for e in events if e.get('event')=='FAULT_CONTEXT']
                self.assertEqual(len(contexts),expected)
                if contexts:self.assertEqual(contexts[0]['reason']['state'],state)

    def test_observer_failure_does_not_stop_or_fail_teleoperation(self):
        outcome,_,events=self.run_main(['TRACKING','TRACKING','SESSION_EXIT'],True)
        self.assertEqual(outcome,0)
        self.assertEqual(events[-1]['event'],'SESSION_END')
        self.assertFalse(any(e.get('event')in('FAULT_CONTEXT','ERROR')for e in events))

    def test_signal_during_tick_is_normal_exit_without_fault_context(self):
        outcome,_,events=self.run_main(['TRACKING','SIGNAL_RELEASE'])
        self.assertEqual(outcome,0)
        self.assertEqual(events[-1]['reason'],'SIGNAL')
        self.assertFalse(any(e.get('event')=='FAULT_CONTEXT'for e in events))

    def test_gripper_fault_is_observed_without_ending_either_arm_control(self):
        for arm in ('right','left'):
            outcome,actions,events=self.run_main(['TRACKING','ERROR'],arm=arm,gripper_enabled=True)
            self.assertEqual(outcome,1)  # The injected arm error still stops normally.
            self.assertIn(('gripper_route',full_teleop.ARM_IPS[arm],arm),actions)
            self.assertEqual(next(e for e in events if 'gripper' in e)['gripper']['status'],'FAULT')
            self.assertLess(actions.index('gripper_cancel'),actions.index('stop'))
            self.assertIn('gripper_close',actions)
            outcome,_,events=self.run_main(['TRACKING','SESSION_EXIT'],arm=arm,gripper_enabled=True)
            self.assertEqual(outcome,0)
            self.assertEqual(events[-1]['event'],'SESSION_END')


if __name__=='__main__':unittest.main()
