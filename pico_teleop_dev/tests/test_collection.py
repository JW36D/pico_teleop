"""Collection boundaries and exit gates with fake devices only."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from button_events import ButtonEventRing, ExitGuard
from collection_session import ButtonCursor, CollectionSession
from raw_episode_writer import RawEpisodeWriter
from raw_dataset import inspect_episode, jsonl
from test_full_teleop import ControllerHarness


class RingTests(unittest.TestCase):
    def setUp(self):
        self.ring = ButtonEventRing(capacity=8)
        self.ns = 1_000_000_000
        self.seq = 0
        self.state = {k:False for k in ('A','X','B','Y')}

    def frame(self, focus=True, step_ms=10, **changes):
        self.ns += step_ms*1_000_000
        self.seq += 1
        self.state.update(changes)
        self.ring.observe({'receive_monotonic_ns':self.ns,'sequence':self.seq,'device_id':'one',
            'app_focus':focus,'input_mode':1,'left':{'primaryButton':self.state['X'], 'secondaryButton':self.state['Y']},
            'right':{'primaryButton':self.state['A'], 'secondaryButton':self.state['B']}})
        return self.ring.events[-1]

    def test_held_at_start_and_after_focus_loss_requires_release(self):
        self.assertEqual(self.frame(A=True)['pressed'],[])
        self.frame(A=False)
        self.assertEqual(self.frame(A=True)['pressed'],['A'])
        self.frame(focus=False)
        self.assertEqual(self.frame(A=True)['pressed'],[])
        self.frame(A=False)
        self.assertEqual(self.frame(A=True)['pressed'],['A'])

    def test_fast_click_is_retained_when_latest_pose_is_released(self):
        self.frame()
        cursor = self.ring.sequence
        self.frame(A=True)
        self.frame(A=False)
        reply = self.ring.read(cursor,self.ring.receiver_id)
        self.assertEqual([e['pressed'] for e in reply['events']],[['A'],[]])

    def test_chord_window_hold_and_single_trigger(self):
        self.frame()
        self.frame(B=True)
        event = self.frame(step_ms=100,Y=True)
        self.assertTrue(event['exit_chord'])
        seq = self.ring.exit_sequence
        self.frame(A=True)
        self.assertEqual(self.ring.exit_sequence,seq)
        self.assertFalse(self.ring.events[-1]['exit_chord'])

    def test_long_hold_then_other_key_is_not_chord(self):
        self.frame()
        self.frame(B=True)
        for _ in range(6):self.frame(step_ms=50)
        self.assertFalse(self.frame(Y=True)['exit_chord'])

    def test_gap_invalidates_held_buttons_and_increments_generation(self):
        self.frame()
        self.frame(B=True)
        event = self.frame(step_ms=200,Y=True)
        self.assertFalse(event['exit_chord'])
        self.assertEqual(self.ring.generation,1)

    def test_overflow_and_restart_do_not_replay_commands(self):
        self.frame()
        cursor = ButtonCursor()
        cursor.consume(self.ring.read(0,None))
        for i in range(12):self.frame(A=i%2==0)
        events,issue = cursor.consume(self.ring.read(cursor.cursor,cursor.receiver_id))
        self.assertEqual(events,[])
        self.assertEqual(issue,'EVENT_OVERFLOW')
        self.frame(A=True)
        events,_ = cursor.consume(self.ring.read(cursor.cursor,cursor.receiver_id))
        self.assertEqual(events[-1]['pressed'],['A'])  # valid release in reset snapshot
        self.ring.invalidate(self.ns+1)
        self.frame(A=True)
        events,_ = cursor.consume(self.ring.read(cursor.cursor,cursor.receiver_id))
        self.assertFalse(any('A' in e['pressed'] for e in events))

    def test_exit_guard_skips_historical_exit_and_latches_new_one(self):
        guard = ExitGuard()
        def status(seq,receiver='one'):
            return {'button_stream':{'receiver_id':receiver,'exit_sequence':seq}}
        self.assertFalse(guard.observe(status(10)))
        self.assertFalse(guard.observe(status(10)))
        self.assertTrue(guard.observe(status(20)))
        self.assertTrue(guard.observe(status(0,'two')))


class FakeWriter:
    def __init__(self):
        self.ready=True; self.error=None; self.actions=[]; self.count=0
    def begin(self,event,number,task):
        self.count+=1; episode='ep'+str(self.count)
        self.actions.append(('begin',episode,event['receive_monotonic_ns']))
        return episode
    def end(self,episode,event,**kwargs):
        self.actions.append(('end',episode,event['receive_monotonic_ns'],kwargs)); return True
    def delete(self,episode,event):
        self.actions.append(('delete',episode)); return True
    def event(self,event):pass
    def submit(self,event):pass


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.ns=1_000_000_000; self.writer=FakeWriter(); self.stops=[]
        self.session=CollectionSession(self.writer,'sorting',lambda:self.stops.append(True),lambda:self.ns)
    def batch(self,*keys,exit=False):
        self.ns+=10_000_000
        self.session.handle_batch([{'receive_monotonic_ns':self.ns,'event_sequence':self.ns,
                                    'pressed':list(keys),'exit_chord':exit}])
    def test_continuous_grip_has_two_independent_episodes_without_stop(self):
        for _ in range(4):self.batch('A')
        self.assertEqual([a[:2] for a in self.writer.actions],
                         [('begin','ep1'),('end','ep1'),('begin','ep2'),('end','ep2')])
        self.assertEqual(self.stops,[])
        self.assertEqual(self.session.deletable_episode_id,'ep2')
    def test_delete_once_and_never_falls_back_to_older_episode(self):
        for _ in range(4):self.batch('A')
        self.batch('X');self.batch('X')
        self.assertEqual([a for a in self.writer.actions if a[0]=='delete'],[('delete','ep2')])
        self.batch('A');self.batch('A');self.batch('X')
        self.assertEqual(self.writer.actions[-1],('delete','ep3'))
    def test_x_during_recording_does_not_schedule_delete(self):
        self.batch('A');self.batch('X');self.batch('A')
        self.assertFalse(any(a[0]=='delete' for a in self.writer.actions))
    def test_a_wins_over_x_in_same_batch_and_exit_wins_over_both(self):
        self.batch('A');self.batch('A','X')
        self.assertEqual(self.writer.actions[-1][0],'end')
        self.batch('A','X',exit=True)
        self.assertEqual(len(self.writer.actions),2)
        self.assertEqual(self.stops,[True])
    def test_new_start_revokes_previous_delete_and_b_y_alone_are_noops(self):
        self.batch('A');self.batch('A');self.batch('A')
        self.assertIsNone(self.session.deletable_episode_id)
        count=len(self.writer.actions)
        self.batch('B');self.batch('Y')
        self.assertEqual(len(self.writer.actions),count)
        self.assertEqual(self.stops,[])
    def test_not_ready_and_stale_event_do_not_claim_recording(self):
        self.session.handle_batch([{'receive_monotonic_ns':self.ns-200_000_000,'event_sequence':1,'pressed':['A']}])
        self.assertIsNone(self.session.active_episode_id)
        self.writer.ready=False
        self.batch('A')
        self.assertIn('未开始录制',self.session.message)
        self.assertIsNone(self.session.active_episode_id)
    def test_exit_marks_current_interrupted_and_retained(self):
        self.batch('A');self.batch(exit=True)
        self.assertEqual(self.writer.actions[-1][3]['interrupted'],True)
        self.assertEqual(self.session.deletable_episode_id,'ep1')


class NewGateTests(ControllerHarness,unittest.TestCase):
    def test_b_and_y_alone_do_not_stop_reanchor_or_cancel_send(self):
        self.active()
        anchor = copy.deepcopy(self.control.mapper.anchor_hand)
        for side in ('right','left'):
            status=self.sample(1,[0,0,-.002])
            status['latest'].setdefault(side,{})['secondaryButton']=True
            report=self.control.tick(status,lambda:status,self.ns)
            self.assertTrue(report['motion_command_sent_this_tick'])
            self.assertEqual(self.control.mapper.anchor_hand,anchor)
        self.assertEqual(self.backend.stops,0)
    def test_exit_after_final_candidate_computation_is_checked_at_emission(self):
        self.active(); before=len(self.backend.sent)
        def candidate(*args):
            self.control.stop_requested=lambda:True
        self.control.pending_joint_goal=([0]*7,[0]*6)
        self.control.final_joint_step=candidate
        # A direct send_result isolates the very last gate from earlier mapper work.
        status=self.sample(1,[0,0,-.002])
        measured=self.feedback.snapshot()
        report=self.control.send_result({'state':'TRACKING','send':True},status,
            self.backend.initial_pose,measured,self.ns,lambda:status)
        self.assertEqual(report['mapping']['state'],'SESSION_EXIT')
        self.assertEqual(len(self.backend.sent),before)
        self.assertFalse(self.control.mapper.active)


class WriterTests(unittest.TestCase):
    def test_boundary_filter_initial_state_two_episodes_and_delete_during_finalize(self):
        with tempfile.TemporaryDirectory() as root:
            writer=RawEpisodeWriter(root,{'mode':'PREVIEW_NO_MOTION'},commit_lag_s=.05)
            t=time.monotonic_ns()
            def sample(seq,stamp):
                return {'source':'left','source_sequence':seq,'session_id':writer.session_id,
                        'receive_monotonic_ns':stamp,'control_monotonic_ns':stamp,
                        'robot_joints_deg':[1,2,3,4,5,370], 'last_actually_sent_command':{'joints_deg':[1]*6}}
            writer.submit(sample(1,t-1))
            first=writer.begin({'receive_monotonic_ns':t},1,'sorting')
            writer.submit(sample(2,t))
            writer.submit(sample(3,t+10))
            writer.end(first,{'receive_monotonic_ns':t+10})
            second=writer.begin({'receive_monotonic_ns':t+10},2,'sorting')
            writer.submit(sample(4,t+20))
            writer.end(second,{'receive_monotonic_ns':t+30})
            writer.delete(second,{'receive_monotonic_ns':t+40})
            writer.close()
            episode=writer.directory/'episodes'/first
            rows=jsonl(episode/'left.jsonl')
            self.assertEqual([r['source_sequence'] for r in rows],[2])
            initial=json.loads((episode/'initial_state.json').read_text())
            self.assertEqual(initial['sides']['left']['source_sequence'],1)
            self.assertEqual(initial['sides']['left']['robot_joints_deg'][-1],370)
            self.assertTrue(inspect_episode(episode)['valid'])
            info=json.loads((episode/'episode.json').read_text())
            self.assertTrue(info['retained']);self.assertIsNone(info['task_success'])
            trash=writer.directory/'.trash'/second
            self.assertTrue(trash.is_dir())
            self.assertEqual([r['source_sequence'] for r in jsonl(trash/'left.jsonl')],[3,4])
            self.assertFalse(json.loads((trash/'episode.json').read_text())['retained'])

    def test_shutdown_without_a_end_is_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            writer=RawEpisodeWriter(root,{'mode':'PREVIEW_NO_MOTION'})
            episode=writer.begin({'receive_monotonic_ns':time.monotonic_ns()},1,'sorting')
            writer.close()
            info=json.loads((writer.directory/'episodes'/episode/'integrity.json').read_text())
            self.assertFalse(info['complete']);self.assertFalse(info['normal_close'])

    def test_disk_failure_does_not_produce_completion_commit(self):
        with tempfile.TemporaryDirectory() as root:
            writer=RawEpisodeWriter(root,{'mode':'PREVIEW_NO_MOTION'})
            with mock.patch('raw_episode_writer.atomic_json',side_effect=OSError('disk full')):
                episode=writer.begin({'receive_monotonic_ns':time.monotonic_ns()},1,'sorting')
                writer.thread.join(timeout=.2)
                with self.assertRaisesRegex(RuntimeError,'disk full'):writer.close()
            self.assertFalse((writer.directory/'episodes'/episode/'integrity.json').exists())


if __name__=='__main__':unittest.main()
