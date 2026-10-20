"""Full coordinator + real Unix IPC + writer; child devices replaced with publishers."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import collection
from button_events import ButtonEventRing
from collection_ipc import Publisher
from raw_dataset import inspect_episode


class RuntimeTests(unittest.TestCase):
    def scenario(self,fail_left=False,fail_display=False,video_enabled=None,external_status=False):
        with tempfile.TemporaryDirectory() as directory:
            cfg=collection.load_collection_config(collection.CONFIG_DIR/'collection.json')
            cfg['data_root']=directory
            if video_enabled is not None:cfg['video']['enabled']=video_enabled
            cfg['writer']=dict(cfg['writer'],commit_lag_s=.05)
            actions=[];workers=[];epoch=[None]
            ring=ButtonEventRing()
            schedule=[(.35,'A',True),(.45,'A',False),(.55,'A',True),(.65,'A',False),
                      (.75,'A',True),(.85,'A',False),(.95,'A',True),(1.05,'A',False),
                      (1.15,'X',True),(1.25,'X',False),(1.35,'X',True),(1.45,'X',False),
                      (1.55,'B',True),(1.6,'Y',True)]
            state={k:False for k in 'AXBY'}
            def observe(stamp):
                ring.observe({'receive_monotonic_ns':stamp,'sequence':stamp,'device_id':'fake',
                    'app_focus':True,'input_mode':1,
                    'left':{'primaryButton':state['X'],'secondaryButton':state['Y']},
                    'right':{'primaryButton':state['A'],'secondaryButton':state['B']}})
            def receiver(path,command='status'):
                now=time.monotonic_ns()
                if epoch[0]:
                    while schedule and now>=epoch[0]+int(schedule[0][0]*1e9):
                        delay,key,down=schedule.pop(0);state[key]=down
                        observe(epoch[0]+int(delay*1e9))
                observe(now)
                if isinstance(command,dict):
                    return ring.read(command['cursor'],command['receiver_id'])
                return {'fresh':True,'age_ms':1.,'host_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        'button_stream':ring.summary()}
            class Worker:
                def __init__(self,command,**kwargs):
                    self.side='cameras' if '--ipc' in command else ('left' if 'left_rotation' in ' '.join(command) else 'right')
                    self.stop=threading.Event();self.code=None;workers.append(self)
                    ipc=command[command.index('--ipc' if self.side=='cameras' else '--collection-socket')+1]
                    session=command[command.index('--session-id')+1]
                    roles=('front','left_wrist','right_wrist') if self.side=='cameras' else (self.side,)
                    if self.side=='cameras':
                        video_file=Path(command[command.index('--video-config')+1])
                        actions.append(('video_enabled',json.loads(video_file.read_text())['video']['enabled']))
                        actions.append(('camera_status_path',command[command.index('--status-path')+1]))
                    self.pubs=[Publisher(ipc,role,session,metadata={'synthetic':True}) for role in roles]
                    self.display=Publisher(ipc,'display',session,metadata={'synthetic':True}) if fail_display and self.side=='cameras' else None
                    if epoch[0] is None:epoch[0]=time.monotonic_ns()
                    def run():
                        try:
                            sequence=0
                            while not self.stop.is_set():
                                stamp=time.monotonic_ns();sequence+=1
                                if self.display and stamp-epoch[0]>.42e9 and not self.display.dropped:
                                    self.display.dropped=1
                                    self.display.submit({'kind':'failure','receive_monotonic_ns':stamp,
                                        'error':'synthetic optional headset display failure'})
                                    actions.append(('display_failure','cameras'))
                                if fail_left and self.side=='left' and stamp-epoch[0]>.48e9:
                                    self.code=1;return
                                for pub in self.pubs:
                                    row={'kind':'frame' if self.side=='cameras' else 'telemetry',
                                         'receive_monotonic_ns':stamp,'synthetic':True,'clutch_active':True}
                                    if self.side=='cameras':row.update(frame_number=sequence)
                                    pub.submit(row,b'synthetic JPEG fixture' if self.side=='cameras' else b'')
                                self.stop.wait(.02)
                        finally:
                            if self.display:self.display.close()
                            for pub in self.pubs:pub.close()
                            if self.code is None:self.code=0
                    self.thread=threading.Thread(target=run,daemon=True);self.thread.start()
                def poll(self):return self.code
                def terminate(self):actions.append(('stop',self.side));self.stop.set()
                def wait(self,timeout):
                    self.thread.join(timeout)
                    if self.thread.is_alive():raise RuntimeError('fake source failed to terminate')
                    actions.append(('wait',self.side));return self.code
            with mock.patch.object(collection,'request_receiver',side_effect=receiver), \
                    mock.patch.object(collection.platform,'platform',return_value='synthetic-offline'), \
                    mock.patch.object(collection.subprocess,'Popen',side_effect=Worker),contextlib.redirect_stdout(io.StringIO()):
                status_path=Path(directory)/'observer.json' if external_status else None
                outcome=collection.run_collection(cfg,collection.CONFIG_DIR/'collection.json',False,'synthetic continuous grip',5,
                                                  status_path=status_path)
                if external_status:
                    actions.append(('final_status',json.loads(status_path.read_text())))
            session=next((Path(directory)/'sessions').iterdir())
            reports=[inspect_episode(p) for p in (session/'episodes').iterdir()]
            trash=[json.loads((p/'episode.json').read_text()) for p in (session/'.trash').iterdir()]
            return outcome,reports,trash,actions

    def test_two_continuous_grip_episodes_delete_once_and_chord_clean_shutdown(self):
        outcome,reports,trash,actions=self.scenario()
        self.assertEqual(outcome,0)
        self.assertEqual(len(reports),1);self.assertEqual(len(trash),1)
        report=reports[0]
        self.assertTrue(report['valid'],report['errors'])
        self.assertTrue(report['integrity']['complete'],report['integrity'])
        self.assertTrue(all(report['sources'][s]>0 for s in ('left','right','front','left_wrist','right_wrist')))
        self.assertIsNone(report['episode']['task_success'])
        self.assertTrue(trash[0]['deleted'])
        self.assertLess(actions.index(('stop','right')),actions.index(('wait','left')))

    def test_control_worker_failure_stops_peer_and_keeps_interrupted_raw(self):
        outcome,reports,trash,actions=self.scenario(True)
        self.assertEqual(outcome,1)
        self.assertEqual(len(reports),1)
        self.assertFalse(reports[0]['integrity']['complete'])
        self.assertEqual(reports[0]['episode']['termination'],'CONTROL_WORKER_EXIT')
        self.assertIn(('stop','right'),actions)

    def test_headset_display_failure_and_transport_loss_do_not_interrupt_recording(self):
        outcome,reports,trash,actions=self.scenario(fail_display=True)
        self.assertIn(('display_failure','cameras'),actions)
        self.assertEqual(outcome,0)
        self.assertEqual(len(reports),1);self.assertEqual(len(trash),1)
        self.assertTrue(reports[0]['integrity']['complete'],reports[0]['integrity'])
        self.assertEqual(reports[0]['episode']['termination'],'A_END')

    def test_no_video_override_reaches_camera_child_and_all_raw_sources_still_record(self):
        outcome,reports,trash,actions=self.scenario(video_enabled=False)
        self.assertIn(('video_enabled',False),actions)
        self.assertEqual(outcome,0)
        self.assertTrue(reports[0]['integrity']['complete'])
        self.assertTrue(all(reports[0]['sources'][s]>0 for s in ('left','right','front','left_wrist','right_wrist')))

    def test_external_observer_snapshot_survives_cleanup_and_matches_camera_overlay(self):
        outcome,reports,trash,actions=self.scenario(external_status=True)
        self.assertEqual(outcome,0)
        state=next(value for name,value in actions if name=='final_status')
        self.assertTrue(state['exiting'])
        self.assertFalse(state['ready'])
        self.assertFalse(state['recording'])
        self.assertEqual(state['episode_number'],2)
        self.assertIn('ended_monotonic_ns',state)
        self.assertTrue(next(value for name,value in actions if name=='camera_status_path').endswith('/observer.json'))
        self.assertTrue(reports[0]['integrity']['complete'])


if __name__=='__main__':unittest.main()
