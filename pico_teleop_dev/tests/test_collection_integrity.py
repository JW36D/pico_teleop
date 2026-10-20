"""Failure and delayed-delivery tests for raw collection, no robot dependencies."""
import json
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from raw_episode_writer import RawEpisodeWriter
from raw_dataset import inspect_episode, jsonl, recover
import raw_dataset
from collection_ipc import Publisher, IngestServer


def sample(writer,sequence,stamp):
    return {'kind':'telemetry','source':'left','source_sequence':sequence,
            'receive_monotonic_ns':stamp,'session_id':writer.session_id}


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def writer(self,**metadata):
        return RawEpisodeWriter(self.directory.name,dict(mode='PREVIEW_NO_MOTION',**metadata),
                                commit_lag_s=.01,history_s=.3)

    def test_checker_accepts_session_or_episode_and_rejects_empty_success(self):
        w=self.writer();t=time.monotonic_ns()
        ep=w.begin({'receive_monotonic_ns':t},1,'source scope')
        w.submit(sample(w,1,t+1));w.end(ep,{'receive_monotonic_ns':t+20});w.close()
        for source in (Path(self.directory.name),w.directory,w.directory/'episodes'/ep):
            output=io.StringIO()
            with mock.patch.object(sys,'argv',['check','--source',str(source)]),contextlib.redirect_stdout(output):
                code=raw_dataset.main()
            self.assertEqual(code,0)
            self.assertEqual(json.loads(output.getvalue())['episode_count'],1)
        empty=Path(self.directory.name)/'empty';empty.mkdir()
        with mock.patch.object(sys,'argv',['check','--source',str(empty)]),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(raw_dataset.main(),1)

    def test_delayed_a_end_cannot_commit_post_boundary_samples(self):
        w=self.writer(boundary_watermarks=True)
        t=time.monotonic_ns()-100_000_000
        ep=w.begin({'receive_monotonic_ns':t},1,'delayed A')
        w.submit(sample(w,1,t+10))
        w.submit(sample(w,2,t+20))
        # More than the commit lag passes before the coordinator delivers A.
        time.sleep(.04)
        w.end(ep,{'receive_monotonic_ns':t+20})
        w.advance_boundary_watermark(t+30)
        w.close()
        path=w.directory/'episodes'/ep
        self.assertEqual([r['source_sequence'] for r in jsonl(path/'left.jsonl')],[1])
        self.assertTrue(inspect_episode(path)['integrity']['complete'])

    def test_late_ipc_samples_arrive_before_end_watermark_and_are_included(self):
        w=self.writer(required_sources=['left'],boundary_watermarks=True)
        t=time.monotonic_ns()
        ep=w.begin({'receive_monotonic_ns':t},1,'delayed IPC')
        w.advance_boundary_watermark(t)
        w.end(ep,{'receive_monotonic_ns':t+20})
        time.sleep(.04)
        self.assertFalse((w.directory/'episodes'/ep/'integrity.json').exists())
        w.submit(sample(w,1,t+10));w.submit(sample(w,2,t+30))
        w.close()
        report=inspect_episode(w.directory/'episodes'/ep)
        self.assertTrue(report['integrity']['complete'])
        self.assertEqual(report['sources']['left'],1)

    def test_source_disconnect_without_end_watermark_is_incomplete(self):
        w=self.writer(required_sources=['left'])
        t=time.monotonic_ns();ep=w.begin({'receive_monotonic_ns':t},1,'source loss')
        w.submit(sample(w,1,t+1));w.end(ep,{'receive_monotonic_ns':t+20})
        w.close()
        report=inspect_episode(w.directory/'episodes'/ep)
        self.assertFalse(report['integrity']['complete'])
        self.assertIn('left:end_watermark_not_observed',report['integrity']['errors'])

    def test_clock_regression_is_visible_and_not_a_complete_record(self):
        w=self.writer();t=time.monotonic_ns();ep=w.begin({'receive_monotonic_ns':t},1,'clock')
        w.submit(sample(w,1,t+20));w.submit(sample(w,2,t+10))
        w.end(ep,{'receive_monotonic_ns':t+30})
        with self.assertRaisesRegex(RuntimeError,'time_regression'):w.close()
        self.assertFalse(inspect_episode(w.directory/'episodes'/ep)['integrity']['complete'])

    def test_bounded_queue_overflow_is_counted_and_marks_incomplete(self):
        w=self.writer();t=time.monotonic_ns();ep=w.begin({'receive_monotonic_ns':t},1,'overflow')
        # Inject an unavailable bounded queue while the writer remains operational.
        import queue
        with mock.patch.object(w.sources,'put_nowait',side_effect=queue.Full):
            self.assertFalse(w.submit(sample(w,1,t+1)))
        w.end(ep,{'receive_monotonic_ns':t+20})
        with self.assertRaisesRegex(RuntimeError,'queue_overflow'):w.close()
        report=inspect_episode(w.directory/'episodes'/ep)
        self.assertFalse(report['integrity']['complete'])
        self.assertEqual(report['integrity']['writer_dropped'],1)

    def test_missing_jpeg_is_located_by_read_only_checker(self):
        w=self.writer();t=time.monotonic_ns();ep=w.begin({'receive_monotonic_ns':t},1,'missing image')
        row=sample(w,1,t+1)
        row.update(source='front',frame_number=12)
        w.submit(row,b'fixture bytes')
        w.end(ep,{'receive_monotonic_ns':t+20});w.close()
        path=w.directory/'episodes'/ep
        image=next((path/'images/front').glob('*.jpg'));image.unlink()
        report=inspect_episode(path)
        self.assertIn('front:missing_image',report['errors'])

    def test_explicit_recovery_never_claims_complete(self):
        w=self.writer();t=time.monotonic_ns();ep=w.begin({'receive_monotonic_ns':t},1,'crash')
        w.close();path=w.directory/'episodes'/ep
        (path/'integrity.json').unlink()
        self.assertIn('missing_completion_commit',inspect_episode(path)['errors'])
        recover(path)
        report=inspect_episode(path)
        self.assertFalse(report['integrity']['complete'])
        self.assertEqual(report['episode']['termination'],'RECOVERED_INTERRUPTION')

    def test_real_ipc_shutdown_drains_all_sources_and_final_counts(self):
        records=[]
        socket=Path(self.directory.name)/'ingest.sock'
        server=IngestServer(socket,lambda row,blob:records.append(row))
        publisher=Publisher(socket,'left','fixture',capacity=128)
        for i in range(70):publisher.submit({'kind':'telemetry','receive_monotonic_ns':time.monotonic_ns()})
        publisher.close();server.close()
        rows=[r for r in records if r['kind']=='telemetry']
        self.assertEqual([r['source_sequence'] for r in rows],list(range(1,71)))
        final=next(r for r in records if r['kind']=='source_closed')
        self.assertEqual(final['last_source_sequence'],70)
        self.assertEqual(final['transport_dropped'],0)
        self.assertEqual(final['undelivered'],0)


if __name__=='__main__':unittest.main()
