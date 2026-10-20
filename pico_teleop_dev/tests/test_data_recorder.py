import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
from pathlib import Path
import tempfile
import unittest

from legacy_recording import DataRecorder
from incremental import MappingError


class DataRecorderTests(unittest.TestCase):
    def test_close_drains_samples_in_order_and_records_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder=DataRecorder(Path(directory)/'session',{'mode':'PREVIEW_NO_MOTION'})
            for tick in range(1000):recorder.append({'tick':tick,'motion_sent':False})
            recorder.close()
            rows=[json.loads(x) for x in (recorder.directory/'samples.jsonl').read_text().splitlines()]
            self.assertEqual([row['tick'] for row in rows],list(range(1000)))
            summary=json.loads((recorder.directory/'summary.json').read_text())
            self.assertTrue(summary['complete']);self.assertEqual(summary['written_samples'],1000)

    def test_writer_failure_is_visible_to_control_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder=DataRecorder(Path(directory)/'session',{})
            recorder.error='disk full'
            with self.assertRaisesRegex(MappingError,'disk full'):recorder.append({'tick':1})
            recorder.close()
            self.assertFalse(json.loads((recorder.directory/'summary.json').read_text())['complete'])


if __name__=='__main__':unittest.main()
