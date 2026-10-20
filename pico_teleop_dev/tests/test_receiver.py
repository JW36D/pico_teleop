import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""Offline checks for stale/invalid data in the future control input boundary."""
import ctypes
import json
import time
import unittest

import receiver as rx


class InputBoundaryTests(unittest.TestCase):
    def setUp(self):
        with rx.lock:
            rx.latest = None
            rx.last_receive_ns = None
            rx.first_sample = None
            rx.first_raw = None
            rx.samples.clear()
            rx.position_min.clear()
            rx.position_max.clear()
            rx.input_min.clear()
            rx.input_max.clear()
            rx.stats.update(frames=0, tracking_frames=0, parse_errors=0,
                            server_connected=True, device_connected=False)

    def frame(self, value):
        msg = rx.DeviceState()
        msg.devID = b'TestDevice'
        msg.stateJson = json.dumps({'functionName': 'Tracking',
                                    'value': json.dumps(value)}).encode()
        rx.callback(None, rx.DEVICE_STATE, 0, ctypes.addressof(msg))

    def test_stale_sample_is_not_fresh(self):
        self.frame({'Controller': {'left': {'pose': '0,0,0,0,0,0,1'}}})
        self.assertTrue(rx.snapshot()['fresh'])
        with rx.lock:
            rx.latest['receive_monotonic_ns'] = time.monotonic_ns() - 400_000_000
        self.assertFalse(rx.snapshot()['fresh'])

    def test_disconnect_invalidates_sample(self):
        self.frame({'Controller': {'right': {'pose': '0,0,0,0,0,0,1'}}})
        rx.callback(None, rx.SERVER_OFF, 0, None)
        self.assertIsNone(rx.snapshot()['latest'])
        self.assertFalse(rx.snapshot()['fresh'])

    def test_malformed_message_does_not_refresh_old_sample(self):
        self.frame({'Head': {'pose': '0,0,0,0,0,0,1'}})
        before = rx.latest['receive_monotonic_ns']
        msg = rx.DeviceState()
        msg.stateJson = b'not json'
        rx.callback(None, rx.DEVICE_STATE, 0, ctypes.addressof(msg))
        self.assertEqual(rx.latest['receive_monotonic_ns'], before)
        self.assertEqual(rx.stats['parse_errors'], 1)

    def test_missing_controller_has_no_fabricated_pose(self):
        self.frame({'Head': {'pose': '0,0,0,0,0,0,1'}})
        self.assertIsNone(rx.latest['left'])
        self.assertIsNone(rx.latest['right'])

    def test_nonfinite_pose_is_not_exposed(self):
        self.frame({'Controller': {'left': {'pose': 'nan,0,0,0,0,0,1'}}})
        self.assertIsNone(rx.latest['left']['pose'])

    def test_only_newest_frame_is_retained(self):
        self.frame({'Controller': {'right': {'pose': '1,0,0,0,0,0,1'}}})
        self.frame({'Controller': {'right': {'pose': '2,0,0,0,0,0,1'}}})
        self.assertEqual(rx.latest['sequence'], 2)
        self.assertEqual(rx.latest['right']['pose'][0], 2)


if __name__ == '__main__':
    unittest.main()
