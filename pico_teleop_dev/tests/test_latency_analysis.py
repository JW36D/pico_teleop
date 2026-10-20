"""Optical edge pairing must preserve causality and expose missing events."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'camera'))
from measure_display_latency import match_edges


class OpticalLatencyTests(unittest.TestCase):
    def test_known_capture_frame_delay_and_polarity(self):
        physical=[(i*60,bool(i%2)) for i in range(24)]
        headset=[(frame+14,polarity) for frame,polarity in physical]
        matched,missing,extra=match_edges(physical,headset,120)
        self.assertEqual(len(matched),24)
        self.assertFalse(missing);self.assertFalse(extra)
        self.assertTrue(all(abs(x['delay_ms']-116.6666666667)<1e-6 for x in matched))

    def test_display_preceding_scene_is_not_a_valid_positive_delay(self):
        matched,missing,extra=match_edges([(100,True)],[(90,True)],120)
        self.assertFalse(matched)
        self.assertEqual(len(missing),1);self.assertEqual(len(extra),1)

    def test_missing_display_transition_is_reported_without_compressing_time(self):
        matched,missing,extra=match_edges([(0,True),(60,False),(120,True)],[(14,True),(134,True)],120)
        self.assertEqual([r['source_frame'] for r in matched],[0,120])
        self.assertEqual(missing,[{'source_frame':60,'polarity':False}])
        self.assertFalse(extra)


if __name__=='__main__': unittest.main()
