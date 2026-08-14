import importlib.util
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


def load_batch_pattern():
    pypylon = types.ModuleType('pypylon')
    pypylon.pylon = mock.MagicMock()
    cv2 = types.ModuleType('cv2')
    cv2.COLOR_BayerRG2BGR = 1

    hololith = types.ModuleType('hololith')
    slm_pkg = types.ModuleType('hololith.SLM')
    slmcontrol = types.ModuleType('hololith.SLM.slmcontrol')
    slm_win = types.ModuleType('hololith.SLM._slm_win')
    slm_win.SLM_OK = 0

    hololith.SLM = slm_pkg
    hololith.Util = types.SimpleNamespace(
        config=types.SimpleNamespace(readConfig=lambda path: object()))
    hololith.Mask = types.SimpleNamespace(
        tonemapper=types.SimpleNamespace(ToneMapper=lambda **kwargs: object()))

    modules = {
        'cv2': cv2,
        'pypylon': pypylon,
        'hololith': hololith,
        'hololith.SLM': slm_pkg,
        'hololith.SLM.slmcontrol': slmcontrol,
        'hololith.SLM._slm_win': slm_win,
    }
    with mock.patch.dict(sys.modules, modules):
        path = os.path.join(os.path.dirname(__file__), 'batch_pattern.py')
        spec = importlib.util.spec_from_file_location('batch_pattern_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class ReferenceGainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bp = load_batch_pattern()

    def test_manual_and_automatic_slot_layouts(self):
        bp = self.bp

        bp.REFERENCE_GAIN_ENABLED = False
        bp.MANUAL_ADVANCE_AFTER_SAVE = True
        self.assertEqual(bp.target_memory_location(0), 1)
        self.assertEqual(bp.max_target_chunk_size(), 128)

        bp.REFERENCE_GAIN_ENABLED = True
        bp.MANUAL_ADVANCE_AFTER_SAVE = True
        self.assertEqual(bp.reference_memory_location(5), 1)
        self.assertEqual(bp.target_memory_location(0), 2)
        self.assertEqual(bp.target_memory_location(5), 7)
        self.assertEqual(bp.max_target_chunk_size(), 127)

        bp.MANUAL_ADVANCE_AFTER_SAVE = False
        self.assertEqual(bp.reference_memory_location(0), 1)
        self.assertEqual(bp.reference_memory_location(1), 3)
        self.assertEqual(bp.target_memory_location(0), 2)
        self.assertEqual(bp.target_memory_location(1), 4)
        self.assertEqual(bp.slm_frames_per_target(), 2)
        self.assertEqual(bp.max_target_chunk_size(), 64)

    def test_reference_metric_tracks_multiplicative_intensity(self):
        bp = self.bp
        bp.REFERENCE_SIGNAL_PERCENTILE = 90.0
        bp.REFERENCE_BACKGROUND_PERCENTILE = 20.0
        bp.REFERENCE_MIN_ROI_PIXELS = 100

        baseline = np.full((100, 100), 100, dtype=np.uint16)
        baseline.flat[:2000] = 1100
        dimmed = np.full((100, 100), 100, dtype=np.uint16)
        dimmed.flat[:2000] = 900

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'reference_gain_log.csv')
            tracker = bp.ReferenceGainTracker(csv_path, 'reference.npy')
            first = tracker.measure(baseline)
            second = tracker.measure(dimmed)
            tracker.record(1, 1, 'target.npy', second, 'OK')

            self.assertAlmostEqual(first['correction_gain'], 1.0, places=6)
            self.assertAlmostEqual(second['relative_intensity'], 0.8, places=6)
            self.assertAlmostEqual(second['correction_gain'], 1.25, places=6)
            with open(csv_path, newline='') as f:
                rows = list(bp.csv.DictReader(f))
            self.assertEqual(rows[0]['mask_name'], 'target.npy')
            self.assertEqual(rows[0]['target_status'], 'OK')


if __name__ == '__main__':
    unittest.main()
