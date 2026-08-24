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
        bp.REFERENCE_REGISTRATION_ENABLED = False
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
            self.assertIn('registration_dx_px', rows[0])
            self.assertIn('registration_dy_px', rows[0])

    def test_translation_is_shift_applied_to_moving_image(self):
        bp = self.bp
        bp.REFERENCE_REGISTRATION_SATURATION_THRESHOLD = np.inf
        bp.REFERENCE_REGISTRATION_DARK_THRESHOLD = -np.inf
        bp.REFERENCE_REGISTRATION_MASK_DILATE_RADIUS = 0
        bp.REFERENCE_REGISTRATION_BLUR_SIGMA = 0.0
        bp.REFERENCE_REGISTRATION_HIGHPASS_SIGMA = 0.0
        bp.REFERENCE_REGISTRATION_METHOD = 'phasecorr'
        bp.REFERENCE_REGISTRATION_MAX_EXPECTED_SHIFT = 50.0

        rng = np.random.default_rng(1234)
        fixed = rng.normal(size=(128, 160))
        moving = np.roll(fixed, shift=(3, -4), axis=(0, 1))
        dx, dy = bp.estimate_reference_translation(fixed, moving)

        self.assertAlmostEqual(dx, 4.0, delta=0.15)
        self.assertAlmostEqual(dy, -3.0, delta=0.15)

    def test_tracker_uses_first_reference_as_registration_anchor(self):
        bp = self.bp
        bp.REFERENCE_REGISTRATION_ENABLED = True
        bp.REFERENCE_REGISTRATION_CHANNEL = 'raw_bayer'
        bp.REFERENCE_REGISTRATION_SATURATION_THRESHOLD = np.inf
        bp.REFERENCE_REGISTRATION_DARK_THRESHOLD = -np.inf
        bp.REFERENCE_REGISTRATION_MASK_DILATE_RADIUS = 0
        bp.REFERENCE_REGISTRATION_BLUR_SIGMA = 0.0
        bp.REFERENCE_REGISTRATION_HIGHPASS_SIGMA = 0.0
        bp.REFERENCE_REGISTRATION_METHOD = 'phasecorr'
        bp.REFERENCE_REGISTRATION_MAX_EXPECTED_SHIFT = 50.0

        rng = np.random.default_rng(4321)
        y, x = np.mgrid[:96, :128]
        envelope = 1200 * np.exp(-((x - 64) ** 2 + (y - 48) ** 2) / (2 * 18 ** 2))
        anchor = (100 + envelope + rng.integers(0, 100, size=envelope.shape)).astype(
            np.uint16)
        moving = np.roll(anchor, shift=(-2, 3), axis=(0, 1))

        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = bp.ReferenceGainTracker(
                os.path.join(temp_dir, 'registration.csv'), 'reference.npy')
            first = tracker.measure(anchor)
            second = tracker.measure(moving)

        self.assertEqual(first['registration_dx_px'], 0.0)
        self.assertEqual(first['registration_dy_px'], 0.0)
        self.assertAlmostEqual(second['registration_dx_px'], -3.0, delta=0.2)
        self.assertAlmostEqual(second['registration_dy_px'], 2.0, delta=0.2)


if __name__ == '__main__':
    unittest.main()
