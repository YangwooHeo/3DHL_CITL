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

    def test_optimized_preprocessor_caches_wfc_and_bypasses_identity_tonemap(self):
        bp = self.bp

        class IdentityToneMapper:
            method = 'power_curve'
            power_curve_max_value = 1023
            power_curve_gamma = 1.0

            def __init__(self):
                self.calls = 0

            def __call__(self, image):
                self.calls += 1
                return image

        tonemapper = IdentityToneMapper()
        wfc = np.arange(48, dtype=np.uint16).reshape(6, 8)
        first_mask = np.arange(16, dtype=np.uint16).reshape(4, 4)
        second_mask = first_mask + 100

        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = os.path.join(temp_dir, 'first.npy')
            second_path = os.path.join(temp_dir, 'second.npy')
            np.save(first_path, first_mask)
            np.save(second_path, second_mask)

            with mock.patch.object(
                    bp, 'load_wavefront_correction', return_value=wfc) as loader:
                preprocessor = bp.SLMFramePreprocessor(
                    'wfc.csv', tonemapper, frame_shape=(6, 8))
                first, _ = preprocessor.prepare_frame(first_path)
                second, _ = preprocessor.prepare_frame(second_path)

        expected_first = np.zeros((6, 8), dtype=np.uint16)
        expected_first[1:5, 2:6] = first_mask
        expected_first = (expected_first + wfc * (
            np.pad(np.ones((4, 4), dtype=np.uint16), ((1, 1), (2, 2))))) % 1024

        loader.assert_called_once_with('wfc.csv')
        self.assertEqual(tonemapper.calls, 0)
        self.assertTrue(first.flags.c_contiguous)
        self.assertTrue(second.flags.c_contiguous)
        np.testing.assert_array_equal(first, expected_first)

    def test_optimized_slot_order_matches_reference_modes(self):
        bp = self.bp

        class IdentityToneMapper:
            method = 'power_curve'
            power_curve_max_value = 1023
            power_curve_gamma = 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = os.path.join(temp_dir, 'reference.npy')
            first_path = os.path.join(temp_dir, 'first.npy')
            second_path = os.path.join(temp_dir, 'second.npy')
            np.save(reference_path, np.full((2, 2), 9, dtype=np.uint16))
            np.save(first_path, np.full((2, 2), 1, dtype=np.uint16))
            np.save(second_path, np.full((2, 2), 2, dtype=np.uint16))

            bp.REFERENCE_GAIN_ENABLED = True
            bp.REFERENCE_MASK_PATH = reference_path
            preprocessor = bp.SLMFramePreprocessor(
                None, IdentityToneMapper(), frame_shape=(2, 2))

            bp.MANUAL_ADVANCE_AFTER_SAVE = True
            manual = bp.build_chunk_slm_frames(
                [first_path, second_path], preprocessor)
            self.assertEqual([int(frame[0, 0]) for frame in manual], [9, 1, 2])

            bp.MANUAL_ADVANCE_AFTER_SAVE = False
            automatic = bp.build_chunk_slm_frames(
                [first_path, second_path], preprocessor)
            self.assertEqual(
                [int(frame[0, 0]) for frame in automatic], [9, 1, 9, 2])

    def test_direct_upload_preserves_frame_and_slot_order(self):
        bp = self.bp

        class FakeSLM:
            def __init__(self):
                self.uploads = []
                self.calls = []

            def displayConstantValue(self, phase_integer):
                self.calls.append(('constant', phase_integer))

            def uploadPhaseMask(self, mask, memory_location):
                self.uploads.append((memory_location, mask))

            def resetDisplayOrder(self, from_idx, to_idx_include):
                self.calls.append(('order', from_idx, to_idx_include))

            def setMemoryPlaybackRange(self, start_frame_idx, end_frame_idx_include):
                self.calls.append(('range', start_frame_idx, end_frame_idx_include))

        slm = FakeSLM()
        pc = types.SimpleNamespace(
            slm_ctrl=slm,
            _resetStartFrame=lambda: slm.calls.append(('start',)))
        frames = [
            np.ascontiguousarray(np.full((2, 3), value, dtype=np.uint16))
            for value in (11, 22, 33)
        ]

        bp.upload_slm_frames(pc, frames)

        self.assertEqual([slot for slot, _ in slm.uploads], [1, 2, 3])
        self.assertEqual(
            [int(frame[0, 0]) for _, frame in slm.uploads], [11, 22, 33])
        self.assertEqual(slm.calls, [
            ('constant', 0), ('order', 1, 3), ('range', 1, 3), ('start',)])


if __name__ == '__main__':
    unittest.main()
