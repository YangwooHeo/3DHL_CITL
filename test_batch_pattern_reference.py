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

    def test_run_static_slot_layout(self):
        bp = self.bp

        bp.REFERENCE_GAIN_ENABLED = False
        self.assertEqual(bp.target_memory_location(0), 1)
        self.assertEqual(bp.max_target_chunk_size(), 128)

        bp.REFERENCE_GAIN_ENABLED = True
        self.assertEqual(bp.reference_memory_location(5), 1)
        self.assertEqual(bp.target_memory_location(0), 2)
        self.assertEqual(bp.target_memory_location(5), 7)
        self.assertEqual(bp.max_target_chunk_size(), 127)

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
            self.assertNotIn('registration_dx_px', rows[0])
            self.assertNotIn('registration_dy_px', rows[0])

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

    def test_optimized_slot_order_matches_run_static(self):
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

            frames = bp.build_chunk_slm_frames(
                [first_path, second_path], preprocessor)
            self.assertEqual([int(frame[0, 0]) for frame in frames], [9, 1, 2])

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
