"""Synthetic tests; numerical integration uses the bundled preprocessing script.

Set CITL_PREPROCESS_SCRIPT to test a different preprocessing script. Tests never
connect to hardware and create all camera frames in a temporary directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
import uuid

import numpy as np

from citl_camera_preprocess import (
    CameraPreprocessConfig, CameraPreprocessor, _default_arguments,
    _report_options, _validate_matrix,
)


SCRIPT = Path(os.environ.get(
    "CITL_PREPROCESS_SCRIPT",
    str(Path(__file__).resolve().parents[1] / "preprocess_hdr_aligned_camera.py")))


class ConfigurationTests(unittest.TestCase):
    @unittest.skipUnless(SCRIPT.is_file(), "set CITL_PREPROCESS_SCRIPT for numerical integration tests")
    def test_numerical_suite_isolated_when_torch_is_loaded(self):
        if "torch" not in sys.modules:
            return  # The numerical cases run directly in this clean process.
        # Pytest imports all test modules at collection time, including Torch.
        # Keep the real numerical coverage while isolating native CPU runtimes,
        # just as the production CameraPreprocessorWorker does.
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests",
                   "-p", "test_citl_camera_preprocess.py", "-v"]
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("skipped=", result.stderr)

    def test_matrix_rejects_corruption(self):
        for value in (np.zeros((3, 3)), np.eye(2), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                _validate_matrix(value)

    def test_report_restores_geometry_and_registration(self):
        result = _report_options({
            "raw_shape": [2168, 4096], "exposures_us": [300, 1000, 5000],
            "geometry_exposure_us": 1000,
            "reference_registration": {"channel": "blue", "enabled": True,
                                       "max_expected_shift_px": 17.0},
        })
        self.assertEqual(result["raw_width"], 4096)
        self.assertEqual(result["geometry_exposure"], 1000)
        self.assertEqual(result["registration_max_shift"], 17)
        self.assertFalse(result["disable_reference_registration"])

    def test_defaults_do_not_parse_or_mutate_application_arguments(self):
        import argparse
        backend = SimpleNamespace(argparse=argparse)

        def parse_args():
            parser = backend.argparse.ArgumentParser()
            parser.add_argument("--camera-value", default="expected")
            return parser.parse_args()

        backend.parse_args = parse_args
        original = sys.argv
        sys.argv = ["application", "--unknown-to-backend"]
        try:
            current = sys.argv
            args = _default_arguments(backend)
            self.assertIs(sys.argv, current)
            self.assertEqual(args.camera_value, "expected")
            self.assertIs(backend.argparse, argparse)
        finally:
            sys.argv = original


@unittest.skipUnless(SCRIPT.is_file(), "set CITL_PREPROCESS_SCRIPT for numerical integration tests")
class CameraPreprocessingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "torch" in sys.modules:
            raise unittest.SkipTest(
                "Native CPU runtime is isolated from Torch; the required subprocess-suite test runs these cases")

    def setUp(self):
        test_root = Path(__file__).resolve().parent
        self.root = test_root / (".citl-test-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.assertEqual(self.root.resolve().parent, test_root)
        self.addCleanup(shutil.rmtree, self.root)
        self.exposures = (300, 1000, 5000)
        self.size = 192
        yy, xx = np.mgrid[:self.size, :self.size]
        self.reference = np.full((self.size, self.size), 20.0)
        rng = np.random.default_rng(23)
        for cy, cx, sigma, power in zip(rng.uniform(20, 172, 35), rng.uniform(20, 172, 35),
                                        rng.uniform(2, 7, 35), rng.uniform(80, 300, 35)):
            self.reference += power * np.exp(-((yy-cy)**2 + (xx-cx)**2)/(2*sigma**2))
        self.lut_path = self.root / "response.npy"
        np.save(self.lut_path, np.array([[0., 0.], [4091., 4091.]]))
        for exposure in self.exposures:
            folder = self.root / f"Exp{exposure}"
            refs = folder / "reference_captures"
            refs.mkdir(parents=True)
            self.write_raw(refs / "training_anchor.raw", self.reference * exposure / 1000)
            self.write_raw(folder / "training_anchor.raw", self.reference * exposure / 1000)
        self.report = {
            "algorithm_version": "raw-reference-hdr-v3",
            "exposures_us": list(self.exposures), "geometry_exposure_us": 1000,
            "anchor_exposure_us": 1000, "homography_pattern_names": ["training_anchor"],
            "raw_shape": [self.size * 2, self.size * 2], "output_size": self.size,
            "output_channels": "blue", "reference_gain_mode": "raw",
            "reference_registration": {"enabled": True, "channel": "blue", "method": "ecc",
                                       "max_expected_shift_px": 6., "downsample": 2},
            "lut": {"path": str(self.lut_path), "rows": 2, "min_dn": 0, "max_dn": 4091},
            "positions": {"z_0": {
                "transforms": {"shared_matrix": np.eye(3).tolist()},
                "reference_registration_summary": {
                    str(e): {"anchor_stem": "training_anchor"} for e in self.exposures},
                "radiometric_calibration": {str(e): {"dark_applied": False,
                    "flat_applied": False, "dark_frames": 0, "flat_frames": 0} for e in self.exposures},
            }},
        }
        self.report_path = self.root / "preprocessing_report.json"
        self.write_report()
        self.options = {"registration_roi_size": self.size, "reference_gain_min_pixels": 20,
                        "reference_gain_low_dn": 20.0}

    def write_report(self):
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")

    def write_raw(self, path, blue):
        mosaic = np.zeros((self.size*2, self.size*2), dtype="<u2")
        mosaic[1::2, 1::2] = np.clip(np.rint(blue), 0, 4095).astype(np.uint16)
        mosaic.tofile(path)

    def processor(self):
        return CameraPreprocessor(CameraPreprocessConfig(
            self.root, SCRIPT, self.lut_path, options=self.options))

    def bracket(self, power=1.0):
        target = np.full((self.size, self.size), 1000.0)
        target[10:40, 10:40] = 0
        target[100:150, 30:80] = 50
        target[130:170, 130:170] = 20000
        targets, refs = {}, {}
        for exposure in self.exposures:
            targets[exposure] = self.root / f"target_{exposure}.raw"
            refs[exposure] = self.root / f"reference_{exposure}.raw"
            self.write_raw(targets[exposure], target * power * exposure / 1000)
            self.write_raw(refs[exposure], self.reference * power * exposure / 1000)
        return targets, refs

    def test_hdr_keeps_dark_measurements_and_masks_saturation(self):
        processor = self.processor()
        targets, refs = self.bracket()
        result = processor.process(targets, refs, "iteration_0000")
        np.testing.assert_allclose(np.median(result.intensity[50:90, 50:90]), 1e6, rtol=.01)
        self.assertTrue(result.valid_mask[20, 20])
        self.assertTrue(result.fallback_mask[20, 20])
        self.assertEqual(result.intensity[20, 20], 0)
        self.assertFalse(result.valid_mask[150, 150])
        self.assertTrue(np.isnan(result.intensity[150, 150]))
        self.assertEqual(processor.provenance["geometry_source"], "training_report")
        self.assertIn("low_full_weight_dn", processor.provenance["unrecorded_options_using_source_defaults"])

    def test_repeated_stem_reestimates_fresh_power_gain(self):
        processor = self.processor()
        targets, refs = self.bracket(1.0)
        first = processor.process(targets, refs, "iteration_0000")
        targets, refs = self.bracket(0.75)
        second = processor.process(targets, refs, "iteration_0000")
        ratio = np.median(second.intensity[50:90, 50:90]) / np.median(first.intensity[50:90, 50:90])
        self.assertAlmostEqual(float(ratio), 1., delta=.01)
        self.assertAlmostEqual(second.diagnostics["reference_gain"]["1000"], 1/.75, delta=.015)

    def test_reference_translation_is_applied_in_correct_direction(self):
        processor = self.processor()
        targets, refs = self.bracket()
        for paths in (targets, refs):
            for path in paths.values():
                mosaic = np.fromfile(path, dtype="<u2").reshape(self.size*2, self.size*2)
                blue = mosaic[1::2, 1::2]
                self.write_raw(path, np.roll(blue, (-2, 3), axis=(0, 1)))
        result = processor.process(targets, refs, "iteration_0001")
        np.testing.assert_allclose(result.diagnostics["reference_shift_xy"]["1000"], [-3, 2], atol=.1)
        self.assertTrue(result.valid_mask[20, 20])
        self.assertEqual(result.intensity[20, 20], 0)
        np.testing.assert_allclose(np.median(result.intensity[110:140, 40:70]), 50000., rtol=.02)

    def test_nonlinear_lut_is_applied_before_exposure_normalization(self):
        dn = np.arange(4092, dtype=float)
        np.save(self.lut_path, np.column_stack([dn, dn**2]))
        self.report["lut"]["rows"] = len(dn)
        self.write_report()
        processor = self.processor()
        targets, refs = self.bracket()
        for exposure in self.exposures:
            factor = np.sqrt(exposure/1000)
            self.write_raw(targets[exposure], np.full((self.size, self.size), 1000.) * factor)
            self.write_raw(refs[exposure], self.reference * factor)
        result = processor.process(targets, refs, "iteration_nonlinear")
        np.testing.assert_allclose(np.median(result.intensity[30:160, 30:160]), 1e9, rtol=.01)

    def test_missing_reference_fails_without_silent_identity_gain(self):
        processor = self.processor()
        targets, refs = self.bracket()
        refs.pop(300)
        with self.assertRaisesRegex(ValueError, "reference exposure keys"):
            processor.process(targets, refs, "iteration_0000")

    def test_report_missing_requires_explicit_refit(self):
        self.report_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "preprocessing_report"):
            self.processor()

    def test_training_dark_calibration_cannot_be_silently_dropped(self):
        self.report["positions"]["z_0"]["radiometric_calibration"]["1000"]["dark_applied"] = True
        self.write_report()
        with self.assertRaisesRegex(ValueError, "provide options.dark_root"):
            self.processor()

    def test_recorded_settings_cannot_be_overridden(self):
        self.options["output_size"] = 128
        with self.assertRaisesRegex(ValueError, "conflicts with training report"):
            self.processor()

    def test_malformed_raw_rejected_before_processing(self):
        processor = self.processor()
        targets, refs = self.bracket()
        targets[300].write_bytes(b"invalid")
        with self.assertRaisesRegex(ValueError, "wrong RAW size"):
            processor.process(targets, refs, "iteration_0000")


if __name__ == "__main__":
    unittest.main()
