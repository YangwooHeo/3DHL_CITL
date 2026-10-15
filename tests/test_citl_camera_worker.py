"""Real camera-calibration subprocess tests; no camera/SLM SDK is used."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from citl_camera_preprocess import CameraPreprocessConfig
from citl_camera_worker import CameraPreprocessorWorker
import test_citl_camera_preprocess as camera_fixtures

SCRIPT = camera_fixtures.SCRIPT


@unittest.skipUnless(SCRIPT.is_file(), "set CITL_PREPROCESS_SCRIPT for numerical integration tests")
class CameraWorkerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = camera_fixtures.CameraPreprocessingIntegrationTests("test_hdr_keeps_dark_measurements_and_masks_saturation")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.config = CameraPreprocessConfig(self.fixture.root, SCRIPT, self.fixture.lut_path,
                                              options=self.fixture.options)
        self.log_path = self.fixture.root / "camera_worker.log"

    def test_persistent_worker_roundtrip_and_fresh_gain(self):
        with CameraPreprocessorWorker(self.config, self.log_path, timeout_s=60) as worker:
            self.assertEqual(worker.exposures_us, self.fixture.exposures)
            self.assertEqual(worker.raw_shape, (384, 384))
            self.assertEqual(worker.output_size, self.fixture.size)
            pid = worker.provenance["worker"]["pid"]
            targets, refs = self.fixture.bracket()
            first = worker.process(targets, refs, "iteration_0000")
            np.testing.assert_allclose(np.median(first.intensity[50:90, 50:90]), 1e6, rtol=.01)
            self.assertTrue(np.isnan(first.intensity[150, 150]))
            self.assertFalse(first.valid_mask[150, 150])
            self.assertTrue(first.fallback_mask[20, 20])
            targets, refs = self.fixture.bracket(.75)
            second = worker.process(targets, refs, "iteration_0001")
            self.assertEqual(worker.provenance["worker"]["pid"], pid)
            np.testing.assert_allclose(np.median(second.intensity[50:90, 50:90]), 1e6, rtol=.01)
            self.assertAlmostEqual(second.diagnostics["reference_gain"]["1000"], 1/.75, delta=.015)
            self.assertFalse(list(worker._output_root.glob("*.npz")))
        self.assertIsNotNone(worker._process.poll())
        self.assertFalse(worker._output_root.exists())
        self.assertTrue(all(path.exists() for path in targets.values()))

    def test_backend_error_has_diagnostics_and_closes_worker(self):
        worker = CameraPreprocessorWorker(self.config, self.log_path, timeout_s=60)
        self.addCleanup(worker.close)
        targets, refs = self.fixture.bracket()
        refs.pop(300)
        with self.assertRaisesRegex(RuntimeError, "reference exposure keys"):
            worker.process(targets, refs, "iteration_bad")
        self.assertIsNotNone(worker._process.poll())
        self.assertFalse(worker._output_root.exists())
        self.assertIn("ValueError", self.log_path.read_text())

    def test_crashed_worker_reports_exit_without_hanging(self):
        worker = CameraPreprocessorWorker(self.config, self.log_path, timeout_s=60)
        self.addCleanup(worker.close)
        worker._process.kill()
        worker._process.wait(timeout=3)
        targets, refs = self.fixture.bracket()
        with self.assertRaisesRegex(RuntimeError, "worker.*(exit code|failed)"):
            worker.process(targets, refs, "iteration_crash")
        self.assertFalse(worker._output_root.exists())

    def test_request_timeout_closes_child_and_private_output_directory(self):
        worker = CameraPreprocessorWorker(self.config, self.log_path, timeout_s=60)
        self.addCleanup(worker.close)
        worker._timeout_s = 1e-9
        targets, refs = self.fixture.bracket()
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            worker.process(targets, refs, "iteration_timeout")
        self.assertIsNotNone(worker._process.poll())
        self.assertFalse(worker._output_root.exists())

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Torch unavailable in this interpreter")
    def test_torch_first_parent_can_process_with_isolated_worker(self):
        # Keep Torch out of this test runner so direct backend tests remain safe.
        script = r'''
import json, sys
from pathlib import Path
import torch
import numpy as np
from citl_camera_preprocess import CameraPreprocessConfig
from citl_camera_worker import CameraPreprocessorWorker
config = CameraPreprocessConfig(**json.loads(sys.argv[1]))
with CameraPreprocessorWorker(config, sys.argv[2], timeout_s=60) as worker:
    root = Path(config.training_root)
    targets = {e: root / f"target_{e}.raw" for e in worker.exposures_us}
    refs = {e: root / f"reference_{e}.raw" for e in worker.exposures_us}
    result = worker.process(targets, refs, "torch_parent")
    np.testing.assert_allclose(np.median(result.intensity[50:90, 50:90]), 1e6, rtol=.01)
print("torch-parent camera-worker roundtrip passed")
'''
        from dataclasses import asdict
        import json
        self.fixture.bracket()
        result = subprocess.run([sys.executable, "-c", script,
                                 json.dumps(asdict(self.config), default=str), str(self.log_path)],
                                capture_output=True, text=True, timeout=90,
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
        self.assertIn("roundtrip passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
