import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import citl_closed_loop as loop


def test_loop_measures_final_mask_and_saves_best(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "save_plots", lambda *a: None)
    proxy = loop.SyntheticProxy()
    seen = []

    def measure(commands, path):
        seen.append(commands.copy())
        return loop.synthetic_measurement(commands, path)

    result = loop.run_loop(proxy, measure, tmp_path, loop.LoopConfig(iters=6, lr=.03))
    assert len(seen) == 7
    assert result["measurements"] == 7
    assert result["final_loss"] < result["initial_loss"]
    np.testing.assert_array_equal(np.load(tmp_path / "phase_final_slm.npy"), seen[-1])
    np.testing.assert_array_equal(np.load(tmp_path / "phase_best_slm.npy"), seen[result["best_iteration"]])
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "complete"


def test_gradient_uses_measurement_not_proxy_error(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "save_plots", lambda *a: None)
    proxy = loop.SyntheticProxy()
    # Proxy is below target over most pixels; a brighter real measurement gives
    # the opposite gradient. Compare the first optimizer step to the measured VJP.
    measured = np.full((32, 40), 2., dtype=np.float32)

    def measure(commands, path):
        return SimpleNamespace(intensity=measured, valid_mask=np.ones_like(measured, bool),
                               confidence=np.ones_like(measured), fallback_mask=np.zeros_like(measured, bool),
                               diagnostics={})

    loop.run_loop(proxy, measure, tmp_path, loop.LoopConfig(iters=1, lr=.02))
    final = np.load(tmp_path / "phase_final_rad_internal.npy")
    np.testing.assert_allclose(final, .63, atol=1e-5)


def test_invalid_pixel_stops_before_next_update(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "save_plots", lambda *a: None)
    proxy = loop.SyntheticProxy()

    def measure(commands, path):
        frame = loop.synthetic_measurement(commands, path)
        if path.name == "0001":
            frame.valid_mask[2, 3] = False
            frame.intensity[2, 3] = np.nan
        return frame

    with pytest.raises(ValueError, match="initial loss support"):
        loop.run_loop(proxy, measure, tmp_path, loop.LoopConfig(iters=2))
    assert not (tmp_path / "phase_final_slm.npy").exists()
    assert (tmp_path / "phase_last_measured_slm.npy").exists()
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed"


def test_reference_then_target_capture_and_raw_roundtrip(tmp_path):
    events = []
    raw = np.arange(24, dtype=np.uint16).reshape(4, 6)

    class Hardware:
        metadata = {"fake": True}
        def upload_phase(self, value):
            events.append(int(value[0, 0]))
        def capture_bracket(self, exposures):
            return {x: raw.copy() for x in exposures}

    class Preprocessor:
        exposures_us = (300, 1000)
        def process(self, target_paths, reference_paths, stem):
            for path in [*target_paths.values(), *reference_paths.values()]:
                np.testing.assert_array_equal(np.fromfile(path, dtype="<u2").reshape(4, 6), raw)
            return "processed"

    measure = loop.LiveMeasurement(Hardware(), Preprocessor(), np.array([[7]], np.uint16))
    assert measure(np.array([[12]], np.uint16), tmp_path) == "processed"
    assert events == [7, 12]


def test_loop_through_real_hdr_worker_and_simulated_sensor(tmp_path, monkeypatch):
    """Exercise the whole numerical path: uploads -> Bayer -> HDR -> measured VJP."""
    import test_citl_camera_preprocess as fixtures
    from citl_camera_preprocess import CameraPreprocessConfig
    from citl_camera_worker import CameraPreprocessorWorker

    if not fixtures.SCRIPT.is_file():
        pytest.skip("set CITL_PREPROCESS_SCRIPT for real HDR backend integration")
    monkeypatch.setattr(loop, "save_plots", lambda *a: None)
    fixture = fixtures.CameraPreprocessingIntegrationTests("test_hdr_keeps_dark_measurements_and_masks_saturation")
    fixture.setUp()
    try:
        size = fixture.size
        proxy = loop.SyntheticProxy()
        proxy.initial_phase = torch.full((size, size), .4)
        proxy.target = torch.full((1, 1, size, size), .5)
        proxy.camera_tensor = lambda a: torch.as_tensor(a, dtype=torch.float32)[None, None] / 1e6

        class Sensor:
            metadata = {"simulated_sensor": True}
            def upload_phase(self, commands):
                self.commands = commands.copy()
            def capture_bracket(self, exposures):
                if self.commands.shape == (1, 1):
                    rate = fixture.reference * 1000
                else:
                    rad = self.commands.astype(np.float32) * (2 * np.pi / 1023)
                    rate = (.92 * np.square(np.sin(rad)) + .04) * 1e6
                result = {}
                for exposure in exposures:
                    mosaic = np.zeros((size * 2, size * 2), dtype=np.uint16)
                    mosaic[1::2, 1::2] = np.clip(np.rint(rate * exposure / 1e6), 0, 4095).astype(np.uint16)
                    result[exposure] = mosaic
                return result

        pp_config = CameraPreprocessConfig(fixture.root, fixtures.SCRIPT, fixture.lut_path,
                                           options=fixture.options)
        with CameraPreprocessorWorker(pp_config, tmp_path / "worker.log", timeout_s=60) as worker:
            measure = loop.LiveMeasurement(Sensor(), worker, np.array([[7]], np.uint16))
            result = loop.run_loop(proxy, measure, tmp_path / "run", loop.LoopConfig(iters=2, lr=.05))
        assert result["measurements"] == 3
        assert result["final_loss"] < result["initial_loss"]
        assert len(list((tmp_path / "run" / "iterations").glob("*/aligned_hdr.npy"))) == 3
    finally:
        fixture.doCleanups()
