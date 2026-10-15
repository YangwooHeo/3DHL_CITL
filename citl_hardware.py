"""Persistent, single-plane SLM + Basler acquisition for the live CITL loop.

The device convention follows ``batch_pattern_and_capture.py``: the SLM memory
API takes native row-major 10-bit uint16 commands, with centered zero padding.
There is no transpose, RGB bit packing, or phase/LUT conversion here. The caller
owns phase quantization. Captures deliberately retain BayerRG12 sensor DN, as
required by ``preprocess_hdr_aligned_camera.py`` (not the batch PNG exporter).

Importing this module never imports vendor SDKs or connects to hardware.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import importlib
import math
from numbers import Integral, Real
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import warnings

import numpy as np


@dataclass(frozen=True)
class HardwareConfig:
    config_path: str
    hololith_root: str | None = None
    camera_serial: str | None = None
    pixel_format: str = "BayerRG12"
    camera_gain: float | None = None
    laser_pwm: int = 152
    average_frames: int = 1
    flush_frames: int = 2
    slm_settle_s: float = 0.30
    laser_settle_s: float = 0.10
    timeout_ms: int = 5000
    slm_timeout_s: float = 15.0
    slm_poll_s: float = 0.05
    raw_shape: tuple[int, int] = (2168, 4096)
    slm_shape: tuple[int, int] = (1200, 1920)

    def __post_init__(self):
        if self.pixel_format != "BayerRG12":
            raise ValueError("Training preprocessing requires unpacked BayerRG12 sensor data")
        for name, lower, upper in (("laser_pwm", 1, 255), ("average_frames", 1, 1),
                                   ("flush_frames", 0, None), ("timeout_ms", 1, None)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < lower or (upper is not None and value > upper):
                if name == "average_frames":
                    raise ValueError("average_frames must be 1: inverse camera LUT and saturation "
                                     "classification must precede any frame averaging")
                raise ValueError(f"{name} must be an integer in [{lower}, {upper or 'unbounded'}]")
        for name in ("slm_settle_s", "laser_settle_s", "slm_timeout_s", "slm_poll_s"):
            value = getattr(self, name)
            positive = name in ("slm_timeout_s", "slm_poll_s")
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0 or (positive and value == 0):
                raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
        for shape in (self.raw_shape, self.slm_shape):
            if len(shape) != 2 or any(isinstance(x, bool) or not isinstance(x, Integral) or x < 1 for x in shape):
                raise ValueError("raw_shape and slm_shape must contain two positive integer dimensions")
        if self.camera_gain is not None and (isinstance(self.camera_gain, bool) or not isinstance(self.camera_gain, Real) or not math.isfinite(self.camera_gain)):
            raise ValueError("camera_gain must be a finite number or null")


def prepare_slm_commands(commands: np.ndarray, shape=(1200, 1920)) -> np.ndarray:
    """Validate device-oriented codes and center-pad exactly like MaskStack."""
    commands = np.asarray(commands)
    if commands.ndim == 3 and commands.shape[2] == 1:
        commands = commands[:, :, 0]
    if commands.ndim != 2 or not commands.size:
        raise ValueError("Exactly one nonempty 2-D SLM mask is required")
    if not np.issubdtype(commands.dtype, np.integer):
        raise TypeError("upload_phase expects integer device commands, not radians")
    if commands.min() < 0 or commands.max() > 1023:
        raise ValueError("SLM commands must be in 0..1023")
    if any(source > target for source, target in zip(commands.shape, shape)):
        raise ValueError(f"Mask {commands.shape} exceeds SLM dimensions {shape}")
    padding = []
    for source, target in zip(commands.shape, shape):
        before = (target - source) // 2
        padding.append((before, target - source - before))
    return np.ascontiguousarray(np.pad(commands, padding), dtype=np.uint16)


class HololithCameraHardware:
    """Open devices once; upload/capture repeatedly; close all on any failure.

    ``capture_bracket`` uses integer microseconds and returns a dict of raw
    uint16 Bayer arrays with untouched sensor DN. One frame per exposure is
    required: averaging before nonlinear inverse LUT/saturation classification
    would differ from the training calibration. The laser is off
    during uploads, Python optimization, disk writes, and between brackets.
    """

    def __init__(self, config: HardwareConfig, *, _sdk=None):
        self.config = config
        self._sdk = _sdk  # injectable vendor boundary for hardware-free tests
        self._camera = None
        self._arduino = None
        self._slm_number = None
        self._slm_open = False
        self._laser_on = False
        self._opened = False
        self._phase_uploaded = False
        self.metadata = {}

    def _load_sdk(self):
        if self._sdk is not None:
            return
        if self.config.hololith_root:
            root = Path(self.config.hololith_root).expanduser().resolve()
            if not (root / "hololith").is_dir():
                raise FileNotFoundError(f"hololith package not found under {root}")
            sys.path.insert(0, str(root))
        try:
            self._sdk = SimpleNamespace(
                pylon=importlib.import_module("pypylon.pylon"),
                slm=importlib.import_module("hololith.SLM._slm_win"),
                read_config=importlib.import_module("hololith.Util.config").readConfig,
                arduino_class=importlib.import_module(
                    "hololith.Arduino.arduinocontrol").ArduinoController,
            )
        except ImportError as exc:
            raise RuntimeError("Live acquisition requires pypylon and the installed hololith "
                               "controller/SLM DLL on the acquisition PC") from exc

    def _check_slm(self, status, operation):
        if status != self._sdk.slm.SLM_OK:
            raise RuntimeError(f"{operation} failed with SLM status {status}")

    def _wait_slm(self):
        deadline = time.monotonic() + self.config.slm_timeout_s
        while True:
            status = self._sdk.slm.SLM_Ctrl_ReadSU(self._slm_number)
            if status == self._sdk.slm.SLM_OK:
                return
            if status != self._sdk.slm.SLM_BS:
                self._check_slm(status, "SLM readiness query")
            if time.monotonic() >= deadline:
                raise TimeoutError("SLM readiness timed out")
            time.sleep(self.config.slm_poll_s)

    def _slm_call(self, name, *args):
        self._wait_slm()
        result = getattr(self._sdk.slm, name)(self._slm_number, *args)
        self._check_slm(result, name)

    def _set_laser(self, pwm):
        # Set before writing so cleanup retries even if the serial reply fails.
        if pwm:
            self._laser_on = True
        self._arduino.writeCommand(f"laser {int(pwm)} 0")
        self._laser_on = bool(pwm)

    def _set_camera_node(self, names, value, *, required=True):
        errors = []
        for name in names:
            try:
                node = getattr(self._camera, name)
                node.SetValue(value)
                return node
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if required:
            raise RuntimeError(f"Cannot configure camera {names}: {'; '.join(errors)}")
        return None

    def open(self):
        if self._opened:
            return self
        self._load_sdk()
        try:
            cfg = self._sdk.read_config(str(self.config.config_path))
            self._arduino = self._sdk.arduino_class(cfg)
            self._arduino.openPort()
            self._set_laser(0)
            # Single-plane manual display needs neither Arduino nor SLM triggers.
            self._arduino.writeCommand("s2m -c")
            self._arduino.writeCommand("m2s -c")

            self._slm_number = int(cfg.slm.slm_number)
            self._check_slm(self._sdk.slm.SLM_Ctrl_Open(self._slm_number), "SLM open")
            self._slm_open = True
            mode = ctypes.c_uint32(99)
            self._slm_call("SLM_Ctrl_ReadVI", ctypes.byref(mode))
            if mode.value != self._sdk.slm.MODE_MEMORY:
                self._slm_call("SLM_Ctrl_WriteVI", self._sdk.slm.MODE_MEMORY)
                self._slm_call("SLM_Ctrl_ReadVI", ctypes.byref(mode))
                if mode.value != self._sdk.slm.MODE_MEMORY:
                    raise RuntimeError("SLM did not enter memory display mode")
            self._slm_call("SLM_Ctrl_WriteTI", 0)
            self._slm_call("SLM_Ctrl_WriteTM", 0)
            self._slm_call("SLM_Ctrl_WriteGS", 0)
            self._slm_call("SLM_Ctrl_WriteMT", 1, 1)
            self._slm_call("SLM_Ctrl_WriteMR", 1, 1)
            self._slm_call("SLM_Ctrl_WriteMP", 1)

            pylon = self._sdk.pylon
            factory = pylon.TlFactory.GetInstance()
            if self.config.camera_serial is None:
                device = factory.CreateFirstDevice()
            else:
                info = pylon.DeviceInfo()
                info.SetSerialNumber(str(self.config.camera_serial))
                device = factory.CreateDevice(info)
            self._camera = pylon.InstantCamera(device)
            self._camera.Open()
            self._set_camera_node(("PixelFormat",), self.config.pixel_format)
            self._set_camera_node(("ExposureAuto",), "Off")
            self._set_camera_node(("GainAuto",), "Off")
            self._set_camera_node(("TriggerMode",), "Off")
            self._set_camera_node(("AcquisitionMode",), "Continuous")
            self._set_camera_node(("ExposureMode",), "Timed", required=False)
            if self.config.camera_gain is not None:
                self._set_camera_node(("Gain", "GainRaw"), float(self.config.camera_gain))
            actual_format = self._camera.PixelFormat.GetValue()
            if actual_format != self.config.pixel_format:
                raise RuntimeError(f"Camera returned unexpected format {actual_format}")
            info = self._camera.GetDeviceInfo()
            self.metadata.update(camera_model=info.GetModelName(), camera_serial=info.GetSerialNumber(),
                                 pixel_format=actual_format, laser_pwm=self.config.laser_pwm,
                                 exposure_unit="us", raw_value_range=[0, 4095])
            for label, names in (("camera_gain", ("Gain", "GainRaw")),
                                 ("camera_black_level", ("BlackLevel", "BlackLevelRaw"))):
                for name in names:
                    try:
                        self.metadata[label] = float(getattr(self._camera, name).GetValue())
                        break
                    except Exception:
                        pass
            self._opened = True
            return self
        except BaseException:
            self.close(suppress_errors=True)
            raise

    def upload_phase(self, slm_commands: np.ndarray):
        """Upload one already quantized, device-oriented phase mask to slot 1."""
        if not self._opened:
            raise RuntimeError("Hardware has not been opened")
        mask = prepare_slm_commands(slm_commands, self.config.slm_shape)
        if self._laser_on:
            self._set_laser(0)
        self._phase_uploaded = False
        # The reference controller explicitly frees the currently displayed
        # slot before overwriting it; otherwise some SLM firmware keeps old data.
        self._slm_call("SLM_Ctrl_WriteGS", 0)
        pointer = mask.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        self._slm_call("SLM_Ctrl_WriteMI", 1, mask.shape[1], mask.shape[0],
                       ctypes.c_uint(0), pointer)
        self._slm_call("SLM_Ctrl_WriteDS", 1)
        self._wait_slm()
        time.sleep(self.config.slm_settle_s)
        self._phase_uploaded = True

    def _grab_raw(self, timeout_ms):
        result = self._camera.RetrieveResult(timeout_ms, self._sdk.pylon.TimeoutHandling_ThrowException)
        try:
            if not result.GrabSucceeded():
                raise RuntimeError(f"Camera grab failed: {result.GetErrorCode()} "
                                   f"{result.GetErrorDescription()}")
            raw = result.Array.copy()
        finally:
            result.Release()
        if raw.dtype != np.uint16 or raw.shape != tuple(self.config.raw_shape):
            raise ValueError(f"Expected raw uint16 Bayer {self.config.raw_shape}, "
                             f"received {raw.dtype} {raw.shape}; check camera ROI/pixel format")
        if raw.max() > 4095:
            raise ValueError("BayerRG12 data must use low 12 bits (DN 0..4095)")
        return raw

    def capture_bracket(self, exposures_us) -> dict[int, np.ndarray]:
        if not self._opened or not self._phase_uploaded:
            raise RuntimeError("Open hardware and upload a phase before capture")
        exposures = list(exposures_us)
        if not exposures or any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in exposures):
            raise ValueError("HDR exposures must be positive integer microseconds")
        exposures = [int(x) for x in exposures]
        if len(set(exposures)) != len(exposures):
            raise ValueError("HDR exposure values must be unique")
        captures = {}
        actual_exposures = {}
        try:
            self._set_laser(self.config.laser_pwm)
            time.sleep(self.config.laser_settle_s)
            for exposure in exposures:
                # Restart acquisition, not the camera connection, so the old
                # exposure's queued/in-flight images cannot enter this bracket.
                if self._camera.IsGrabbing():
                    self._camera.StopGrabbing()
                node = self._set_camera_node(("ExposureTime", "ExposureTimeAbs"), float(exposure))
                actual = float(node.GetValue())
                if not np.isfinite(actual) or abs(actual - exposure) > max(1.0, exposure * 1e-3):
                    raise RuntimeError(f"Requested exposure {exposure} us, camera uses {actual} us")
                actual_exposures[str(exposure)] = actual
                self._camera.StartGrabbing(self._sdk.pylon.GrabStrategy_LatestImageOnly)
                timeout = max(self.config.timeout_ms, int(actual / 1000) + 2000)
                for _ in range(self.config.flush_frames):
                    self._grab_raw(timeout)
                captures[exposure] = self._grab_raw(timeout)
            self.metadata["actual_exposures_us"] = actual_exposures
            return captures
        finally:
            # Laser shutdown is attempted even if stopping acquisition fails.
            try:
                if self._camera.IsGrabbing():
                    self._camera.StopGrabbing()
            finally:
                self._set_laser(0)

    def close(self, *, suppress_errors=False):
        errors = []

        def attempt(action):
            try:
                action()
            except Exception as exc:
                errors.append(exc)

        if self._arduino is not None:
            attempt(lambda: self._set_laser(0))
        if self._camera is not None:
            attempt(lambda: self._camera.StopGrabbing() if self._camera.IsGrabbing() else None)
            attempt(self._camera.Close)
            self._camera = None
        if self._slm_open:
            attempt(lambda: self._slm_call("SLM_Ctrl_WriteGS", 0))
            attempt(lambda: self._check_slm(self._sdk.slm.SLM_Ctrl_Close(self._slm_number), "SLM close"))
            self._slm_open = False
        if self._arduino is not None:
            attempt(self._arduino.closePort)
            self._arduino = None
        self._opened = False
        self._phase_uploaded = False
        if errors:
            message = "Hardware cleanup error(s): " + "; ".join(str(e) for e in errors)
            if suppress_errors:
                warnings.warn(message, RuntimeWarning)
            else:
                raise RuntimeError(message) from errors[0]

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(suppress_errors=exc_type is not None)
        return False
