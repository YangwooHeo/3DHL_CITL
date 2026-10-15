"""Frozen shared-optical proxy and intensity contracts for live CITL.

The optical implementation is imported from ``proxy_source_dir`` so a live run
uses the same simulator, target placement, and checkpoint migration as the
offline optimizer. This module never imports or starts a hardware controller.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi


def _level_max(value):
    if isinstance(value, bool) or int(value) != value or not 1 <= value <= 32767:
        raise ValueError("phase_level_max must be an integer in [1, 32767]")
    return int(value)


def quantize_phase_ste(phase, level_max=1023):
    """Uploaded linear command's phase in forward; identity derivative backward.

    Rounding, wrap, and the 1023 denominator exactly match the offline SLM
    export. The proxy Jacobian is evaluated at this uploaded phase, rather than
    at an unquantized Adam parameter that was never displayed on the SLM.
    """
    level_max = _level_max(level_max)
    if phase.ndim != 2 or phase.is_complex() or not bool(torch.isfinite(phase).all()):
        raise ValueError("Phase must be a finite real 2D tensor")
    quantized = torch.round(phase.remainder(TWO_PI) * (level_max / TWO_PI)) * (TWO_PI / level_max)
    return quantized.detach() + (phase - phase.detach())


def phase_to_slm(phase, level_max=1023, transpose=True, flip_first_axis=True):
    """Internal [Nx, Ny] radians -> unsigned SLM file/display [height, width]."""
    level_max = _level_max(level_max)
    if phase.ndim != 2 or phase.is_complex() or not bool(torch.isfinite(phase).all()):
        raise ValueError("Phase must be a finite real 2D tensor")
    levels = torch.round(phase.detach().remainder(TWO_PI) * (level_max / TWO_PI))
    if flip_first_axis:
        levels = levels.flip(0)
    if transpose:
        levels = levels.T
    return np.ascontiguousarray(levels.cpu().numpy(), dtype=np.uint16)


def slm_to_phase(array, *, device="cpu", level_max=1023, transpose=True,
                 flip_first_axis=True):
    """Inverse command orientation; useful for auditing the displayed mask."""
    level_max = _level_max(level_max)
    array = np.asarray(array)
    if (array.ndim != 2 or np.iscomplexobj(array) or not np.isfinite(array).all()
            or (array < 0).any() or (array > level_max).any()):
        raise ValueError("SLM command must be finite 2D values in [0, level_max]")
    if not np.equal(array, np.round(array)).all():
        raise ValueError("SLM command must contain integer drive levels")
    if transpose:
        array = array.T
    if flip_first_axis:
        array = array[::-1, :]
    return torch.tensor(np.ascontiguousarray(array), device=device,
                        dtype=torch.float32) * (TWO_PI / level_max)


def native_center_crop(value, size):
    """Camera and quality masks share this parity-preserving native crop."""
    if value.ndim < 2 or min(value.shape[-2:]) < size:
        raise ValueError(f"Camera shape {tuple(value.shape)} cannot contain native {size}x{size} ROI")
    height, width = value.shape[-2:]
    if (height - size) % 2 or (width - size) % 2:
        raise ValueError("Native camera crop must preserve pixel-center parity; resizing is forbidden")
    y, x = (height - size) // 2, (width - size) // 2
    return value[..., y:y + size, x:x + size]


def _single_image(value, *, device=None, dtype=torch.float32):
    if isinstance(value, torch.Tensor):
        result = value.to(device=device, dtype=dtype)
    else:
        result = torch.as_tensor(np.ascontiguousarray(value), device=device, dtype=dtype)
    if result.ndim == 2:
        result = result[None, None]
    if result.ndim != 4 or result.shape[:2] != (1, 1):
        raise ValueError(f"Expected one HW or 1x1xHxW image, got {tuple(result.shape)}")
    return result


def make_camera_loss(mode="shape", roi_fraction=1.0):
    """Masked data loss in checkpoint camera units, without display scaling.

    Shape loss matches the offline optimizer on all-valid images. With quality
    weights, both brightness means and the final reduction use only valid
    samples. Spatial training losses (SSIM/FFT) are deliberately not exposed:
    filling invalid pixels would change those losses and their gradients.
    """
    if mode not in ("mse", "shape"):
        raise ValueError("Live masked loss must be 'mse' or 'shape'")
    if not math.isfinite(roi_fraction) or not 0 < roi_fraction <= 1:
        raise ValueError("loss_roi_fraction must be in (0, 1]")

    def loss(prediction, target, mask=None):
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[:2] != (1, 1):
            raise ValueError("Prediction and target must have the same 1x1xHxW shape")
        weights = (torch.ones_like(prediction) if mask is None else
                   _single_image(mask, device=prediction.device, dtype=prediction.dtype))
        if weights.shape != prediction.shape or not bool(torch.isfinite(weights).all()):
            raise ValueError("Quality weights must be finite and match the camera ROI")
        if bool((weights < 0).any()) or bool((weights > 1).any()):
            raise ValueError("Quality weights must lie in [0, 1]")
        height, width = prediction.shape[-2:]
        rh, rw = max(1, round(height * roi_fraction)), max(1, round(width * roi_fraction))
        y, x = (height - rh) // 2, (width - rw) // 2
        prediction = prediction[..., y:y + rh, x:x + rw]
        target = target[..., y:y + rh, x:x + rw]
        weights = weights[..., y:y + rh, x:x + rw]
        positive = weights > 0
        total_weight = weights.sum()
        if not bool(total_weight > 0):
            raise ValueError("No valid camera pixels remain inside the loss ROI")
        if not bool(torch.isfinite(prediction[positive]).all()) or not bool(torch.isfinite(target[positive]).all()):
            raise ValueError("Nonfinite intensity at a valid camera pixel")
        if bool((prediction[positive] < 0).any()) or bool((target[positive] < 0).any()):
            raise ValueError("Camera and target intensities must be nonnegative")
        # torch.where also prevents NaN in an invalid pixel from poisoning a
        # reduction or an otherwise zero upstream derivative.
        p = torch.where(positive, prediction, torch.zeros_like(prediction))
        t = torch.where(positive, target, torch.zeros_like(target))
        if mode == "shape":
            pmean = (p * weights).sum() / total_weight
            tmean = (t * weights).sum() / total_weight
            if not bool(tmean > 0):
                raise ValueError("Shape loss needs positive target signal in valid pixels")
            p = p / pmean.clamp_min(1e-8)
            t = t / tmean.clamp_min(1e-8)
            errors = F.smooth_l1_loss(p, t, reduction="none")
        else:
            errors = (p - t).square()
        return (errors * weights).sum() / total_weight

    return loss


def measured_proxy_gradient(prediction, measured, target, loss_fn, mask=None):
    """Return measured loss and dL/df to backpropagate through ``prediction``.

    Usage: ``loss, upstream = measured_proxy_gradient(...); prediction.backward(upstream)``.
    Measurement is an independent leaf; no surrogate forward value or proxy
    residual participates in dL/df.
    """
    if measured.shape != prediction.shape or measured.device != prediction.device:
        raise ValueError("Measured image and proxy prediction must share shape/device")
    camera_leaf = measured.detach().clone().requires_grad_(True)
    value = loss_fn(camera_leaf, target, mask)
    upstream, = torch.autograd.grad(value, camera_leaf)
    if not bool(torch.isfinite(value)) or not bool(torch.isfinite(upstream).all()):
        raise FloatingPointError("Nonfinite measured loss or dL/df")
    return value.detach(), upstream.detach()


def _load_optical_modules(source_dir):
    source_dir = Path(source_dir).expanduser().resolve()
    required = ("shared_optical_phase_optimizer.py", "shared_optical_phase_system.py",
                "train_fno_axicon.py", "axicon_simulator.py")
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise ValueError(f"proxy_source_dir is missing {missing}: {source_dir}")
    # A stale mbvam from this checkout would produce a plausible but different
    # optical Jacobian. Require one coherent source tree per Python process.
    names = ("shared_optical_phase_optimizer", "shared_optical_phase_system",
             "train_fno_axicon", "axicon_simulator", "axicon_optical", "fno_losses", "mbvam")
    for name, module in tuple(sys.modules.items()):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in names):
            location = getattr(module, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(source_dir):
                raise RuntimeError(f"Optical module {name} already loaded from {location}; "
                                   "run CITL in a fresh Python process with the intended proxy_source_dir")
    sys.path.insert(0, str(source_dir))
    importlib.invalidate_caches()
    optimizer = importlib.import_module("shared_optical_phase_optimizer")
    system = importlib.import_module("shared_optical_phase_system")
    return source_dir, optimizer, system


class ProxyAdapter:
    """One frozen CSSNet checkpoint, one target, one absolute physical z plane."""

    @classmethod
    def from_args(cls, args):
        if isinstance(args, dict):
            args = SimpleNamespace(**args)
        self = cls()
        self.source_dir, self._optimizer, optical = _load_optical_modules(args.proxy_source_dir)
        self.device = torch.device(getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.level_max = _level_max(getattr(args, "phase_level_max", 1023))
        self.transpose_phase = bool(getattr(args, "transpose_phase", True))
        self.flip_phase_first_axis = bool(getattr(args, "flip_phase_first_axis", True))
        self.model, self.cfg = self._optimizer.load_proxy(Path(args.checkpoint), self.device)
        self.model_size = int(self.cfg["model_size"])
        self.camera_scale_mode = self.cfg.get("camera_scale_mode", "global_percentile")
        if self.camera_scale_mode not in ("raw", "global_percentile"):
            raise ValueError("Live CITL requires fixed raw/global_percentile camera scaling; sample_norm is unsupported")
        self.camera_scale = 1.0
        if self.camera_scale_mode == "global_percentile":
            scale = self.cfg.get("camera_scale_actual")
            if scale is None:
                scale = self.cfg.get("camera_scale")
            if scale is None or not math.isfinite(float(scale)) or float(scale) <= 0:
                raise ValueError("Checkpoint needs saved positive camera_scale_actual or camera_scale")
            self.camera_scale = float(scale)
        self.camera_black_level = float(self.cfg.get("camera_black_level", 0.0))
        if not math.isfinite(self.camera_black_level):
            raise ValueError("Checkpoint camera_black_level must be finite")
        z_mm = float(args.z_mm)
        if not math.isfinite(z_mm) or z_mm <= 0:
            raise ValueError("z_mm must be a positive absolute physical distance")
        config_path = getattr(args, "system_config", None)
        overrides = {} if config_path is None else json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("system_config must contain a JSON object")
        required_size = int(self.cfg.get("optical_input_size", 1024))
        if overrides.get("roi_size", required_size) != required_size:
            raise ValueError("system roi_size must match checkpoint optical_input_size")
        overrides["roi_size"] = required_size
        self.system = optical.build_system(z_mm * 1e-3, self.device, overrides,
                                          preset=getattr(args, "system_preset", "legacy_training"))
        shape = (self.system.beam_config.Nx, self.system.beam_config.Ny)
        self.initial_phase = self._optimizer.load_phase(
            Path(args.phase), shape, self.device, getattr(args, "phase_units", "slm"),
            self.level_max, self.transpose_phase, self.flip_phase_first_axis)
        self.target, target_report = optical.prepare_target(
            Path(args.target), self.model_size, self.system.beam_config,
            self.system.settings["upsample_factor"],
            mode=getattr(args, "target_mode", "legacy_optimizer"),
            target_pixel_size_m=float(getattr(args, "target_pixel_size_um", 2.0)) * 1e-6,
            transpose=bool(getattr(args, "transpose_target", False)),
            flip_ud=bool(getattr(args, "flip_target_ud", False)),
            flip_lr=bool(getattr(args, "flip_target_lr", True)),
            legacy_optimizer_roi_size=int(getattr(args, "legacy_optimizer_roi_size", 1024)))
        target_peak = getattr(args, "target_peak", None)
        if target_peak is not None:
            if not math.isfinite(float(target_peak)) or float(target_peak) <= 0:
                raise ValueError("target_peak must be positive in checkpoint camera units")
            self.target = self.target / self.target.amax().clamp_min(1e-8) * float(target_peak)
        roi = getattr(args, "loss_roi_fraction", None)
        self.loss_roi_fraction = float(self.cfg.get("loss_roi_fraction", 1.0) if roi is None else roi)
        self.loss_mode = getattr(args, "loss", "shape")
        self.loss_fn = make_camera_loss(self.loss_mode, self.loss_roi_fraction)
        reference_args = SimpleNamespace(
            reference_check=getattr(args, "reference_check", "auto"),
            reference_field=(None if getattr(args, "reference_field", None) is None else Path(args.reference_field)),
            phase=Path(args.phase), z_mm=z_mm,
            z_reference_mm=getattr(args, "z_reference_mm", 6.0))
        if reference_args.reference_check not in ("off", "auto", "required"):
            raise ValueError("reference_check must be off, auto, or required")
        self.reference_path = self._optimizer.find_reference_field(reference_args, self.cfg)
        self._reference_checked = False
        self.metadata = {
            "proxy_source_dir": str(self.source_dir), "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_cfg": self.cfg, "system": self.system.settings,
            "target": {**target_report, "effective_peak": float(self.target.max()),
                       "effective_mean": float(self.target.mean())},
            "camera": {"scale_mode": self.camera_scale_mode, "scale": self.camera_scale,
                       "black_level": self.camera_black_level, "model_size": self.model_size,
                       "formula": "max(aligned_HDR - camera_black_level, 0) / (saved_scale + 1e-8); raw mode omits division",
                       "native_crop_only": True},
            "phase": {"level_max": self.level_max, "transpose": self.transpose_phase,
                      "flip_first_axis": self.flip_phase_first_axis,
                      "response": "linear [0, level_max] -> [0, 2pi]; no extra drive LUT",
                      "proxy_evaluation": "quantized uploaded phase with identity straight-through derivative"},
            "physical_z_mm": z_mm, "training_z_reference_mm": reference_args.z_reference_mm,
            "loss": self.loss_mode, "loss_roi_fraction": self.loss_roi_fraction,
            "reference_field_check": None,
        }
        return self

    def phase_to_slm(self, phase):
        return phase_to_slm(phase, self.level_max, self.transpose_phase, self.flip_phase_first_axis)

    def predict(self, phase):
        field = self.system.forward(quantize_phase_ste(phase, self.level_max))
        if self.reference_path is not None and not self._reference_checked:
            self.metadata["reference_field_check"] = self._optimizer.compare_reference_field(
                field, self.reference_path, self.cfg)
            self._reference_checked = True
        prediction = self._optimizer.camera_from_complex(self.model, field, self.cfg)
        if prediction.shape != self.target.shape or not bool(torch.isfinite(prediction).all()):
            raise ValueError(f"Proxy returned invalid prediction shape/data: {tuple(prediction.shape)}")
        return prediction

    def camera_mask(self, valid_mask):
        mask = native_center_crop(_single_image(valid_mask, device=self.device), self.model_size)
        if not bool(torch.isfinite(mask).all()) or bool((mask < 0).any()) or bool((mask > 1).any()):
            raise ValueError("Camera quality mask must contain finite weights in [0, 1]")
        return mask

    mask_tensor = camera_mask

    def camera_tensor(self, aligned_image, valid_mask=None):
        """Apply the trainer's exact black level and fixed camera scale once."""
        camera = native_center_crop(_single_image(aligned_image, device=self.device), self.model_size)
        if valid_mask is not None:
            mask = self.camera_mask(valid_mask)
            if mask.shape != camera.shape:
                raise ValueError("Camera and mask crop shapes differ")
            camera = torch.where(mask > 0, camera, torch.zeros_like(camera))
        if not bool(torch.isfinite(camera).all()):
            raise ValueError("Aligned camera has nonfinite pixels; supply a validity mask or repair capture")
        camera = (camera - self.camera_black_level).clamp_min(0.0)
        if self.camera_scale_mode == "global_percentile":
            camera = camera / (self.camera_scale + 1e-8)
        return camera

    def loss(self, measured, mask=None):
        return self.loss_fn(measured, self.target, mask)
