# -*- coding: utf-8 -*-
"""
CITL-style phase update for an SLM + axicon system using a trained FNO proxy.

This script is intentionally single-target and single-z-plane.  The optimized
phase remains in the differentiable graph through:

    phase -> axicon propagation field -> FNO camera proxy -> target loss

and, when the FNO was trained with SLM phase conditioning:

    phase -> FNO phase-conditioning channels -> FNO camera proxy -> target loss

The FNO parameters are frozen; only the SLM phase mask is updated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from scipy.ndimage import zoom

from axicon_simulator import build_axicon_transfer_function, load_fno_proxy
from mbvam.Beam.holobeam import HoloBeam
from mbvam.Beam.holobeamconfig import HoloBeamConfig
from train_fno_axicon import (
    apply_loss_roi,
    log_display_tensor,
    predict_camera,
    resize_batch,
    simple_ssim_loss,
    target_mean_normalized_smooth_l1_loss,
)


TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# User-editable run configuration
# ---------------------------------------------------------------------------
#
# Paste long absolute paths here and run this file directly.  CLI arguments are
# still supported and override these defaults when provided.
SCRIPT_CONFIG = {
    "target": None,  # Path(r"C:\...\target.png")
    "phase": None,  # Path(r"C:\...\initial_phase.npy")
    "fno": None,  # Path(r"C:\...\best.pt")
    "output_dir": Path("citl_axicon_fno_output"),

    # Optimization
    "iters": 1,
    "lr": 0.03,
    "optimizer": "adam",
    "loss_mode": "mixed_visual",
    "w_log_display_mse": 0.0,
    "w_log_display_ssim": 1.0,
    "w_mean_norm_mse": 1.0,
    "w_mean_norm_smooth_l1": 0.0,
    "phase_tv_weight": 0.0,
    "grad_clip": None,
    "print_every": 1,

    # Axicon system. These mirror axicon_simulator.py defaults.
    "z_target": 0.01149,
    "roi_size": 1024,
    "upsample_factor": 20,
    "axicon_grating_pitch": 1.396e-6,
    "propagation_medium_index": 1.471,
    "axicon_angle_in_medium": False,
    "margin_factor": 5000,
    "show_transfer_function_plot": False,

    # Beam config
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wavelength": 0.473e-6,
    "focal_slm": 0.016,
    "ps_slm_physical": 8e-6 * 0.8,
    "nx_physical": 1600,
    "ny_physical": 1200,
    "binning_factor": 1,
    "axis_angle": [1.0, 0.0, 0.0],
    "z_plane_sampling_rate": 0.5,
    "amplitude_profile_type": "gaussian",
    "gaussian_beam_waist": 0.00638708 * 0.8,

    # Orientation and target handling
    "phase_units": "slm",
    "phase_level_max": 1023.0,
    "transpose_phase": True,
    "flip_phase_first_axis": True,
    "transpose_output_field": True,
    # "auto": direct for .npy / already-grid-sized targets, phase_optimization for raw PNG targets.
    "target_preprocess_mode": "auto",
    "target_original_pixel_size": 2e-6,
    "phase_optimization_roi_size": 1600,
    "phase_optimization_crop_to_current_roi": True,
    "phase_optimization_internal_transpose": False,
    "transpose_target": False,
    "flip_target_ud": False,
    "flip_target_lr": False,
    "loss_roi_fraction": None,
}


LOWER_IS_BETTER = {
    "loss",
    "raw_mse",
    "log_display_mse",
    "max_norm_mse",
    "mean_norm_mse",
    "mean_norm_smooth_l1",
    "ssim_loss",
}


def safe_stem(path: Path) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in path.stem)


def parse_axis_angle(text: str) -> list[float]:
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(values) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "axis angle must be comma-separated with 3 or 4 values, e.g. 1,0,0"
        )
    return values


def build_beam_config_from_args(args: argparse.Namespace) -> HoloBeamConfig:
    beam_config = HoloBeamConfig()
    beam_config.lambda_ = float(args.wavelength)
    beam_config.focal_SLM = float(args.focal_slm)
    beam_config.psSLM_physical = float(args.ps_slm_physical)
    beam_config.binningFactor = int(args.binning_factor)
    beam_config.Nx_physical = int(args.nx_physical)
    beam_config.Ny_physical = int(args.ny_physical)
    beam_config.axis_angle = args.axis_angle
    beam_config.z_plane_sampling_rate = float(args.z_plane_sampling_rate)
    beam_config.amplitude_profile_type = args.amplitude_profile_type
    beam_config.gaussian_beam_waist = float(args.gaussian_beam_waist)
    beam_config.device = torch.device(args.device)
    return beam_config


def load_target_array(target_path: Path) -> np.ndarray:
    target_path = Path(target_path)
    if target_path.suffix.lower() == ".npy":
        arr = np.load(target_path)
        if np.iscomplexobj(arr):
            arr = np.abs(arr) ** 2
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            if arr.shape[0] in (2, 3):
                arr = arr[0]
            elif arr.shape[-1] in (2, 3):
                arr = arr[..., 0]
    else:
        arr = np.asarray(Image.open(target_path).convert("L"), dtype=np.float32) / 255.0

    if arr.ndim != 2:
        raise ValueError(f"{target_path} must resolve to a 2D target; got shape {arr.shape}")
    return np.nan_to_num(np.asarray(arr, dtype=np.float32))


def center_crop_or_pad_tensor(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    if height > target_h:
        y0 = (height - target_h) // 2
        x = x[..., y0:y0 + target_h, :]
        height = target_h
    if width > target_w:
        x0 = (width - target_w) // 2
        x = x[..., :, x0:x0 + target_w]
        width = target_w

    pad_top = max(0, (target_h - height) // 2)
    pad_bottom = max(0, target_h - height - pad_top)
    pad_left = max(0, (target_w - width) // 2)
    pad_right = max(0, target_w - width - pad_left)
    if pad_top or pad_bottom or pad_left or pad_right:
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    return x


def center_crop_or_pad_numpy(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    height, width = arr.shape
    out = np.zeros((target_h, target_w), dtype=arr.dtype)
    y0_dst = max(0, (target_h - height) // 2)
    x0_dst = max(0, (target_w - width) // 2)
    y0_src = max(0, (height - target_h) // 2)
    x0_src = max(0, (width - target_w) // 2)
    copy_h = min(height - y0_src, target_h - y0_dst)
    copy_w = min(width - x0_src, target_w - x0_dst)
    if copy_h > 0 and copy_w > 0:
        out[y0_dst:y0_dst + copy_h, x0_dst:x0_dst + copy_w] = arr[
            y0_src:y0_src + copy_h,
            x0_src:x0_src + copy_w,
        ]
    return out


def phase_optimization_target_array(
    arr: np.ndarray,
    beam_config: HoloBeamConfig,
    original_pixel_size: float = 2e-6,
    internal_transpose: bool = False,
) -> np.ndarray:
    """
    Match the physical target scaling used for GS phase optimization.

    Phase_optimization_axicon_adam_3D.py returns final_image.T because the GS
    optimizer works in HoloBeam's internal (x, y) array order.  The FNO/CITL
    camera path normally has transpose_output_field=True, so its prediction is
    already in image/display (y, x) order.  For target loss against that FNO
    camera output, keep internal_transpose=False.
    """
    desired_pixel_size = (beam_config.abbe_res_x, beam_config.abbe_res_y)
    rescaled = zoom(
        arr,
        (
            float(original_pixel_size) / float(desired_pixel_size[1]),
            float(original_pixel_size) / float(desired_pixel_size[0]),
        ),
        order=1,
    )
    final_image = center_crop_or_pad_numpy(
        rescaled.astype(np.float32, copy=False),
        target_h=beam_config.Ny,
        target_w=beam_config.Nx,
    )
    if internal_transpose:
        final_image = final_image.T
    return final_image.astype(np.float32, copy=False)


def target_preprocess_report(
    source_shape: tuple[int, int],
    beam_config: HoloBeamConfig,
    original_pixel_size: float,
    phase_optimization_roi_size: int,
    current_roi_size: int,
    target_size: int,
) -> dict[str, float]:
    source_h, source_w = source_shape
    target_resize = float(target_size) / float(current_roi_size)
    source_horizontal_scale = (
        float(original_pixel_size)
        / float(beam_config.abbe_res_x)
        * float(phase_optimization_roi_size)
        / float(beam_config.Nx)
        * target_resize
    )
    source_vertical_scale = (
        float(original_pixel_size)
        / float(beam_config.abbe_res_y)
        * float(phase_optimization_roi_size)
        / float(beam_config.Ny)
        * target_resize
    )
    direct_horizontal_scale = float(target_size) / float(source_w)
    direct_vertical_scale = float(target_size) / float(source_h)
    return {
        "abbe_res_x_um": float(beam_config.abbe_res_x) * 1e6,
        "abbe_res_y_um": float(beam_config.abbe_res_y) * 1e6,
        "source_horizontal_scale_px_per_source_px": source_horizontal_scale,
        "source_vertical_scale_px_per_source_px": source_vertical_scale,
        "direct_horizontal_scale_px_per_source_px": direct_horizontal_scale,
        "direct_vertical_scale_px_per_source_px": direct_vertical_scale,
        "relative_horizontal_magnification_vs_direct_resize": (
            source_horizontal_scale / max(direct_horizontal_scale, 1e-12)
        ),
        "relative_vertical_magnification_vs_direct_resize": (
            source_vertical_scale / max(direct_vertical_scale, 1e-12)
        ),
    }


def resolve_target_preprocess_mode(
    target_path: Path,
    source_shape: tuple[int, int],
    requested_mode: str,
    target_size: int,
    current_roi_size: int | None,
) -> str:
    if requested_mode in {"direct", "phase_optimization"}:
        return requested_mode
    if requested_mode != "auto":
        raise ValueError(
            "target_preprocess_mode must be 'auto', 'direct', or 'phase_optimization'; "
            f"got {requested_mode!r}"
        )

    source_h, source_w = source_shape
    grid_sizes = {int(target_size)}
    if current_roi_size is not None:
        grid_sizes.add(int(current_roi_size))

    # NPY targets are usually already simulated/camera-grid data. Do not apply
    # the raw-PNG physical magnification correction unless explicitly requested.
    if Path(target_path).suffix.lower() == ".npy":
        return "direct"
    if source_h in grid_sizes and source_w in grid_sizes:
        return "direct"
    return "phase_optimization"


def load_target_image(
    target_path: Path,
    target_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    beam_config: HoloBeamConfig | None = None,
    preprocess_mode: str = "direct",
    original_pixel_size: float = 2e-6,
    phase_optimization_roi_size: int = 1600,
    current_roi_size: int | None = None,
    crop_phase_optimization_to_current_roi: bool = True,
    phase_optimization_internal_transpose: bool = False,
    transpose: bool = False,
    flip_ud: bool = False,
    flip_lr: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    arr = load_target_array(target_path)

    if transpose:
        arr = arr.T
    if flip_ud:
        arr = arr[::-1, :]
    if flip_lr:
        arr = arr[:, ::-1]

    arr = np.ascontiguousarray(arr)
    arr = arr - float(arr.min())
    arr = arr / (float(arr.max()) + 1e-8)
    source_shape = arr.shape

    if preprocess_mode == "direct":
        target = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
        target = resize_batch(target, int(target_size), mode="bilinear").clamp(0.0, 1.0)
        report = {
            "requested_mode": preprocess_mode,
            "mode": "direct",
            "source_shape": list(source_shape),
            "direct_horizontal_scale_px_per_source_px": float(target_size) / float(source_shape[1]),
            "direct_vertical_scale_px_per_source_px": float(target_size) / float(source_shape[0]),
        }
        return target, report

    if current_roi_size is None:
        current_roi_size = int(target_size)
    requested_mode = preprocess_mode
    preprocess_mode = resolve_target_preprocess_mode(
        target_path=target_path,
        source_shape=source_shape,
        requested_mode=requested_mode,
        target_size=int(target_size),
        current_roi_size=int(current_roi_size),
    )

    if preprocess_mode == "direct":
        target = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
        target = resize_batch(target, int(target_size), mode="bilinear").clamp(0.0, 1.0)
        report = {
            "requested_mode": requested_mode,
            "mode": "direct",
            "source_shape": list(source_shape),
            "direct_horizontal_scale_px_per_source_px": float(target_size) / float(source_shape[1]),
            "direct_vertical_scale_px_per_source_px": float(target_size) / float(source_shape[0]),
            "auto_reason": "npy_or_already_grid_sized_target",
        }
        return target, report

    if preprocess_mode != "phase_optimization":
        raise RuntimeError(
            f"Unhandled resolved target preprocessing mode: {preprocess_mode!r}"
        )
    if beam_config is None:
        raise ValueError("beam_config is required when target_preprocess_mode='phase_optimization'.")

    arr = phase_optimization_target_array(
        arr,
        beam_config,
        original_pixel_size=float(original_pixel_size),
        internal_transpose=bool(phase_optimization_internal_transpose),
    )
    target = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
    target = resize_batch(target, int(phase_optimization_roi_size), mode="bilinear")
    if crop_phase_optimization_to_current_roi:
        target = center_crop_or_pad_tensor(target, int(current_roi_size), int(current_roi_size))
    target = resize_batch(target, int(target_size), mode="bilinear").clamp(0.0, 1.0)
    target = target / target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    report = target_preprocess_report(
        source_shape=source_shape,
        beam_config=beam_config,
        original_pixel_size=float(original_pixel_size),
        phase_optimization_roi_size=int(phase_optimization_roi_size),
        current_roi_size=int(current_roi_size),
        target_size=int(target_size),
    )
    report.update({
        "requested_mode": requested_mode,
        "mode": "phase_optimization",
        "source_shape": list(source_shape),
        "phase_optimization_roi_size": int(phase_optimization_roi_size),
        "current_roi_size": int(current_roi_size),
        "target_size": int(target_size),
        "crop_phase_optimization_to_current_roi": bool(crop_phase_optimization_to_current_roi),
        "phase_optimization_internal_transpose": bool(phase_optimization_internal_transpose),
    })
    return target, report


def _image_phase_to_levels(path: Path, phase_level_max: float) -> np.ndarray:
    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float32)
    return arr * (float(phase_level_max) / 255.0)


def load_initial_phase(
    phase_path: Path,
    beam_config: HoloBeamConfig,
    phase_units: str = "slm",
    phase_level_max: float = 1023.0,
    transpose_phase: bool = True,
    flip_phase_first_axis: bool = True,
) -> torch.Tensor:
    phase_path = Path(phase_path)
    if phase_path.suffix.lower() == ".npy":
        phase_np = np.load(phase_path)
    elif phase_path.suffix.lower() in {".png", ".tif", ".tiff", ".bmp"}:
        phase_np = _image_phase_to_levels(phase_path, phase_level_max)
    else:
        raise ValueError(f"Unsupported phase file type: {phase_path.suffix}")

    phase_np = np.squeeze(phase_np)
    if phase_np.ndim != 2:
        raise ValueError(f"{phase_path} must contain a 2D phase mask; got {phase_np.shape}")

    if transpose_phase:
        phase_np = phase_np.T
    if flip_phase_first_axis:
        phase_np = phase_np[::-1, :]

    phase_np = np.ascontiguousarray(phase_np, dtype=np.float32)
    expected_shape = (beam_config.Nx, beam_config.Ny)
    if phase_np.shape != expected_shape:
        raise ValueError(
            f"{phase_path} has shape {phase_np.shape} after orientation correction, "
            f"but HoloBeam expects {expected_shape}. Check --transpose-phase, "
            f"--flip-phase-first-axis, or the beam size arguments."
        )

    if phase_units == "slm":
        phase_rad = phase_np * (TWO_PI / float(phase_level_max))
    elif phase_units == "rad":
        phase_rad = phase_np
    else:
        raise ValueError(f"Unsupported phase_units={phase_units!r}")

    return torch.from_numpy(phase_rad).to(device=beam_config.device, dtype=beam_config.fdtype)


def phase_to_slm_file_array(
    phase_rad: torch.Tensor,
    phase_level_max: float = 1023.0,
    transpose_phase: bool = True,
    flip_phase_first_axis: bool = True,
) -> np.ndarray:
    phase = (phase_rad.detach() % TWO_PI).cpu().numpy()
    levels = np.round(phase * (float(phase_level_max) / TWO_PI))
    levels = np.remainder(levels, float(phase_level_max) + 1.0)

    if flip_phase_first_axis:
        levels = levels[::-1, :]
    if transpose_phase:
        levels = levels.T

    return np.ascontiguousarray(levels.astype(np.int16))


def phase_to_fno_condition(phase_rad: torch.Tensor, fno_cfg: dict) -> torch.Tensor:
    phase = phase_rad.transpose(0, 1)
    if fno_cfg.get("phase_flip", True):
        phase = torch.flip(phase, dims=(1,))
    return phase.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)


def scale_amp_for_fno(amp: torch.Tensor, fno_cfg: dict, eps: float = 1e-8) -> torch.Tensor:
    mode = fno_cfg.get("field_scale_mode", "global_percentile")
    if mode == "raw":
        return amp
    if mode == "global_percentile":
        scale = fno_cfg.get("field_amp_scale_actual", None) or fno_cfg.get("field_amp_scale", None)
        if scale is None:
            raise ValueError(
                "FNO cfg is missing field_amp_scale_actual/field_amp_scale. "
                "Use a checkpoint saved by train_fno_axicon.py after dataset creation."
            )
        return amp / (float(scale) + eps)
    if mode == "sample_norm":
        percentile = float(fno_cfg.get("field_amp_percentile", 99.9))
        scale = torch.quantile(amp.detach().reshape(-1), percentile / 100.0)
        return torch.clamp(amp / (scale + eps), 0.0, 1.0)
    raise ValueError(f"Unsupported field_scale_mode={mode!r}")


def phase_to_fno_field(
    phase_rad: torch.Tensor,
    beam: HoloBeam,
    cone_angle: float,
    upsample_factor: int,
    h_asm,
    roi_size: int,
    propagation_medium_index: float,
    axicon_angle_in_medium: bool,
    axicon_transverse_frequency: float,
    fno_cfg: dict,
    transpose_output_field: bool = True,
) -> torch.Tensor:
    field = beam.propagateToVolume_Axicon2(
        axicon_angle=cone_angle,
        upsample_factor=upsample_factor,
        phase_mask=phase_rad,
        H_asm=h_asm,
        roi_size=roi_size,
        convert_to_intensity=False,
        apply_spatial_filter=True,
        n_medium=propagation_medium_index,
        axicon_angle_in_medium=axicon_angle_in_medium,
        axicon_transverse_frequency=axicon_transverse_frequency,
    )
    field = torch.squeeze(field)
    if field.ndim != 2:
        raise ValueError(f"Expected a single z-plane field; got {field.shape}")
    if transpose_output_field:
        field = field.transpose(0, 1)

    amp = torch.abs(field).to(torch.float32)
    amp_scaled = scale_amp_for_fno(amp, fno_cfg)
    phase_norm = amp.clamp_min(1e-8)
    cos_p = (field.real.to(torch.float32) / phase_norm).clamp(-1.0, 1.0)
    sin_p = (field.imag.to(torch.float32) / phase_norm).clamp(-1.0, 1.0)
    fno_field = torch.stack([amp_scaled, cos_p, sin_p], dim=0).unsqueeze(0)
    return resize_batch(fno_field, int(fno_cfg["model_size"]), mode="bilinear")


def wrapped_phase_tv(phase_rad: torch.Tensor) -> torch.Tensor:
    phase_complex = torch.stack([torch.cos(phase_rad), torch.sin(phase_rad)], dim=0)
    dx = phase_complex[:, 1:, :] - phase_complex[:, :-1, :]
    dy = phase_complex[:, :, 1:] - phase_complex[:, :, :-1]
    return dx.abs().mean() + dy.abs().mean()


def target_mean_normalized_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    scale = target.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return F.mse_loss(pred / scale, target / scale)


def max_normalized_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = pred / pred.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    target_n = target / target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return F.mse_loss(pred_n, target_n)


def camera_loss_components(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    pred_d = log_display_tensor(pred)
    target_d = log_display_tensor(target)
    ssim_loss = simple_ssim_loss(pred_d, target_d)
    return {
        "raw_mse": F.mse_loss(pred, target),
        "log_display_mse": F.mse_loss(pred_d, target_d),
        "log_display_ssim": 1.0 - ssim_loss,
        "ssim_loss": ssim_loss,
        "max_norm_mse": max_normalized_mse_loss(pred, target),
        "mean_norm_mse": target_mean_normalized_mse_loss(pred, target),
        "mean_norm_smooth_l1": target_mean_normalized_smooth_l1_loss(pred, target),
    }


def detach_components(components: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in components.items()}


def camera_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: str = "display_mse",
    roi_fraction: float = 1.0,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_roi, target_roi = apply_loss_roi(pred, target, roi_fraction)
    components = camera_loss_components(pred_roi, target_roi)

    if mode == "display_mse":
        return components["log_display_mse"], detach_components(components)
    if mode == "norm_mse":
        return components["max_norm_mse"], detach_components(components)
    if mode == "raw_mse":
        return components["raw_mse"], detach_components(components)
    if mode == "mixed_visual":
        weights = weights or {}
        total = pred.new_tensor(0.0)
        total = total + float(weights.get("w_log_display_mse", 0.0)) * components["log_display_mse"]
        total = total + float(weights.get("w_log_display_ssim", 1.0)) * components["ssim_loss"]
        total = total + float(weights.get("w_mean_norm_mse", 1.0)) * components["mean_norm_mse"]
        total = total + float(weights.get("w_mean_norm_smooth_l1", 0.0)) * components["mean_norm_smooth_l1"]
        summary = detach_components(components)
        summary["loss"] = float(total.detach().cpu().item())
        return total, summary
    raise ValueError(f"Unknown loss mode: {mode}")


def compute_camera_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    roi_fraction: float = 1.0,
    loss_mode: str = "mixed_visual",
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    with torch.no_grad():
        loss, components = camera_loss(
            pred,
            target,
            mode=loss_mode,
            roi_fraction=roi_fraction,
            weights=weights,
        )
        components["loss"] = float(loss.detach().cpu().item())
    return components


def compare_metric_dicts(initial: dict[str, float], updated: dict[str, float]) -> dict[str, dict[str, float]]:
    comparison = {}
    for key in sorted(set(initial) & set(updated)):
        init_v = float(initial[key])
        upd_v = float(updated[key])
        delta = upd_v - init_v
        improvement = init_v - upd_v if key in LOWER_IS_BETTER else upd_v - init_v
        denom = max(abs(init_v), 1e-8)
        comparison[key] = {
            "initial": init_v,
            "updated": upd_v,
            "delta_updated_minus_initial": delta,
            "improvement": improvement,
            "improvement_percent": 100.0 * improvement / denom,
        }
    return comparison


def build_axicon_system(
    beam_config: HoloBeamConfig,
    z_target: float,
    upsample_factor: int,
    propagation_medium_index: float,
    axicon_grating_pitch: float,
    axicon_angle_in_medium: bool,
    margin_factor: float,
    show_transfer_function_plot: bool = False,
):
    axicon_transverse_frequency = 1.0 / float(axicon_grating_pitch)
    axicon_na_air_equiv = beam_config.lambda_ * axicon_transverse_frequency
    if axicon_na_air_equiv >= 1.0:
        raise ValueError("Axicon grating pitch gives an invalid free-space NA >= 1.")
    cone_angle = float(np.arcsin(axicon_na_air_equiv))
    cone_angle_medium = float(np.arcsin(axicon_na_air_equiv / propagation_medium_index))
    print(
        f"Axicon air-equivalent NA={axicon_na_air_equiv:.4f}, "
        f"theta_air={cone_angle:.4f} rad, "
        f"theta_medium={cone_angle_medium:.4f} rad at n={propagation_medium_index:.3f}"
    )
    print(f"Single physical z target: {z_target * 1000:.3f} mm")

    beam = HoloBeam(beam_config)
    beam.phase_mask_iter = torch.zeros(
        (beam_config.Nx, beam_config.Ny),
        device=beam_config.device,
        dtype=beam_config.fdtype,
    )
    beam.slm_amplitude_profile = beam.buildSLMAmplitudeProfile()
    beam.beam_mean_amplitude_iter = torch.tensor(
        1.0,
        device=beam_config.device,
        dtype=beam_config.fdtype,
    )

    z_eval_planes = torch.tensor(
        [float(z_target)],
        device=beam_config.device,
        dtype=beam_config.fdtype,
    )
    h_asm = build_axicon_transfer_function(
        beam,
        show_transfer_function_plot,
        upsample_factor=upsample_factor,
        z_query=z_eval_planes,
        n_medium=propagation_medium_index,
        axicon_angle=cone_angle,
        axicon_angle_in_medium=axicon_angle_in_medium,
        axicon_transverse_frequency=axicon_transverse_frequency,
        margin_factor=margin_factor,
    )
    return beam, cone_angle, axicon_transverse_frequency, h_asm


class AxiconFNOCITLSolver:
    def __init__(
        self,
        beam: HoloBeam,
        fno_model: nn.Module,
        fno_cfg: dict,
        cone_angle: float,
        upsample_factor: int,
        h_asm,
        roi_size: int,
        propagation_medium_index: float,
        axicon_angle_in_medium: bool,
        axicon_transverse_frequency: float,
        transpose_output_field: bool = True,
    ):
        self.beam = beam
        self.fno_model = fno_model
        self.fno_cfg = fno_cfg
        self.cone_angle = cone_angle
        self.upsample_factor = upsample_factor
        self.h_asm = h_asm
        self.roi_size = roi_size
        self.propagation_medium_index = propagation_medium_index
        self.axicon_angle_in_medium = axicon_angle_in_medium
        self.axicon_transverse_frequency = axicon_transverse_frequency
        self.transpose_output_field = transpose_output_field

        for param in self.fno_model.parameters():
            param.requires_grad_(False)
        self.fno_model.eval()

    def forward(self, phase_rad: torch.Tensor):
        field = phase_to_fno_field(
            phase_rad,
            self.beam,
            self.cone_angle,
            self.upsample_factor,
            self.h_asm,
            self.roi_size,
            self.propagation_medium_index,
            self.axicon_angle_in_medium,
            self.axicon_transverse_frequency,
            self.fno_cfg,
            transpose_output_field=self.transpose_output_field,
        )
        phase_cond = phase_to_fno_condition(phase_rad, self.fno_cfg).to(field.device)
        pred, field_like = predict_camera(
            self.fno_model,
            field,
            phase_cond,
            self.fno_cfg,
            train_noise=False,
        )
        sim_intensity = field[:, 0:1].pow(2).clamp_min(0.0)
        return pred, sim_intensity, field, field_like

    def evaluate(self, phase_rad: torch.Tensor):
        with torch.no_grad():
            pred, sim_intensity, _, _ = self.forward(phase_rad.detach())
        return pred, sim_intensity

    def optimize(
        self,
        init_phase: torch.Tensor,
        target: torch.Tensor,
        iters: int = 1,
        lr: float = 0.03,
        optimizer_name: str = "adam",
        loss_mode: str = "display_mse",
        loss_weights: dict[str, float] | None = None,
        phase_tv_weight: float = 0.0,
        grad_clip: float | None = None,
        print_every: int = 1,
    ):
        phase_var = nn.Parameter(init_phase.detach().clone())
        if optimizer_name == "adam":
            optimizer = optim.Adam([phase_var], lr=lr)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD([phase_var], lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer={optimizer_name!r}")

        roi_fraction = float(self.fno_cfg.get("loss_roi_fraction", 1.0))
        best_loss = float("inf")
        best_phase = init_phase.detach().clone()
        loss_history: list[float] = []
        component_history: list[dict[str, float]] = []
        grad_last = None
        pred_last = None
        sim_last = None

        for idx in range(int(iters)):
            optimizer.zero_grad(set_to_none=True)
            pred, sim_intensity, _, _ = self.forward(phase_var)
            loss, components = camera_loss(
                pred,
                target,
                mode=loss_mode,
                roi_fraction=roi_fraction,
                weights=loss_weights,
            )
            if phase_tv_weight > 0:
                loss = loss + float(phase_tv_weight) * wrapped_phase_tv(phase_var)

            loss.backward()
            if grad_clip is not None and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_([phase_var], float(grad_clip))

            grad_last = phase_var.grad.detach().clone()
            optimizer.step()
            with torch.no_grad():
                phase_var.remainder_(TWO_PI)

            loss_value = float(loss.detach().cpu().item())
            components["loss"] = loss_value
            loss_history.append(loss_value)
            component_history.append(components)
            if loss_value < best_loss:
                best_loss = loss_value
                best_phase = phase_var.detach().clone()

            pred_last = pred.detach()
            sim_last = sim_intensity.detach()

            if print_every and ((idx + 1) % print_every == 0 or idx == 0 or idx + 1 == iters):
                grad_abs = grad_last.abs()
                print(
                    f"  [{idx + 1:4d}/{iters}] loss={loss_value:.6f} "
                    f"ssim={components['log_display_ssim']:.4f} "
                    f"mean_norm_mse={components['mean_norm_mse']:.4f} "
                    f"grad(mean={grad_last.mean().item():+.3e}, "
                    f"absmax={grad_abs.max().item():.3e})"
                )

        return {
            "best_phase": best_phase,
            "final_phase": phase_var.detach().clone(),
            "loss_history": np.asarray(loss_history, dtype=np.float32),
            "component_history": component_history,
            "last_grad": grad_last,
            "last_pred": pred_last,
            "last_sim_intensity": sim_last,
        }


def _display_scaled(arr: np.ndarray, percentile: float = 99.9) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32))
    scale = max(float(np.percentile(arr, percentile)), 1e-8)
    return np.clip(arr / scale, 0.0, 1.0)


def save_comparison_preview(
    target: torch.Tensor,
    initial_pred: torch.Tensor,
    updated_pred: torch.Tensor,
    initial_sim_intensity: torch.Tensor,
    updated_sim_intensity: torch.Tensor,
    init_phase: torch.Tensor,
    updated_phase: torch.Tensor,
    grad: torch.Tensor | None,
    loss_history: np.ndarray,
    metrics_comparison: dict[str, dict[str, float]],
    save_path: Path,
    roi_fraction: float = 1.0,
):
    target_view = apply_loss_roi(target, target, roi_fraction)[0][0, 0].detach().cpu().numpy()
    initial_pred_view = apply_loss_roi(initial_pred, initial_pred, roi_fraction)[0][0, 0].detach().cpu().numpy()
    updated_pred_view = apply_loss_roi(updated_pred, updated_pred, roi_fraction)[0][0, 0].detach().cpu().numpy()
    initial_sim_view = apply_loss_roi(initial_sim_intensity, initial_sim_intensity, roi_fraction)[0][0, 0].detach().cpu().numpy()
    updated_sim_view = apply_loss_roi(updated_sim_intensity, updated_sim_intensity, roi_fraction)[0][0, 0].detach().cpu().numpy()
    init_np = (init_phase.detach() % TWO_PI).cpu().numpy()
    updated_np = (updated_phase.detach() % TWO_PI).cpu().numpy()
    diff_np = np.angle(np.exp(1j * (updated_np - init_np))).astype(np.float32)
    grad_np = None if grad is None else grad.detach().cpu().numpy()

    pred_values = np.concatenate([initial_pred_view.reshape(-1), updated_pred_view.reshape(-1)])
    pred_scale = max(float(np.percentile(pred_values, 99.9)), 1e-8)
    sim_values = np.concatenate([initial_sim_view.reshape(-1), updated_sim_view.reshape(-1)])
    sim_scale = max(float(np.percentile(sim_values, 99.9)), 1e-8)
    pred_delta = updated_pred_view - initial_pred_view
    pred_delta_scale = max(float(np.percentile(np.abs(pred_delta), 99.0)), 1e-8)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.reshape(3, 4)
    axes[0, 0].imshow(target_view, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Target")
    axes[0, 1].imshow(np.clip(initial_pred_view / pred_scale, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Initial FNO camera")
    axes[0, 2].imshow(np.clip(updated_pred_view / pred_scale, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Updated FNO camera")
    axes[0, 3].imshow(pred_delta, cmap="seismic", vmin=-pred_delta_scale, vmax=pred_delta_scale)
    axes[0, 3].set_title("Updated - initial camera")

    axes[1, 0].imshow(np.clip(initial_sim_view / sim_scale, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Initial axicon field input")
    axes[1, 1].imshow(np.clip(updated_sim_view / sim_scale, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Updated axicon field input")
    axes[1, 2].imshow(init_np, cmap="hsv", vmin=0, vmax=TWO_PI)
    axes[1, 2].set_title("Initial phase")
    axes[1, 3].imshow(updated_np, cmap="hsv", vmin=0, vmax=TWO_PI)
    axes[1, 3].set_title("Updated phase")

    metric_names = ["loss", "raw_mse", "log_display_mse", "mean_norm_mse", "log_display_ssim"]
    x = np.arange(len(metric_names))
    initial_vals = [metrics_comparison[name]["initial"] for name in metric_names]
    updated_vals = [metrics_comparison[name]["updated"] for name in metric_names]
    axes[2, 0].plot(loss_history)
    axes[2, 0].set_title("Optimization loss")
    axes[2, 0].set_xlabel("Iteration")
    axes[2, 1].bar(x - 0.18, initial_vals, width=0.36, label="initial")
    axes[2, 1].bar(x + 0.18, updated_vals, width=0.36, label="updated")
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(metric_names, rotation=35, ha="right")
    axes[2, 1].set_title("Metrics")
    axes[2, 1].legend(fontsize=8)
    axes[2, 2].imshow(diff_np, cmap="twilight", vmin=-math.pi, vmax=math.pi)
    axes[2, 2].set_title("Wrapped phase update")
    if grad_np is not None:
        grad_scale = max(float(np.percentile(np.abs(grad_np), 99.0)), 1e-8)
        axes[2, 3].imshow(grad_np, cmap="seismic", vmin=-grad_scale, vmax=grad_scale)
    axes[2, 3].set_title("Last gradient")

    summary = (
        f"loss {metrics_comparison['loss']['initial']:.4g} -> "
        f"{metrics_comparison['loss']['updated']:.4g}; "
        f"SSIM {metrics_comparison['log_display_ssim']['initial']:.4f} -> "
        f"{metrics_comparison['log_display_ssim']['updated']:.4f}; "
        f"mean-norm MSE {metrics_comparison['mean_norm_mse']['initial']:.4g} -> "
        f"{metrics_comparison['mean_norm_mse']['updated']:.4g}"
    )

    for ax in axes.flat:
        if ax not in (axes[2, 0], axes[2, 1]):
            ax.axis("off")
    fig.suptitle(summary, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_metrics_report(
    json_path: Path,
    csv_path: Path,
    initial_metrics: dict[str, float],
    updated_metrics: dict[str, float],
    comparison: dict[str, dict[str, float]],
):
    payload = {
        "initial": initial_metrics,
        "updated": updated_metrics,
        "comparison": comparison,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "initial",
                "updated",
                "delta_updated_minus_initial",
                "improvement",
                "improvement_percent",
            ],
        )
        writer.writeheader()
        for metric, values in comparison.items():
            row = {"metric": metric}
            row.update(values)
            writer.writerow(row)


def write_run_config(path: Path, args: argparse.Namespace, fno_cfg: dict):
    serializable_args = vars(args).copy()
    serializable_args["target"] = str(serializable_args["target"])
    serializable_args["phase"] = str(serializable_args["phase"])
    serializable_args["fno"] = str(serializable_args["fno"])
    serializable_args["output_dir"] = str(serializable_args["output_dir"])
    serializable_args["axis_angle"] = list(serializable_args["axis_angle"])
    payload = {"args": serializable_args, "fno_cfg": fno_cfg}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-plane axicon + FNO proxy phase-mask update."
    )
    parser.add_argument("--target", type=Path, default=SCRIPT_CONFIG["target"], help="Target PNG or NPY.")
    parser.add_argument("--phase", type=Path, default=SCRIPT_CONFIG["phase"], help="Initial phase mask NPY/PNG.")
    parser.add_argument("--fno", type=Path, default=SCRIPT_CONFIG["fno"], help="Trained FNO checkpoint, e.g. best.pt.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_CONFIG["output_dir"])
    parser.add_argument("--iters", type=int, default=SCRIPT_CONFIG["iters"], help="Number of phase update iterations.")
    parser.add_argument("--lr", type=float, default=SCRIPT_CONFIG["lr"], help="Phase optimizer learning rate.")
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default=SCRIPT_CONFIG["optimizer"])
    parser.add_argument(
        "--loss-mode",
        choices=("display_mse", "norm_mse", "raw_mse", "mixed_visual"),
        default=SCRIPT_CONFIG["loss_mode"],
    )
    parser.add_argument("--w-log-display-mse", type=float, default=SCRIPT_CONFIG["w_log_display_mse"])
    parser.add_argument("--w-log-display-ssim", type=float, default=SCRIPT_CONFIG["w_log_display_ssim"])
    parser.add_argument("--w-mean-norm-mse", type=float, default=SCRIPT_CONFIG["w_mean_norm_mse"])
    parser.add_argument("--w-mean-norm-smooth-l1", type=float, default=SCRIPT_CONFIG["w_mean_norm_smooth_l1"])
    parser.add_argument("--phase-tv-weight", type=float, default=SCRIPT_CONFIG["phase_tv_weight"])
    parser.add_argument("--grad-clip", type=float, default=SCRIPT_CONFIG["grad_clip"])
    parser.add_argument("--print-every", type=int, default=SCRIPT_CONFIG["print_every"])
    parser.add_argument("--device", default=SCRIPT_CONFIG["device"])

    parser.add_argument("--z-target", type=float, default=SCRIPT_CONFIG["z_target"], help="Single target z plane [m].")
    parser.add_argument("--roi-size", type=int, default=SCRIPT_CONFIG["roi_size"])
    parser.add_argument("--upsample-factor", type=int, default=SCRIPT_CONFIG["upsample_factor"])
    parser.add_argument("--axicon-grating-pitch", type=float, default=SCRIPT_CONFIG["axicon_grating_pitch"])
    parser.add_argument("--propagation-medium-index", type=float, default=SCRIPT_CONFIG["propagation_medium_index"])
    parser.add_argument(
        "--axicon-angle-in-medium",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["axicon_angle_in_medium"],
    )
    parser.add_argument("--margin-factor", type=float, default=SCRIPT_CONFIG["margin_factor"])
    parser.add_argument(
        "--show-transfer-function-plot",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["show_transfer_function_plot"],
    )

    parser.add_argument("--wavelength", type=float, default=SCRIPT_CONFIG["wavelength"])
    parser.add_argument("--focal-slm", type=float, default=SCRIPT_CONFIG["focal_slm"])
    parser.add_argument("--ps-slm-physical", type=float, default=SCRIPT_CONFIG["ps_slm_physical"])
    parser.add_argument("--nx-physical", type=int, default=SCRIPT_CONFIG["nx_physical"])
    parser.add_argument("--ny-physical", type=int, default=SCRIPT_CONFIG["ny_physical"])
    parser.add_argument("--binning-factor", type=int, default=SCRIPT_CONFIG["binning_factor"])
    parser.add_argument("--axis-angle", type=parse_axis_angle, default=SCRIPT_CONFIG["axis_angle"])
    parser.add_argument("--z-plane-sampling-rate", type=float, default=SCRIPT_CONFIG["z_plane_sampling_rate"])
    parser.add_argument("--amplitude-profile-type", choices=("gaussian", "flat_top"), default=SCRIPT_CONFIG["amplitude_profile_type"])
    parser.add_argument("--gaussian-beam-waist", type=float, default=SCRIPT_CONFIG["gaussian_beam_waist"])

    parser.add_argument("--phase-units", choices=("slm", "rad"), default=SCRIPT_CONFIG["phase_units"])
    parser.add_argument("--phase-level-max", type=float, default=SCRIPT_CONFIG["phase_level_max"])
    parser.add_argument(
        "--transpose-phase",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["transpose_phase"],
        help="Match axicon_simulator.py phase loading orientation.",
    )
    parser.add_argument(
        "--flip-phase-first-axis",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["flip_phase_first_axis"],
        help="Match axicon_simulator.py phase loading orientation.",
    )
    parser.add_argument(
        "--transpose-output-field",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["transpose_output_field"],
        help="Match axicon_simulator.py FNO field orientation.",
    )
    parser.add_argument(
        "--target-preprocess-mode",
        choices=("auto", "direct", "phase_optimization"),
        default=SCRIPT_CONFIG["target_preprocess_mode"],
    )
    parser.add_argument(
        "--target-original-pixel-size",
        type=float,
        default=SCRIPT_CONFIG["target_original_pixel_size"],
        help="Physical pixel size of source target PNG used by phase optimization [m].",
    )
    parser.add_argument(
        "--phase-optimization-roi-size",
        type=int,
        default=SCRIPT_CONFIG["phase_optimization_roi_size"],
        help="ROI size used when the phase mask was originally optimized.",
    )
    parser.add_argument(
        "--phase-optimization-crop-to-current-roi",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["phase_optimization_crop_to_current_roi"],
    )
    parser.add_argument(
        "--phase-optimization-internal-transpose",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["phase_optimization_internal_transpose"],
        help="Use the GS optimizer's internal x/y target orientation. Leave false for FNO camera/display loss.",
    )
    parser.add_argument("--transpose-target", action=argparse.BooleanOptionalAction, default=SCRIPT_CONFIG["transpose_target"])
    parser.add_argument("--flip-target-ud", action=argparse.BooleanOptionalAction, default=SCRIPT_CONFIG["flip_target_ud"])
    parser.add_argument("--flip-target-lr", action=argparse.BooleanOptionalAction, default=SCRIPT_CONFIG["flip_target_lr"])
    parser.add_argument("--loss-roi-fraction", type=float, default=SCRIPT_CONFIG["loss_roi_fraction"])
    return parser


def resolve_required_path(value, field_name: str) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(
            f"Set SCRIPT_CONFIG['{field_name}'] near the top of this file, "
            f"or pass --{field_name.replace('_', '-')} on the command line."
        )
    return Path(value).expanduser().resolve()


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.target = resolve_required_path(args.target, "target")
    args.phase = resolve_required_path(args.phase, "phase")
    args.fno = resolve_required_path(args.fno, "fno")
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    beam_config = build_beam_config_from_args(args)
    fno_model, fno_cfg = load_fno_proxy(args.fno, device)
    if args.loss_roi_fraction is not None:
        fno_cfg["loss_roi_fraction"] = float(args.loss_roi_fraction)
    loss_weights = {
        "w_log_display_mse": float(args.w_log_display_mse),
        "w_log_display_ssim": float(args.w_log_display_ssim),
        "w_mean_norm_mse": float(args.w_mean_norm_mse),
        "w_mean_norm_smooth_l1": float(args.w_mean_norm_smooth_l1),
    }

    beam, cone_angle, axicon_transverse_frequency, h_asm = build_axicon_system(
        beam_config=beam_config,
        z_target=float(args.z_target),
        upsample_factor=int(args.upsample_factor),
        propagation_medium_index=float(args.propagation_medium_index),
        axicon_grating_pitch=float(args.axicon_grating_pitch),
        axicon_angle_in_medium=bool(args.axicon_angle_in_medium),
        margin_factor=float(args.margin_factor),
        show_transfer_function_plot=bool(args.show_transfer_function_plot),
    )

    init_phase = load_initial_phase(
        args.phase,
        beam_config,
        phase_units=args.phase_units,
        phase_level_max=args.phase_level_max,
        transpose_phase=args.transpose_phase,
        flip_phase_first_axis=args.flip_phase_first_axis,
    )
    target, preprocess_report = load_target_image(
        args.target,
        target_size=int(fno_cfg["model_size"]),
        device=device,
        dtype=torch.float32,
        beam_config=beam_config,
        preprocess_mode=args.target_preprocess_mode,
        original_pixel_size=float(args.target_original_pixel_size),
        phase_optimization_roi_size=int(args.phase_optimization_roi_size),
        current_roi_size=int(args.roi_size),
        crop_phase_optimization_to_current_roi=bool(args.phase_optimization_crop_to_current_roi),
        phase_optimization_internal_transpose=bool(args.phase_optimization_internal_transpose),
        transpose=args.transpose_target,
        flip_ud=args.flip_target_ud,
        flip_lr=args.flip_target_lr,
    )
    args.target_preprocess_report = preprocess_report
    if preprocess_report.get("requested_mode") == "auto":
        print(f"[Target] auto preprocessing selected: {preprocess_report.get('mode')}")
    if preprocess_report.get("mode") == "phase_optimization":
        print(
            "[Target] phase-optimization scale correction: "
            f"abbe=({preprocess_report['abbe_res_x_um']:.6f}, "
            f"{preprocess_report['abbe_res_y_um']:.6f}) um; "
            f"relative magnification vs direct resize="
            f"({preprocess_report['relative_horizontal_magnification_vs_direct_resize']:.6f}x, "
            f"{preprocess_report['relative_vertical_magnification_vs_direct_resize']:.6f}x)"
        )
    else:
        print("[Target] using direct target resize; phase-optimization scale correction disabled.")

    solver = AxiconFNOCITLSolver(
        beam=beam,
        fno_model=fno_model,
        fno_cfg=fno_cfg,
        cone_angle=cone_angle,
        upsample_factor=int(args.upsample_factor),
        h_asm=h_asm,
        roi_size=int(args.roi_size),
        propagation_medium_index=float(args.propagation_medium_index),
        axicon_angle_in_medium=bool(args.axicon_angle_in_medium),
        axicon_transverse_frequency=axicon_transverse_frequency,
        transpose_output_field=bool(args.transpose_output_field),
    )

    print("Running axicon + FNO phase update...")
    result = solver.optimize(
        init_phase=init_phase,
        target=target,
        iters=int(args.iters),
        lr=float(args.lr),
        optimizer_name=args.optimizer,
        loss_mode=args.loss_mode,
        loss_weights=loss_weights,
        phase_tv_weight=float(args.phase_tv_weight),
        grad_clip=args.grad_clip,
        print_every=int(args.print_every),
    )

    updated_phase = result["final_phase"]
    roi_fraction = float(fno_cfg.get("loss_roi_fraction", 1.0))
    initial_pred, initial_sim_intensity = solver.evaluate(init_phase)
    updated_pred, updated_sim_intensity = solver.evaluate(updated_phase)
    initial_metrics = compute_camera_metrics(
        initial_pred,
        target,
        roi_fraction=roi_fraction,
        loss_mode=args.loss_mode,
        weights=loss_weights,
    )
    updated_metrics = compute_camera_metrics(
        updated_pred,
        target,
        roi_fraction=roi_fraction,
        loss_mode=args.loss_mode,
        weights=loss_weights,
    )
    metrics_comparison = compare_metric_dicts(initial_metrics, updated_metrics)

    stem = f"{safe_stem(args.target)}_z{args.z_target * 1000:.3f}mm"
    phase_path = args.output_dir / f"{stem}_updated_phase.npy"
    phase_rad_path = args.output_dir / f"{stem}_updated_phase_rad_internal.npy"
    loss_path = args.output_dir / f"{stem}_loss.npy"
    component_history_path = args.output_dir / f"{stem}_loss_components.json"
    initial_pred_path = args.output_dir / f"{stem}_initial_fno_pred.npy"
    updated_pred_path = args.output_dir / f"{stem}_updated_fno_pred.npy"
    initial_sim_path = args.output_dir / f"{stem}_initial_axicon_field_input_intensity.npy"
    updated_sim_path = args.output_dir / f"{stem}_updated_axicon_field_input_intensity.npy"
    grad_path = args.output_dir / f"{stem}_last_grad_internal.npy"
    preview_path = args.output_dir / f"{stem}_preview.png"
    metrics_json_path = args.output_dir / f"{stem}_metrics.json"
    metrics_csv_path = args.output_dir / f"{stem}_metrics.csv"
    config_path = args.output_dir / f"{stem}_run_config.json"

    np.save(
        phase_path,
        phase_to_slm_file_array(
            updated_phase,
            phase_level_max=args.phase_level_max,
            transpose_phase=args.transpose_phase,
            flip_phase_first_axis=args.flip_phase_first_axis,
        ),
    )
    np.save(phase_rad_path, (updated_phase.detach() % TWO_PI).cpu().numpy().astype(np.float32))
    np.save(loss_path, result["loss_history"])
    with open(component_history_path, "w", encoding="utf-8") as f:
        json.dump(result["component_history"], f, indent=2)
    np.save(initial_pred_path, initial_pred[0, 0].detach().cpu().numpy().astype(np.float32))
    np.save(updated_pred_path, updated_pred[0, 0].detach().cpu().numpy().astype(np.float32))
    np.save(initial_sim_path, initial_sim_intensity[0, 0].detach().cpu().numpy().astype(np.float32))
    np.save(updated_sim_path, updated_sim_intensity[0, 0].detach().cpu().numpy().astype(np.float32))
    if result["last_grad"] is not None:
        np.save(grad_path, result["last_grad"].detach().cpu().numpy().astype(np.float32))

    save_comparison_preview(
        target=target,
        initial_pred=initial_pred,
        updated_pred=updated_pred,
        initial_sim_intensity=initial_sim_intensity,
        updated_sim_intensity=updated_sim_intensity,
        init_phase=init_phase,
        updated_phase=updated_phase,
        grad=result["last_grad"],
        loss_history=result["loss_history"],
        metrics_comparison=metrics_comparison,
        save_path=preview_path,
        roi_fraction=roi_fraction,
    )
    save_metrics_report(
        metrics_json_path,
        metrics_csv_path,
        initial_metrics,
        updated_metrics,
        metrics_comparison,
    )
    write_run_config(config_path, args, fno_cfg)

    print(f"Saved updated phase: {phase_path}")
    print(f"Saved preview: {preview_path}")
    print(f"Saved metrics: {metrics_json_path}")
    print(
        "Metrics initial -> updated: "
        f"loss {initial_metrics['loss']:.6f} -> {updated_metrics['loss']:.6f}, "
        f"SSIM {initial_metrics['log_display_ssim']:.6f} -> {updated_metrics['log_display_ssim']:.6f}, "
        f"mean_norm_mse {initial_metrics['mean_norm_mse']:.6f} -> {updated_metrics['mean_norm_mse']:.6f}"
    )


if __name__ == "__main__":
    main()
