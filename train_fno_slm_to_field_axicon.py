# -*- coding: utf-8 -*-
"""
Two-stage spectral-operator trainer for SLM-phase-to-axicon-field learning.

Stage 1:
    input  = SLM phase (+ coordinate grid + z channel)
    output = propagated electric field as [real(E), imag(E)]
    target = clean synthetic forward propagation from 1.Forward_Sim

Stage 2:
    keep the same model/output head
    predict [real(E), imag(E)], form intensity = |E|^2
    fine tune against aligned camera intensity, with an optional field anchor

This deliberately avoids changing the head to intensity in stage 2.  The
camera loss is intensity-to-intensity, but the model still carries a complex
field representation learned from synthetic propagation.

``annular_fno`` restricts each learned Fourier convolution to the physical
support produced by convolving the axicon ring with the square GS bandwidth.
The same support can also be used for stage-1 losses and diagnostics.
"""

import argparse
import csv
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from train_fno_axicon import (
    AxiconFNO2d,
    evenly_spaced_sample_indices,
    resize_batch,
    sample_type_from_id,
    split_dataset,
    visual_loss_per_sample,
    simple_ssim_loss_per_sample,
)


DEFAULT_ROOT = (
    r"G:\공유 드라이브\taylorlab\3DHL\CITL\Fourier Neural Operator_Training phase masks"
    r"\06_14_2026_sample3_z6mm"
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def optional_int(text):
    if text is None:
        return None
    if isinstance(text, int):
        return text
    if str(text).strip().lower() in {"none", "null", "-1"}:
        return None
    return int(text)


def parse_int_list(text):
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        raw_parts = text
    else:
        text = str(text).strip()
        if not text:
            return []
        raw_parts = re.split(r"[,;\s]+", text)
    values = []
    for part in raw_parts:
        if str(part).strip():
            value = int(part)
            if value <= 0:
                raise ValueError("Angular orders must be positive integers.")
            values.append(value)
    return sorted(set(values))


def normalized_stem(path):
    stem = Path(path).stem
    if stem.startswith("sine") and len(stem) > 4 and stem[4].isdigit():
        stem = "sine_" + stem[4:]
    parts = []
    for part in stem.split("_"):
        parts.append(str(int(part)) if part.isdigit() else part)
    return "_".join(parts)


def parse_z_mm_from_path(path, default=None):
    text = str(path).replace("\\", "/")
    matches = re.findall(r"z[_-]?(-?\d+(?:\.\d+)?)\s*mm", text, flags=re.IGNORECASE)
    if matches:
        return float(matches[-1])
    return default


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_sample(values, max_values, rng):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values.astype(np.float32)
    if max_values is not None and values.size > max_values:
        idx = rng.choice(values.size, size=int(max_values), replace=False)
        values = values[idx]
    return values.astype(np.float32, copy=False)


def quantile_display(arr, q_min=0.001, q_max=0.999, eps=1e-8):
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32))
    lo = float(np.quantile(arr.reshape(-1), q_min))
    hi = float(np.quantile(arr.reshape(-1), q_max))
    return np.clip((arr - lo) / (hi - lo + eps), 0.0, 1.0)


def angle_display(real, imag):
    return np.angle(real + 1j * imag).astype(np.float32)


def plain_decimal_tick(value, _pos=None):
    if not np.isfinite(value):
        return ""
    value = float(value)
    abs_value = abs(value)
    if abs_value < 1e-12:
        return "0"
    if abs_value >= 100:
        text = f"{value:.0f}"
    elif abs_value >= 10:
        text = f"{value:.1f}"
    elif abs_value >= 1:
        text = f"{value:.3f}"
    elif abs_value >= 0.01:
        text = f"{value:.4f}"
    elif abs_value >= 0.001:
        text = f"{value:.5f}"
    else:
        text = f"{value:.6f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def apply_plain_decimal_yaxis(ax):
    formatter = FuncFormatter(plain_decimal_tick)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(formatter)
    ax.yaxis.offsetText.set_visible(False)


def resize_wrapped_phase_for_display(phase, size):
    phase_sin = resize_batch(torch.sin(phase), size, mode="bilinear")
    phase_cos = resize_batch(torch.cos(phase), size, mode="bilinear")
    return torch.atan2(phase_sin, phase_cos)


def field_intensity(field):
    return field[:, 0:1].pow(2) + field[:, 1:2].pow(2)


def mean_per_sample(x):
    return x.flatten(start_dim=1).mean(dim=1)


def per_image_standardize(x, eps=1e-6):
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SLMToFieldDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dirs,
        phase_dir="0.Phase_Masks",
        field_dir="1.Forward_Sim",
        camera_dir="3.Aligned_Camera",
        model_size=608,
        fov_crop_size=608,
        phase_spatial_mode="center_square",
        phase_flip_lr=True,
        phase_units="levels",
        phase_level_max=1023.0,
        field_scale_mode="global_percentile",
        field_amp_percentile=99.9,
        field_amp_scale=None,
        camera_scale_mode="global_percentile",
        camera_percentile=99.9,
        camera_scale=None,
        camera_black_level=0.0,
        require_camera=False,
        z_mm=None,
        scale_sample_pixels=8192,
        max_samples=None,
        seed=42,
    ):
        self.root_dirs = [Path(p) for p in root_dirs]
        self.phase_dir = phase_dir
        self.field_dir = field_dir
        self.camera_dir = camera_dir
        self.model_size = int(model_size)
        self.fov_crop_size = optional_int(fov_crop_size)
        self.phase_spatial_mode = str(phase_spatial_mode).lower()
        self.phase_flip_lr = bool(phase_flip_lr)
        self.phase_units = phase_units
        self.phase_level_max = float(phase_level_max)
        self.field_scale_mode = field_scale_mode
        self.field_amp_percentile = float(field_amp_percentile)
        self.field_amp_scale = field_amp_scale
        self.camera_scale_mode = camera_scale_mode
        self.camera_percentile = float(camera_percentile)
        self.camera_scale = camera_scale
        self.camera_black_level = float(camera_black_level)
        self.require_camera = bool(require_camera)
        self.scale_sample_pixels = int(scale_sample_pixels) if scale_sample_pixels else None
        self.rng = np.random.default_rng(seed)

        self.valid_scale_modes = {"raw", "sample_norm", "global_percentile"}
        if self.field_scale_mode not in self.valid_scale_modes:
            raise ValueError(f"field_scale_mode must be one of {sorted(self.valid_scale_modes)}")
        if self.camera_scale_mode not in self.valid_scale_modes:
            raise ValueError(f"camera_scale_mode must be one of {sorted(self.valid_scale_modes)}")
        if self.phase_units not in {"levels", "radians"}:
            raise ValueError("phase_units must be 'levels' or 'radians'")
        if self.phase_spatial_mode not in {"center_square", "stretch"}:
            raise ValueError("phase_spatial_mode must be 'center_square' or 'stretch'")

        if z_mm is None:
            self.z_by_root = {
                str(root): parse_z_mm_from_path(root, default=0.0)
                for root in self.root_dirs
            }
        elif len(z_mm) == 1:
            self.z_by_root = {str(root): float(z_mm[0]) for root in self.root_dirs}
        elif len(z_mm) == len(self.root_dirs):
            self.z_by_root = {
                str(root): float(z)
                for root, z in zip(self.root_dirs, z_mm)
            }
        else:
            raise ValueError("--z-mm must be omitted, one value, or one value per --root-dir")

        self.samples = self._discover_samples()
        if max_samples is not None:
            self.samples = self.samples[:int(max_samples)]
        if not self.samples:
            raise RuntimeError("No paired phase/field samples were found.")

        first_phase = np.load(self.samples[0]["phase"], mmap_mode="r")
        first_phase = np.squeeze(first_phase)
        if first_phase.ndim != 2:
            raise ValueError(
                f"Phase mask must be 2D; got {first_phase.shape} in {self.samples[0]['phase']}"
            )
        phase_side = min(first_phase.shape)
        self.phase_shape_info = {
            "original": [int(first_phase.shape[0]), int(first_phase.shape[1])],
            "pre_resize": (
                [int(phase_side), int(phase_side)]
                if self.phase_spatial_mode == "center_square"
                else [int(first_phase.shape[0]), int(first_phase.shape[1])]
            ),
            "model": [self.model_size, self.model_size],
            "mode": self.phase_spatial_mode,
        }

        if self.field_scale_mode == "global_percentile" and self.field_amp_scale is None:
            self.field_amp_scale = self._compute_global_field_amp_scale()
        if self.require_camera and self.camera_scale_mode == "global_percentile" and self.camera_scale is None:
            self.camera_scale = self._compute_global_camera_scale()

        print(f">>> Loaded {len(self.samples)} paired samples")
        print(f">>> First sample ids: {', '.join(s['id'] for s in self.samples[:5])}")
        print(f">>> z by root: {self.z_by_root}")
        print(
            ">>> Phase spatial mapping: "
            f"mode={self.phase_spatial_mode}, "
            f"original={tuple(self.phase_shape_info['original'])}, "
            f"pre_resize={tuple(self.phase_shape_info['pre_resize'])}, "
            f"model={tuple(self.phase_shape_info['model'])}"
        )
        print(f">>> Field scale: mode={self.field_scale_mode}, amp_scale={self.field_amp_scale}")
        if self.require_camera:
            print(f">>> Camera scale: mode={self.camera_scale_mode}, scale={self.camera_scale}")

    def _path_map(self, directory):
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Expected directory not found: {directory}")
        out = {}
        for path in sorted(directory.glob("*.npy")):
            key = normalized_stem(path)
            if key in out:
                print(f"[skip] duplicate normalized id {key}: {out[key].name}, {path.name}")
                continue
            out[key] = path
        return out

    def _discover_samples(self):
        samples = []
        for root in self.root_dirs:
            if not root.exists():
                raise FileNotFoundError(f"Root not found: {root}")

            phase_map = self._path_map(root / self.phase_dir)
            field_map = self._path_map(root / self.field_dir)
            camera_map = {}
            camera_path = root / self.camera_dir
            if camera_path.is_dir():
                camera_map = self._path_map(camera_path)
            elif self.require_camera:
                raise FileNotFoundError(f"Camera directory not found: {camera_path}")

            common = sorted(set(phase_map) & set(field_map))
            if self.require_camera:
                common = sorted(set(common) & set(camera_map))

            missing_field = sorted(set(phase_map) - set(field_map))
            missing_phase = sorted(set(field_map) - set(phase_map))
            missing_camera = sorted(set(common) - set(camera_map)) if self.require_camera else []
            if missing_field:
                print(f">>> {root.name}: phase without field={len(missing_field)} first={missing_field[:5]}")
            if missing_phase:
                print(f">>> {root.name}: field without phase={len(missing_phase)} first={missing_phase[:5]}")
            if missing_camera:
                print(f">>> {root.name}: paired phase/field without camera={len(missing_camera)}")

            z_value = self.z_by_root[str(root)]
            root_tag = root.name
            for sample_id in common:
                samples.append({
                    "id": sample_id,
                    "root_tag": root_tag,
                    "phase": phase_map[sample_id],
                    "field": field_map[sample_id],
                    "camera": camera_map.get(sample_id),
                    "z_mm": float(z_value),
                })
        return sorted(samples, key=lambda s: (s["root_tag"], s["id"]))

    def __len__(self):
        return len(self.samples)

    def _center_crop_2d(self, arr, name="array"):
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D before crop; got {arr.shape}")
        if self.fov_crop_size is None:
            return arr
        crop = int(self.fov_crop_size)
        h, w = arr.shape
        if crop > h or crop > w:
            raise ValueError(f"fov_crop_size={crop} exceeds {name} shape {arr.shape}")
        y0 = (h - crop) // 2
        x0 = (w - crop) // 2
        return arr[y0:y0 + crop, x0:x0 + crop]

    def _resize_chw(self, tensor, mode="bilinear"):
        if tensor.shape[-2:] == (self.model_size, self.model_size):
            return tensor.float()
        kwargs = {"mode": mode}
        if mode in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        return F.interpolate(
            tensor.unsqueeze(0),
            size=(self.model_size, self.model_size),
            **kwargs,
        ).squeeze(0).float()

    def _split_channel_array(self, arr):
        if arr.ndim != 3:
            return None
        if arr.shape[0] in (2, 3):
            return arr
        if arr.shape[-1] in (2, 3):
            return np.moveaxis(arr, -1, 0)
        return None

    def _field_amp_phase(self, field_arr, path):
        field_arr = np.squeeze(field_arr)
        if np.iscomplexobj(field_arr):
            if field_arr.ndim != 2:
                raise ValueError(f"Complex field must be 2D; got {field_arr.shape} in {path}")
            amp = np.abs(field_arr).astype(np.float32)
            phase = np.angle(field_arr).astype(np.float32)
            cos_p = np.cos(phase).astype(np.float32)
            sin_p = np.sin(phase).astype(np.float32)
            return amp, cos_p, sin_p

        arr = np.asarray(field_arr, dtype=np.float32)
        chw = self._split_channel_array(arr)
        if chw is None:
            if arr.ndim != 2:
                raise ValueError(f"Field must be 2D, 2ch, 3ch, or complex; got {arr.shape} in {path}")
            arr = np.clip(np.nan_to_num(arr), 0.0, None)
            amp = np.sqrt(arr).astype(np.float32)
            return amp, np.ones_like(amp, dtype=np.float32), np.zeros_like(amp, dtype=np.float32)
        if chw.shape[0] == 2:
            real = chw[0].astype(np.float32)
            imag = chw[1].astype(np.float32)
            amp = np.sqrt(real ** 2 + imag ** 2)
            phase = np.arctan2(imag, real)
            return amp, np.cos(phase).astype(np.float32), np.sin(phase).astype(np.float32)

        amp = chw[0].astype(np.float32)
        cos_p = chw[1].astype(np.float32)
        sin_p = chw[2].astype(np.float32)
        norm = np.sqrt(cos_p ** 2 + sin_p ** 2)
        cos_p = cos_p / np.maximum(norm, 1e-6)
        sin_p = sin_p / np.maximum(norm, 1e-6)
        return amp, cos_p.astype(np.float32), sin_p.astype(np.float32)

    def _compute_global_field_amp_scale(self):
        values = []
        for sample in tqdm(self.samples, desc="Estimating field amplitude scale"):
            arr = np.load(sample["field"])
            amp, _, _ = self._field_amp_phase(arr, sample["field"])
            amp = self._center_crop_2d(amp, name=f"field amp {sample['id']}")
            values.append(finite_sample(amp.reshape(-1), self.scale_sample_pixels, self.rng))
        values = np.concatenate([v for v in values if v.size]) if values else np.array([], dtype=np.float32)
        if values.size == 0:
            return 1.0
        return max(float(np.percentile(values, self.field_amp_percentile)), 1e-8)

    def _compute_global_camera_scale(self):
        values = []
        for sample in tqdm(self.samples, desc="Estimating camera scale"):
            if sample["camera"] is None:
                continue
            arr = self._load_camera_array(sample["camera"])
            arr = np.clip(arr - self.camera_black_level, 0.0, None)
            values.append(finite_sample(arr.reshape(-1), self.scale_sample_pixels, self.rng))
        values = np.concatenate([v for v in values if v.size]) if values else np.array([], dtype=np.float32)
        if values.size == 0:
            return 1.0
        return max(float(np.percentile(values, self.camera_percentile)), 1e-8)

    def _scale_amp(self, amp):
        amp = np.clip(np.nan_to_num(np.asarray(amp, dtype=np.float32)), 0.0, None)
        if self.field_scale_mode == "raw":
            return amp.astype(np.float32)
        if self.field_scale_mode == "sample_norm":
            scale = max(float(np.percentile(amp, self.field_amp_percentile)), 1e-8)
            return (amp / scale).astype(np.float32)
        if self.field_scale_mode == "global_percentile":
            return (amp / (float(self.field_amp_scale) + 1e-8)).astype(np.float32)
        raise RuntimeError(f"Unhandled field_scale_mode: {self.field_scale_mode}")

    def _load_field(self, path):
        arr = np.load(path)
        amp, cos_p, sin_p = self._field_amp_phase(arr, path)
        amp = self._center_crop_2d(amp, name=f"field amp {Path(path).name}")
        cos_p = self._center_crop_2d(cos_p, name=f"field cos {Path(path).name}")
        sin_p = self._center_crop_2d(sin_p, name=f"field sin {Path(path).name}")

        amp = self._scale_amp(amp)
        real = amp * cos_p
        imag = amp * sin_p
        field = torch.from_numpy(np.stack([real, imag], axis=0).astype(np.float32))
        field = self._resize_chw(field, mode="bilinear")
        amp_tensor = torch.sqrt(field[0:1].pow(2) + field[1:2].pow(2)).clamp_min(0.0)
        return field.float(), amp_tensor.float(), amp_tensor.pow(2).float()

    def _load_phase(self, path):
        phase = np.load(path).astype(np.float32)
        phase = np.squeeze(phase)
        if phase.ndim != 2:
            raise ValueError(f"Phase mask must be 2D; got {phase.shape} in {path}")
        if self.phase_spatial_mode == "center_square":
            side = min(phase.shape)
            y0 = (phase.shape[0] - side) // 2
            x0 = (phase.shape[1] - side) // 2
            phase = phase[y0:y0 + side, x0:x0 + side]
        if self.phase_flip_lr:
            phase = phase[:, ::-1].copy()
        if self.phase_units == "levels":
            phase = phase * (2.0 * np.pi / self.phase_level_max)
        phase = np.mod(phase, 2.0 * np.pi).astype(np.float32)
        return torch.from_numpy(phase).unsqueeze(0).float()

    def _load_camera_array(self, path):
        arr = np.load(path)
        if np.iscomplexobj(arr):
            arr = np.abs(arr) ** 2
        arr = np.squeeze(arr).astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Camera array must be 2D; got {arr.shape} in {path}")
        return self._center_crop_2d(arr, name=f"camera {Path(path).name}")

    def _load_camera(self, path):
        arr = np.clip(np.nan_to_num(self._load_camera_array(path)) - self.camera_black_level, 0.0, None)
        if self.camera_scale_mode == "sample_norm":
            scale = max(float(np.percentile(arr, self.camera_percentile)), 1e-8)
            arr = arr / scale
        elif self.camera_scale_mode == "global_percentile":
            arr = arr / (float(self.camera_scale) + 1e-8)
        elif self.camera_scale_mode != "raw":
            raise RuntimeError(f"Unhandled camera_scale_mode: {self.camera_scale_mode}")
        cam = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
        return self._resize_chw(cam, mode="bilinear").float()

    def __getitem__(self, idx):
        sample = self.samples[int(idx)]
        field, amp, intensity = self._load_field(sample["field"])
        out = {
            "id": sample["id"],
            "root_tag": sample["root_tag"],
            "phase": self._load_phase(sample["phase"]),
            "field": field,
            "field_amp": amp,
            "field_intensity": intensity,
            "z_mm": torch.tensor(sample["z_mm"], dtype=torch.float32),
        }
        if self.require_camera and sample["camera"] is not None:
            out["camera"] = self._load_camera(sample["camera"])
        return out


# ---------------------------------------------------------------------------
# Model input
# ---------------------------------------------------------------------------

def make_grid(batch, height, width, device, dtype, radial=True):
    yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    pieces = [x, y]
    if radial:
        pieces.append(torch.sqrt(x.pow(2) + y.pow(2)))
    grid = torch.stack(pieces, dim=0).unsqueeze(0)
    return grid.repeat(batch, 1, 1, 1)


def _physical_axis(n_source_pixels, n_out, pixel_size, device, dtype):
    return torch.linspace(
        -(float(n_source_pixels) - 1.0) / 2.0,
        (float(n_source_pixels) - 1.0) / 2.0,
        int(n_out),
        device=device,
        dtype=dtype,
    ) * float(pixel_size)


def make_axicon_coordinate_grid(size, cfg, device, dtype):
    scope = str(cfg.get("axicon_map_scope", "roi")).lower()
    slm_pixel = float(cfg.get("slm_pixel_size_m", 6.4e-6))
    upsample = int(cfg.get("axicon_upsample_factor", 20))
    upsample = max(1, upsample)
    upsampled_pixel = slm_pixel / upsample

    if scope == "slm":
        nx = int(cfg.get("slm_nx", 1600)) * upsample
        ny = int(cfg.get("slm_ny", 1200)) * upsample
        x = _physical_axis(nx, size, upsampled_pixel, device, dtype)
        y = _physical_axis(ny, size, upsampled_pixel, device, dtype)
    elif scope == "roi":
        crop = cfg.get("fov_crop_size", None)
        crop = int(crop) if crop is not None else int(cfg.get("axicon_roi_size", 1024))
        x = _physical_axis(crop, size, upsampled_pixel, device, dtype)
        y = _physical_axis(crop, size, upsampled_pixel, device, dtype)
    else:
        raise ValueError(f"Unknown axicon_map_scope={scope!r}; expected 'roi' or 'slm'")

    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx, yy


def model_field_pixel_size_m(cfg, height=None, width=None):
    """Physical sampling of the cropped synthetic field represented by the model."""
    height = int(height if height is not None else cfg["model_size"])
    width = int(width if width is not None else cfg["model_size"])
    crop = cfg.get("fov_crop_size", None)
    crop = int(crop) if crop is not None else int(cfg.get("axicon_roi_size", 1024))
    slm_pixel = float(cfg.get("slm_pixel_size_m", 6.4e-6))
    upsample = max(1, int(cfg.get("axicon_upsample_factor", 20)))
    source_pixel = slm_pixel / upsample

    # _physical_axis spans crop source samples, including both end points.
    dx = source_pixel * max(crop - 1, 1) / max(width - 1, 1)
    dy = source_pixel * max(crop - 1, 1) / max(height - 1, 1)
    return float(dy), float(dx)


def effective_spectrum_geometry(cfg, height=None, width=None):
    height = int(height if height is not None else cfg["model_size"])
    width = int(width if width is not None else cfg["model_size"])
    dy, dx = model_field_pixel_size_m(cfg, height=height, width=width)
    gs_pixel = cfg.get("gs_effective_pixel_size_m", None)
    gs_pixel = float(gs_pixel if gs_pixel is not None else cfg.get("slm_pixel_size_m", 6.4e-6))
    if gs_pixel <= 0:
        raise ValueError("GS effective pixel size must be positive.")

    bandwidth_fraction = float(cfg.get("gs_bandwidth_nyquist_fraction", 1.0))
    if bandwidth_fraction <= 0:
        raise ValueError("GS bandwidth Nyquist fraction must be positive.")
    f_nyquist = 1.0 / (2.0 * gs_pixel)
    bandwidth_axis = bandwidth_fraction * f_nyquist
    bandwidth_corner = math.sqrt(2.0) * bandwidth_axis
    ring_frequency = float(cfg.get(
        "axicon_transverse_frequency_actual",
        1.0 / float(cfg.get("axicon_grating_pitch_m", 1.396e-6)),
    ))
    if ring_frequency <= 0:
        raise ValueError("Axicon transverse frequency must be positive.")

    return {
        "height": height,
        "width": width,
        "field_dy_m": dy,
        "field_dx_m": dx,
        "df_y_cyc_per_m": 1.0 / (height * dy),
        "df_x_cyc_per_m": 1.0 / (width * dx),
        "model_nyquist_y_cyc_per_m": 1.0 / (2.0 * dy),
        "model_nyquist_x_cyc_per_m": 1.0 / (2.0 * dx),
        "gs_effective_pixel_size_m": gs_pixel,
        "gs_nyquist_cyc_per_m": f_nyquist,
        "gs_bandwidth_axis_cyc_per_m": bandwidth_axis,
        "gs_bandwidth_corner_cyc_per_m": bandwidth_corner,
        "axicon_ring_cyc_per_m": ring_frequency,
        "ring_over_gs_nyquist": ring_frequency / f_nyquist,
        "axis_halfwidth_over_ring": bandwidth_axis / ring_frequency,
        "corner_halfwidth_over_ring": bandwidth_corner / ring_frequency,
        "ring_bin_x": ring_frequency * width * dx,
        "ring_bin_y": ring_frequency * height * dy,
    }


_EFFECTIVE_SPECTRUM_CACHE = {}


def effective_axicon_spectrum_weight(height, width, cfg, device, dtype, rfft=False):
    """Return the convolution support of an axicon ring and GS-mask bandwidth.

    For ``square_minkowski`` this is the exact support of a radius-f_r ring
    convolved with the square SLM Nyquist support [-B, B]^2.  A radial envelope
    is available for direct comparison with the conservative +/-sqrt(2) B rule.
    """
    geometry = effective_spectrum_geometry(cfg, height=height, width=width)
    shape = str(cfg.get("effective_spectrum_shape", "square_minkowski")).lower()
    soft_fn = max(0.0, float(cfg.get("effective_spectrum_soft_edge_fn", 0.05)))
    cache_key = (
        int(height), int(width), bool(rfft), str(device), str(dtype), shape,
        geometry["field_dy_m"], geometry["field_dx_m"],
        geometry["axicon_ring_cyc_per_m"],
        geometry["gs_bandwidth_axis_cyc_per_m"], soft_fn,
    )
    cached = _EFFECTIVE_SPECTRUM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    fy = torch.fft.fftfreq(height, d=geometry["field_dy_m"], device=device, dtype=dtype)
    if rfft:
        fx = torch.fft.rfftfreq(width, d=geometry["field_dx_m"], device=device, dtype=dtype)
    else:
        fx = torch.fft.fftfreq(width, d=geometry["field_dx_m"], device=device, dtype=dtype)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    abs_x = fxx.abs()
    abs_y = fyy.abs()
    ring = geometry["axicon_ring_cyc_per_m"]
    bandwidth = geometry["gs_bandwidth_axis_cyc_per_m"]
    soft = soft_fn * geometry["gs_nyquist_cyc_per_m"]

    if shape == "square_minkowski":
        # A frequency k is in ring (+) square iff the circle radius lies between
        # the minimum and maximum distances from k to the centered square.
        distance_min = torch.sqrt(
            F.relu(abs_x - bandwidth).pow(2) + F.relu(abs_y - bandwidth).pow(2)
        )
        distance_max = torch.sqrt((abs_x + bandwidth).pow(2) + (abs_y + bandwidth).pow(2))
        if soft > 0:
            weight = torch.sigmoid((ring - distance_min) / soft)
            weight = weight * torch.sigmoid((distance_max - ring) / soft)
        else:
            weight = ((distance_min <= ring) & (ring <= distance_max)).to(dtype)
    elif shape == "radial_envelope":
        rho = torch.sqrt(fxx.pow(2) + fyy.pow(2))
        halfwidth = geometry["gs_bandwidth_corner_cyc_per_m"]
        if soft > 0:
            weight = torch.sigmoid((halfwidth - torch.abs(rho - ring)) / soft)
        else:
            weight = (torch.abs(rho - ring) <= halfwidth).to(dtype)
    else:
        raise ValueError(
            f"Unknown effective_spectrum_shape={shape!r}; expected "
            "'square_minkowski' or 'radial_envelope'."
        )

    weight = weight.clamp(0.0, 1.0)
    _EFFECTIVE_SPECTRUM_CACHE[cache_key] = weight
    return weight


def make_axicon_physics_maps(batch, size, cfg, device, dtype):
    xx, yy = make_axicon_coordinate_grid(size, cfg, device, dtype)
    radius = torch.sqrt(xx.pow(2) + yy.pow(2))

    grating_pitch = float(cfg.get("axicon_grating_pitch_m", 1.396e-6))
    transverse_frequency = float(cfg.get(
        "axicon_transverse_frequency_actual",
        cfg.get("axicon_transverse_frequency", 0.0),
    ))
    if transverse_frequency <= 0:
        transverse_frequency = 1.0 / grating_pitch
    phase_sign = float(cfg.get("axicon_phase_sign", -1.0))
    axicon_phase = phase_sign * 2.0 * math.pi * transverse_frequency * radius
    axicon_cos = torch.cos(axicon_phase).unsqueeze(0).unsqueeze(0)
    axicon_sin = torch.sin(axicon_phase).unsqueeze(0).unsqueeze(0)

    radius_scale = radius.amax().clamp_min(torch.as_tensor(1e-12, device=device, dtype=dtype))
    radius_norm = (radius / radius_scale).unsqueeze(0).unsqueeze(0)

    waist = float(cfg.get("axicon_gaussian_waist_m", 0.00638708 * 0.8))
    aperture = torch.exp(-radius.pow(2) / max(waist, 1e-12) ** 2)
    aperture = aperture / aperture.mean().clamp_min(torch.as_tensor(1e-12, device=device, dtype=dtype))
    aperture = aperture.unsqueeze(0).unsqueeze(0)

    return {
        "axicon_cos": axicon_cos.repeat(batch, 1, 1, 1),
        "axicon_sin": axicon_sin.repeat(batch, 1, 1, 1),
        "axicon_radius": radius_norm.repeat(batch, 1, 1, 1),
        "axicon_aperture": aperture.repeat(batch, 1, 1, 1),
    }


def build_phase_fno_input(phase, z_mm, cfg):
    size = int(cfg["model_size"])
    phase_cos = torch.cos(phase)
    phase_sin = torch.sin(phase)

    lowpass = cfg.get("phase_lowpass_size")
    if lowpass is not None and int(lowpass) > 0 and int(lowpass) < size:
        lowpass = int(lowpass)
        phase_cos = resize_batch(phase_cos, lowpass, mode="bilinear")
        phase_sin = resize_batch(phase_sin, lowpass, mode="bilinear")

    phase_cos = resize_batch(phase_cos, size, mode="bilinear")
    phase_sin = resize_batch(phase_sin, size, mode="bilinear")
    phase_norm = torch.sqrt(phase_cos.pow(2) + phase_sin.pow(2)).clamp_min(1e-6)
    slm_cos = phase_cos / phase_norm
    slm_sin = phase_sin / phase_norm
    pieces = [slm_cos, slm_sin]

    axicon_maps = None
    if (
        cfg.get("use_axicon_phase", True)
        or cfg.get("use_slm_axicon_product", True)
        or cfg.get("use_axicon_radius", True)
        or cfg.get("use_axicon_aperture", False)
    ):
        axicon_maps = make_axicon_physics_maps(
            phase.shape[0], size, cfg, phase.device, phase.dtype
        )

    if cfg.get("use_axicon_phase", True):
        pieces.extend([axicon_maps["axicon_cos"], axicon_maps["axicon_sin"]])

    if cfg.get("use_slm_axicon_product", True):
        ax_cos = axicon_maps["axicon_cos"]
        ax_sin = axicon_maps["axicon_sin"]
        product_cos = slm_cos * ax_cos - slm_sin * ax_sin
        product_sin = slm_sin * ax_cos + slm_cos * ax_sin
        pieces.extend([product_cos, product_sin])

    if cfg.get("use_axicon_radius", True):
        pieces.append(axicon_maps["axicon_radius"])

    if cfg.get("use_axicon_aperture", False):
        pieces.append(axicon_maps["axicon_aperture"])

    if cfg.get("use_grid", True):
        pieces.append(make_grid(
            phase.shape[0],
            size,
            size,
            phase.device,
            phase.dtype,
            radial=cfg.get("use_radial_coord", True),
        ))

    if cfg.get("use_z_channel", True):
        z_ref = float(cfg.get("z_ref_mm", 10.0))
        z = z_mm.to(device=phase.device, dtype=phase.dtype).view(-1, 1, 1, 1) / z_ref
        pieces.append(z.expand(-1, 1, size, size))

    return torch.cat(pieces, dim=1)


def infer_phase_input_channels(cfg):
    channels = 2
    if cfg.get("use_axicon_phase", True):
        channels += 2
    if cfg.get("use_slm_axicon_product", True):
        channels += 2
    if cfg.get("use_axicon_radius", True):
        channels += 1
    if cfg.get("use_axicon_aperture", False):
        channels += 1
    if cfg.get("use_grid", True):
        channels += 3 if cfg.get("use_radial_coord", True) else 2
    if cfg.get("use_z_channel", True):
        channels += 1
    return channels


def phase_input_channel_names(cfg):
    names = ["slm_cos", "slm_sin"]
    if cfg.get("use_axicon_phase", True):
        names.extend(["axicon_cos", "axicon_sin"])
    if cfg.get("use_slm_axicon_product", True):
        names.extend(["slm_times_axicon_cos", "slm_times_axicon_sin"])
    if cfg.get("use_axicon_radius", True):
        names.append("axicon_radius_norm")
    if cfg.get("use_axicon_aperture", False):
        names.append("axicon_gaussian_aperture")
    if cfg.get("use_grid", True):
        names.extend(["x_norm", "y_norm"])
        if cfg.get("use_radial_coord", True):
            names.append("r_norm")
    if cfg.get("use_z_channel", True):
        names.append("z_mm_over_ref")
    return names


# ---------------------------------------------------------------------------
# Operator models
# ---------------------------------------------------------------------------

def channel_index_map(names):
    return {name: idx for idx, name in enumerate(names)}


class PointwiseMLP2d(nn.Module):
    def __init__(self, in_ch, out_ch, width, depth, zero_init=True):
        super().__init__()
        depth = max(0, int(depth))
        width = max(1, int(width))
        if depth <= 0:
            self.net = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            layers = [nn.Conv2d(in_ch, width, kernel_size=1), nn.GELU()]
            for _ in range(depth - 1):
                layers.extend([nn.Conv2d(width, width, kernel_size=1), nn.GELU()])
            layers.append(nn.Conv2d(width, out_ch, kernel_size=1))
            self.net = nn.Sequential(*layers)
        if zero_init:
            last = self.net[-1] if isinstance(self.net, nn.Sequential) else self.net
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.net(x)


class AnnularSpectralConv2d(nn.Module):
    """FNO spectral convolution restricted to the physical axicon support."""

    def __init__(
        self,
        in_channels,
        out_channels,
        model_size,
        cfg,
        low_modes=0,
        support_threshold=0.01,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.model_size = int(model_size)
        self.low_modes = max(0, int(low_modes))
        threshold = float(support_threshold)
        if not 0.0 <= threshold < 1.0:
            raise ValueError("Annular support threshold must lie in [0, 1).")

        support = effective_axicon_spectrum_weight(
            self.model_size,
            self.model_size,
            cfg,
            device=torch.device("cpu"),
            dtype=torch.float32,
            rfft=True,
        ).clone()
        active = support >= threshold
        if self.low_modes:
            low = torch.zeros_like(active)
            my = min(self.low_modes, self.model_size // 2)
            mx = min(self.low_modes, self.model_size // 2 + 1)
            low[:my, :mx] = True
            low[-my:, :mx] = True
            active |= low
            support = torch.where(low, torch.ones_like(support), support)

        active_indices = torch.nonzero(active.reshape(-1), as_tuple=False).flatten()
        if active_indices.numel() == 0:
            raise ValueError("The effective axicon spectrum support is empty on the model grid.")
        taper = support.reshape(-1).index_select(0, active_indices)
        self.n_active_modes = int(active_indices.numel())
        self.support_fraction = self.n_active_modes / float(support.numel())
        self.register_buffer("active_indices", active_indices)
        self.register_buffer("active_taper", taper)
        self.register_buffer("support_weight", support)

        scale = 1.0 / math.sqrt(self.in_channels * self.out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(
                self.in_channels,
                self.out_channels,
                self.n_active_modes,
                dtype=torch.cfloat,
            )
        )

    def forward(self, x):
        batch, _, height, width = x.shape
        if height != self.model_size or width != self.model_size:
            raise ValueError(
                f"Annular FNO was built for {self.model_size}x{self.model_size}, "
                f"received {height}x{width}."
            )
        x_ft = torch.fft.rfft2(x, norm="ortho")
        x_flat = x_ft.reshape(batch, self.in_channels, -1)
        x_active = torch.index_select(x_flat, -1, self.active_indices)
        out_active = torch.einsum("bik,iok->bok", x_active, self.weight)
        out_active = out_active * self.active_taper.view(1, 1, -1)
        out_flat = torch.zeros(
            batch,
            self.out_channels,
            x_flat.shape[-1],
            device=x.device,
            dtype=x_ft.dtype,
        )
        out_flat = out_flat.index_copy(-1, self.active_indices, out_active)
        out_ft = out_flat.reshape(batch, self.out_channels, height, width // 2 + 1)
        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class AnnularFNOBlock2d(nn.Module):
    def __init__(self, width, model_size, cfg, low_modes=0, support_threshold=0.01, groups=4):
        super().__init__()
        self.spectral = AnnularSpectralConv2d(
            width,
            width,
            model_size=model_size,
            cfg=cfg,
            low_modes=low_modes,
            support_threshold=support_threshold,
        )
        self.local = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(min(groups, width), width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.local(x)))


class AnnularAxiconFNO2d(nn.Module):
    """FNO whose global branch learns the convolution-expanded axicon ring."""

    def __init__(
        self,
        in_ch,
        out_ch,
        width,
        depth,
        mlp_width,
        model_size,
        cfg,
        low_modes=0,
        support_threshold=0.01,
    ):
        super().__init__()
        self.lift = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
        )
        self.blocks = nn.ModuleList([
            AnnularFNOBlock2d(
                width,
                model_size=model_size,
                cfg=cfg,
                low_modes=low_modes,
                support_threshold=support_threshold,
            )
            for _ in range(int(depth))
        ])
        self.project = nn.Sequential(
            nn.Conv2d(width, mlp_width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mlp_width, out_ch, kernel_size=1),
        )
        first = self.blocks[0].spectral
        self.n_active_modes = first.n_active_modes
        self.support_fraction = first.support_fraction

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.project(x)


class PhysicsConstrainedSpectralPropagator2d(nn.Module):
    """Compact ASM-like surrogate with a learnable radial Fourier multiplier."""

    def __init__(
        self,
        in_ch,
        channel_names,
        model_size,
        width=16,
        depth=2,
        radial_bins=256,
        transfer_amp_scale=2.0,
        angular_orders=None,
        angular_amp_scale=0.25,
        angular_phase_scale=0.50,
        pupil_phase_scale=0.25,
        pupil_amp_scale=0.10,
        residual_scale=0.10,
        support_radius=1.0,
        support_softness=0.05,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.channel_names = list(channel_names)
        self.channel_idx = channel_index_map(self.channel_names)
        self.model_size = int(model_size)
        self.radial_bins = max(8, int(radial_bins))
        self.transfer_amp_scale = float(transfer_amp_scale)
        self.angular_orders = parse_int_list(angular_orders)
        self.n_angular = len(self.angular_orders)
        self.angular_amp_scale = float(angular_amp_scale)
        self.angular_phase_scale = float(angular_phase_scale)
        self.pupil_phase_scale = float(pupil_phase_scale)
        self.pupil_amp_scale = float(pupil_amp_scale)
        self.residual_scale = float(residual_scale)
        self.support_radius = float(support_radius)
        self.support_softness = max(float(support_softness), 1e-6)

        self.radial_log_amp = nn.Parameter(torch.zeros(self.radial_bins))
        self.radial_phase = nn.Parameter(torch.zeros(self.radial_bins))
        self.output_log_gain = nn.Parameter(torch.zeros(()))
        if self.n_angular:
            self.angular_log_amp_cos = nn.Parameter(torch.zeros(self.n_angular, self.radial_bins))
            self.angular_log_amp_sin = nn.Parameter(torch.zeros(self.n_angular, self.radial_bins))
            self.angular_phase_cos = nn.Parameter(torch.zeros(self.n_angular, self.radial_bins))
            self.angular_phase_sin = nn.Parameter(torch.zeros(self.n_angular, self.radial_bins))
        else:
            self.angular_log_amp_cos = None
            self.angular_log_amp_sin = None
            self.angular_phase_cos = None
            self.angular_phase_sin = None

        self.pupil_calibration = PointwiseMLP2d(
            in_ch=self.in_ch,
            out_ch=2,
            width=width,
            depth=depth,
            zero_init=True,
        )
        self.post_residual = PointwiseMLP2d(
            in_ch=self.in_ch + 2,
            out_ch=2,
            width=width,
            depth=depth,
            zero_init=True,
        )

        fy = torch.fft.fftfreq(self.model_size)
        fx = torch.fft.fftfreq(self.model_size)
        fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
        rho = torch.sqrt(fxx.pow(2) + fyy.pow(2))
        theta = torch.atan2(fyy, fxx)
        rho = rho / rho.max().clamp_min(1e-8)
        bin_pos = rho * (self.radial_bins - 1)
        bin_lo = torch.floor(bin_pos).long().clamp(0, self.radial_bins - 1)
        bin_hi = (bin_lo + 1).clamp(0, self.radial_bins - 1)
        bin_frac = (bin_pos - bin_lo.to(bin_pos.dtype)).clamp(0.0, 1.0)
        if self.support_radius >= 0.999:
            support = torch.ones_like(rho)
        else:
            support = torch.sigmoid((self.support_radius - rho) / self.support_softness)
        self.register_buffer("rho_norm", rho)
        self.register_buffer("bin_lo", bin_lo)
        self.register_buffer("bin_hi", bin_hi)
        self.register_buffer("bin_frac", bin_frac)
        self.register_buffer("support_taper", support)
        if self.n_angular:
            order_tensor = torch.as_tensor(self.angular_orders, dtype=theta.dtype).view(-1, 1, 1)
            self.register_buffer("angular_cos_basis", torch.cos(order_tensor * theta.unsqueeze(0)))
            self.register_buffer("angular_sin_basis", torch.sin(order_tensor * theta.unsqueeze(0)))
        else:
            self.register_buffer("angular_cos_basis", torch.empty(0, self.model_size, self.model_size))
            self.register_buffer("angular_sin_basis", torch.empty(0, self.model_size, self.model_size))

    def _channel(self, x, name):
        idx = self.channel_idx.get(name)
        if idx is None:
            return None
        return x[:, idx:idx + 1]

    def _complex_pupil(self, x):
        product_cos = self._channel(x, "slm_times_axicon_cos")
        product_sin = self._channel(x, "slm_times_axicon_sin")
        if product_cos is not None and product_sin is not None:
            real, imag = product_cos, product_sin
        else:
            slm_cos = self._channel(x, "slm_cos")
            slm_sin = self._channel(x, "slm_sin")
            if slm_cos is None or slm_sin is None:
                raise RuntimeError("spectral operator requires slm_cos/slm_sin input channels.")
            ax_cos = self._channel(x, "axicon_cos")
            ax_sin = self._channel(x, "axicon_sin")
            if ax_cos is None or ax_sin is None:
                real, imag = slm_cos, slm_sin
            else:
                real = slm_cos * ax_cos - slm_sin * ax_sin
                imag = slm_sin * ax_cos + slm_cos * ax_sin

        aperture = self._channel(x, "axicon_gaussian_aperture")
        if aperture is not None:
            real = real * aperture
            imag = imag * aperture

        calib = self.pupil_calibration(x)
        amp_corr = torch.exp(self.pupil_amp_scale * torch.tanh(calib[:, 0:1]))
        phase_corr = self.pupil_phase_scale * torch.tanh(calib[:, 1:2])
        phase_cos = torch.cos(phase_corr)
        phase_sin = torch.sin(phase_corr)
        real_cal = real * phase_cos - imag * phase_sin
        imag_cal = imag * phase_cos + real * phase_sin
        return torch.complex(real_cal[:, 0] * amp_corr[:, 0], imag_cal[:, 0] * amp_corr[:, 0])

    def _radial_lookup(self, values):
        if values.ndim == 1:
            lo = values[self.bin_lo]
            hi = values[self.bin_hi]
        else:
            leading_shape = tuple(values.shape[:-1])
            flat_lo = self.bin_lo.reshape(-1)
            flat_hi = self.bin_hi.reshape(-1)
            lo = torch.index_select(values, values.ndim - 1, flat_lo)
            hi = torch.index_select(values, values.ndim - 1, flat_hi)
            lo = lo.reshape(*leading_shape, *self.bin_lo.shape)
            hi = hi.reshape(*leading_shape, *self.bin_hi.shape)
        return lo * (1.0 - self.bin_frac) + hi * self.bin_frac

    def transfer_function(self):
        log_amp = self.transfer_amp_scale * torch.tanh(self._radial_lookup(self.radial_log_amp))
        phase = self._radial_lookup(self.radial_phase)
        if self.n_angular:
            amp_cos = self._radial_lookup(self.angular_log_amp_cos)
            amp_sin = self._radial_lookup(self.angular_log_amp_sin)
            phase_cos = self._radial_lookup(self.angular_phase_cos)
            phase_sin = self._radial_lookup(self.angular_phase_sin)
            angular_log_amp = (
                amp_cos * self.angular_cos_basis
                + amp_sin * self.angular_sin_basis
            ).sum(dim=0)
            angular_phase = (
                phase_cos * self.angular_cos_basis
                + phase_sin * self.angular_sin_basis
            ).sum(dim=0)
            log_amp = log_amp + self.angular_amp_scale * torch.tanh(angular_log_amp)
            phase = phase + self.angular_phase_scale * torch.tanh(angular_phase)
        amp = torch.exp(log_amp) * self.support_taper
        return torch.polar(amp, phase)

    def forward(self, x):
        pupil = self._complex_pupil(x)
        spectrum = torch.fft.fft2(pupil, norm="ortho")
        propagated = torch.fft.ifft2(spectrum * self.transfer_function().unsqueeze(0), norm="ortho")
        out = torch.stack([propagated.real, propagated.imag], dim=1)
        out = torch.exp(self.output_log_gain) * out
        if self.residual_scale > 0:
            residual = self.post_residual(torch.cat([x, out], dim=1))
            out = out + self.residual_scale * residual
        return out

    def regularization_terms(self):
        amp_diff = self.radial_log_amp[1:] - self.radial_log_amp[:-1]
        phase_diff = self.radial_phase[1:] - self.radial_phase[:-1]
        terms = {
            "spectral_amp_smooth": amp_diff.pow(2).mean(),
            "spectral_phase_smooth": phase_diff.pow(2).mean(),
            "spectral_amp_l2": self.radial_log_amp.pow(2).mean(),
            "spectral_gain_l2": self.output_log_gain.pow(2),
        }
        if self.n_angular:
            angular_amp = torch.cat([
                self.angular_log_amp_cos,
                self.angular_log_amp_sin,
            ], dim=0)
            angular_phase = torch.cat([
                self.angular_phase_cos,
                self.angular_phase_sin,
            ], dim=0)
            terms.update({
                "spectral_angular_amp_l2": angular_amp.pow(2).mean(),
                "spectral_angular_phase_l2": angular_phase.pow(2).mean(),
                "spectral_angular_amp_smooth": (angular_amp[:, 1:] - angular_amp[:, :-1]).pow(2).mean(),
                "spectral_angular_phase_smooth": (angular_phase[:, 1:] - angular_phase[:, :-1]).pow(2).mean(),
            })
        return terms

    def diagnostic_arrays(self):
        with torch.no_grad():
            h = self.transfer_function().detach().cpu()
            return {
                "transfer_amp": h.abs().numpy(),
                "transfer_phase": torch.angle(h).numpy(),
                "radial_log_amp": self.radial_log_amp.detach().cpu().numpy(),
                "radial_phase": self.radial_phase.detach().cpu().numpy(),
                "angular_orders": np.array(self.angular_orders, dtype=np.int64),
            }


def build_operator_model(cfg, in_ch):
    mode = cfg.get("operator_mode", "fno")
    channel_names = phase_input_channel_names(cfg)
    if mode == "fno":
        return AxiconFNO2d(
            in_ch=in_ch,
            out_ch=2,
            width=cfg["width"],
            modes_y=cfg["modes_y"],
            modes_x=cfg["modes_x"],
            depth=cfg["depth"],
            mlp_width=cfg["mlp_width"],
        )
    if mode == "annular_fno":
        return AnnularAxiconFNO2d(
            in_ch=in_ch,
            out_ch=2,
            width=cfg["width"],
            depth=cfg["depth"],
            mlp_width=cfg["mlp_width"],
            model_size=cfg["model_size"],
            cfg=cfg,
            low_modes=cfg["annular_fno_low_modes"],
            support_threshold=cfg["effective_spectrum_support_threshold"],
        )
    if mode == "spectral":
        return PhysicsConstrainedSpectralPropagator2d(
            in_ch=in_ch,
            channel_names=channel_names,
            model_size=cfg["model_size"],
            width=cfg["spectral_width"],
            depth=cfg["spectral_depth"],
            radial_bins=cfg["spectral_radial_bins"],
            transfer_amp_scale=cfg["spectral_transfer_amp_scale"],
            angular_orders=cfg["spectral_angular_orders"],
            angular_amp_scale=cfg["spectral_angular_amp_scale"],
            angular_phase_scale=cfg["spectral_angular_phase_scale"],
            pupil_phase_scale=cfg["spectral_pupil_phase_scale"],
            pupil_amp_scale=cfg["spectral_pupil_amp_scale"],
            residual_scale=cfg["spectral_residual_scale"],
            support_radius=cfg["spectral_support_radius"],
            support_softness=cfg["spectral_support_softness"],
        )
    raise ValueError(f"Unknown operator_mode={mode!r}")


def model_regularization_loss(model, cfg, device):
    if not hasattr(model, "regularization_terms"):
        return torch.zeros((), device=device), {}
    terms = model.regularization_terms()
    weights = {
        "spectral_amp_smooth": cfg.get("w_spectral_amp_smooth", 0.0),
        "spectral_phase_smooth": cfg.get("w_spectral_phase_smooth", 0.0),
        "spectral_amp_l2": cfg.get("w_spectral_amp_l2", 0.0),
        "spectral_gain_l2": cfg.get("w_spectral_gain_l2", 0.0),
        "spectral_angular_amp_l2": cfg.get("w_spectral_angular_l2", 0.0),
        "spectral_angular_phase_l2": cfg.get("w_spectral_angular_l2", 0.0),
        "spectral_angular_amp_smooth": cfg.get("w_spectral_angular_smooth", 0.0),
        "spectral_angular_phase_smooth": cfg.get("w_spectral_angular_smooth", 0.0),
    }
    total = torch.zeros((), device=device)
    weighted_terms = {}
    for key, value in terms.items():
        value = value.to(device)
        weight = float(weights.get(key, 0.0))
        weighted_terms[key] = value
        if weight:
            total = total + weight * value
    return total, weighted_terms


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def gradient_l1_per_sample(pred, target):
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return mean_per_sample(torch.abs(pred_dx - target_dx)) + mean_per_sample(torch.abs(pred_dy - target_dy))


def intensity_mean_norm_smooth_l1_per_sample(pred_int, target_int, eps=1e-6):
    scale = target_int.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return mean_per_sample(F.smooth_l1_loss(pred_int / scale, target_int / scale, reduction="none"))


def phase_circular_loss_per_sample(pred, target, amp_floor=0.02, weight_power=1.0, eps=1e-6):
    pred_amp = torch.sqrt(pred[:, 0:1].pow(2) + pred[:, 1:2].pow(2)).clamp_min(eps)
    target_amp = torch.sqrt(target[:, 0:1].pow(2) + target[:, 1:2].pow(2)).clamp_min(eps)
    pred_unit = pred / pred_amp
    target_unit = target / target_amp
    dot = (pred_unit * target_unit).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    phase_err = 1.0 - dot

    max_amp = target_amp.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    mask = (target_amp >= float(amp_floor) * max_amp).to(target.dtype)
    weights = mask * target_amp.pow(float(weight_power))
    numerator = (weights * phase_err).flatten(start_dim=1).sum(dim=1)
    denominator = weights.flatten(start_dim=1).sum(dim=1).clamp_min(eps)
    return numerator / denominator


def spectrum_logmag_l1_per_sample(pred, target):
    pred_c = torch.complex(pred[:, 0], pred[:, 1])
    target_c = torch.complex(target[:, 0], target[:, 1])
    pred_mag = torch.log1p(torch.abs(torch.fft.fft2(pred_c, norm="ortho"))).unsqueeze(1)
    target_mag = torch.log1p(torch.abs(torch.fft.fft2(target_c, norm="ortho"))).unsqueeze(1)
    return mean_per_sample(torch.abs(per_image_standardize(pred_mag) - per_image_standardize(target_mag)))


def _field_fft(field):
    field_c = torch.complex(field[:, 0], field[:, 1])
    return torch.fft.fft2(field_c, norm="ortho")


def _weighted_standardize_spectrum(values, weight, eps=1e-6):
    weight = weight.unsqueeze(0)
    denominator = weight.sum().clamp_min(eps)
    mean = (values * weight).sum(dim=(-2, -1), keepdim=True) / denominator
    variance = ((values - mean).pow(2) * weight).sum(dim=(-2, -1), keepdim=True) / denominator
    return (values - mean) / torch.sqrt(variance + eps)


def effective_spectrum_complex_nmse_per_sample(pred, target, cfg, eps=1e-8):
    pred_fft = _field_fft(pred)
    target_fft = _field_fft(target)
    weight = effective_axicon_spectrum_weight(
        pred.shape[-2], pred.shape[-1], cfg, pred.device, pred.dtype, rfft=False
    ).unsqueeze(0)
    numerator = (weight * (pred_fft - target_fft).abs().pow(2)).sum(dim=(-2, -1))
    denominator = (weight * target_fft.abs().pow(2)).sum(dim=(-2, -1)).clamp_min(eps)
    return numerator / denominator


def effective_spectrum_logmag_l1_per_sample(pred, target, cfg, eps=1e-6):
    pred_mag = torch.log1p(_field_fft(pred).abs())
    target_mag = torch.log1p(_field_fft(target).abs())
    weight = effective_axicon_spectrum_weight(
        pred.shape[-2], pred.shape[-1], cfg, pred.device, pred.dtype, rfft=False
    )
    pred_norm = _weighted_standardize_spectrum(pred_mag, weight, eps=eps)
    target_norm = _weighted_standardize_spectrum(target_mag, weight, eps=eps)
    weighted_error = weight.unsqueeze(0) * torch.abs(pred_norm - target_norm)
    return weighted_error.sum(dim=(-2, -1)) / weight.sum().clamp_min(eps)


def effective_spectrum_energy_fraction_per_sample(field, cfg, eps=1e-8):
    spectrum_energy = _field_fft(field).abs().pow(2)
    weight = effective_axicon_spectrum_weight(
        field.shape[-2], field.shape[-1], cfg, field.device, field.dtype, rfft=False
    ).unsqueeze(0)
    selected = (weight * spectrum_energy).sum(dim=(-2, -1))
    total = spectrum_energy.sum(dim=(-2, -1)).clamp_min(eps)
    return selected / total


def synthetic_field_loss_per_sample(pred, target, cfg):
    pred_int = field_intensity(pred)
    target_int = field_intensity(target)
    pred_amp = torch.sqrt(pred_int.clamp_min(0.0))
    target_amp = torch.sqrt(target_int.clamp_min(0.0))

    zeros = pred.new_zeros(pred.shape[0])
    complex_mse = mean_per_sample((pred - target).pow(2))
    complex_smooth_l1 = mean_per_sample(F.smooth_l1_loss(pred, target, reduction="none"))
    amp_smooth_l1 = mean_per_sample(F.smooth_l1_loss(pred_amp, target_amp, reduction="none"))
    intensity_mean_norm_l1 = intensity_mean_norm_smooth_l1_per_sample(pred_int, target_int)
    phase_circular = phase_circular_loss_per_sample(
        pred,
        target,
        amp_floor=cfg["phase_loss_amp_floor"],
        weight_power=cfg["phase_loss_weight_power"],
    )
    grad_l1 = gradient_l1_per_sample(pred, target) if cfg["w_field_grad"] > 0 else zeros
    spectrum = spectrum_logmag_l1_per_sample(pred, target) if cfg["w_spectrum"] > 0 else zeros
    effective_spectrum_complex = (
        effective_spectrum_complex_nmse_per_sample(pred, target, cfg)
        if cfg["w_effective_spectrum_complex"] > 0 else zeros
    )
    effective_spectrum_logmag = (
        effective_spectrum_logmag_l1_per_sample(pred, target, cfg)
        if cfg["w_effective_spectrum_logmag"] > 0 else zeros
    )

    total = (
        cfg["w_complex_mse"] * complex_mse
        + cfg["w_complex_smooth_l1"] * complex_smooth_l1
        + cfg["w_amp_smooth_l1"] * amp_smooth_l1
        + cfg["w_intensity_mean_norm_l1"] * intensity_mean_norm_l1
        + cfg["w_phase_circular"] * phase_circular
        + cfg["w_field_grad"] * grad_l1
        + cfg["w_spectrum"] * spectrum
        + cfg["w_effective_spectrum_complex"] * effective_spectrum_complex
        + cfg["w_effective_spectrum_logmag"] * effective_spectrum_logmag
    )
    return total, {
        "complex_mse": complex_mse,
        "complex_smooth_l1": complex_smooth_l1,
        "amp_smooth_l1": amp_smooth_l1,
        "intensity_mean_norm_l1": intensity_mean_norm_l1,
        "phase_circular": phase_circular,
        "grad_l1": grad_l1,
        "spectrum": spectrum,
        "effective_spectrum_complex_nmse": effective_spectrum_complex,
        "effective_spectrum_logmag_l1": effective_spectrum_logmag,
        "pred_intensity_mse": mean_per_sample((pred_int - target_int).pow(2)),
    }


def stage2_camera_loss_per_sample(pred_field, target_field, camera, cfg):
    pred_int = field_intensity(pred_field).clamp_min(0.0)
    camera_loss, camera_comps = visual_loss_per_sample(
        pred_int,
        camera,
        w_log_display_smooth_l1=cfg["stage2_w_log_display_smooth_l1"],
        w_ssim=cfg["stage2_w_ssim"],
        w_grad=cfg["stage2_w_grad"],
        w_fft=cfg["stage2_w_fft"],
        w_mean_norm_l1=cfg["stage2_w_mean_norm_l1"],
        w_peak=cfg["stage2_w_peak"],
        w_dark=cfg["stage2_w_dark"],
        peak_margin=cfg["stage2_peak_margin"],
        peak_top_fraction=cfg["stage2_peak_top_fraction"],
        dark_margin=cfg["stage2_dark_margin"],
        dark_top_fraction=cfg["stage2_dark_top_fraction"],
    )

    comps = {f"camera_{k}": v for k, v in camera_comps.items()}
    total = camera_loss
    if cfg["stage2_field_anchor_weight"] > 0:
        anchor, anchor_comps = synthetic_field_loss_per_sample(pred_field, target_field, cfg)
        total = total + cfg["stage2_field_anchor_weight"] * anchor
        comps["field_anchor"] = anchor
        comps["anchor_complex_mse"] = anchor_comps["complex_mse"]
        comps["anchor_phase_circular"] = anchor_comps["phase_circular"]
    else:
        comps["field_anchor"] = pred_field.new_zeros(pred_field.shape[0])
    comps["camera_loss"] = camera_loss
    return total, comps


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def radial_profile_batch(x, n_bins=64, eps=1e-6):
    batch, _, height, width = x.shape
    yy = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
    y, xcoord = torch.meshgrid(yy, xx, indexing="ij")
    r = torch.sqrt(xcoord.pow(2) + y.pow(2)).clamp_max(math.sqrt(2.0))
    bins = torch.clamp((r / math.sqrt(2.0) * n_bins).long(), max=n_bins - 1)
    flat_bins = bins.reshape(-1)
    counts = torch.bincount(flat_bins, minlength=n_bins).to(dtype=x.dtype).clamp_min(eps)
    values = x.reshape(batch, -1)
    profiles = []
    for b in range(batch):
        sums = torch.zeros(n_bins, device=x.device, dtype=x.dtype)
        sums.scatter_add_(0, flat_bins, values[b])
        profiles.append(sums / counts)
    return torch.stack(profiles, dim=0)


def field_metrics_batch(pred, target, phase_amp_floor=0.02, radial_bins=64, cfg=None, eps=1e-8):
    pred_c = torch.complex(pred[:, 0], pred[:, 1])
    target_c = torch.complex(target[:, 0], target[:, 1])
    pred_int = field_intensity(pred)
    target_int = field_intensity(target)
    target_energy = target_c.abs().pow(2).flatten(start_dim=1).sum(dim=1).clamp_min(eps)
    pred_energy = pred_c.abs().pow(2).flatten(start_dim=1).sum(dim=1).clamp_min(eps)

    complex_mse = mean_per_sample((pred - target).pow(2))
    complex_nmse = ((pred_c - target_c).abs().pow(2).flatten(start_dim=1).sum(dim=1) / target_energy)

    inner = (torch.conj(target_c) * pred_c).flatten(start_dim=1).sum(dim=1)
    overlap = inner.abs().pow(2) / (target_energy * pred_energy)
    rot = torch.conj(inner) / inner.abs().clamp_min(eps)
    pred_aligned = pred_c * rot.view(-1, 1, 1)
    aligned_nmse = ((pred_aligned - target_c).abs().pow(2).flatten(start_dim=1).sum(dim=1) / target_energy)

    pred_phase = torch.atan2(pred[:, 1:2], pred[:, 0:1])
    target_phase = torch.atan2(target[:, 1:2], target[:, 0:1])
    dphi = torch.atan2(torch.sin(pred_phase - target_phase), torch.cos(pred_phase - target_phase))
    target_amp = torch.sqrt(target_int.clamp_min(0.0))
    max_amp = target_amp.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    weights = (target_amp >= phase_amp_floor * max_amp).to(target.dtype) * target_amp
    wsum = weights.flatten(start_dim=1).sum(dim=1).clamp_min(eps)
    phase_mae = (weights * dphi.abs()).flatten(start_dim=1).sum(dim=1) / wsum
    phase_rmse = torch.sqrt((weights * dphi.pow(2)).flatten(start_dim=1).sum(dim=1) / wsum)

    amp_mse = mean_per_sample((torch.sqrt(pred_int.clamp_min(0.0)) - torch.sqrt(target_int.clamp_min(0.0))).pow(2))
    intensity_mse = mean_per_sample((pred_int - target_int).pow(2))
    ssim = 1.0 - simple_ssim_loss_per_sample(pred_int, target_int)

    pred_prof = radial_profile_batch(pred_int, n_bins=radial_bins)
    target_prof = radial_profile_batch(target_int, n_bins=radial_bins)
    scale = target_prof.mean(dim=1, keepdim=True).clamp_min(eps)
    radial_profile_mse = ((pred_prof / scale - target_prof / scale).pow(2)).mean(dim=1)

    spectrum = spectrum_logmag_l1_per_sample(pred, target)
    metrics = {
        "complex_mse": complex_mse,
        "complex_nmse": complex_nmse,
        "global_phase_aligned_nmse": aligned_nmse,
        "coherent_overlap": overlap,
        "amp_mse": amp_mse,
        "intensity_mse": intensity_mse,
        "intensity_ssim": ssim,
        "phase_mae_rad": phase_mae,
        "phase_rmse_rad": phase_rmse,
        "radial_profile_mse": radial_profile_mse,
        "fft_logmag_l1": spectrum,
    }
    if cfg is not None:
        metrics.update({
            "effective_spectrum_complex_nmse": effective_spectrum_complex_nmse_per_sample(
                pred, target, cfg, eps=eps
            ),
            "effective_spectrum_logmag_l1": effective_spectrum_logmag_l1_per_sample(
                pred, target, cfg
            ),
            "target_effective_spectrum_energy_fraction": effective_spectrum_energy_fraction_per_sample(
                target, cfg, eps=eps
            ),
            "pred_effective_spectrum_energy_fraction": effective_spectrum_energy_fraction_per_sample(
                pred, cfg, eps=eps
            ),
        })
    return metrics


def camera_metrics_batch(pred_field, camera, radial_bins=64, eps=1e-8):
    pred_int = field_intensity(pred_field).clamp_min(0.0)
    raw_mse = mean_per_sample((pred_int - camera).pow(2))
    ssim = 1.0 - simple_ssim_loss_per_sample(pred_int, camera)
    pred_prof = radial_profile_batch(pred_int, n_bins=radial_bins)
    cam_prof = radial_profile_batch(camera, n_bins=radial_bins)
    scale = cam_prof.mean(dim=1, keepdim=True).clamp_min(eps)
    radial_mse = ((pred_prof / scale - cam_prof / scale).pow(2)).mean(dim=1)
    return {
        "camera_raw_mse": raw_mse,
        "camera_ssim": ssim,
        "camera_radial_profile_mse": radial_mse,
    }


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def batch_to_device(batch, device):
    out = {
        "phase": batch["phase"].to(device, non_blocking=True),
        "field": batch["field"].to(device, non_blocking=True),
        "z_mm": batch["z_mm"].to(device, non_blocking=True),
        "id": batch["id"],
    }
    if "camera" in batch:
        out["camera"] = batch["camera"].to(device, non_blocking=True)
    return out


def normalized_group_weights(dataset, train_indices, raw_weights, enabled):
    names = ("systematic", "real", "pert", "other")
    counts = {name: 0 for name in names}
    for idx in train_indices:
        group = sample_type_from_id(dataset.samples[int(idx)]["id"])
        counts[group if group in counts else "other"] += 1

    weights = {name: float(raw_weights.get(name, 1.0)) if enabled else 1.0 for name in names}
    total = sum(counts.values())
    mean_weight = sum(counts[name] * weights[name] for name in names) / max(total, 1)
    norm = {name: weights[name] / max(mean_weight, 1e-8) for name in names}
    print(">>> Group weights:")
    for name in names:
        print(f">>>   {name}: count={counts[name]} raw={weights[name]:.4g} normalized={norm[name]:.4g}")
    return norm, counts


def run_one_epoch(model, loader, optimizer, device, cfg, stage, train):
    model.train(train)
    amp_enabled = bool(cfg["use_amp"] and device.type == "cuda")
    scaler = cfg.get("_scaler")
    totals = {"loss": 0.0}
    n_seen = 0
    iterator = tqdm(loader, desc=f"{stage} {'train' if train else 'val'}", leave=False)

    for batch in iterator:
        batch = batch_to_device(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                x = build_phase_fno_input(batch["phase"], batch["z_mm"], cfg)
                pred_field = model(x)
                if stage == "stage1":
                    per_sample_loss, comps = synthetic_field_loss_per_sample(pred_field, batch["field"], cfg)
                elif stage == "stage2":
                    if "camera" not in batch:
                        raise RuntimeError("Stage 2 requires camera targets.")
                    per_sample_loss, comps = stage2_camera_loss_per_sample(
                        pred_field, batch["field"], batch["camera"], cfg
                    )
                else:
                    raise ValueError(f"Unknown stage: {stage}")

                weights = pred_field.new_ones(pred_field.shape[0])
                if cfg["use_group_loss_weights"]:
                    weights = pred_field.new_tensor([
                        cfg["group_loss_weights_normalized"].get(sample_type_from_id(sid), 1.0)
                        for sid in batch["id"]
                    ])
                loss = (weights * per_sample_loss).mean()
                reg_loss, reg_terms = model_regularization_loss(model, cfg, pred_field.device)
                loss = loss + reg_loss
                if reg_terms:
                    comps = dict(comps)
                    comps["model_regularization"] = reg_loss.expand(pred_field.shape[0])
                    for key, value in reg_terms.items():
                        comps[key] = value.expand(pred_field.shape[0])

        if train:
            if amp_enabled:
                scaler.scale(loss).backward()
                if cfg["grad_clip"] is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg["grad_clip"] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()

        batch_size = pred_field.shape[0]
        n_seen += batch_size
        totals["loss"] += float(loss.detach().item()) * batch_size
        for key, value in comps.items():
            totals.setdefault(key, 0.0)
            totals[key] += float(value.detach().mean().item()) * batch_size
        iterator.set_postfix(loss=f"{loss.detach().item():.4g}")

    return {key: value / max(n_seen, 1) for key, value in totals.items()}


def train_stage(model, train_loader, val_loader, device, cfg, stage, stage_dir, lr, epochs):
    stage_dir = ensure_dir(stage_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    cfg["_scaler"] = torch.amp.GradScaler("cuda", enabled=bool(cfg["use_amp"] and device.type == "cuda"))
    history = {"train": [], "val": []}
    best_val = float("inf")

    for epoch in range(int(epochs)):
        train_stats = run_one_epoch(model, train_loader, optimizer, device, cfg, stage, train=True)
        val_stats = run_one_epoch(model, val_loader, optimizer, device, cfg, stage, train=False)
        history["train"].append(train_stats)
        history["val"].append(val_stats)

        print(
            f"[{stage}] epoch {epoch + 1:03d}/{epochs} "
            f"train={train_stats['loss']:.6g} val={val_stats['loss']:.6g}"
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(stage_dir / "best.pt", model, optimizer, cfg, stage, epoch + 1, best_val)
        save_checkpoint(stage_dir / "last.pt", model, optimizer, cfg, stage, epoch + 1, val_stats["loss"])

        with open(stage_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        plot_history(history, stage_dir / "loss_curve.png", stage)

    ckpt = torch.load(stage_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    save_model_diagnostics(model, stage_dir / "model_diagnostics.png")
    return history


def save_checkpoint(path, model, optimizer, cfg, stage, epoch, val_loss):
    dump_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "cfg": dump_cfg,
        "stage": stage,
        "epoch": int(epoch),
        "val_loss": float(val_loss),
    }, path)


def load_model_from_checkpoint(model, checkpoint, device, strict=True):
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    return ckpt


def evaluate(model, loader, dataset, indices, device, cfg, stage, out_dir):
    model.eval()
    rows = []
    index_iter = iter(indices)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {stage}"):
            current_indices = [next(index_iter) for _ in batch["id"]]
            b = batch_to_device(batch, device)
            x = build_phase_fno_input(b["phase"], b["z_mm"], cfg)
            pred = model(x)
            metrics = field_metrics_batch(
                pred,
                b["field"],
                phase_amp_floor=cfg["phase_loss_amp_floor"],
                radial_bins=cfg["radial_bins"],
                cfg=cfg,
            )
            if stage == "stage2" and "camera" in b:
                metrics.update(camera_metrics_batch(pred, b["camera"], radial_bins=cfg["radial_bins"]))
            metrics_cpu = {k: v.detach().cpu().numpy() for k, v in metrics.items()}
            for j, sample_id in enumerate(batch["id"]):
                sample = dataset.samples[int(current_indices[j])]
                row = {
                    "idx": int(current_indices[j]),
                    "id": sample_id,
                    "split": "eval",
                    "group": sample_type_from_id(sample_id),
                    "root_tag": sample["root_tag"],
                    "z_mm": float(sample["z_mm"]),
                }
                for key, values in metrics_cpu.items():
                    row[key] = float(values[j])
                rows.append(row)

    out_dir = ensure_dir(out_dir)
    csv_path = out_dir / f"{stage}_per_sample_metrics.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    write_metric_summary(rows, out_dir / f"{stage}_metric_summary.json")
    plot_metric_summary(rows, out_dir / f"{stage}_metrics.png", stage)
    return rows


def write_metric_summary(rows, path):
    summary = {}
    if rows:
        numeric_keys = [
            k for k, v in rows[0].items()
            if isinstance(v, (int, float)) and k not in {"idx", "z_mm"}
        ]
        for key in numeric_keys:
            vals = np.array([float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))])
            if vals.size:
                summary[key] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "p90": float(np.percentile(vals, 90)),
                    "max": float(np.max(vals)),
                }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Plots / previews
# ---------------------------------------------------------------------------

def plot_history(history, path, stage):
    if not history.get("train"):
        return
    train_loss = [row["loss"] for row in history["train"]]
    val_loss = [row["loss"] for row in history["val"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_loss, label="train")
    ax.plot(val_loss, label="val")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"{stage} loss")
    ax.grid(True, alpha=0.25)
    apply_plain_decimal_yaxis(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    keys = sorted(k for k in history["val"][-1].keys() if k != "loss")
    if keys:
        ncols = 3
        nrows = int(math.ceil(len(keys) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows))
        axes = np.asarray(axes).reshape(-1)
        for ax, key in zip(axes, keys):
            train_vals = [row.get(key, np.nan) for row in history["train"]]
            val_vals = [row.get(key, np.nan) for row in history["val"]]
            ax.plot(train_vals, label="train")
            ax.plot(val_vals, label="val")
            ax.set_title(key)
            finite_vals = np.array(train_vals + val_vals, dtype=np.float64)
            finite_vals = finite_vals[np.isfinite(finite_vals)]
            if finite_vals.size and np.all(finite_vals > 0):
                ax.set_yscale("log")
            apply_plain_decimal_yaxis(ax)
            ax.grid(True, alpha=0.25)
        for ax in axes[len(keys):]:
            ax.axis("off")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(path).with_name("loss_components.png"), dpi=150)
        plt.close(fig)


def plot_metric_summary(rows, path, stage):
    if not rows:
        return
    candidate_keys = [
        "complex_nmse",
        "global_phase_aligned_nmse",
        "coherent_overlap",
        "phase_rmse_rad",
        "intensity_ssim",
        "effective_spectrum_complex_nmse",
        "effective_spectrum_logmag_l1",
        "target_effective_spectrum_energy_fraction",
        "pred_effective_spectrum_energy_fraction",
        "camera_raw_mse",
        "camera_ssim",
    ]
    keys = [k for k in candidate_keys if k in rows[0]]
    if not keys:
        return
    ncols = 2
    nrows = int(math.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows))
    axes = np.asarray(axes).reshape(-1)
    x = np.arange(len(rows))
    colors = {
        "systematic": "tab:blue",
        "real": "tab:green",
        "pert": "tab:orange",
        "other": "tab:gray",
    }
    groups = [row["group"] for row in rows]
    for ax, key in zip(axes, keys):
        vals = np.array([float(row[key]) for row in rows])
        for group in sorted(set(groups)):
            mask = np.array([g == group for g in groups])
            ax.scatter(x[mask], vals[mask], s=10, alpha=0.7, label=group, color=colors.get(group, "tab:gray"))
        ax.set_title(key)
        ax.grid(True, alpha=0.25)
        finite_vals = vals[np.isfinite(vals)]
        if (
            key not in {"coherent_overlap", "intensity_ssim", "camera_ssim"}
            and finite_vals.size
            and np.all(finite_vals > 0)
        ):
            ax.set_yscale("log")
        apply_plain_decimal_yaxis(ax)
    for ax in axes[len(keys):]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{stage} per-sample metrics", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def standard_fno_effective_support_coverage(cfg):
    size = int(cfg["model_size"])
    weight = effective_axicon_spectrum_weight(
        size, size, cfg, torch.device("cpu"), torch.float32, rfft=True
    )
    selected = torch.zeros_like(weight, dtype=torch.bool)
    my = min(int(cfg.get("modes_y", 0)), size)
    mx = min(int(cfg.get("modes_x", 0)), size // 2 + 1)
    if my > 0 and mx > 0:
        selected[:my, :mx] = True
        selected[-my:, :mx] = True
    return float((weight * selected).sum() / weight.sum().clamp_min(1e-8))


def save_effective_spectrum_diagnostic(cfg, path):
    size = int(cfg["model_size"])
    geometry = effective_spectrum_geometry(cfg, height=size, width=size)
    weight = effective_axicon_spectrum_weight(
        size, size, cfg, torch.device("cpu"), torch.float32, rfft=False
    ).numpy()
    threshold = float(cfg["effective_spectrum_support_threshold"])
    active_fraction = float(np.mean(weight >= threshold))
    fno_coverage = standard_fno_effective_support_coverage(cfg)

    shifted = np.fft.fftshift(weight)
    fx = np.fft.fftshift(np.fft.fftfreq(size, d=geometry["field_dx_m"])) / 1000.0
    fy = np.fft.fftshift(np.fft.fftfreq(size, d=geometry["field_dy_m"])) / 1000.0
    center = size // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    im = axes[0].imshow(
        shifted,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        extent=[fx[0], fx[-1], fy[0], fy[-1]],
        origin="lower",
    )
    axes[0].set_title("Convolution-expanded axicon support")
    axes[0].set_xlabel("fx (kcyc/m)")
    axes[0].set_ylabel("fy (kcyc/m)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)

    axes[1].plot(fx, shifted[center], label="fy = 0")
    axes[1].axvline(geometry["axicon_ring_cyc_per_m"] / 1000.0, color="tab:green", ls="--", label="+f_r")
    axes[1].axvline(-geometry["axicon_ring_cyc_per_m"] / 1000.0, color="tab:green", ls="--", label="-f_r")
    axes[1].set_xlabel("fx (kcyc/m)")
    axes[1].set_ylabel("support weight")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("Central spectral cross-section")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    axes[2].axis("off")
    lines = [
        f"shape: {cfg['effective_spectrum_shape']}",
        f"axicon pitch: {cfg['axicon_grating_pitch_m'] * 1e6:.4f} um",
        f"f_r: {geometry['axicon_ring_cyc_per_m'] / 1000.0:.2f} kcyc/m",
        f"GS pixel: {geometry['gs_effective_pixel_size_m'] * 1e6:.4f} um",
        f"f_N: {geometry['gs_nyquist_cyc_per_m'] / 1000.0:.2f} kcyc/m",
        f"axis B/f_r: {100.0 * geometry['axis_halfwidth_over_ring']:.2f}%",
        f"corner B/f_r: {100.0 * geometry['corner_halfwidth_over_ring']:.2f}%",
        f"ring radius: {geometry['ring_bin_x']:.2f} FFT bins",
        f"active grid: {100.0 * active_fraction:.2f}%",
        f"legacy FNO weighted coverage: {100.0 * fno_coverage:.2f}%",
    ]
    axes[2].text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=10)
    axes[2].set_title("Physical support summary")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_model_diagnostics(model, path):
    if not hasattr(model, "diagnostic_arrays"):
        return
    arrays = model.diagnostic_arrays()
    amp = arrays["transfer_amp"]
    phase = arrays["transfer_phase"]
    radial_log_amp = arrays["radial_log_amp"]
    radial_phase = arrays["radial_phase"]
    angular_orders = arrays.get("angular_orders", np.array([], dtype=np.int64))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.reshape(-1)
    axes[0].plot(np.arange(radial_log_amp.size), radial_log_amp)
    axes[0].set_title("H_eff radial log-amplitude parameter")
    axes[0].set_xlabel("radial bin")
    axes[0].grid(True, alpha=0.25)
    apply_plain_decimal_yaxis(axes[0])

    axes[1].plot(np.arange(radial_phase.size), radial_phase)
    axes[1].set_title("H_eff radial phase parameter")
    axes[1].set_xlabel("radial bin")
    axes[1].grid(True, alpha=0.25)
    apply_plain_decimal_yaxis(axes[1])

    im_amp = axes[2].imshow(quantile_display(amp), cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("|H_eff(fx, fy)|")
    axes[2].axis("off")
    fig.colorbar(im_amp, ax=axes[2], fraction=0.035, pad=0.02)

    im_phase = axes[3].imshow(phase, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi)
    axes[3].set_title("angle H_eff(fx, fy)")
    axes[3].axis("off")
    cbar = fig.colorbar(im_phase, ax=axes[3], fraction=0.035, pad=0.02)
    cbar.set_ticks([-np.pi, 0.0, np.pi])
    cbar.set_ticklabels(["-3.14", "0", "3.14"])

    if angular_orders.size:
        fig.suptitle(f"H_eff angular orders: {', '.join(str(int(v)) for v in angular_orders)}", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_previews(model, dataset, indices, device, cfg, stage, out_dir):
    out_dir = ensure_dir(out_dir)
    model.eval()
    for idx in indices:
        sample = dataset[int(idx)]
        with torch.no_grad():
            phase = sample["phase"].unsqueeze(0).to(device)
            z_mm = sample["z_mm"].view(1).to(device)
            target = sample["field"].unsqueeze(0).to(device)
            pred = model(build_phase_fno_input(phase, z_mm, cfg))

        phase_np = resize_wrapped_phase_for_display(phase, cfg["model_size"])[0, 0].detach().cpu().numpy()
        phase_np = np.mod(phase_np, 2.0 * np.pi)
        target_np = target[0].detach().cpu().numpy()
        pred_np = pred[0].detach().cpu().numpy()
        target_int = (target_np[0] ** 2 + target_np[1] ** 2)
        pred_int = (pred_np[0] ** 2 + pred_np[1] ** 2)
        target_amp = np.sqrt(np.maximum(target_int, 0.0))
        pred_amp = np.sqrt(np.maximum(pred_int, 0.0))
        target_phase = angle_display(target_np[0], target_np[1])
        pred_phase = angle_display(pred_np[0], pred_np[1])
        phase_err = np.angle(np.exp(1j * (pred_phase - target_phase))).astype(np.float32)

        if stage == "stage2" and "camera" in sample:
            camera_np = sample["camera"][0].numpy()
            fig, axes = plt.subplots(2, 4, figsize=(15, 7))
            panels = [
                (phase_np, "SLM phase", "twilight", "slm_phase"),
                (camera_np, "Camera", "magma", "scalar"),
                (pred_int, "Pred |E|^2", "magma", "scalar"),
                (np.abs(pred_int - camera_np), "|Pred-camera|", "inferno", "scalar"),
                (target_amp, "Synthetic amp", "viridis", "scalar"),
                (pred_amp, "Pred amp", "viridis", "scalar"),
                (target_phase, "Synthetic phase", "twilight_shifted", "field_phase"),
                (pred_phase, "Pred phase", "twilight_shifted", "field_phase"),
            ]
        else:
            fig, axes = plt.subplots(3, 3, figsize=(12, 10))
            panels = [
                (phase_np, "SLM phase", "twilight", "slm_phase"),
                (target_amp, "Target amp", "viridis", "scalar"),
                (target_phase, "Target phase", "twilight_shifted", "field_phase"),
                (np.abs(pred_np[0] - target_np[0]) + np.abs(pred_np[1] - target_np[1]), "Field L1 error", "inferno", "scalar"),
                (pred_amp, "Pred amp", "viridis", "scalar"),
                (pred_phase, "Pred phase", "twilight_shifted", "field_phase"),
                None,
                (np.abs(pred_amp - target_amp), "Amp error", "inferno", "scalar"),
                (phase_err, "Phase error", "coolwarm", "phase_error"),
            ]

        for ax, panel in zip(axes.reshape(-1), panels):
            if panel is None:
                ax.axis("off")
                continue
            arr, title, cmap, kind = panel
            if kind == "slm_phase":
                im = ax.imshow(np.mod(arr, 2.0 * np.pi), cmap=cmap, vmin=0.0, vmax=2.0 * np.pi)
            elif kind in {"field_phase", "phase_error"}:
                im = ax.imshow(arr, cmap=cmap, vmin=-np.pi, vmax=np.pi)
            else:
                im = ax.imshow(quantile_display(arr), cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            if kind == "slm_phase":
                cbar.set_ticks([0.0, np.pi, 2.0 * np.pi])
                cbar.set_ticklabels(["0", "3.14", "6.28"])
            elif kind in {"field_phase", "phase_error"}:
                cbar.set_ticks([-np.pi, 0.0, np.pi])
                cbar.set_ticklabels(["-3.14", "0", "3.14"])
        for ax in axes.reshape(-1)[len(panels):]:
            ax.axis("off")
        fig.suptitle(f"{sample['id']}  z={float(sample['z_mm']):.3g} mm", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"{sample['id']}.png", dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Two-stage SLM phase -> electric field FNO trainer for axicon propagation."
    )
    parser.add_argument("--root-dir", type=Path, nargs="+", default=[Path(DEFAULT_ROOT)])
    parser.add_argument("--output-dir", type=Path, default=Path("runs_fno_slm_to_field"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--stage", choices=["stage1", "stage2", "both", "eval_stage1", "eval_stage2"], default="both")
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--stage2-checkpoint", type=Path, default=None)

    parser.add_argument("--phase-dir", type=str, default="0.Phase_Masks")
    parser.add_argument("--field-dir", type=str, default="1.Forward_Sim")
    parser.add_argument("--camera-dir", type=str, default="3.Aligned_Camera")
    parser.add_argument("--fov-crop-size", type=optional_int, default=608)
    parser.add_argument("--model-size", type=int, default=608)
    parser.add_argument(
        "--phase-spatial-mode",
        choices=["center_square", "stretch"],
        default="center_square",
        help=(
            "center_square preserves phase x/y scale before square resize; "
            "stretch reproduces the legacy rectangular-to-square mapping."
        ),
    )
    parser.add_argument("--phase-units", choices=["levels", "radians"], default="levels")
    parser.add_argument("--phase-level-max", type=float, default=1023.0)
    parser.add_argument("--no-phase-flip-lr", action="store_true")
    parser.add_argument("--phase-lowpass-size", type=optional_int, default=None)
    parser.add_argument("--z-mm", type=float, nargs="*", default=None)
    parser.add_argument("--z-ref-mm", type=float, default=10.0)
    parser.add_argument("--no-z-channel", action="store_true")
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--no-radial-coord", action="store_true")
    parser.add_argument("--no-axicon-phase", action="store_true")
    parser.add_argument("--no-slm-axicon-product", action="store_true")
    parser.add_argument("--no-axicon-radius", action="store_true")
    parser.add_argument("--use-axicon-aperture", action="store_true")
    parser.add_argument("--axicon-map-scope", choices=["roi", "slm"], default="roi")
    parser.add_argument("--axicon-roi-size", type=int, default=1024)
    parser.add_argument("--slm-nx", type=int, default=1600)
    parser.add_argument("--slm-ny", type=int, default=1200)
    parser.add_argument("--slm-pixel-size-m", type=float, default=6.4e-6)
    parser.add_argument("--axicon-upsample-factor", type=int, default=20)
    parser.add_argument("--axicon-grating-pitch-m", type=float, default=1.396e-6)
    parser.add_argument("--axicon-transverse-frequency", type=float, default=0.0)
    parser.add_argument("--axicon-wavelength-m", type=float, default=0.473e-6)
    parser.add_argument("--axicon-medium-index", type=float, default=1.471)
    parser.add_argument("--axicon-phase-sign", type=float, default=-1.0)
    parser.add_argument("--axicon-gaussian-waist-m", type=float, default=0.00638708 * 0.8)
    parser.add_argument(
        "--gs-effective-pixel-size-m",
        type=float,
        default=None,
        help="Effective GS-mask pixel pitch. Defaults to --slm-pixel-size-m.",
    )
    parser.add_argument(
        "--gs-bandwidth-nyquist-fraction",
        type=float,
        default=1.0,
        help="GS phasor bandwidth B/f_N along each square-grid axis.",
    )
    parser.add_argument(
        "--effective-spectrum-shape",
        choices=["square_minkowski", "radial_envelope"],
        default="square_minkowski",
        help="Exact ring (+) square support or its conservative radial envelope.",
    )
    parser.add_argument(
        "--effective-spectrum-soft-edge-fn",
        type=float,
        default=0.05,
        help="Support-edge transition width as a fraction of GS f_N.",
    )
    parser.add_argument(
        "--effective-spectrum-support-threshold",
        type=float,
        default=0.01,
        help="Minimum soft support weight retained by annular_fno.",
    )

    parser.add_argument("--field-scale-mode", choices=["raw", "sample_norm", "global_percentile"], default="global_percentile")
    parser.add_argument("--field-amp-percentile", type=float, default=99.9)
    parser.add_argument("--field-amp-scale", type=float, default=None)
    parser.add_argument("--camera-scale-mode", choices=["raw", "sample_norm", "global_percentile"], default="global_percentile")
    parser.add_argument("--camera-percentile", type=float, default=99.9)
    parser.add_argument("--camera-scale", type=float, default=None)
    parser.add_argument("--camera-black-level", type=float, default=0.0)
    parser.add_argument("--scale-sample-pixels", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--modes-y", type=int, default=96)
    parser.add_argument("--modes-x", type=int, default=96)
    parser.add_argument("--mlp-width", type=int, default=128)
    parser.add_argument("--operator-mode", choices=["fno", "annular_fno", "spectral"], default="fno")
    parser.add_argument(
        "--annular-fno-low-modes",
        type=int,
        default=0,
        help="Optional legacy low-frequency square retained alongside the annular support.",
    )
    parser.add_argument("--spectral-width", type=int, default=16)
    parser.add_argument("--spectral-depth", type=int, default=2)
    parser.add_argument("--spectral-radial-bins", type=int, default=256)
    parser.add_argument("--spectral-transfer-amp-scale", type=float, default=2.0)
    parser.add_argument("--spectral-angular-orders", type=str, default="")
    parser.add_argument("--spectral-angular-amp-scale", type=float, default=0.25)
    parser.add_argument("--spectral-angular-phase-scale", type=float, default=0.50)
    parser.add_argument("--spectral-pupil-phase-scale", type=float, default=0.25)
    parser.add_argument("--spectral-pupil-amp-scale", type=float, default=0.10)
    parser.add_argument("--spectral-residual-scale", type=float, default=0.10)
    parser.add_argument("--spectral-support-radius", type=float, default=1.0)
    parser.add_argument("--spectral-support-softness", type=float, default=0.05)

    parser.add_argument("--stage1-epochs", type=int, default=120)
    parser.add_argument("--stage2-epochs", type=int, default=60)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-train-ratio", type=float, default=0.8)
    parser.add_argument("--sys-train-ratio", type=float, default=0.9)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--use-group-loss-weights", action="store_true")
    parser.add_argument("--weight-systematic", type=float, default=1.0)
    parser.add_argument("--weight-real", type=float, default=1.0)
    parser.add_argument("--weight-pert", type=float, default=1.0)
    parser.add_argument("--weight-other", type=float, default=1.0)

    parser.add_argument("--w-complex-mse", type=float, default=0.20)
    parser.add_argument("--w-complex-smooth-l1", type=float, default=1.00)
    parser.add_argument("--w-amp-smooth-l1", type=float, default=0.30)
    parser.add_argument("--w-intensity-mean-norm-l1", type=float, default=0.20)
    parser.add_argument("--w-phase-circular", type=float, default=0.30)
    parser.add_argument("--w-field-grad", type=float, default=0.05)
    parser.add_argument("--w-spectrum", type=float, default=0.05)
    parser.add_argument("--w-effective-spectrum-complex", type=float, default=0.0)
    parser.add_argument("--w-effective-spectrum-logmag", type=float, default=0.0)
    parser.add_argument("--w-spectral-amp-smooth", type=float, default=1e-4)
    parser.add_argument("--w-spectral-phase-smooth", type=float, default=1e-4)
    parser.add_argument("--w-spectral-amp-l2", type=float, default=1e-5)
    parser.add_argument("--w-spectral-gain-l2", type=float, default=1e-5)
    parser.add_argument("--w-spectral-angular-smooth", type=float, default=1e-4)
    parser.add_argument("--w-spectral-angular-l2", type=float, default=1e-4)
    parser.add_argument("--phase-loss-amp-floor", type=float, default=0.02)
    parser.add_argument("--phase-loss-weight-power", type=float, default=1.0)

    parser.add_argument("--stage2-field-anchor-weight", type=float, default=0.05)
    parser.add_argument("--stage2-w-log-display-smooth-l1", type=float, default=0.0)
    parser.add_argument("--stage2-w-ssim", type=float, default=1.0)
    parser.add_argument("--stage2-w-grad", type=float, default=0.10)
    parser.add_argument("--stage2-w-fft", type=float, default=0.0)
    parser.add_argument("--stage2-w-mean-norm-l1", type=float, default=1.0)
    parser.add_argument("--stage2-w-peak", type=float, default=0.0)
    parser.add_argument("--stage2-w-dark", type=float, default=0.0)
    parser.add_argument("--stage2-peak-margin", type=float, default=0.10)
    parser.add_argument("--stage2-peak-top-fraction", type=float, default=0.002)
    parser.add_argument("--stage2-dark-margin", type=float, default=0.10)
    parser.add_argument("--stage2-dark-top-fraction", type=float, default=0.002)

    parser.add_argument("--n-vis-train", type=int, default=16)
    parser.add_argument("--n-vis-val", type=int, default=8)
    parser.add_argument("--radial-bins", type=int, default=64)
    return parser


def config_from_args(args):
    axicon_frequency = (
        float(args.axicon_transverse_frequency)
        if float(args.axicon_transverse_frequency) > 0
        else 1.0 / float(args.axicon_grating_pitch_m)
    )
    return {
        "root_dir": [str(p) for p in args.root_dir],
        "phase_dir": args.phase_dir,
        "field_dir": args.field_dir,
        "camera_dir": args.camera_dir,
        "fov_crop_size": args.fov_crop_size,
        "model_size": args.model_size,
        "phase_spatial_mode": args.phase_spatial_mode,
        "phase_units": args.phase_units,
        "phase_level_max": args.phase_level_max,
        "phase_flip_lr": not args.no_phase_flip_lr,
        "phase_lowpass_size": args.phase_lowpass_size,
        "z_mm": args.z_mm,
        "z_ref_mm": args.z_ref_mm,
        "use_z_channel": not args.no_z_channel,
        "use_grid": not args.no_grid,
        "use_radial_coord": not args.no_radial_coord,
        "use_axicon_phase": not args.no_axicon_phase,
        "use_slm_axicon_product": not args.no_slm_axicon_product,
        "use_axicon_radius": not args.no_axicon_radius,
        "use_axicon_aperture": args.use_axicon_aperture,
        "axicon_map_scope": args.axicon_map_scope,
        "axicon_roi_size": args.axicon_roi_size,
        "slm_nx": args.slm_nx,
        "slm_ny": args.slm_ny,
        "slm_pixel_size_m": args.slm_pixel_size_m,
        "axicon_upsample_factor": args.axicon_upsample_factor,
        "axicon_grating_pitch_m": args.axicon_grating_pitch_m,
        "axicon_transverse_frequency": args.axicon_transverse_frequency,
        "axicon_transverse_frequency_actual": axicon_frequency,
        "axicon_wavelength_m": args.axicon_wavelength_m,
        "axicon_medium_index": args.axicon_medium_index,
        "axicon_na_air_equiv": float(args.axicon_wavelength_m) * axicon_frequency,
        "axicon_phase_sign": args.axicon_phase_sign,
        "axicon_gaussian_waist_m": args.axicon_gaussian_waist_m,
        "gs_effective_pixel_size_m": args.gs_effective_pixel_size_m,
        "gs_bandwidth_nyquist_fraction": args.gs_bandwidth_nyquist_fraction,
        "effective_spectrum_shape": args.effective_spectrum_shape,
        "effective_spectrum_soft_edge_fn": args.effective_spectrum_soft_edge_fn,
        "effective_spectrum_support_threshold": args.effective_spectrum_support_threshold,
        "field_scale_mode": args.field_scale_mode,
        "field_amp_percentile": args.field_amp_percentile,
        "field_amp_scale": args.field_amp_scale,
        "camera_scale_mode": args.camera_scale_mode,
        "camera_percentile": args.camera_percentile,
        "camera_scale": args.camera_scale,
        "camera_black_level": args.camera_black_level,
        "scale_sample_pixels": args.scale_sample_pixels,
        "max_samples": args.max_samples,
        "width": args.width,
        "depth": args.depth,
        "modes_y": args.modes_y,
        "modes_x": args.modes_x,
        "mlp_width": args.mlp_width,
        "operator_mode": args.operator_mode,
        "annular_fno_low_modes": args.annular_fno_low_modes,
        "spectral_width": args.spectral_width,
        "spectral_depth": args.spectral_depth,
        "spectral_radial_bins": args.spectral_radial_bins,
        "spectral_transfer_amp_scale": args.spectral_transfer_amp_scale,
        "spectral_angular_orders": parse_int_list(args.spectral_angular_orders),
        "spectral_angular_amp_scale": args.spectral_angular_amp_scale,
        "spectral_angular_phase_scale": args.spectral_angular_phase_scale,
        "spectral_pupil_phase_scale": args.spectral_pupil_phase_scale,
        "spectral_pupil_amp_scale": args.spectral_pupil_amp_scale,
        "spectral_residual_scale": args.spectral_residual_scale,
        "spectral_support_radius": args.spectral_support_radius,
        "spectral_support_softness": args.spectral_support_softness,
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "stage1_lr": args.stage1_lr,
        "stage2_lr": args.stage2_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "real_train_ratio": args.real_train_ratio,
        "sys_train_ratio": args.sys_train_ratio,
        "use_amp": args.use_amp,
        "grad_clip": args.grad_clip,
        "use_group_loss_weights": args.use_group_loss_weights,
        "group_loss_weights_raw": {
            "systematic": args.weight_systematic,
            "real": args.weight_real,
            "pert": args.weight_pert,
            "other": args.weight_other,
        },
        "w_complex_mse": args.w_complex_mse,
        "w_complex_smooth_l1": args.w_complex_smooth_l1,
        "w_amp_smooth_l1": args.w_amp_smooth_l1,
        "w_intensity_mean_norm_l1": args.w_intensity_mean_norm_l1,
        "w_phase_circular": args.w_phase_circular,
        "w_field_grad": args.w_field_grad,
        "w_spectrum": args.w_spectrum,
        "w_effective_spectrum_complex": args.w_effective_spectrum_complex,
        "w_effective_spectrum_logmag": args.w_effective_spectrum_logmag,
        "w_spectral_amp_smooth": args.w_spectral_amp_smooth,
        "w_spectral_phase_smooth": args.w_spectral_phase_smooth,
        "w_spectral_amp_l2": args.w_spectral_amp_l2,
        "w_spectral_gain_l2": args.w_spectral_gain_l2,
        "w_spectral_angular_smooth": args.w_spectral_angular_smooth,
        "w_spectral_angular_l2": args.w_spectral_angular_l2,
        "phase_loss_amp_floor": args.phase_loss_amp_floor,
        "phase_loss_weight_power": args.phase_loss_weight_power,
        "stage2_field_anchor_weight": args.stage2_field_anchor_weight,
        "stage2_w_log_display_smooth_l1": args.stage2_w_log_display_smooth_l1,
        "stage2_w_ssim": args.stage2_w_ssim,
        "stage2_w_grad": args.stage2_w_grad,
        "stage2_w_fft": args.stage2_w_fft,
        "stage2_w_mean_norm_l1": args.stage2_w_mean_norm_l1,
        "stage2_w_peak": args.stage2_w_peak,
        "stage2_w_dark": args.stage2_w_dark,
        "stage2_peak_margin": args.stage2_peak_margin,
        "stage2_peak_top_fraction": args.stage2_peak_top_fraction,
        "stage2_dark_margin": args.stage2_dark_margin,
        "stage2_dark_top_fraction": args.stage2_dark_top_fraction,
        "radial_bins": args.radial_bins,
    }


def save_split(path, dataset, train_idx, val_idx, cfg):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "split_rule": "train_fno_axicon.split_dataset with real/pert parent grouping",
            "real_train_ratio": cfg["real_train_ratio"],
            "sys_train_ratio": cfg["sys_train_ratio"],
            "train": [dataset.samples[i]["id"] for i in train_idx],
            "val": [dataset.samples[i]["id"] for i in val_idx],
        }, f, indent=2)


def make_loaders(dataset, train_idx, val_idx, cfg, device):
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        Subset(dataset, train_idx + val_idx),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, eval_loader


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("--require-cuda was set, but CUDA is not available.")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    cfg = config_from_args(args)
    require_camera = args.stage in {"stage2", "both", "eval_stage2"}
    dataset = SLMToFieldDataset(
        root_dirs=args.root_dir,
        phase_dir=args.phase_dir,
        field_dir=args.field_dir,
        camera_dir=args.camera_dir,
        model_size=args.model_size,
        fov_crop_size=args.fov_crop_size,
        phase_spatial_mode=args.phase_spatial_mode,
        phase_flip_lr=not args.no_phase_flip_lr,
        phase_units=args.phase_units,
        phase_level_max=args.phase_level_max,
        field_scale_mode=args.field_scale_mode,
        field_amp_percentile=args.field_amp_percentile,
        field_amp_scale=args.field_amp_scale,
        camera_scale_mode=args.camera_scale_mode,
        camera_percentile=args.camera_percentile,
        camera_scale=args.camera_scale,
        camera_black_level=args.camera_black_level,
        require_camera=require_camera,
        z_mm=args.z_mm,
        scale_sample_pixels=args.scale_sample_pixels,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    cfg["field_amp_scale_actual"] = dataset.field_amp_scale
    cfg["camera_scale_actual"] = dataset.camera_scale
    cfg["z_by_root"] = dataset.z_by_root
    cfg["phase_shape_info"] = dataset.phase_shape_info

    _, _, train_idx, val_idx = split_dataset(
        dataset,
        real_train_ratio=args.real_train_ratio,
        sys_train_ratio=args.sys_train_ratio,
        seed=args.seed,
    )
    group_weights, group_counts = normalized_group_weights(
        dataset,
        train_idx,
        cfg["group_loss_weights_raw"],
        enabled=args.use_group_loss_weights,
    )
    cfg["group_loss_weights_normalized"] = group_weights
    cfg["train_group_counts"] = group_counts
    spectrum_geometry = effective_spectrum_geometry(cfg)
    support_cpu = effective_axicon_spectrum_weight(
        cfg["model_size"],
        cfg["model_size"],
        cfg,
        torch.device("cpu"),
        torch.float32,
        rfft=False,
    )
    support_threshold = float(cfg["effective_spectrum_support_threshold"])
    spectrum_geometry["active_grid_fraction"] = float(
        (support_cpu >= support_threshold).float().mean()
    )
    spectrum_geometry["mean_support_weight"] = float(support_cpu.mean())
    spectrum_geometry["legacy_fno_weighted_coverage"] = standard_fno_effective_support_coverage(cfg)
    conservative_outer_frequency = (
        spectrum_geometry["axicon_ring_cyc_per_m"]
        + spectrum_geometry["gs_bandwidth_corner_cyc_per_m"]
    )
    spectrum_geometry["conservative_outer_frequency_cyc_per_m"] = conservative_outer_frequency
    spectrum_geometry["support_clipped_by_model_nyquist"] = bool(
        conservative_outer_frequency
        > min(
            spectrum_geometry["model_nyquist_x_cyc_per_m"],
            spectrum_geometry["model_nyquist_y_cyc_per_m"],
        )
    )
    cfg["effective_spectrum_geometry"] = spectrum_geometry

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(args.output_dir / run_name)
    stage1_dir = ensure_dir(run_dir / "stage1_synthetic_field")
    stage2_dir = ensure_dir(run_dir / "stage2_camera_finetune")
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    save_split(run_dir / "split.json", dataset, train_idx, val_idx, cfg)
    save_effective_spectrum_diagnostic(cfg, run_dir / "effective_spectrum_support.png")

    in_ch = infer_phase_input_channels(cfg)
    model = build_operator_model(cfg, in_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Input channels: {in_ch}")
    print(f"Input channel names: {', '.join(phase_input_channel_names(cfg))}")
    print(f"Operator mode: {cfg['operator_mode']}")
    if cfg["operator_mode"] == "annular_fno":
        print(
            "Annular FNO: "
            f"active_modes={model.n_active_modes}, "
            f"rfft_fraction={100.0 * model.support_fraction:.2f}%, "
            f"low_modes={cfg['annular_fno_low_modes']}"
        )
    if cfg["operator_mode"] == "spectral":
        print(
            "Spectral operator: "
            f"width={cfg['spectral_width']}, depth={cfg['spectral_depth']}, "
            f"radial_bins={cfg['spectral_radial_bins']}, "
            f"angular_orders={cfg['spectral_angular_orders'] or 'none'}, "
            f"residual_scale={cfg['spectral_residual_scale']:.4g}"
        )
    print(
        "Axicon conditioning: "
        f"scope={cfg['axicon_map_scope']}, "
        f"pitch={cfg['axicon_grating_pitch_m']:.4g} m, "
        f"f_r={cfg['axicon_transverse_frequency_actual']:.4g} 1/m, "
        f"NA_air={cfg['axicon_na_air_equiv']:.4f}"
    )
    print(
        "Effective spectrum: "
        f"shape={cfg['effective_spectrum_shape']}, "
        f"GS_pixel={spectrum_geometry['gs_effective_pixel_size_m'] * 1e6:.4f} um, "
        f"f_N={spectrum_geometry['gs_nyquist_cyc_per_m'] / 1000.0:.2f} kcyc/m, "
        f"ring={spectrum_geometry['axicon_ring_cyc_per_m'] / 1000.0:.2f} kcyc/m "
        f"({spectrum_geometry['ring_bin_x']:.2f} bins), "
        f"corner_halfwidth/ring={100.0 * spectrum_geometry['corner_halfwidth_over_ring']:.2f}%, "
        f"active={100.0 * spectrum_geometry['active_grid_fraction']:.2f}%"
    )
    if spectrum_geometry["support_clipped_by_model_nyquist"]:
        print(
            "WARNING: the conservative effective spectrum extends beyond the "
            "model-grid Nyquist limit. Increase model sampling density or reduce "
            "the requested GS bandwidth before interpreting annular losses."
        )
    if cfg["operator_mode"] == "fno":
        print(
            "Legacy FNO effective-support coverage: "
            f"{100.0 * spectrum_geometry['legacy_fno_weighted_coverage']:.2f}%"
        )
    print(
        "Effective-spectrum loss weights: "
        f"complex={cfg['w_effective_spectrum_complex']:.4g}, "
        f"logmag={cfg['w_effective_spectrum_logmag']:.4g}"
    )
    print(f"Model parameters: {n_params / 1e6:.3f} M")
    print(f"Run dir: {run_dir}")

    train_loader, val_loader, eval_loader = make_loaders(dataset, train_idx, val_idx, cfg, device)
    if args.dry_run:
        batch = next(iter(train_loader))
        b = batch_to_device(batch, device)
        with torch.no_grad():
            x = build_phase_fno_input(b["phase"], b["z_mm"], cfg)
            y = model(x)
            dry_metrics = field_metrics_batch(
                y,
                b["field"],
                phase_amp_floor=cfg["phase_loss_amp_floor"],
                radial_bins=cfg["radial_profile_bins"],
                cfg=cfg,
            )
        print(f"Dry run batch phase: {tuple(b['phase'].shape)}")
        print(f"Dry run model input: {tuple(x.shape)}")
        print(f"Dry run channels:    {', '.join(phase_input_channel_names(cfg))}")
        print(f"Dry run output:      {tuple(y.shape)}")
        print(f"Dry run field target:{tuple(b['field'].shape)}")
        print(
            "Dry run effective spectrum: "
            f"complex_nmse={dry_metrics['effective_spectrum_complex_nmse'].mean().item():.6f}, "
            f"logmag_l1={dry_metrics['effective_spectrum_logmag_l1'].mean().item():.6f}, "
            f"target_energy={dry_metrics['target_effective_spectrum_energy_fraction'].mean().item():.4f}, "
            f"pred_energy={dry_metrics['pred_effective_spectrum_energy_fraction'].mean().item():.4f}"
        )
        if "camera" in b:
            print(f"Dry run camera:      {tuple(b['camera'].shape)}")
        print("Dry run complete.")
        return

    if args.stage in {"stage1", "both"}:
        train_stage(
            model,
            train_loader,
            val_loader,
            device,
            cfg,
            stage="stage1",
            stage_dir=stage1_dir,
            lr=args.stage1_lr,
            epochs=args.stage1_epochs,
        )
        evaluate(model, eval_loader, dataset, train_idx + val_idx, device, cfg, "stage1", stage1_dir)
        save_previews(
            model,
            dataset,
            evenly_spaced_sample_indices(train_idx, args.n_vis_train),
            device,
            cfg,
            "stage1",
            stage1_dir / "samples_train",
        )
        save_previews(
            model,
            dataset,
            evenly_spaced_sample_indices(val_idx, args.n_vis_val),
            device,
            cfg,
            "stage1",
            stage1_dir / "samples_val",
        )

    if args.stage in {"stage2", "both"}:
        if args.stage == "stage2":
            ckpt = args.stage1_checkpoint
            if ckpt is None:
                raise RuntimeError("--stage1-checkpoint is required when --stage stage2 is used.")
            print(f">>> Loading stage1 checkpoint: {ckpt}")
            load_model_from_checkpoint(model, ckpt, device)
        else:
            load_model_from_checkpoint(model, stage1_dir / "best.pt", device)

        train_stage(
            model,
            train_loader,
            val_loader,
            device,
            cfg,
            stage="stage2",
            stage_dir=stage2_dir,
            lr=args.stage2_lr,
            epochs=args.stage2_epochs,
        )
        evaluate(model, eval_loader, dataset, train_idx + val_idx, device, cfg, "stage2", stage2_dir)
        save_previews(
            model,
            dataset,
            evenly_spaced_sample_indices(train_idx, args.n_vis_train),
            device,
            cfg,
            "stage2",
            stage2_dir / "samples_train",
        )
        save_previews(
            model,
            dataset,
            evenly_spaced_sample_indices(val_idx, args.n_vis_val),
            device,
            cfg,
            "stage2",
            stage2_dir / "samples_val",
        )

    if args.stage == "eval_stage1":
        ckpt = args.stage1_checkpoint or stage1_dir / "best.pt"
        print(f">>> Loading stage1 checkpoint: {ckpt}")
        load_model_from_checkpoint(model, ckpt, device)
        evaluate(model, eval_loader, dataset, train_idx + val_idx, device, cfg, "stage1", stage1_dir)
        save_previews(model, dataset, evenly_spaced_sample_indices(val_idx, args.n_vis_val), device, cfg, "stage1", stage1_dir / "samples_eval")

    if args.stage == "eval_stage2":
        ckpt = args.stage2_checkpoint or stage2_dir / "best.pt"
        print(f">>> Loading stage2 checkpoint: {ckpt}")
        load_model_from_checkpoint(model, ckpt, device)
        evaluate(model, eval_loader, dataset, train_idx + val_idx, device, cfg, "stage2", stage2_dir)
        save_previews(model, dataset, evenly_spaced_sample_indices(val_idx, args.n_vis_val), device, cfg, "stage2", stage2_dir / "samples_eval")

    print(f"Done. Outputs: {run_dir}")


if __name__ == "__main__":
    main()
