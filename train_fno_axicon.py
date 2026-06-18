# -*- coding: utf-8 -*-
"""
Fourier Neural Operator trainer for SLM-axicon camera proxy learning.

This file is intentionally separate from train_residual_unet.py. It keeps the
same data contract:

Workflow-folder layout:

    pool/
      0.Phase_Mask/*.npy
      2.Aligned_Camera/*.npy
      1.Forward_Sim/*.npy

Files are paired by normalized stem, so minor naming differences such as
`sine80_890_0.npy` vs `sine_80_890_0.npy` and `..._09.npy` vs `..._9.npy`
can still be matched.

Default model behavior is non-residual:

    [simulation electric field + optional SLM/radial conditioning]
        -> FNO
        -> predicted camera intensity

Set PREDICTION_MODE = 'direct_field' if you want the FNO to generate a complex
E_out and use |E_out|^2 as the final intensity.
"""

import csv
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AxiconFieldDataset(Dataset):
    def __init__(self, root_dir,
                 workflow_field_dir='1.Forward_Sim',
                 workflow_camera_dir='2.Aligned_Camera',
                 workflow_phase_dir='0.Phase_Mask',
                 sim_size=1024,
                 phase_flip_lr=True,
                 field_scale_mode='global_percentile',
                 field_amp_percentile=99.9,
                 field_amp_scale=None,
                 camera_scale_mode='global_percentile',
                 camera_percentile=99.9,
                 camera_scale=None,
                 camera_black_level=0.0,
                 fov_crop_size=None):
        self.root_dir = Path(root_dir)
        self.workflow_field_dir = workflow_field_dir
        self.workflow_camera_dir = workflow_camera_dir
        self.workflow_phase_dir = workflow_phase_dir
        self.sim_size = int(sim_size)
        self.phase_flip_lr = phase_flip_lr
        self.field_scale_mode = field_scale_mode
        self.field_amp_percentile = field_amp_percentile
        self.field_amp_scale = field_amp_scale
        self.camera_scale_mode = camera_scale_mode
        self.camera_percentile = camera_percentile
        self.camera_scale = camera_scale
        self.camera_black_level = camera_black_level
        self.fov_crop_size = None if fov_crop_size is None else int(fov_crop_size)
        self.valid_scale_modes = {'sample_norm', 'global_percentile', 'raw'}

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root not found: {self.root_dir}")
        self._validate_scale_mode(self.field_scale_mode, 'field_scale_mode')
        self._validate_scale_mode(self.camera_scale_mode, 'camera_scale_mode')
        if self.fov_crop_size is not None and self.fov_crop_size <= 0:
            raise ValueError(f"fov_crop_size must be positive or None; got {self.fov_crop_size}")

        self.samples = self._discover_workflow_samples()
        print(f">>> Loaded {len(self.samples)} workflow samples from {self.root_dir}")
        if self.samples:
            print(f">>> First sample ids: {', '.join(s['id'] for s in self.samples[:5])}")
        if self.camera_scale_mode == 'global_percentile' and self.camera_scale is None:
            self.camera_scale = self._compute_global_camera_scale()
        if self.field_scale_mode == 'global_percentile' and self.field_amp_scale is None:
            self.field_amp_scale = self._compute_global_field_amp_scale()
        print(f">>> Camera scale mode: {self.camera_scale_mode}, scale={self.camera_scale}")
        print(f">>> Field scale mode: {self.field_scale_mode}, amp_scale={self.field_amp_scale}")
        if self.fov_crop_size is not None:
            print(f">>> Native center FOV crop: {self.fov_crop_size} x {self.fov_crop_size}")

    def _normalized_stem(self, path):
        stem = Path(path).stem
        if stem.startswith('sine') and len(stem) > 4 and stem[4].isdigit():
            stem = 'sine_' + stem[4:]
        parts = []
        for part in stem.split('_'):
            parts.append(str(int(part)) if part.isdigit() else part)
        return '_'.join(parts)

    def _path_map_by_normalized_stem(self, directory, patterns):
        directory = self.root_dir / directory
        if not directory.is_dir():
            raise FileNotFoundError(f"Expected workflow directory not found: {directory}")

        paths = []
        for pattern in patterns:
            paths.extend(directory.glob(pattern))

        path_map = {}
        for path in sorted(p for p in paths if p.is_file()):
            key = self._normalized_stem(path)
            if key in path_map:
                print(f"[skip] duplicate normalized id {key}: {path_map[key].name}, {path.name}")
                continue
            path_map[key] = path
        return path_map

    def _discover_workflow_samples(self):
        field_map = self._path_map_by_normalized_stem(self.workflow_field_dir, ('*.npy',))
        camera_map = self._path_map_by_normalized_stem(self.workflow_camera_dir, ('*.npy',))
        phase_map = self._path_map_by_normalized_stem(self.workflow_phase_dir, ('*.npy',))

        common_ids = sorted(set(field_map) & set(camera_map))
        samples = []
        for sample_id in common_ids:
            phase_p = phase_map.get(sample_id)
            if phase_p is None:
                print(f"[skip] {sample_id} (missing phase)")
                continue
            samples.append({
                'id': sample_id,
                'field': field_map[sample_id],
                'camera': camera_map[sample_id],
                'phase': phase_p,
            })

        missing_camera = sorted(set(field_map) - set(camera_map))
        missing_field = sorted(set(camera_map) - set(field_map))
        if missing_camera:
            print(f">>> Workflow samples with simulation but no camera: {len(missing_camera)}")
            print(f">>>   first: {', '.join(missing_camera[:8])}")
        if missing_field:
            print(f">>> Workflow samples with camera but no simulation: {len(missing_field)}")
            print(f">>>   first: {', '.join(missing_field[:8])}")
        return samples

    def __len__(self):
        return len(self.samples)

    def _validate_scale_mode(self, mode, name):
        if mode not in self.valid_scale_modes:
            valid = ', '.join(sorted(self.valid_scale_modes))
            raise ValueError(f"{name} must be one of {{{valid}}}; got {mode}")

    def _resize_chw(self, tensor, target_size, mode='bilinear'):
        if tensor.shape[-2:] == (target_size, target_size):
            return tensor
        kwargs = {'mode': mode}
        if mode in ('bilinear', 'bicubic'):
            kwargs['align_corners'] = False
        return F.interpolate(tensor.unsqueeze(0), size=(target_size, target_size),
                             **kwargs).squeeze(0)

    def _normalize_unit(self, arr, eps=1e-8):
        arr = np.nan_to_num(np.asarray(arr, dtype=np.float32))
        min_v = float(arr.min())
        max_v = float(arr.max())
        if min_v < 0.0 or max_v > 1.0:
            arr = arr - min_v
            max_v = float(arr.max())
            arr = arr / (max_v + eps)
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def _center_crop_2d(self, arr, source_name='array'):
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"{source_name} must be 2D before FOV cropping; got {arr.shape}")
        if self.fov_crop_size is None:
            return arr

        crop = self.fov_crop_size
        height, width = arr.shape
        if crop > height or crop > width:
            raise ValueError(
                f"fov_crop_size={crop} exceeds {source_name} shape {arr.shape}"
            )
        y0 = (height - crop) // 2
        x0 = (width - crop) // 2
        return arr[y0:y0 + crop, x0:x0 + crop]

    def _split_channel_array(self, arr):
        if arr.ndim != 3:
            return None
        if arr.shape[0] in (2, 3):
            return arr
        if arr.shape[-1] in (2, 3):
            return np.moveaxis(arr, -1, 0)
        return None

    def _load_camera_array(self, path):
        if path.suffix.lower() != '.npy':
            raise ValueError(f"Camera files must be .npy in workflow layout; got {path}")
        arr = np.load(path)
        if np.iscomplexobj(arr):
            arr = np.abs(arr) ** 2
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"{path} must be 2D camera data; got {arr.shape}")
        arr = self._center_crop_2d(arr, source_name=f"camera {path.name}")
        return np.nan_to_num(arr.astype(np.float32))

    def _camera_to_training_scale(self, arr, eps=1e-8):
        arr = np.nan_to_num(np.asarray(arr, dtype=np.float32))
        if self.camera_scale_mode == 'sample_norm':
            return self._normalize_unit(arr, eps=eps)

        arr = arr - float(self.camera_black_level)
        arr = np.clip(arr, 0.0, None)
        if self.camera_scale_mode == 'global_percentile':
            return (arr / (float(self.camera_scale) + eps)).astype(np.float32)
        if self.camera_scale_mode == 'raw':
            return arr.astype(np.float32)
        raise RuntimeError(f"Unhandled camera_scale_mode: {self.camera_scale_mode}")

    def _load_camera(self, path):
        arr = self._camera_to_training_scale(self._load_camera_array(path))
        cam = torch.from_numpy(arr).unsqueeze(0)
        return self._resize_chw(cam, self.sim_size, mode='bilinear').float()

    def _extract_field_amplitude(self, field_arr):
        field_arr = np.squeeze(field_arr)
        if np.iscomplexobj(field_arr):
            return np.abs(field_arr).astype(np.float32)

        arr = np.asarray(field_arr, dtype=np.float32)
        chw = self._split_channel_array(arr)
        if chw is None:
            intensity = self._normalize_unit(arr) if self.field_scale_mode == 'sample_norm' else np.clip(arr, 0.0, None)
            return np.sqrt(intensity).astype(np.float32)
        if chw.shape[0] == 2:
            return np.sqrt(chw[0].astype(np.float32) ** 2 + chw[1].astype(np.float32) ** 2)
        return chw[0].astype(np.float32)

    def _compute_global_camera_scale(self, eps=1e-8):
        values = []
        for s in self.samples:
            arr = self._load_camera_array(s['camera']) - float(self.camera_black_level)
            arr = np.clip(arr, 0.0, None)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                values.append(finite.reshape(-1))
        if not values:
            return 1.0
        return max(float(np.percentile(np.concatenate(values), self.camera_percentile)), eps)

    def _compute_global_field_amp_scale(self, eps=1e-8):
        values = []
        for s in self.samples:
            arr = np.load(s['field'])
            amp = self._extract_field_amplitude(arr)
            amp = self._center_crop_2d(amp, source_name=f"field amplitude {s['field'].name}")
            finite = amp[np.isfinite(amp)]
            if finite.size:
                values.append(finite.reshape(-1))
        if not values:
            return 1.0
        return max(float(np.percentile(np.concatenate(values), self.field_amp_percentile)), eps)

    def _scale_field_amplitude(self, amp, eps=1e-8):
        amp = np.clip(np.nan_to_num(np.asarray(amp, dtype=np.float32)), 0.0, None)
        if self.field_scale_mode == 'sample_norm':
            scale = float(np.percentile(amp, self.field_amp_percentile))
            return np.clip(amp / (scale + eps), 0.0, 1.0).astype(np.float32)
        if self.field_scale_mode == 'global_percentile':
            return (amp / (float(self.field_amp_scale) + eps)).astype(np.float32)
        if self.field_scale_mode == 'raw':
            return amp.astype(np.float32)
        raise RuntimeError(f"Unhandled field_scale_mode: {self.field_scale_mode}")

    def _field_to_channels(self, field_arr, eps=1e-8):
        field_arr = np.squeeze(field_arr)

        if np.iscomplexobj(field_arr):
            if field_arr.ndim != 2:
                raise ValueError(f"Complex simulation field must be 2D; got {field_arr.shape}")
            amp = np.abs(field_arr).astype(np.float32)
            phase = np.angle(field_arr).astype(np.float32)
            cos_p = np.cos(phase).astype(np.float32)
            sin_p = np.sin(phase).astype(np.float32)
        else:
            arr = np.asarray(field_arr, dtype=np.float32)
            chw = self._split_channel_array(arr)
            if chw is None:
                if arr.ndim != 2:
                    raise ValueError(f"Simulation field must be 2D, 2ch, or 3ch; got {arr.shape}")
                intensity = self._normalize_unit(arr) if self.field_scale_mode == 'sample_norm' else np.clip(arr, 0.0, None)
                amp = np.sqrt(intensity).astype(np.float32)
                cos_p = np.ones_like(amp, dtype=np.float32)
                sin_p = np.zeros_like(amp, dtype=np.float32)
            elif chw.shape[0] == 2:
                real = chw[0].astype(np.float32)
                imag = chw[1].astype(np.float32)
                amp = np.sqrt(real ** 2 + imag ** 2)
                phase = np.arctan2(imag, real)
                cos_p = np.cos(phase).astype(np.float32)
                sin_p = np.sin(phase).astype(np.float32)
            else:
                amp = chw[0].astype(np.float32)
                cos_p = chw[1].astype(np.float32)
                sin_p = chw[2].astype(np.float32)

        amp = self._center_crop_2d(amp, source_name='field amplitude')
        cos_p = self._center_crop_2d(cos_p, source_name='field phase cosine')
        sin_p = self._center_crop_2d(sin_p, source_name='field phase sine')
        amp = self._scale_field_amplitude(amp, eps=eps)
        field = torch.from_numpy(np.stack([amp, cos_p, sin_p], axis=0))
        field = self._resize_chw(field, self.sim_size, mode='bilinear')

        phase_norm = torch.sqrt(field[1:2] ** 2 + field[2:3] ** 2).clamp_min(1e-6)
        field[1:3] = field[1:3] / phase_norm
        field[0:1] = field[0:1].clamp_min(0.0)
        sim_intensity = field[0:1].pow(2).clamp_min(0.0)
        return field.float(), sim_intensity.float()

    def _load_field(self, path):
        if path.suffix.lower() != '.npy':
            raise ValueError(f"Simulation field files must be .npy in workflow layout; got {path}")
        arr = np.load(path)
        return self._field_to_channels(arr)

    def _load_phase(self, path):
        phase = np.load(path).astype(np.float32)
        if self.phase_flip_lr:
            phase = phase[:, ::-1].copy()
        phase_rad = phase * (2 * np.pi / 1023.0)
        return torch.from_numpy(phase_rad).unsqueeze(0).float()

    def __getitem__(self, idx):
        s = self.samples[idx]
        field, sim = self._load_field(s['field'])
        return {
            'field': field,
            'sim': sim,
            'phase': self._load_phase(s['phase']),
            'camera': self._load_camera(s['camera']),
            'id': s['id'],
        }


# ---------------------------------------------------------------------------
# FNO model
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_y=32, modes_x=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_y = int(modes_y)
        self.modes_x = int(modes_x)
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weight_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels,
                                self.modes_y, self.modes_x, dtype=torch.cfloat)
        )
        self.weight_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels,
                                self.modes_y, self.modes_x, dtype=torch.cfloat)
        )

    def _compl_mul2d(self, x, weights):
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x):
        batch, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x, norm='ortho')
        out_ft = torch.zeros(
            batch, self.out_channels, height, width // 2 + 1,
            device=x.device, dtype=torch.cfloat,
        )

        my = min(self.modes_y, height)
        mx = min(self.modes_x, width // 2 + 1)
        out_ft[:, :, :my, :mx] = self._compl_mul2d(
            x_ft[:, :, :my, :mx],
            self.weight_pos[:, :, :my, :mx],
        )
        out_ft[:, :, -my:, :mx] = self._compl_mul2d(
            x_ft[:, :, -my:, :mx],
            self.weight_neg[:, :, :my, :mx],
        )
        return torch.fft.irfft2(out_ft, s=(height, width), norm='ortho')


class FNOBlock2d(nn.Module):
    def __init__(self, width, modes_y, modes_x, groups=4):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_y, modes_x)
        self.local = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(min(groups, width), width)

    def forward(self, x):
        x = self.spectral(x) + self.local(x)
        x = self.norm(x)
        return F.gelu(x)


class AxiconFNO2d(nn.Module):
    def __init__(self, in_ch, out_ch=1, width=32, modes_y=32, modes_x=32,
                 depth=5, mlp_width=128):
        super().__init__()
        self.lift = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
        )
        self.blocks = nn.ModuleList([
            FNOBlock2d(width, modes_y, modes_x)
            for _ in range(depth)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(width, mlp_width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mlp_width, out_ch, kernel_size=1),
        )

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.project(x)


# ---------------------------------------------------------------------------
# Conditioning / prediction
# ---------------------------------------------------------------------------

def resize_batch(x, size, mode='bilinear'):
    if x.shape[-2:] == (size, size):
        return x
    kwargs = {'mode': mode}
    if mode in ('bilinear', 'bicubic'):
        kwargs['align_corners'] = False
    return F.interpolate(x, size=(size, size), **kwargs)


def make_grid(batch, height, width, device, dtype, radial=True):
    yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    y, x = torch.meshgrid(yy, xx, indexing='ij')
    channels = [x, y]
    if radial:
        r = torch.sqrt(x ** 2 + y ** 2)
        channels.append(r)
    grid = torch.stack(channels, dim=0).unsqueeze(0)
    return grid.repeat(batch, 1, 1, 1)


def make_axicon_phase_channels(batch, height, width, device, dtype,
                               slope_rad_per_pixel=None):
    if slope_rad_per_pixel is None:
        return None
    yy = torch.arange(height, device=device, dtype=dtype) - (height - 1) / 2
    xx = torch.arange(width, device=device, dtype=dtype) - (width - 1) / 2
    y, x = torch.meshgrid(yy, xx, indexing='ij')
    phase = slope_rad_per_pixel * torch.sqrt(x ** 2 + y ** 2)
    ax = torch.stack([torch.cos(phase), torch.sin(phase)], dim=0).unsqueeze(0)
    return ax.repeat(batch, 1, 1, 1)


def make_radial_mask(batch, height, width, device, dtype,
                     radius=None, width_param=None):
    if radius is None or width_param is None or width_param <= 0:
        return None
    yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    y, x = torch.meshgrid(yy, xx, indexing='ij')
    r = torch.sqrt(x ** 2 + y ** 2)
    mask = torch.exp(-0.5 * ((r - float(radius)) / float(width_param)) ** 2)
    return mask.unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1, 1)


def condition_slm_phase(phase, target_size,
                        weight=1.0,
                        lowpass_size=None,
                        ring_radius=None,
                        ring_width=None):
    """
    Build SLM phase conditioning channels without letting the full phase map
    dominate as a spatial shortcut.

    Phase is converted to cos/sin before any resizing/low-pass operation to
    avoid wrap discontinuities. Use weight < 1 or disable SLM phase entirely
    when the phase pattern starts imprinting onto the predicted camera image.
    """
    phase_cos = torch.cos(phase)
    phase_sin = torch.sin(phase)

    if lowpass_size is not None and int(lowpass_size) > 0 and int(lowpass_size) < target_size:
        low = int(lowpass_size)
        phase_cos = resize_batch(phase_cos, low, mode='bilinear')
        phase_sin = resize_batch(phase_sin, low, mode='bilinear')

    phase_cos = resize_batch(phase_cos, target_size, mode='bilinear')
    phase_sin = resize_batch(phase_sin, target_size, mode='bilinear')

    phase_norm = torch.sqrt(phase_cos ** 2 + phase_sin ** 2).clamp_min(1e-6)
    phase_ch = torch.cat([phase_cos / phase_norm, phase_sin / phase_norm], dim=1)

    mask = make_radial_mask(
        phase_ch.shape[0], target_size, target_size,
        phase_ch.device, phase_ch.dtype,
        radius=ring_radius, width_param=ring_width,
    )
    if mask is not None:
        phase_ch = phase_ch * mask

    return float(weight) * phase_ch


def build_fno_input(field, phase,
                    use_slm_phase=True,
                    slm_phase_weight=1.0,
                    slm_phase_lowpass_size=None,
                    slm_phase_ring_radius=None,
                    slm_phase_ring_width=None,
                    use_grid=True,
                    use_radial_coord=True,
                    use_axicon_phase_map=False,
                    axicon_slope_rad_per_pixel=None):
    batch, _, height, width = field.shape
    pieces = [field, field[:, 0:1].pow(2).clamp_min(0.0)]

    if use_slm_phase:
        pieces.append(condition_slm_phase(
            phase, height,
            weight=slm_phase_weight,
            lowpass_size=slm_phase_lowpass_size,
            ring_radius=slm_phase_ring_radius,
            ring_width=slm_phase_ring_width,
        ))

    if use_grid:
        pieces.append(make_grid(batch, height, width, field.device, field.dtype,
                                radial=use_radial_coord))

    if use_axicon_phase_map:
        ax = make_axicon_phase_channels(
            batch, height, width, field.device, field.dtype,
            slope_rad_per_pixel=axicon_slope_rad_per_pixel,
        )
        if ax is not None:
            pieces.append(ax)

    return torch.cat(pieces, dim=1)


def field_to_real_imag(field):
    amp = field[:, 0:1].clamp_min(0.0)
    phase_vec = field[:, 1:3]
    phase_norm = torch.sqrt((phase_vec ** 2).sum(dim=1, keepdim=True)).clamp_min(1e-6)
    cos_p = phase_vec[:, 0:1] / phase_norm
    sin_p = phase_vec[:, 1:2] / phase_norm
    return amp * cos_p, amp * sin_p


def scale_direct_field(raw_out, sim_field,
                       field_out_scale=4.0,
                       field_out_scale_mode='field_rms',
                       activation='tanh',
                       noise_std=0.0):
    if activation == 'tanh':
        e_out = torch.tanh(raw_out[:, 0:2])
    elif activation == 'softsign':
        e_out = F.softsign(raw_out[:, 0:2])
    elif activation == 'linear':
        e_out = raw_out[:, 0:2]
    else:
        raise ValueError(f"Unknown direct field activation: {activation}")

    sim_real, sim_imag = field_to_real_imag(sim_field)
    if field_out_scale_mode == 'field_rms':
        sim_power = sim_real ** 2 + sim_imag ** 2
        scale = field_out_scale * torch.sqrt(
            sim_power.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        )
    elif field_out_scale_mode == 'absolute':
        scale = float(field_out_scale)
    else:
        raise ValueError(f"Unknown field_out_scale_mode: {field_out_scale_mode}")

    e_out = scale * e_out
    if noise_std and noise_std > 0:
        e_out = e_out + (noise_std * scale) * torch.randn_like(e_out)
    return e_out


def predict_camera(model, field, phase, cfg, train_noise=False):
    x = build_fno_input(
        field, phase,
        use_slm_phase=cfg['use_slm_phase'],
        slm_phase_weight=cfg['slm_phase_weight'],
        slm_phase_lowpass_size=cfg['slm_phase_lowpass_size'],
        slm_phase_ring_radius=cfg['slm_phase_ring_radius'],
        slm_phase_ring_width=cfg['slm_phase_ring_width'],
        use_grid=cfg['use_grid'],
        use_radial_coord=cfg['use_radial_coord'],
        use_axicon_phase_map=cfg['use_axicon_phase_map'],
        axicon_slope_rad_per_pixel=cfg['axicon_slope_rad_per_pixel'],
    )
    raw = model(x)

    if cfg['prediction_mode'] == 'intensity':
        if cfg['intensity_activation'] == 'softplus':
            pred = F.softplus(raw[:, 0:1])
        elif cfg['intensity_activation'] == 'relu':
            pred = F.relu(raw[:, 0:1])
        elif cfg['intensity_activation'] == 'square':
            pred = raw[:, 0:1].pow(2)
        elif cfg['intensity_activation'] == 'linear_clamp':
            pred = raw[:, 0:1].clamp_min(0.0)
        else:
            raise ValueError(f"Unknown intensity_activation: {cfg['intensity_activation']}")
        field_like = raw[:, 0:1]
    elif cfg['prediction_mode'] == 'direct_field':
        e_out = scale_direct_field(
            raw, field,
            field_out_scale=cfg['field_out_scale'],
            field_out_scale_mode=cfg['field_out_scale_mode'],
            activation=cfg['direct_field_activation'],
            noise_std=cfg['direct_field_noise_std'] if train_noise else 0.0,
        )
        pred = e_out[:, 0:1].pow(2) + e_out[:, 1:2].pow(2)
        field_like = e_out
    else:
        raise ValueError(f"Unknown prediction_mode: {cfg['prediction_mode']}")

    return pred.clamp_min(0.0), field_like


def center_roi_bounds(height, width, fraction):
    if fraction is None:
        return 0, height, 0, width
    fraction = float(fraction)
    if fraction <= 0:
        raise ValueError(f"ROI fraction must be > 0 or None; got {fraction}")
    if fraction >= 1:
        return 0, height, 0, width

    roi_h = max(1, int(round(height * fraction)))
    roi_w = max(1, int(round(width * fraction)))
    y0 = (height - roi_h) // 2
    x0 = (width - roi_w) // 2
    return y0, y0 + roi_h, x0, x0 + roi_w


def crop_center_roi(x, fraction):
    y0, y1, x0, x1 = center_roi_bounds(x.shape[-2], x.shape[-1], fraction)
    return x[..., y0:y1, x0:x1]


def apply_loss_roi(pred, target, fraction):
    if fraction is None or float(fraction) >= 1.0:
        return pred, target
    return crop_center_roi(pred, fraction), crop_center_roi(target, fraction)


def infer_input_channels(use_slm_phase=True, use_grid=True,
                         use_radial_coord=True, use_axicon_phase_map=False,
                         axicon_slope_rad_per_pixel=None):
    ch = 4  # amp/cos/sin + sim intensity
    if use_slm_phase:
        ch += 2
    if use_grid:
        ch += 3 if use_radial_coord else 2
    if use_axicon_phase_map and axicon_slope_rad_per_pixel is not None:
        ch += 2
    return ch


# ---------------------------------------------------------------------------
# Losses / metrics
# ---------------------------------------------------------------------------

def per_image_standardize(x, eps=1e-6):
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (x - mean) / std


def log_display_tensor(x, eps=1e-6):
    x = x.clamp_min(0.0)
    scale = x.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return per_image_standardize(torch.log1p(x / scale), eps=eps)


def mean_per_sample(x):
    return x.flatten(start_dim=1).mean(dim=1)


def gradient_loss_per_sample(pred, target):
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return (mean_per_sample(torch.abs(pred_dx - target_dx)) +
            mean_per_sample(torch.abs(pred_dy - target_dy)))


def gradient_loss(pred, target):
    return gradient_loss_per_sample(pred, target).mean()


def simple_ssim_loss_per_sample(pred, target, window=11,
                                c1=0.01 ** 2, c2=0.03 ** 2):
    pad = window // 2
    mu_x = F.avg_pool2d(pred, window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(pred * pred, window, stride=1, padding=pad) - mu_x ** 2
    sigma_y = F.avg_pool2d(target * target, window, stride=1, padding=pad) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, window, stride=1, padding=pad) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return mean_per_sample(1.0 - ssim.clamp(-1.0, 1.0))


def simple_ssim_loss(pred, target, window=11, c1=0.01 ** 2, c2=0.03 ** 2):
    return simple_ssim_loss_per_sample(
        pred, target, window=window, c1=c1, c2=c2).mean()


def highpass(x, kernel_size=31):
    pad = kernel_size // 2
    low = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode='reflect'),
                       kernel_size, stride=1)
    return x - low


def fft_texture_loss_per_sample(pred, target, highpass_kernel=31):
    pred_hp = highpass(pred, kernel_size=highpass_kernel)
    target_hp = highpass(target, kernel_size=highpass_kernel)
    pred_mag = torch.log1p(torch.abs(torch.fft.rfft2(pred_hp, norm='ortho')))
    target_mag = torch.log1p(torch.abs(torch.fft.rfft2(target_hp, norm='ortho')))
    return mean_per_sample(torch.abs(
        per_image_standardize(pred_mag) - per_image_standardize(target_mag)))


def fft_texture_loss(pred, target, highpass_kernel=31):
    return fft_texture_loss_per_sample(
        pred, target, highpass_kernel=highpass_kernel).mean()


def bright_excess_loss_per_sample(pred, target, margin=0.10,
                                  top_fraction=0.002, eps=1e-6):
    """
    Penalize small, very bright hallucinations that visual/log losses can miss.

    The threshold follows the local target plus a per-image mean margin, so real
    bright camera structures are allowed while predicted-only hot spots are not.
    """
    scale = target.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    excess = F.relu(pred - target - float(margin) * scale) / scale
    excess = excess.pow(2).flatten(start_dim=1)
    if top_fraction is not None and top_fraction > 0:
        k = max(1, int(excess.shape[1] * float(top_fraction)))
        excess = torch.topk(excess, k=k, dim=1).values
    return excess.mean(dim=1)


def bright_excess_loss(pred, target, margin=0.10, top_fraction=0.002, eps=1e-6):
    return bright_excess_loss_per_sample(
        pred, target, margin=margin, top_fraction=top_fraction, eps=eps).mean()


def dark_deficit_loss_per_sample(pred, target, margin=0.10,
                                 top_fraction=0.002, eps=1e-6):
    """
    Penalize small, very dark hallucinations such as local pore artifacts.

    This is the symmetric counterpart to bright_excess_loss.
    """
    scale = target.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    deficit = F.relu(target - pred - float(margin) * scale) / scale
    deficit = deficit.pow(2).flatten(start_dim=1)
    if top_fraction is not None and top_fraction > 0:
        k = max(1, int(deficit.shape[1] * float(top_fraction)))
        deficit = torch.topk(deficit, k=k, dim=1).values
    return deficit.mean(dim=1)


def dark_deficit_loss(pred, target, margin=0.10, top_fraction=0.002, eps=1e-6):
    return dark_deficit_loss_per_sample(
        pred, target, margin=margin, top_fraction=top_fraction, eps=eps).mean()


def target_mean_normalized_smooth_l1_loss_per_sample(pred, target, eps=1e-6):
    scale = target.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    loss_map = F.smooth_l1_loss(pred / scale, target / scale, reduction='none')
    return mean_per_sample(loss_map)


def target_mean_normalized_smooth_l1_loss(pred, target, eps=1e-6):
    return target_mean_normalized_smooth_l1_loss_per_sample(
        pred, target, eps=eps).mean()


def visual_loss_per_sample(pred, target,
                           w_log_display_smooth_l1=0.0, w_ssim=1.0,
                           w_grad=0.1, w_fft=0.0,
                           w_mean_norm_l1=1.0, w_peak=0.0, w_dark=0.0,
                           peak_margin=0.10, peak_top_fraction=0.002,
                           dark_margin=0.10, dark_top_fraction=0.002):
    pred_d = target_d = None
    if w_log_display_smooth_l1 > 0 or w_ssim > 0 or w_grad > 0 or w_fft > 0:
        pred_d = log_display_tensor(pred)
        target_d = log_display_tensor(target)

    zeros = pred.new_zeros(pred.shape[0])
    log_display_smooth_l1 = (
        mean_per_sample(F.smooth_l1_loss(pred_d, target_d, reduction='none'))
        if w_log_display_smooth_l1 > 0 else zeros
    )
    ssim = simple_ssim_loss_per_sample(pred_d, target_d) if w_ssim > 0 else zeros
    grad = gradient_loss_per_sample(pred_d, target_d) if w_grad > 0 else zeros
    fft = fft_texture_loss_per_sample(pred_d, target_d) if w_fft > 0 else zeros
    mean_norm_l1 = (
        target_mean_normalized_smooth_l1_loss_per_sample(pred, target)
        if w_mean_norm_l1 > 0 else zeros
    )
    peak = bright_excess_loss_per_sample(
        pred, target,
        margin=peak_margin,
        top_fraction=peak_top_fraction,
    ) if w_peak > 0 else zeros
    dark = dark_deficit_loss_per_sample(
        pred, target,
        margin=dark_margin,
        top_fraction=dark_top_fraction,
    ) if w_dark > 0 else zeros
    total = (w_log_display_smooth_l1 * log_display_smooth_l1 +
             w_ssim * ssim + w_grad * grad + w_fft * fft +
             w_mean_norm_l1 * mean_norm_l1 +
             w_peak * peak + w_dark * dark)
    return total, {
        'log_display_smooth_l1': log_display_smooth_l1,
        'ssim': ssim,
        'grad': grad,
        'fft': fft,
        'mean_norm_l1': mean_norm_l1,
        'peak': peak,
        'dark': dark,
        'raw_mse': mean_per_sample((pred - target).pow(2)),
    }


def visual_loss(pred, target, **kwargs):
    per_sample_loss, per_sample_comps = visual_loss_per_sample(
        pred, target, **kwargs)
    return per_sample_loss.mean(), {
        key: value.mean().item() for key, value in per_sample_comps.items()
    }


def quantile_normalize_np(arr, q_min=0.001, q_max=0.999, eps=1e-8):
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32))
    v_min = float(np.quantile(arr.reshape(-1), q_min))
    v_max = float(np.quantile(arr.reshape(-1), q_max))
    arr = np.clip(arr, v_min, v_max)
    return (arr - v_min) / (v_max - v_min + eps)


def sample_type_from_id(sample_id):
    sample_id = str(sample_id)
    if sample_id.startswith('sys_'):
        return 'systematic'
    if sample_id.startswith('real_'):
        return 'real'
    if sample_id.startswith('pert_'):
        return 'pert'
    return 'other'


def normalize_group_loss_weights(dataset, train_indices, raw_weights, enabled=True):
    group_names = ('systematic', 'real', 'pert', 'other')
    counts = {name: 0 for name in group_names}
    for idx in train_indices:
        group = sample_type_from_id(dataset.samples[int(idx)]['id'])
        counts[group if group in counts else 'other'] += 1

    weights = {
        name: float(raw_weights.get(name, 1.0)) if enabled else 1.0
        for name in group_names
    }
    total_count = sum(counts.values())
    if total_count == 0:
        raise ValueError("Cannot normalize group loss weights for an empty training split")
    mean_weight = sum(counts[name] * weights[name] for name in group_names) / total_count
    if mean_weight <= 0:
        raise ValueError(f"Mean group loss weight must be positive; got {mean_weight}")

    normalized = {name: weights[name] / mean_weight for name in group_names}
    print(">>> Training group loss weights (count, raw -> normalized):")
    for name in group_names:
        print(f">>>   {name}: {counts[name]}, {weights[name]:.4g} -> {normalized[name]:.4g}")
    return normalized, counts


def pert_parent_id_from_id(sample_id, real_ids=None):
    sample_id = str(sample_id)
    if not sample_id.startswith('pert_'):
        return None

    if real_ids:
        for real_id in sorted((str(r) for r in real_ids), key=len, reverse=True):
            if sample_id.startswith(f"pert_{real_id}_"):
                return real_id

    stem = sample_id[len('pert_'):]
    for marker in ['_annrand_', '_anngradflow_']:
        if marker in stem:
            return stem.split(marker, 1)[0]
    return stem


def sample_group_from_id(sample_id):
    sample_id = str(sample_id)
    if sample_id.startswith('pert_'):
        return 'pert'
    if sample_id.startswith('real_'):
        return 'real'
    if sample_id.startswith('sys_zernike_combo'):
        return 'zernike_combo'
    if sample_id.startswith('sys_zernike_j'):
        return 'zernike_single'
    if sample_id.startswith('sys_ramp') or sample_id.startswith('sys_linear_ramp'):
        return 'linear_ramp'
    if sample_id.startswith('sys_checker'):
        return 'checkerboard'
    if sample_id.startswith('sys_square_ring'):
        return 'square_ring'
    if sample_id.startswith('sys_radial_ring'):
        return 'radial_ring'
    if sample_id.startswith('sys_local_bump'):
        return 'local_bump'
    if sample_id.startswith('sys_sine'):
        return 'sine'
    if sample_id.startswith('sys_'):
        return 'systematic_other'
    if sample_id.startswith('sine'):
        return 'sine'
    if sample_id.startswith('checker'):
        return 'checkerboard'
    if sample_id.startswith('linear_ramp'):
        return 'linear_ramp'
    return 'other'


def split_dataset(dataset, train_ratio=0.8, seed=42):
    rng = random.Random(seed)

    sys_idx = []
    real_idx_by_id = {}
    pert_idx_by_parent = {}
    other_idx = []
    typed_samples = []

    for idx, sample in enumerate(dataset.samples):
        sample_type = sample_type_from_id(sample['id'])
        typed_samples.append((idx, sample['id'], sample_type))
        if sample_type == 'systematic':
            sys_idx.append(idx)
        elif sample_type == 'real':
            real_idx_by_id[sample['id']] = idx

    real_ids = set(real_idx_by_id)
    for idx, sample_id, sample_type in typed_samples:
        if sample_type == 'pert':
            parent_id = pert_parent_id_from_id(sample_id, real_ids=real_ids)
            pert_idx_by_parent.setdefault(parent_id, []).append(idx)
        elif sample_type not in {'systematic', 'real'}:
            other_idx.append(idx)

    for indices in pert_idx_by_parent.values():
        indices.sort()

    real_parent_ids = sorted(set(real_idx_by_id) | set(pert_idx_by_parent))

    if sys_idx or real_parent_ids:
        rng.shuffle(real_parent_ids)
        rng.shuffle(other_idx)

        n_parent_train = int(round(len(real_parent_ids) * train_ratio))
        n_other_train = int(round(len(other_idx) * train_ratio))
        train_parent_ids = set(real_parent_ids[:n_parent_train])
        val_parent_ids = set(real_parent_ids[n_parent_train:])

        def grouped_indices(parent_ids):
            out = []
            for parent_id in sorted(parent_ids):
                if parent_id in real_idx_by_id:
                    out.append(real_idx_by_id[parent_id])
                out.extend(pert_idx_by_parent.get(parent_id, []))
            return out

        grouped_train_idx = grouped_indices(train_parent_ids)
        grouped_val_idx = grouped_indices(val_parent_ids)

        train_idx = sys_idx + grouped_train_idx + other_idx[:n_other_train]
        val_idx = grouped_val_idx + other_idx[n_other_train:]

        train_real_count = sum(1 for i in grouped_train_idx
                               if sample_type_from_id(dataset.samples[i]['id']) == 'real')
        val_real_count = sum(1 for i in grouped_val_idx
                             if sample_type_from_id(dataset.samples[i]['id']) == 'real')
        train_pert_count = sum(1 for i in grouped_train_idx
                               if sample_type_from_id(dataset.samples[i]['id']) == 'pert')
        val_pert_count = sum(1 for i in grouped_val_idx
                             if sample_type_from_id(dataset.samples[i]['id']) == 'pert')

        print(
            f"Prefix/group split: {len(train_idx)} train / {len(val_idx)} val "
            f"(sys train-fixed={len(sys_idx)}, "
            f"real-parent train/val={len(train_parent_ids)}/{len(val_parent_ids)}, "
            f"real samples train/val={train_real_count}/{val_real_count}, "
            f"pert samples train/val={train_pert_count}/{val_pert_count}, "
            f"other train/val={n_other_train}/{len(other_idx) - n_other_train}, "
            f"ratio={train_ratio})"
        )
        orphan_parents = sorted(pid for pid in pert_idx_by_parent if pid not in real_idx_by_id)
        if orphan_parents:
            print(
                "[warn] pert samples without matching real parent were split as their own parent groups: "
                + ", ".join(orphan_parents[:8])
                + (" ..." if len(orphan_parents) > 8 else "")
            )
    else:
        n = len(dataset)
        n_train = int(round(n * train_ratio))
        indices = list(range(n))
        rng.shuffle(indices)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        print(f"Split: {len(train_idx)} train / {len(val_idx)} val (ratio={train_ratio})")

    return Subset(dataset, train_idx), Subset(dataset, val_idx), train_idx, val_idx


def evenly_spaced_sample_indices(indices, n_samples):
    """Pick visualization samples across the full dataset-index span."""
    if n_samples is None or n_samples <= 0 or not indices:
        return []

    ordered = sorted(int(i) for i in indices)
    if len(ordered) <= n_samples:
        return ordered
    if n_samples == 1:
        return [ordered[len(ordered) // 2]]

    positions = np.linspace(0, len(ordered) - 1, int(n_samples))
    selected = []
    seen = set()
    for pos in positions:
        idx = ordered[int(round(float(pos)))]
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)

    if len(selected) < n_samples:
        for idx in ordered:
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
                if len(selected) == n_samples:
                    break
    return selected


def print_torch_device_diagnostics():
    cuda_available = torch.cuda.is_available()
    print("=== Torch device diagnostics ===")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch CUDA build: {torch.version.cuda}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if cuda_available:
        current = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current)
        print(f"Active CUDA device: cuda:{current} - {props.name}")
        print(f"CUDA capability: {props.major}.{props.minor}")
        print(f"Total VRAM: {props.total_memory / 1024**3:.2f} GiB")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    else:
        print("No CUDA accelerator is visible to this Python process; training will run on CPU.")
    print("================================")


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def prepare_batch(batch, device, model_size):
    field = resize_batch(batch['field'].to(device), model_size, mode='bilinear')
    sim = resize_batch(batch['sim'].to(device), model_size, mode='bilinear')
    phase = batch['phase'].to(device)
    camera = resize_batch(batch['camera'].to(device), model_size, mode='bilinear')
    return field, sim, phase, camera


def train_one_run(model, train_loader, val_loader, optimizer, device, cfg, run_dir):
    amp_enabled = cfg['use_amp'] and device == 'cuda'
    loss_roi_fraction = cfg.get('loss_roi_fraction', 1.0)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    best_val = float('inf')
    history = {
        'train': [], 'val': [],
        'train_unweighted': [],
        'train_systematic': [], 'train_real_family': [], 'train_other': [],
        'train_log_display_smooth_l1': [], 'val_log_display_smooth_l1': [],
        'train_ssim': [], 'val_ssim': [],
        'train_grad': [], 'val_grad': [],
        'train_fft': [], 'val_fft': [],
        'train_mean_norm_l1': [], 'val_mean_norm_l1': [],
        'train_peak': [], 'val_peak': [],
        'train_dark': [], 'val_dark': [],
        'train_raw_mse': [], 'val_raw_mse': [],
    }

    for epoch in range(cfg['epochs']):
        model.train()
        sums = {k: 0.0 for k in ['loss', 'unweighted_loss',
                                  'log_display_smooth_l1', 'ssim', 'grad', 'fft',
                                  'mean_norm_l1', 'peak', 'dark', 'raw_mse']}
        group_loss_sums = {'systematic': 0.0, 'real_family': 0.0, 'other': 0.0}
        group_counts = {'systematic': 0, 'real_family': 0, 'other': 0}
        n_seen = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']} [train]"):
            field, _, phase, camera = prepare_batch(batch, device, cfg['model_size'])

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                pred, _ = predict_camera(model, field, phase, cfg, train_noise=True)
                pred_loss, camera_loss = apply_loss_roi(pred, camera, loss_roi_fraction)
                per_sample_loss, per_sample_comps = visual_loss_per_sample(
                    pred_loss, camera_loss,
                    w_log_display_smooth_l1=cfg['w_log_display_smooth_l1'],
                    w_ssim=cfg['w_ssim'],
                    w_grad=cfg['w_grad'],
                    w_fft=cfg['w_fft'],
                    w_mean_norm_l1=cfg['w_mean_norm_l1'],
                    w_peak=cfg['w_peak'],
                    w_dark=cfg['w_dark'],
                    peak_margin=cfg['peak_margin'],
                    peak_top_fraction=cfg['peak_top_fraction'],
                    dark_margin=cfg['dark_margin'],
                    dark_top_fraction=cfg['dark_top_fraction'],
                )
                sample_types = [sample_type_from_id(sample_id) for sample_id in batch['id']]
                sample_weights = pred.new_tensor([
                    cfg['group_loss_weights_normalized'].get(sample_type, 1.0)
                    for sample_type in sample_types
                ])
                loss = (sample_weights * per_sample_loss).mean()

            scaler.scale(loss).backward()
            if cfg['grad_clip'] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            b = field.shape[0]
            sums['loss'] += loss.item() * b
            sums['unweighted_loss'] += per_sample_loss.detach().sum().item()
            for key, values in per_sample_comps.items():
                sums[key] += values.detach().sum().item()
            for sample_type, sample_loss in zip(
                    sample_types, per_sample_loss.detach().cpu().tolist()):
                if sample_type == 'systematic':
                    family = 'systematic'
                elif sample_type in {'real', 'pert'}:
                    family = 'real_family'
                else:
                    family = 'other'
                group_loss_sums[family] += float(sample_loss)
                group_counts[family] += 1
            n_seen += b

        train_stats = {k: v / max(n_seen, 1) for k, v in sums.items()}
        train_group_stats = {
            name: (group_loss_sums[name] / group_counts[name]
                   if group_counts[name] else float('nan'))
            for name in group_loss_sums
        }

        model.eval()
        sums = {k: 0.0 for k in ['loss', 'log_display_smooth_l1', 'ssim', 'grad', 'fft',
                                  'mean_norm_l1', 'peak', 'dark', 'raw_mse']}
        n_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                field, _, phase, camera = prepare_batch(batch, device, cfg['model_size'])
                pred, _ = predict_camera(model, field, phase, cfg, train_noise=False)
                pred_loss, camera_loss = apply_loss_roi(pred, camera, loss_roi_fraction)
                loss, comps = visual_loss(
                    pred_loss, camera_loss,
                    w_log_display_smooth_l1=cfg['w_log_display_smooth_l1'],
                    w_ssim=cfg['w_ssim'],
                    w_grad=cfg['w_grad'],
                    w_fft=cfg['w_fft'],
                    w_mean_norm_l1=cfg['w_mean_norm_l1'],
                    w_peak=cfg['w_peak'],
                    w_dark=cfg['w_dark'],
                    peak_margin=cfg['peak_margin'],
                    peak_top_fraction=cfg['peak_top_fraction'],
                    dark_margin=cfg['dark_margin'],
                    dark_top_fraction=cfg['dark_top_fraction'],
                )
                b = field.shape[0]
                sums['loss'] += loss.item() * b
                for k in comps:
                    sums[k] += comps[k] * b
                n_seen += b

        val_stats = {k: v / max(n_seen, 1) for k, v in sums.items()}

        history['train'].append(train_stats['loss'])
        history['val'].append(val_stats['loss'])
        history['train_unweighted'].append(train_stats['unweighted_loss'])
        history['train_systematic'].append(train_group_stats['systematic'])
        history['train_real_family'].append(train_group_stats['real_family'])
        history['train_other'].append(train_group_stats['other'])
        for k in ['log_display_smooth_l1', 'ssim', 'grad', 'fft', 'mean_norm_l1',
                  'peak', 'dark', 'raw_mse']:
            history[f'train_{k}'].append(train_stats[k])
            history[f'val_{k}'].append(val_stats[k])

        print(
            f"  Epoch {epoch + 1}: "
            f"train_weighted={train_stats['loss']:.5f} "
            f"train_unweighted={train_stats['unweighted_loss']:.5f} "
            f"train_sys={train_group_stats['systematic']:.5f} "
            f"train_real_family={train_group_stats['real_family']:.5f} "
            f"val_loss={val_stats['loss']:.5f} "
            f"val_mean_norm_l1={val_stats['mean_norm_l1']:.5f} "
            f"val_ssim={val_stats['ssim']:.5f} "
            f"val_grad={val_stats['grad']:.5f} "
            f"raw_mse={val_stats['raw_mse']:.5f}"
        )

        if val_stats['loss'] < best_val:
            best_val = val_stats['loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_loss': best_val,
                'cfg': cfg,
            }, run_dir / 'best.pt')

    return history


def visualize_samples(model, dataset, indices, device, cfg, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    roi_fraction = cfg.get('loss_roi_fraction', 1.0)
    has_roi = roi_fraction is not None and float(roi_fraction) < 1.0

    model.eval()
    with torch.no_grad():
        for idx in indices:
            batch = dataset[idx]
            field = resize_batch(batch['field'].unsqueeze(0).to(device),
                                 cfg['model_size'], mode='bilinear')
            sim = resize_batch(batch['sim'].unsqueeze(0).to(device),
                               cfg['model_size'], mode='bilinear')
            phase = batch['phase'].unsqueeze(0).to(device)
            cam = resize_batch(batch['camera'].unsqueeze(0).to(device),
                               cfg['model_size'], mode='bilinear')

            pred, _ = predict_camera(model, field, phase, cfg, train_noise=False)
            pred_roi, cam_roi = apply_loss_roi(pred, cam, roi_fraction)

            sim_np = sim[0, 0].cpu().numpy()
            cam_np = cam[0, 0].cpu().numpy()
            pred_np = pred[0, 0].cpu().numpy()
            cam_roi_np = cam_roi[0, 0].cpu().numpy()
            pred_roi_np = pred_roi[0, 0].cpu().numpy()

            sim_d = quantile_normalize_np(sim_np, q_max=0.999)
            cam_d = quantile_normalize_np(cam_np, q_max=0.999)
            pred_d = quantile_normalize_np(pred_np, q_max=0.999)
            err_d = np.abs(pred_d - cam_d)
            cam_roi_d = quantile_normalize_np(cam_roi_np, q_max=0.999)
            pred_roi_d = quantile_normalize_np(pred_roi_np, q_max=0.999)

            display_mse = float(np.mean((pred_roi_d - cam_roi_d) ** 2))
            raw_mse = float(np.mean((pred_roi_np - cam_roi_np) ** 2))

            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            ax[0].imshow(sim_d, cmap='gray', vmin=0, vmax=1)
            ax[0].set_title(f"Sim - {batch['id']}")
            ax[1].imshow(cam_d, cmap='gray', vmin=0, vmax=1)
            ax[1].set_title("Camera (target)")
            ax[2].imshow(pred_d, cmap='gray', vmin=0, vmax=1)
            metric_prefix = "ROI " if has_roi else ""
            ax[2].set_title(f"FNO\n{metric_prefix}Raw-MSE: {raw_mse:.4f}\n{metric_prefix}Display-MSE: {display_mse:.4f}")
            ax[3].imshow(err_d, cmap='hot', vmin=0, vmax=0.5)
            ax[3].set_title("|Pred - Cam|")
            if has_roi:
                y0, y1, x0, x1 = center_roi_bounds(pred_d.shape[0], pred_d.shape[1], roi_fraction)
                for a in ax:
                    a.add_patch(plt.Rectangle(
                        (x0 - 0.5, y0 - 0.5),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        edgecolor='deepskyblue',
                        linewidth=0.8,
                        linestyle=(0, (3, 3)),
                        alpha=0.9,
                    ))
            for a in ax:
                a.axis('off')
            plt.tight_layout()
            safe = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in batch['id'])
            plt.savefig(save_dir / f'sample_{safe}.png', dpi=120)
            plt.close()


def evaluate_all(model, dataset, device, cfg):
    model.eval()
    rows = []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='Final eval'):
            batch = dataset[idx]
            field = resize_batch(batch['field'].unsqueeze(0).to(device),
                                 cfg['model_size'], mode='bilinear')
            phase = batch['phase'].unsqueeze(0).to(device)
            cam = resize_batch(batch['camera'].unsqueeze(0).to(device),
                               cfg['model_size'], mode='bilinear')
            pred, _ = predict_camera(model, field, phase, cfg, train_noise=False)
            pred_eval, cam_eval = apply_loss_roi(pred, cam, cfg.get('loss_roi_fraction', 1.0))
            pred_d = log_display_tensor(pred_eval)
            cam_d = log_display_tensor(cam_eval)
            display_mse = F.mse_loss(pred_d, cam_d).item()
            raw_mse = F.mse_loss(pred_eval, cam_eval).item()
            ssim = 1.0 - simple_ssim_loss(pred_eval, cam_eval).item()
            rows.append((idx, batch['id'], raw_mse, display_mse, ssim))
    return rows


def plot_loss_curve(history, run_dir):
    if not history['train']:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(history['train'], label='train weighted', color='steelblue')
    if history.get('train_unweighted'):
        plt.plot(history['train_unweighted'], label='train unweighted',
                 color='steelblue', linestyle='--', alpha=0.7)
    plt.plot(history['val'], label='val', color='coral')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(run_dir) / 'loss_curve.png', dpi=140)
    plt.close()

    group_series = [
        ('train_systematic', 'systematic', 'tab:blue'),
        ('train_real_family', 'real + pert', 'tab:green'),
        ('train_other', 'other', 'tab:gray'),
    ]
    plt.figure(figsize=(8, 5))
    plotted = False
    for key, label, color in group_series:
        values = np.asarray(history.get(key, []), dtype=np.float64)
        if values.size and np.isfinite(values).any():
            plt.plot(values, label=label, color=color)
            plotted = True
    if plotted:
        plt.xlabel('Epoch')
        plt.ylabel('Unweighted per-sample loss')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(run_dir) / 'group_loss_curve.png', dpi=140)
    plt.close()


def write_per_sample_metrics(rows, dataset, train_idx, val_idx, run_dir, metric_prefix=''):
    run_dir = Path(run_dir)
    train_ids = {dataset.samples[i]['id'] for i in train_idx}
    val_ids = {dataset.samples[i]['id'] for i in val_idx}
    records = []

    for idx, sid, raw_mse, display_mse, ssim in rows:
        if sid in train_ids:
            split = 'train'
        elif sid in val_ids:
            split = 'val'
        else:
            split = 'unknown'
        records.append({
            'index': int(idx),
            'id': sid,
            'type': sample_type_from_id(sid),
            'group': sample_group_from_id(sid),
            'split': split,
            'raw_mse': float(raw_mse),
            'display_log_mse': float(display_mse),
            'ssim': float(ssim),
        })

    with open(run_dir / 'per_sample_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'index', 'id', 'type', 'group', 'split',
            f'{metric_prefix}raw_mse',
            f'{metric_prefix}display_log_mse',
            f'{metric_prefix}ssim',
        ])
        for record in records:
            writer.writerow([
                record['index'],
                record['id'],
                record['type'],
                record['group'],
                record['split'],
                record['raw_mse'],
                record['display_log_mse'],
                record['ssim'],
            ])

    summary = {}
    for record in records:
        key = (record['type'], record['group'], record['split'])
        bucket = summary.setdefault(key, {'n': 0, 'raw': [], 'display': [], 'ssim': []})
        bucket['n'] += 1
        bucket['raw'].append(record['raw_mse'])
        bucket['display'].append(record['display_log_mse'])
        bucket['ssim'].append(record['ssim'])

    with open(run_dir / 'per_sample_metrics_summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'type', 'group', 'split', 'n',
            f'mean_{metric_prefix}raw_mse',
            f'median_{metric_prefix}raw_mse',
            f'mean_{metric_prefix}display_log_mse',
            f'median_{metric_prefix}display_log_mse',
            f'mean_{metric_prefix}ssim',
            f'median_{metric_prefix}ssim',
        ])
        for (sample_type, group, split), bucket in sorted(summary.items()):
            writer.writerow([
                sample_type, group, split, bucket['n'],
                float(np.mean(bucket['raw'])),
                float(np.median(bucket['raw'])),
                float(np.mean(bucket['display'])),
                float(np.median(bucket['display'])),
                float(np.mean(bucket['ssim'])),
                float(np.median(bucket['ssim'])),
            ])

    return records


def plot_per_sample_metrics(records, run_dir, metric_prefix=''):
    if not records:
        return
    run_dir = Path(run_dir)

    group_colors = {
        'sine': 'tab:blue',
        'linear_ramp': 'tab:cyan',
        'checkerboard': 'tab:orange',
        'square_ring': 'tab:brown',
        'radial_ring': 'tab:red',
        'zernike_single': 'tab:purple',
        'zernike_combo': 'tab:pink',
        'local_bump': 'tab:green',
        'real': 'black',
        'pert': 'tab:olive',
        'systematic_other': 'tab:gray',
        'other': 'tab:gray',
    }
    split_markers = {'train': 'o', 'val': 'x', 'unknown': '.'}

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, metric_key, title, log_y in [
        (axes[0], 'raw_mse', f"{metric_prefix}raw MSE", True),
        (axes[1], 'display_log_mse', f"{metric_prefix}display log MSE", True),
        (axes[2], 'ssim', f"{metric_prefix}SSIM", False),
    ]:
        for group in sorted({r['group'] for r in records}):
            for split in ['train', 'val', 'unknown']:
                subset = [r for r in records if r['group'] == group and r['split'] == split]
                if not subset:
                    continue
                ax.scatter(
                    [r['index'] for r in subset],
                    [r[metric_key] for r in subset],
                    c=group_colors.get(group, 'tab:gray'),
                    marker=split_markers.get(split, '.'),
                    s=14 if split == 'train' else 28,
                    alpha=0.8,
                    label=f"{group} / {split}",
                )
        ax.set_title(f"Per-sample {title} by dataset index")
        ax.set_ylabel(title)
        if log_y and any(r[metric_key] > 0 for r in records):
            ax.set_yscale('log')
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("dataset index")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), fontsize=7, loc='best', ncol=3)
    plt.tight_layout()
    plt.savefig(run_dir / 'per_sample_mse_plot.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric_key, title in [
        (axes[0], 'raw_mse', f"{metric_prefix}raw MSE"),
        (axes[1], 'display_log_mse', f"{metric_prefix}display log MSE"),
        (axes[2], 'ssim', f"{metric_prefix}SSIM"),
    ]:
        for group in sorted({r['group'] for r in records}):
            vals = [r[metric_key] for r in records if r['group'] == group]
            if vals:
                ax.hist(vals, bins=30, alpha=0.45, label=group,
                        color=group_colors.get(group, 'tab:gray'))
        ax.set_title(f"Distribution of {title}")
        ax.set_xlabel(title)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / 'per_sample_mse_hist.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    POOL_DIR = r'H:\Shared drives\taylorlab\3DHL\CITL\Fourier Neural Operator_Training phase masks\06_14_2026_sample3_z6mm'
    OUTPUT_DIR = r'C:\REVAMP\Yangwoo Heo\SLM_to_Axicon_Optimiaztion\FNO_train_898sample_FOVfix'
    RUN_NAME = datetime.now().strftime('%Y%m%d_%H%M%S')
    EVAL_ONLY_RUN_DIR = None

    # Data
    WORKFLOW_FIELD_DIR = '1.Forward_Sim'
    WORKFLOW_CAMERA_DIR = '3.Aligned_Camera'
    WORKFLOW_PHASE_DIR = '0.Phase_Masks'
    # Set to None for full-frame training. 608 is a native ~60% crop of 1024
    # and avoids resizing the valid FOV back to the original resolution.
    FOV_CROP_SIZE = 608
    SIM_SIZE = FOV_CROP_SIZE if FOV_CROP_SIZE is not None else 1024
    MODEL_SIZE = SIM_SIZE
    PHASE_FLIP = True
    FIELD_SCALE_MODE = 'global_percentile'   # 'raw', 'global_percentile', or 'sample_norm'
    CAMERA_SCALE_MODE = 'global_percentile'  # 'raw', 'global_percentile', or 'sample_norm'
    FIELD_AMP_PERCENTILE = 99.9
    CAMERA_PERCENTILE = 99.9
    FIELD_AMP_SCALE = None
    CAMERA_SCALE = None
    CAMERA_BLACK_LEVEL = 0.0

    # FNO architecture
    PREDICTION_MODE = 'intensity'  # 'intensity' or 'direct_field'
    WIDTH = 12
    INTENSITY_ACTIVATION = 'softplus'
    DEPTH = 3
    MODES_Y = 120  # isolated blur trial: previous good run used 32
    MODES_X = 120
    MLP_WIDTH = 128
    OUTPUT_BIAS_INIT = -4.0  # softplus(-4) gives a dark, nonzero initial output

    # Axicon/SLM-aware conditioning
    USE_SLM_PHASE = False
    SLM_PHASE_WEIGHT = 0.10
    SLM_PHASE_LOWPASS_SIZE = 64
    SLM_PHASE_RING_RADIUS = None  # normalized radius in camera coordinates, e.g. 0.55
    SLM_PHASE_RING_WIDTH = None   # normalized Gaussian width, e.g. 0.08
    USE_GRID = True
    USE_RADIAL_COORD = True
    USE_AXICON_PHASE_MAP = False
    AXICON_SLOPE_RAD_PER_PIXEL = None  # set if fixed NA/lambda/pixel slope is known

    # Only used when PREDICTION_MODE == 'direct_field'
    FIELD_OUT_SCALE = 4.0
    FIELD_OUT_SCALE_MODE = 'field_rms'
    DIRECT_FIELD_ACTIVATION = 'tanh'
    DIRECT_FIELD_NOISE_STD = 0.05

    # Training
    REQUIRE_CUDA = False
    TRAIN_RATIO = 0.6
    BATCH_SIZE = 3
    EPOCHS = 100
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 1
    USE_AMP = False  # FFT + complex weights are safer in full fp32 for the first run
    GRAD_CLIP = 1.0
    USE_GROUP_LOSS_WEIGHTS = True
    GROUP_LOSS_WEIGHTS = {
        'systematic': 1.0,
        'real': 5.0,
        'pert': 5.0,
        'other': 1.0,
    }

    # Loss
    W_LOG_DISPLAY_SMOOTH_L1 = 0.0
    W_SSIM = 1.0
    W_GRAD = 0.10
    W_FFT = 0.0
    W_MEAN_NORM_L1 = 1.0
    W_PEAK = 0.0
    W_DARK = 0.0
    PEAK_MARGIN = 0.10
    PEAK_TOP_FRACTION = 0.002
    DARK_MARGIN = 0.10
    DARK_TOP_FRACTION = 0.002
    # A native FOV crop is already the supervised domain. When it is disabled,
    # retain the previous full-frame behavior with an 80% center loss ROI.
    LOSS_ROI_FRACTION = 1.0 if FOV_CROP_SIZE is not None else 0.80

    # Visualization
    N_VIS_TRAIN = 30
    N_VIS_VAL = 10
    SEED = 42

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print_torch_device_diagnostics()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if REQUIRE_CUDA and DEVICE != 'cuda':
        raise RuntimeError("REQUIRE_CUDA=True but CUDA is not available to this Python process.")
    eval_only = EVAL_ONLY_RUN_DIR is not None
    run_dir = Path(EVAL_ONLY_RUN_DIR) if eval_only else Path(OUTPUT_DIR) / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        'pool_dir': str(POOL_DIR),
        'workflow_field_dir': WORKFLOW_FIELD_DIR,
        'workflow_camera_dir': WORKFLOW_CAMERA_DIR,
        'workflow_phase_dir': WORKFLOW_PHASE_DIR,
        'fov_crop_size': FOV_CROP_SIZE,
        'sim_size': SIM_SIZE,
        'model_size': MODEL_SIZE,
        'phase_flip': PHASE_FLIP,
        'field_scale_mode': FIELD_SCALE_MODE,
        'camera_scale_mode': CAMERA_SCALE_MODE,
        'field_amp_percentile': FIELD_AMP_PERCENTILE,
        'camera_percentile': CAMERA_PERCENTILE,
        'prediction_mode': PREDICTION_MODE,
        'intensity_activation': INTENSITY_ACTIVATION,
        'width': WIDTH,
        'depth': DEPTH,
        'modes_y': MODES_Y,
        'modes_x': MODES_X,
        'mlp_width': MLP_WIDTH,
        'output_bias_init': OUTPUT_BIAS_INIT,
        'use_slm_phase': USE_SLM_PHASE,
        'slm_phase_weight': SLM_PHASE_WEIGHT,
        'slm_phase_lowpass_size': SLM_PHASE_LOWPASS_SIZE,
        'slm_phase_ring_radius': SLM_PHASE_RING_RADIUS,
        'slm_phase_ring_width': SLM_PHASE_RING_WIDTH,
        'use_grid': USE_GRID,
        'use_radial_coord': USE_RADIAL_COORD,
        'use_axicon_phase_map': USE_AXICON_PHASE_MAP,
        'axicon_slope_rad_per_pixel': AXICON_SLOPE_RAD_PER_PIXEL,
        'field_out_scale': FIELD_OUT_SCALE,
        'field_out_scale_mode': FIELD_OUT_SCALE_MODE,
        'direct_field_activation': DIRECT_FIELD_ACTIVATION,
        'direct_field_noise_std': DIRECT_FIELD_NOISE_STD,
        'train_ratio': TRAIN_RATIO,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'num_workers': NUM_WORKERS,
        'require_cuda': REQUIRE_CUDA,
        'use_amp': USE_AMP,
        'grad_clip': GRAD_CLIP,
        'use_group_loss_weights': USE_GROUP_LOSS_WEIGHTS,
        'group_loss_weights_raw': GROUP_LOSS_WEIGHTS,
        'w_log_display_smooth_l1': W_LOG_DISPLAY_SMOOTH_L1,
        'w_ssim': W_SSIM,
        'w_grad': W_GRAD,
        'w_fft': W_FFT,
        'w_mean_norm_l1': W_MEAN_NORM_L1,
        'w_peak': W_PEAK,
        'w_dark': W_DARK,
        'peak_margin': PEAK_MARGIN,
        'peak_top_fraction': PEAK_TOP_FRACTION,
        'dark_margin': DARK_MARGIN,
        'dark_top_fraction': DARK_TOP_FRACTION,
        'loss_roi_fraction': LOSS_ROI_FRACTION,
        'seed': SEED,
        'device': DEVICE,
    }

    print(f"=== FNO run: {RUN_NAME} ===")
    print(f"Device: {DEVICE}")
    print(f"Output dir: {run_dir}")

    dataset = AxiconFieldDataset(
        root_dir=POOL_DIR,
        workflow_field_dir=WORKFLOW_FIELD_DIR,
        workflow_camera_dir=WORKFLOW_CAMERA_DIR,
        workflow_phase_dir=WORKFLOW_PHASE_DIR,
        sim_size=SIM_SIZE,
        phase_flip_lr=PHASE_FLIP,
        field_scale_mode=FIELD_SCALE_MODE,
        field_amp_percentile=FIELD_AMP_PERCENTILE,
        field_amp_scale=FIELD_AMP_SCALE,
        camera_scale_mode=CAMERA_SCALE_MODE,
        camera_percentile=CAMERA_PERCENTILE,
        camera_scale=CAMERA_SCALE,
        camera_black_level=CAMERA_BLACK_LEVEL,
        fov_crop_size=FOV_CROP_SIZE,
    )
    cfg['field_amp_scale_actual'] = dataset.field_amp_scale
    cfg['camera_scale_actual'] = dataset.camera_scale

    if len(dataset) < 2:
        raise RuntimeError(f"Need at least 2 samples, found {len(dataset)}")

    split_path = run_dir / 'split.json'
    if eval_only and split_path.exists():
        with open(split_path, 'r', encoding='utf-8') as f:
            split_dump = json.load(f)
        id_to_idx = {sample['id']: i for i, sample in enumerate(dataset.samples)}
        train_idx = [id_to_idx[sid] for sid in split_dump.get('train', []) if sid in id_to_idx]
        val_idx = [id_to_idx[sid] for sid in split_dump.get('val', []) if sid in id_to_idx]
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
    else:
        train_set, val_set, train_idx, val_idx = split_dataset(
            dataset, train_ratio=TRAIN_RATIO, seed=SEED)
        if not eval_only:
            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'split_rule': (
                        'sys_* -> train fixed; real_* parent groups -> random train/val; '
                        'pert_* -> same split as matching real_* parent; others -> random train/val'
                    ),
                    'train_ratio_for_real_and_other': TRAIN_RATIO,
                    'train': [dataset.samples[i]['id'] for i in train_idx],
                    'val': [dataset.samples[i]['id'] for i in val_idx],
                }, f, indent=2)

    normalized_group_weights, train_group_counts = normalize_group_loss_weights(
        dataset,
        train_idx,
        GROUP_LOSS_WEIGHTS,
        enabled=USE_GROUP_LOSS_WEIGHTS,
    )
    cfg['group_loss_weights_normalized'] = normalized_group_weights
    cfg['train_group_counts'] = train_group_counts

    with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)

    pin_memory = DEVICE == 'cuda'
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin_memory)

    in_ch = infer_input_channels(
        use_slm_phase=USE_SLM_PHASE,
        use_grid=USE_GRID,
        use_radial_coord=USE_RADIAL_COORD,
        use_axicon_phase_map=USE_AXICON_PHASE_MAP,
        axicon_slope_rad_per_pixel=AXICON_SLOPE_RAD_PER_PIXEL,
    )
    out_ch = 2 if PREDICTION_MODE == 'direct_field' else 1

    model = AxiconFNO2d(
        in_ch=in_ch,
        out_ch=out_ch,
        width=WIDTH,
        modes_y=MODES_Y,
        modes_x=MODES_X,
        depth=DEPTH,
        mlp_width=MLP_WIDTH,
    ).to(DEVICE)
    if (PREDICTION_MODE == 'intensity' and INTENSITY_ACTIVATION == 'softplus' and
            OUTPUT_BIAS_INIT is not None):
        nn.init.constant_(model.project[-1].bias, float(OUTPUT_BIAS_INIT))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Input channels: {in_ch}")
    print(f"Model parameters: {n_params / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if eval_only:
        ckpt = torch.load(run_dir / 'best.pt', map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        history = {'train': [], 'val': []}
    else:
        history = train_one_run(model, train_loader, val_loader,
                                optimizer, DEVICE, cfg, run_dir)
        plot_loss_curve(history, run_dir)
        print(">>> Loading best checkpoint for final evaluation...")
        ckpt = torch.load(run_dir / 'best.pt', map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])

    rows = evaluate_all(model, dataset, DEVICE, cfg)
    roi_fraction = cfg.get('loss_roi_fraction', 1.0)
    metric_prefix = 'roi_' if roi_fraction is not None and float(roi_fraction) < 1.0 else ''
    metric_records = write_per_sample_metrics(
        rows, dataset, train_idx, val_idx, run_dir, metric_prefix=metric_prefix)
    plot_per_sample_metrics(metric_records, run_dir, metric_prefix=metric_prefix)

    vis_train = evenly_spaced_sample_indices(train_idx, N_VIS_TRAIN)
    vis_val = evenly_spaced_sample_indices(val_idx, N_VIS_VAL)
    if vis_train:
        visualize_samples(model, dataset, vis_train, DEVICE, cfg,
                          save_dir=run_dir / 'samples_train')
    if vis_val:
        visualize_samples(model, dataset, vis_val, DEVICE, cfg,
                          save_dir=run_dir / 'samples_val')

    print(f"\nDone. All outputs in: {run_dir}")
