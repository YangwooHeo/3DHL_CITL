"""Calibrate a compact physics proxy for the aligned SLM/axicon system.

The calibration data uses the same workflow-folder contract as
``train_fno_axicon.py``::

    pool/
      0.Phase_Masks/*.npy
      1.Forward_Sim/*.npy       # not read by this script
      3.Aligned_Camera/*.npy

Phase and camera files are paired by the same normalized-stem rule used by the
FNO trainer. The optical defaults are imported from ``axicon_simulator.py`` so
z, NA, upsampling, ROI, medium, orientation, and spatial filtering cannot drift
between simulation and calibration.

The learned proxy is deliberately low dimensional:

* A signed, DC-preserving (N*P) x (N*P) kernel describes sub-pixel SLM cross
  talk on voltage-proportional drive values before conversion to phase. N is
  the support in physical SLM pixels and P is the sub-pixel factor.
* Zernike coefficients describe SLM-plane wavefront error.
* A bounded, coarse source map describes smooth illumination non-uniformity.
* A polar complex transfer correction describes deterministic coherent
  camera-path aberrations with aggressive radial/azimuthal capacity.
* A bounded propagation-distance refinement describes a small camera-plane
  defocus around the simulator's fixed, full-resolution propagation distance.
* A bounded two-axis axicon-centre displacement describes lateral alignment
  through a differentiable shift-equivalent propagation path.
* A positive log-parameterized scalar describes camera/throughput gain.

The visual data term and grouped train/validation split are shared with the FNO
trainer. In particular, target-mean-normalized Smooth L1 keeps the gain
identifiable; independently RMS-normalizing prediction and target would cancel
the camera-scale parameter's gradient.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from axicon_simulator import (
    DEFAULT_APPLY_SPATIAL_FILTER,
    DEFAULT_ASM_MARGIN_FACTOR,
    DEFAULT_AXICON_ANGLE_IN_MEDIUM,
    DEFAULT_AXICON_DUTY_CYCLE,
    DEFAULT_AXICON_GRATING_PITCH_M,
    DEFAULT_AXICON_LATERAL_SHIFT_X_UM,
    DEFAULT_AXICON_LATERAL_SHIFT_Y_UM,
    DEFAULT_AXICON_PHASE_DEPTH_RAD,
    DEFAULT_AXICON_PROFILE,
    DEFAULT_AXICON_RADIAL_OFFSET,
    DEFAULT_FLIP_PHASE_FIRST_AXIS,
    DEFAULT_PHASE_LEVEL_MAX,
    DEFAULT_PROPAGATION_MEDIUM_INDEX,
    DEFAULT_ROI_SIZE,
    DEFAULT_TRANSPOSE_OUTPUT_FIELD,
    DEFAULT_TRANSPOSE_PHASE,
    DEFAULT_UPSAMPLE_FACTOR,
    DEFAULT_Z_TARGET_M,
    build_axicon_transfer_function,
    build_beam_config,
)
from mbvam.Beam.holobeam import HoloBeam
from train_fno_axicon import (
    normalize_group_loss_weights,
    sample_type_from_id,
    split_dataset,
    visual_loss_per_sample,
)


DEFAULT_POOL_DIR = (
    r"H:\Shared drives\taylorlab\3DHL\CITL\Fourier Neural Operator_Training phase masks"
    r"\06_14_2026_sample3_z6mm"
)


def torch_load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalized_sample_stem(path: Path) -> str:
    """Normalize IDs exactly like ``AxiconFieldDataset`` in the FNO trainer."""
    stem = Path(path).stem
    if stem.startswith("sine") and len(stem) > 4 and stem[4].isdigit():
        stem = "sine_" + stem[4:]
    return "_".join(str(int(part)) if part.isdigit() else part
                    for part in stem.split("_"))


def center_crop_numpy(array: np.ndarray, crop_size: int | None,
                      source_name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{source_name} must be 2D; got {array.shape}")
    if crop_size is None:
        return array
    height, width = array.shape
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"FOV crop {crop_size} exceeds {source_name} shape {array.shape}"
        )
    y0 = (height - crop_size) // 2
    x0 = (width - crop_size) // 2
    return array[y0:y0 + crop_size, x0:x0 + crop_size]


def center_crop_tensor(array: torch.Tensor, crop_size: int | None) -> torch.Tensor:
    if crop_size is None:
        return array
    height, width = array.shape[-2:]
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"FOV crop {crop_size} exceeds prediction shape {(height, width)}"
        )
    y0 = (height - crop_size) // 2
    x0 = (width - crop_size) // 2
    return array[..., y0:y0 + crop_size, x0:x0 + crop_size]


class ProxyCalibrationDataset(Dataset):
    """Read phase/camera pairs from the current FNO workflow layout."""

    def __init__(
        self,
        root_dir: Path | str,
        phase_dir: str = "0.Phase_Masks",
        camera_dir: str = "3.Aligned_Camera",
        fov_crop_size: int | None = 608,
        phase_level_max: float = DEFAULT_PHASE_LEVEL_MAX,
        phase_transpose: bool = DEFAULT_TRANSPOSE_PHASE,
        phase_flip_first_axis: bool = DEFAULT_FLIP_PHASE_FIRST_AXIS,
        expected_phase_shape: tuple[int, int] | None = None,
        camera_black_level: float = 0.0,
        camera_percentile: float = 99.9,
        camera_scale: float | None = None,
        scale_sample_pixels: int = 8192,
        seed: int = 42,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.phase_dir = phase_dir
        self.camera_dir = camera_dir
        self.fov_crop_size = fov_crop_size
        self.phase_level_max = float(phase_level_max)
        self.phase_transpose = bool(phase_transpose)
        self.phase_flip_first_axis = bool(phase_flip_first_axis)
        self.expected_phase_shape = expected_phase_shape
        self.camera_black_level = float(camera_black_level)
        self.camera_percentile = float(camera_percentile)
        self.scale_sample_pixels = int(scale_sample_pixels)
        self.seed = int(seed)

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")
        if self.phase_level_max <= 0:
            raise ValueError("phase_level_max must be positive")
        if not 0 < self.camera_percentile <= 100:
            raise ValueError("camera_percentile must be in (0, 100]")
        if self.scale_sample_pixels <= 0:
            raise ValueError("scale_sample_pixels must be positive")

        self.samples = self._discover_samples()
        if not self.samples:
            raise RuntimeError(f"No paired phase/camera samples found under {self.root_dir}")
        self.camera_scale = (
            float(camera_scale) if camera_scale is not None
            else self._estimate_camera_scale()
        )
        if not np.isfinite(self.camera_scale) or self.camera_scale <= 0:
            raise ValueError(f"camera_scale must be finite and positive; got {self.camera_scale}")

        print(f">>> Loaded {len(self.samples)} phase/camera pairs from {self.root_dir}")
        print(f">>> Camera scale: p{self.camera_percentile:g} ~= {self.camera_scale:.6g}")
        print(f">>> First sample ids: {', '.join(s['id'] for s in self.samples[:5])}")

    def _path_map(self, directory_name: str) -> dict[str, Path]:
        directory = self.root_dir / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Expected workflow directory not found: {directory}")
        result: dict[str, Path] = {}
        for path in sorted(directory.glob("*.npy")):
            sample_id = normalized_sample_stem(path)
            if sample_id in result:
                raise ValueError(
                    f"Duplicate normalized sample id {sample_id!r}: "
                    f"{result[sample_id].name}, {path.name}"
                )
            result[sample_id] = path
        return result

    def _discover_samples(self) -> list[dict[str, Path | str]]:
        phase_map = self._path_map(self.phase_dir)
        camera_map = self._path_map(self.camera_dir)
        common_ids = sorted(set(phase_map) & set(camera_map))
        missing_camera = sorted(set(phase_map) - set(camera_map))
        missing_phase = sorted(set(camera_map) - set(phase_map))
        if missing_camera:
            print(f">>> Phase files without camera match: {len(missing_camera)}")
            print(f">>>   first: {', '.join(missing_camera[:8])}")
        if missing_phase:
            print(f">>> Camera files without phase match: {len(missing_phase)}")
            print(f">>>   first: {', '.join(missing_phase[:8])}")
        return [
            {"id": sample_id, "phase": phase_map[sample_id],
             "camera": camera_map[sample_id]}
            for sample_id in common_ids
        ]

    def _load_camera_raw(self, path: Path) -> np.ndarray:
        array = np.load(path)
        if np.iscomplexobj(array):
            array = np.abs(array) ** 2
        array = np.squeeze(array)
        if array.ndim != 2:
            raise ValueError(f"{path} must contain 2D camera data; got {array.shape}")
        array = center_crop_numpy(array, self.fov_crop_size, f"camera {path.name}")
        array = np.nan_to_num(array.astype(np.float32), copy=False)
        return np.clip(array - self.camera_black_level, 0.0, None)

    def _estimate_camera_scale(self) -> float:
        """Estimate a dataset percentile without concatenating every full image."""
        rng = np.random.default_rng(self.seed)
        sampled = []
        for sample in self.samples:
            values = self._load_camera_raw(sample["camera"]).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size > self.scale_sample_pixels:
                indices = rng.choice(values.size, self.scale_sample_pixels, replace=False)
                values = values[indices]
            if values.size:
                sampled.append(values)
        if not sampled:
            return 1.0
        return max(float(np.percentile(np.concatenate(sampled),
                                       self.camera_percentile)), 1e-8)

    def _load_slm_drive(self, path: Path) -> torch.Tensor:
        """Load voltage-proportional SLM drive without phase wrapping."""
        drive = np.load(path)
        drive = np.squeeze(drive)
        if drive.ndim != 2:
            raise ValueError(f"{path} must contain a 2D phase map; got {drive.shape}")
        if self.phase_transpose:
            drive = drive.T
        if self.phase_flip_first_axis:
            drive = drive[::-1, :]
        drive = np.ascontiguousarray(drive, dtype=np.float32)
        if self.expected_phase_shape is not None and drive.shape != self.expected_phase_shape:
            raise ValueError(
                f"{path.name} has shape {drive.shape} after orientation correction; "
                f"expected {self.expected_phase_shape}"
            )
        if not np.isfinite(drive).all():
            raise ValueError(f"{path.name} contains non-finite SLM command values")
        tolerance = max(1e-6 * self.phase_level_max, 1e-6)
        drive_min = float(drive.min())
        drive_max = float(drive.max())
        if drive_min < -tolerance or drive_max > self.phase_level_max + tolerance:
            raise ValueError(
                f"{path.name} has SLM command range [{drive_min}, {drive_max}], "
                f"outside [0, {self.phase_level_max}]. The cross-talk model "
                "expects voltage-proportional command levels, not wrapped radians."
            )

        # This is a linear change of units only.  In particular, do not use
        # modulo/remainder or a complex phasor here: pixels at levels 0 and
        # phase_level_max represent different drive voltages to the cross-talk
        # filter even though their ideal complex phases are equivalent.
        drive /= self.phase_level_max
        return torch.from_numpy(drive)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        camera = self._load_camera_raw(sample["camera"]) / self.camera_scale
        return {
            "slm_drive": self._load_slm_drive(sample["phase"]).float(),
            "camera": torch.from_numpy(camera).unsqueeze(0).float(),
            "id": sample["id"],
        }


class AxiconProxyParameters(nn.Module):
    """Low-dimensional, physically constrained calibration parameters."""

    def __init__(self, num_zernike: int = 20,
                 source_grid_shape: tuple[int, int] = (64, 48),
                 source_max_deviation: float = 0.30,
                 crosstalk_kernel_size: int = 1,
                 crosstalk_subpixel_factor: int = 1,
                 xy_enabled: bool = False,
                 xy_initial_x_m: float = 0.0,
                 xy_initial_y_m: float = 0.0,
                 xy_min_x_m: float | None = None,
                 xy_max_x_m: float | None = None,
                 xy_min_y_m: float | None = None,
                 xy_max_y_m: float | None = None,
                 z_enabled: bool = False,
                 z_initial_m: float = DEFAULT_Z_TARGET_M,
                 z_min_m: float | None = None,
                 z_max_m: float | None = None,
                 transfer_enabled: bool = True,
                 transfer_radial_bins: int = 128,
                 transfer_azimuthal_order: int = 96,
                 transfer_angular_samples: int = 256,
                 transfer_max_log_amplitude: float = 1.0,
                 transfer_max_phase_rad: float = math.pi) -> None:
        super().__init__()
        if num_zernike <= 0:
            raise ValueError("num_zernike must be positive")
        if min(source_grid_shape) <= 0:
            raise ValueError("source_grid_shape values must be positive")
        if not 0 <= source_max_deviation < 1:
            raise ValueError("source_max_deviation must be in [0, 1)")
        if crosstalk_kernel_size <= 0:
            raise ValueError("crosstalk_kernel_size must be positive")
        if crosstalk_subpixel_factor <= 0:
            raise ValueError("crosstalk_subpixel_factor must be positive")
        if transfer_radial_bins < 2:
            raise ValueError("transfer_radial_bins must be at least 2")
        if transfer_azimuthal_order < 0:
            raise ValueError("transfer_azimuthal_order must be non-negative")
        minimum_angular_samples = 2 * transfer_azimuthal_order + 1
        if transfer_angular_samples < minimum_angular_samples:
            raise ValueError(
                "transfer_angular_samples must be at least "
                f"2 * transfer_azimuthal_order + 1 ({minimum_angular_samples})"
            )
        if transfer_max_log_amplitude <= 0 or transfer_max_phase_rad <= 0:
            raise ValueError("transfer amplitude/phase limits must be positive")
        if xy_min_x_m is None:
            xy_min_x_m = xy_initial_x_m - 25e-6
        if xy_max_x_m is None:
            xy_max_x_m = xy_initial_x_m + 25e-6
        if xy_min_y_m is None:
            xy_min_y_m = xy_initial_y_m - 25e-6
        if xy_max_y_m is None:
            xy_max_y_m = xy_initial_y_m + 25e-6
        if not all(math.isfinite(value) for value in (
            xy_initial_x_m, xy_initial_y_m,
            xy_min_x_m, xy_max_x_m, xy_min_y_m, xy_max_y_m,
        )):
            raise ValueError("axicon x/y shifts and bounds must be finite")
        if not xy_min_x_m < xy_initial_x_m < xy_max_x_m:
            raise ValueError(
                "x bounds must satisfy xy_min_x_m < xy_initial_x_m < xy_max_x_m"
            )
        if not xy_min_y_m < xy_initial_y_m < xy_max_y_m:
            raise ValueError(
                "y bounds must satisfy xy_min_y_m < xy_initial_y_m < xy_max_y_m"
            )
        if z_min_m is None:
            z_min_m = z_initial_m - 0.15e-3
        if z_max_m is None:
            z_max_m = z_initial_m + 0.15e-3
        if not 0 < z_min_m < z_initial_m < z_max_m:
            raise ValueError(
                "z bounds must satisfy 0 < z_min_m < z_initial_m < z_max_m"
            )
        self.zernike_coeffs = nn.Parameter(torch.zeros(num_zernike))
        self.source_latent = nn.Parameter(
            torch.zeros(1, 1, source_grid_shape[0], source_grid_shape[1])
        )
        self.log_camera_scale = nn.Parameter(torch.zeros(()))
        self.source_max_deviation = float(source_max_deviation)
        self.crosstalk_kernel_size = int(crosstalk_kernel_size)
        self.crosstalk_subpixel_factor = int(crosstalk_subpixel_factor)
        self.crosstalk_effective_kernel_size = (
            self.crosstalk_kernel_size * self.crosstalk_subpixel_factor
        )
        # Model-III-like free taps at sub-pixel resolution.  The learned raw
        # residual is mean-centered in slm_crosstalk_kernel(), so its sum is
        # zero and adding it to the discrete delta kernel preserves uniform
        # voltage exactly. Unlike the old softmax parameterization, signed
        # asymmetric lobes are representable and initialization is identity.
        self.crosstalk_kernel_residual = nn.Parameter(torch.zeros(
            self.crosstalk_effective_kernel_size,
            self.crosstalk_effective_kernel_size,
        ))
        self.xy_enabled = bool(xy_enabled)
        self.register_buffer(
            "xy_initial_m",
            torch.tensor([xy_initial_x_m, xy_initial_y_m], dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "xy_min_m",
            torch.tensor([xy_min_x_m, xy_min_y_m], dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "xy_max_m",
            torch.tensor([xy_max_x_m, xy_max_y_m], dtype=torch.float32),
            persistent=True,
        )
        normalized_xy = (
            2.0 * (self.xy_initial_m - self.xy_min_m)
            / (self.xy_max_m - self.xy_min_m) - 1.0
        ).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        self.xy_latent = nn.Parameter(torch.atanh(normalized_xy))
        self.z_enabled = bool(z_enabled)
        self.register_buffer(
            "z_initial_m", torch.tensor(float(z_initial_m)), persistent=True
        )
        self.register_buffer(
            "z_min_m", torch.tensor(float(z_min_m)), persistent=True
        )
        self.register_buffer(
            "z_max_m", torch.tensor(float(z_max_m)), persistent=True
        )
        normalized_z = (
            2.0 * (float(z_initial_m) - float(z_min_m))
            / (float(z_max_m) - float(z_min_m)) - 1.0
        )
        normalized_z = min(max(normalized_z, -1.0 + 1e-6), 1.0 - 1e-6)
        self.z_latent = nn.Parameter(
            torch.tensor(math.atanh(normalized_z), dtype=torch.float32)
        )
        self.transfer_enabled = bool(transfer_enabled)
        self.transfer_radial_bins = int(transfer_radial_bins)
        self.transfer_azimuthal_order = int(transfer_azimuthal_order)
        self.transfer_angular_samples = int(transfer_angular_samples)
        self.transfer_max_log_amplitude = float(transfer_max_log_amplitude)
        self.transfer_max_phase_rad = float(transfer_max_phase_rad)

        # Real polar Fourier series.  Cosine includes m=0; sine starts at m=1.
        # All-zero initialization makes the correction exactly C(rho, phi)=1.
        cos_shape = (self.transfer_radial_bins,
                     self.transfer_azimuthal_order + 1)
        sin_shape = (self.transfer_radial_bins,
                     self.transfer_azimuthal_order)
        self.transfer_log_amp_cos = nn.Parameter(torch.zeros(cos_shape))
        self.transfer_log_amp_sin = nn.Parameter(torch.zeros(sin_shape))
        self.transfer_phase_cos = nn.Parameter(torch.zeros(cos_shape))
        self.transfer_phase_sin = nn.Parameter(torch.zeros(sin_shape))

        # Coordinate grids and trigonometric bases are deterministic and can be
        # cached without entering checkpoints/state_dict.
        self._transfer_basis_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._transfer_grid_cache: dict[tuple, torch.Tensor] = {}
        self._propagation_phase_cache: dict[tuple, torch.Tensor] = {}
        self._detached_z_kernel_cache: dict[tuple, torch.Tensor] = {}
        self._lateral_frequency_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._detached_xy_kernel_cache: dict[tuple, torch.Tensor] = {}

    def source_map(self, shape: tuple[int, int]) -> torch.Tensor:
        latent = F.interpolate(
            self.source_latent,
            size=shape,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        return 1.0 + self.source_max_deviation * torch.tanh(latent)

    def camera_scale(self) -> torch.Tensor:
        return torch.exp(self.log_camera_scale.clamp(-20.0, 20.0))

    def axicon_lateral_shift_m(self, use_gradient: bool = True) -> torch.Tensor:
        """Return bounded physical axicon-centre shifts [axis0, axis1]."""
        latent = self.xy_latent if use_gradient else self.xy_latent.detach()
        midpoint = 0.5 * (self.xy_min_m + self.xy_max_m)
        half_range = 0.5 * (self.xy_max_m - self.xy_min_m)
        return midpoint + half_range * torch.tanh(latent)

    def delta_xy_m(self, use_gradient: bool = True) -> torch.Tensor:
        return self.axicon_lateral_shift_m(use_gradient) - self.xy_initial_m

    def invalidate_xy_kernel_cache(self) -> None:
        self._detached_xy_kernel_cache.clear()

    def slm_crosstalk_kernel(self, use_gradient: bool = True) -> torch.Tensor:
        """Return a signed, unit-sum sub-pixel voltage-domain kernel."""
        raw_residual = (
            self.crosstalk_kernel_residual
            if use_gradient else self.crosstalk_kernel_residual.detach()
        )
        residual = raw_residual - raw_residual.mean()
        return self.slm_crosstalk_identity() + residual

    def slm_crosstalk_identity(self) -> torch.Tensor:
        """Return the discrete-delta kernel for the configured sub-pixel grid."""
        identity = torch.zeros_like(self.crosstalk_kernel_residual)
        anchor = (self.crosstalk_effective_kernel_size - 1) // 2
        identity[anchor, anchor] = 1.0
        return identity

    def slm_crosstalk_residual(self, use_gradient: bool = True) -> torch.Tensor:
        """Return the learned signed deviation from the identity kernel."""
        return (
            self.slm_crosstalk_kernel(use_gradient)
            - self.slm_crosstalk_identity()
        )

    def filter_slm_drive(self, slm_drive: torch.Tensor,
                         use_gradient: bool = True) -> torch.Tensor:
        """Upsample drive, then apply sub-pixel cross talk without wrapping."""
        if slm_drive.ndim != 2:
            raise ValueError(f"Expected a 2D SLM drive map; got {slm_drive.shape}")
        if self.crosstalk_subpixel_factor > 1:
            slm_drive = slm_drive.repeat_interleave(
                self.crosstalk_subpixel_factor, dim=0
            ).repeat_interleave(self.crosstalk_subpixel_factor, dim=1)
        padding_before = (self.crosstalk_effective_kernel_size - 1) // 2
        padding_after = self.crosstalk_effective_kernel_size - 1 - padding_before
        drive_4d = slm_drive.unsqueeze(0).unsqueeze(0)
        if padding_before or padding_after:
            drive_4d = F.pad(
                drive_4d,
                (padding_before, padding_after, padding_before, padding_after),
                mode="replicate",
            )
        kernel = self.slm_crosstalk_kernel(use_gradient).unsqueeze(0).unsqueeze(0)
        return F.conv2d(drive_4d, kernel).squeeze(0).squeeze(0)

    def slm_phase_from_drive(self, slm_drive: torch.Tensor,
                             use_gradient: bool = True) -> torch.Tensor:
        """Convert filtered drive to radians without modulo or phase wrapping."""
        filtered_drive = self.filter_slm_drive(slm_drive, use_gradient)
        return filtered_drive * (2.0 * math.pi)

    def z_position_m(self, use_gradient: bool = True) -> torch.Tensor:
        """Return a smoothly bounded physical propagation distance in metres."""
        latent = self.z_latent if use_gradient else self.z_latent.detach()
        midpoint = 0.5 * (self.z_min_m + self.z_max_m)
        half_range = 0.5 * (self.z_max_m - self.z_min_m)
        return midpoint + half_range * torch.tanh(latent)

    def delta_z_m(self, use_gradient: bool = True) -> torch.Tensor:
        """Return the ROI refinement relative to the full simulator distance."""
        return self.z_position_m(use_gradient=use_gradient) - self.z_initial_m

    def invalidate_z_kernel_cache(self) -> None:
        """Discard the detached ROI kernel after a z optimizer update."""
        self._detached_z_kernel_cache.clear()

    def transfer_parameters(self) -> tuple[nn.Parameter, ...]:
        parameters = (
            self.transfer_log_amp_cos,
            self.transfer_log_amp_sin,
            self.transfer_phase_cos,
            self.transfer_phase_sin,
        )
        return tuple(parameter for parameter in parameters if parameter.numel() > 0)

    def _transfer_angular_basis(self, device: torch.device,
                                dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), dtype, self.transfer_azimuthal_order,
               self.transfer_angular_samples)
        cached = self._transfer_basis_cache.get(key)
        if cached is not None:
            return cached
        angles = torch.linspace(
            -math.pi, math.pi, self.transfer_angular_samples + 1,
            device=device, dtype=dtype,
        )
        cos_orders = torch.arange(
            self.transfer_azimuthal_order + 1, device=device, dtype=dtype
        )
        sin_orders = torch.arange(
            1, self.transfer_azimuthal_order + 1, device=device, dtype=dtype
        )
        cos_basis = torch.cos(cos_orders[:, None] * angles[None, :])
        sin_basis = torch.sin(sin_orders[:, None] * angles[None, :])
        self._transfer_basis_cache[key] = (cos_basis, sin_basis)
        return cos_basis, sin_basis

    def transfer_polar_maps(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return bounded log-amplitude and phase on a polar control grid."""
        dtype = self.transfer_phase_cos.dtype
        device = self.transfer_phase_cos.device
        cos_basis, sin_basis = self._transfer_angular_basis(device, dtype)

        raw_log_amp = self.transfer_log_amp_cos @ cos_basis
        raw_phase = self.transfer_phase_cos @ cos_basis
        if self.transfer_azimuthal_order > 0:
            raw_log_amp = raw_log_amp + self.transfer_log_amp_sin @ sin_basis
            raw_phase = raw_phase + self.transfer_phase_sin @ sin_basis

        log_amp = self.transfer_max_log_amplitude * torch.tanh(raw_log_amp)
        phase = self.transfer_max_phase_rad * torch.tanh(raw_phase)
        # Remove the two unidentifiable gauges: global amplitude is represented
        # by camera_scale and a phase piston cannot change measured intensity.
        # Mean removal can expand the tanh bounds, so enforce them once more.
        log_amp = (log_amp - log_amp.mean()).clamp(
            -self.transfer_max_log_amplitude,
            self.transfer_max_log_amplitude,
        )
        phase = (phase - phase.mean()).clamp(
            -self.transfer_max_phase_rad,
            self.transfer_max_phase_rad,
        )
        return log_amp, phase

    def _transfer_cartesian_grid(self, shape: tuple[int, int],
                                 device: torch.device,
                                 dtype: torch.dtype) -> torch.Tensor:
        key = (tuple(shape), str(device), dtype)
        cached = self._transfer_grid_cache.get(key)
        if cached is not None:
            return cached
        height, width = shape
        fy = torch.fft.fftfreq(height, device=device, dtype=dtype)
        fx = torch.fft.fftfreq(width, device=device, dtype=dtype)
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        radius_max = torch.sqrt(fy.abs().max().square() + fx.abs().max().square())
        angle = torch.atan2(fy[:, None], fx[None, :])
        grid = torch.stack(
            (angle / math.pi, 2.0 * radius / radius_max.clamp_min(1e-12) - 1.0),
            dim=-1,
        ).unsqueeze(0)
        self._transfer_grid_cache[key] = grid
        return grid

    def transfer_correction_maps(self, shape: tuple[int, int],
                                 dtype: torch.dtype | None = None,
                                 device: torch.device | None = None,
                                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return complex correction, log-amplitude, and phase on an FFT grid."""
        dtype = dtype or self.transfer_phase_cos.dtype
        device = device or self.transfer_phase_cos.device
        polar_log_amp, polar_phase = self.transfer_polar_maps()
        polar_log_amp = polar_log_amp.to(device=device, dtype=dtype)
        polar_phase = polar_phase.to(device=device, dtype=dtype)
        grid = self._transfer_cartesian_grid(shape, device, dtype)

        def sample(polar_map: torch.Tensor) -> torch.Tensor:
            return F.grid_sample(
                polar_map[None, None], grid,
                mode="bilinear", padding_mode="border", align_corners=True,
            )[0, 0]

        log_amp = sample(polar_log_amp)
        phase = sample(polar_phase)
        log_amp = (log_amp - log_amp.mean()).clamp(
            -self.transfer_max_log_amplitude,
            self.transfer_max_log_amplitude,
        )
        phase = (phase - phase.mean()).clamp(
            -self.transfer_max_phase_rad,
            self.transfer_max_phase_rad,
        )
        correction = torch.polar(log_amp.exp(), phase)
        return correction, log_amp, phase

    def _propagation_phase_slope(
        self,
        shape: tuple[int, int],
        pixel_size_m: float,
        wavelength_m: float,
        propagation_medium_index: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[tuple, torch.Tensor]:
        """Return d(ASM phase)/dz with the unobservable piston removed."""
        key = (
            tuple(shape), str(device), dtype, float(pixel_size_m),
            float(wavelength_m), float(propagation_medium_index),
        )
        cached = self._propagation_phase_cache.get(key)
        if cached is not None:
            return key, cached
        height, width = shape
        fy = torch.fft.fftfreq(
            height, d=pixel_size_m, device=device, dtype=dtype
        )
        fx = torch.fft.fftfreq(
            width, d=pixel_size_m, device=device, dtype=dtype
        )
        scaled_fy = wavelength_m * fy[:, None] / propagation_medium_index
        scaled_fx = wavelength_m * fx[None, :] / propagation_medium_index
        gamma_sq = 1.0 - scaled_fy.square() - scaled_fx.square()
        gamma = torch.sqrt(gamma_sq.clamp_min(0.0))
        wave_number = (
            2.0 * math.pi * propagation_medium_index / wavelength_m
        )
        phase_slope = wave_number * (gamma - 1.0)
        # Cropping spreads a little spectral energy outside the physical band.
        # Leaving that leakage unchanged preserves identity at delta_z=0.
        phase_slope = torch.where(
            gamma_sq >= 0.0, phase_slope, torch.zeros_like(phase_slope)
        )
        self._propagation_phase_cache[key] = phase_slope
        return key, phase_slope

    def _z_refinement_kernel(
        self,
        shape: tuple[int, int],
        pixel_size_m: float,
        wavelength_m: float,
        propagation_medium_index: float,
        device: torch.device,
        dtype: torch.dtype,
        enable_z_grad: bool,
    ) -> torch.Tensor:
        geometry_key, phase_slope = self._propagation_phase_slope(
            shape,
            pixel_size_m,
            wavelength_m,
            propagation_medium_index,
            device,
            dtype,
        )
        if enable_z_grad:
            return torch.exp(1j * phase_slope * self.delta_z_m(True))
        cached = self._detached_z_kernel_cache.get(geometry_key)
        if cached is None:
            cached = torch.exp(1j * phase_slope * self.delta_z_m(False)).detach()
            self._detached_z_kernel_cache[geometry_key] = cached
        return cached

    def _lateral_frequency_grid(
        self,
        shape: tuple[int, int],
        pixel_size_m: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[tuple, torch.Tensor, torch.Tensor]:
        key = (tuple(shape), str(device), dtype, float(pixel_size_m))
        cached = self._lateral_frequency_cache.get(key)
        if cached is None:
            frequency_axis0 = torch.fft.fftfreq(
                shape[0], d=pixel_size_m, device=device, dtype=dtype
            )[:, None]
            frequency_axis1 = torch.fft.fftfreq(
                shape[1], d=pixel_size_m, device=device, dtype=dtype
            )[None, :]
            cached = (frequency_axis0, frequency_axis1)
            self._lateral_frequency_cache[key] = cached
        return key, cached[0], cached[1]

    def _xy_translation_kernel(
        self,
        shape: tuple[int, int],
        pixel_size_m: float,
        device: torch.device,
        dtype: torch.dtype,
        enable_xy_grad: bool,
    ) -> torch.Tensor:
        geometry_key, frequency_axis0, frequency_axis1 = (
            self._lateral_frequency_grid(shape, pixel_size_m, device, dtype)
        )
        if enable_xy_grad:
            shift = self.axicon_lateral_shift_m(True)
            phase = -2.0 * math.pi * (
                frequency_axis0 * shift[0] + frequency_axis1 * shift[1]
            )
            return torch.exp(1j * phase)
        cached = self._detached_xy_kernel_cache.get(geometry_key)
        if cached is None:
            shift = self.axicon_lateral_shift_m(False)
            phase = -2.0 * math.pi * (
                frequency_axis0 * shift[0] + frequency_axis1 * shift[1]
            )
            cached = torch.exp(1j * phase).detach()
            self._detached_xy_kernel_cache[geometry_key] = cached
        return cached

    def apply_transfer_correction(
        self,
        field: torch.Tensor,
        *,
        pixel_size_m: float | None = None,
        wavelength_m: float | None = None,
        propagation_medium_index: float = 1.0,
        enable_z_grad: bool = False,
        enable_xy_grad: bool = False,
    ) -> torch.Tensor:
        """Apply transfer, delta-z, and output translation in one ROI FFT."""
        if not self.transfer_enabled and not self.z_enabled and not self.xy_enabled:
            return field
        if field.ndim != 2:
            raise ValueError(f"transfer correction expects a 2D field; got {field.shape}")
        multiplier = None
        if self.transfer_enabled:
            multiplier, _, _ = self.transfer_correction_maps(
                tuple(field.shape), dtype=field.real.dtype, device=field.device
            )
        if self.z_enabled:
            if pixel_size_m is None or wavelength_m is None:
                raise ValueError(
                    "pixel_size_m and wavelength_m are required when z refinement is enabled"
                )
            z_kernel = self._z_refinement_kernel(
                tuple(field.shape),
                pixel_size_m,
                wavelength_m,
                propagation_medium_index,
                field.device,
                field.real.dtype,
                enable_z_grad,
            )
            multiplier = z_kernel if multiplier is None else multiplier * z_kernel
        if self.xy_enabled:
            if pixel_size_m is None:
                raise ValueError(
                    "pixel_size_m is required when xy refinement is enabled"
                )
            xy_kernel = self._xy_translation_kernel(
                tuple(field.shape),
                pixel_size_m,
                field.device,
                field.real.dtype,
                enable_xy_grad,
            )
            multiplier = xy_kernel if multiplier is None else multiplier * xy_kernel
        spectrum = torch.fft.fft2(field, norm="ortho")
        return torch.fft.ifft2(spectrum * multiplier, norm="ortho")

    def transfer_regularization_terms(self) -> dict[str, torch.Tensor]:
        parameters = self.transfer_parameters()
        coefficient_l2 = sum(parameter.square().mean() for parameter in parameters)
        radial_smooth = sum(
            (parameter[1:] - parameter[:-1]).square().mean()
            for parameter in parameters
        )

        cos_order = torch.arange(
            self.transfer_azimuthal_order + 1,
            device=self.transfer_phase_cos.device,
            dtype=self.transfer_phase_cos.dtype,
        )
        cos_weight = (cos_order / max(self.transfer_azimuthal_order, 1)).square()
        angular_order = (
            (self.transfer_log_amp_cos.square() * cos_weight).mean()
            + (self.transfer_phase_cos.square() * cos_weight).mean()
        )
        if self.transfer_azimuthal_order > 0:
            sin_order = torch.arange(
                1, self.transfer_azimuthal_order + 1,
                device=self.transfer_phase_sin.device,
                dtype=self.transfer_phase_sin.dtype,
            )
            sin_weight = (sin_order / self.transfer_azimuthal_order).square()
            angular_order = angular_order + (
                (self.transfer_log_amp_sin.square() * sin_weight).mean()
                + (self.transfer_phase_sin.square() * sin_weight).mean()
            )
        return {
            "transfer_l2": coefficient_l2,
            "transfer_radial_smooth": radial_smooth,
            "transfer_angular_order": angular_order,
        }


def base_illumination(beam: HoloBeam) -> torch.Tensor:
    cfg = beam.beam_config
    if cfg.amplitude_profile_type == "gaussian":
        profile = beam.buildGaussianSourceProfile(cfg.gaussian_beam_waist)
    else:
        profile = beam.buildFlatTopSourceProfile()
    return profile.detach().to(device=cfg.device, dtype=cfg.fdtype)


def combined_illumination(base_profile: torch.Tensor,
                          proxy: AxiconProxyParameters) -> torch.Tensor:
    source = proxy.source_map(tuple(base_profile.shape))
    combined = base_profile * source
    return combined / combined.abs().mean().clamp_min(1e-12)


def axicon_forward_proxy(
    beam: HoloBeam,
    proxy: AxiconProxyParameters,
    slm_drive: torch.Tensor,
    base_profile: torch.Tensor,
    h_asm,
    cone_angle: float,
    upsample_factor: int,
    roi_size: int,
    propagation_medium_index: float,
    axicon_angle_in_medium: bool,
    axicon_transverse_frequency: float,
    axicon_profile: str,
    axicon_phase_depth: float,
    axicon_duty_cycle: float,
    axicon_radial_offset: float,
    fixed_axicon_shift_x_m: float,
    fixed_axicon_shift_y_m: float,
    apply_spatial_filter: bool,
    fov_crop_size: int | None,
    transpose_output: bool,
    require_grad: bool,
    enable_z_grad: bool = False,
    enable_xy_grad: bool = False,
) -> torch.Tensor:
    """Return one camera-domain prediction with simulator-matched propagation."""
    if proxy.xy_enabled and beam._normalize_axicon_profile(axicon_profile) != "continuous":
        raise ValueError(
            "Learnable axicon x/y displacement requires axicon_profile='continuous'"
        )
    beam.zernike_coeffs = (
        proxy.zernike_coeffs if require_grad else proxy.zernike_coeffs.detach()
    )
    profile = combined_illumination(base_profile, proxy)
    if not require_grad:
        profile = profile.detach()
    # Pixel cross talk is linear in the voltage-proportional SLM drive.  Only
    # after that spatial interaction do we convert to radians.  There is
    # intentionally no phase wrapping anywhere in this path.
    slm_phase = proxy.slm_phase_from_drive(
        slm_drive, use_gradient=require_grad
    )
    subpixel_factor = proxy.crosstalk_subpixel_factor
    if subpixel_factor > 1:
        profile = profile.repeat_interleave(
            subpixel_factor, dim=0
        ).repeat_interleave(subpixel_factor, dim=1)
    remaining_upsample_factor = upsample_factor // subpixel_factor
    if proxy.xy_enabled:
        learned_shift = proxy.axicon_lateral_shift_m(
            use_gradient=require_grad and enable_xy_grad
        )
        sample_offset_x = learned_shift[0]
        sample_offset_y = learned_shift[1]
        direct_axicon_shift_x = 0.0
        direct_axicon_shift_y = 0.0
    else:
        sample_offset_x = 0.0
        sample_offset_y = 0.0
        direct_axicon_shift_x = fixed_axicon_shift_x_m
        direct_axicon_shift_y = fixed_axicon_shift_y_m
    beam_amp = torch.ones((), device=slm_phase.device, dtype=slm_phase.dtype)
    volume = beam.propagateToVolume_Axicon2(
        axicon_angle=cone_angle,
        upsample_factor=remaining_upsample_factor,
        phase_mask=slm_phase,
        beam_mean_amplitude=beam_amp,
        slm_amplitude_profile=profile,
        H_asm=h_asm,
        convert_to_intensity=False,
        roi_size=roi_size,
        apply_spatial_filter=apply_spatial_filter,
        n_medium=propagation_medium_index,
        axicon_angle_in_medium=axicon_angle_in_medium,
        axicon_transverse_frequency=axicon_transverse_frequency,
        axicon_profile=axicon_profile,
        axicon_phase_depth=axicon_phase_depth,
        axicon_duty_cycle=axicon_duty_cycle,
        axicon_radial_offset=axicon_radial_offset,
        slm_input_subpixel_factor=subpixel_factor,
        slm_field_sample_offset_x=sample_offset_x,
        slm_field_sample_offset_y=sample_offset_y,
        apply_slm_field_sample_offset=proxy.xy_enabled,
        axicon_lateral_shift_x=direct_axicon_shift_x,
        axicon_lateral_shift_y=direct_axicon_shift_y,
    )
    field = proxy.apply_transfer_correction(
        volume[:, :, 0],
        pixel_size_m=float(beam.beam_config.psSLM) / upsample_factor,
        wavelength_m=float(beam.beam_config.lambda_),
        propagation_medium_index=propagation_medium_index,
        enable_z_grad=require_grad and enable_z_grad,
        enable_xy_grad=require_grad and enable_xy_grad,
    )
    intensity = field.abs().square()
    if transpose_output:
        intensity = intensity.transpose(0, 1)
    intensity = center_crop_tensor(intensity, fov_crop_size)
    intensity = intensity * proxy.camera_scale()
    return intensity.unsqueeze(0).unsqueeze(0)


def match_target_shape(prediction: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[-2:] == target.shape[-2:]:
        return prediction
    pred_h, pred_w = prediction.shape[-2:]
    target_h, target_w = target.shape[-2:]
    if pred_h >= target_h and pred_w >= target_w:
        y0 = (pred_h - target_h) // 2
        x0 = (pred_w - target_w) // 2
        return prediction[..., y0:y0 + target_h, x0:x0 + target_w]
    raise ValueError(
        f"Prediction {prediction.shape[-2:]} is smaller than target "
        f"{target.shape[-2:]}; resizing would change the physical pixel scale"
    )


def proxy_regularization(proxy: AxiconProxyParameters,
                         source_shape: tuple[int, int], cfg: dict,
                         enable_z_grad: bool = True,
                         enable_xy_grad: bool = True) -> tuple[torch.Tensor, dict]:
    source = proxy.source_map(source_shape)
    source_prior = (source - 1.0).square().mean()
    source_smooth = (
        (source[1:, :] - source[:-1, :]).square().mean()
        + (source[:, 1:] - source[:, :-1]).square().mean()
    )
    zernike_l2 = proxy.zernike_coeffs.square().mean()
    z_range = (proxy.z_max_m - proxy.z_min_m).clamp_min(1e-12)
    z_prior = (
        (proxy.z_position_m(use_gradient=enable_z_grad) - proxy.z_initial_m)
        / z_range
    ).square()
    xy_range = (proxy.xy_max_m - proxy.xy_min_m).clamp_min(1e-12)
    xy_prior = (
        proxy.delta_xy_m(use_gradient=enable_xy_grad) / xy_range
    ).square().mean()
    transfer_terms = proxy.transfer_regularization_terms()
    total = (
        cfg["w_source_prior"] * source_prior
        + cfg["w_source_smooth"] * source_smooth
        + cfg["w_zernike_l2"] * zernike_l2
        + cfg["w_z_prior"] * z_prior
        + cfg["w_xy_prior"] * xy_prior
        + cfg["w_transfer_l2"] * transfer_terms["transfer_l2"]
        + cfg["w_transfer_radial_smooth"] * transfer_terms["transfer_radial_smooth"]
        + cfg["w_transfer_angular_order"] * transfer_terms["transfer_angular_order"]
    )
    return total, {
        "source_prior": source_prior.detach(),
        "source_smooth": source_smooth.detach(),
        "zernike_l2": zernike_l2.detach(),
        "z_prior": z_prior.detach(),
        "xy_prior": xy_prior.detach(),
        **{name: value.detach() for name, value in transfer_terms.items()},
    }


def visual_loss(prediction: torch.Tensor, target: torch.Tensor,
                cfg: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses, components = visual_loss_per_sample(
        prediction,
        target,
        w_log_display_smooth_l1=cfg["w_log_display_smooth_l1"],
        w_ssim=cfg["w_ssim"],
        w_grad=cfg["w_grad"],
        w_fft=cfg["w_fft"],
        w_mean_norm_l1=cfg["w_mean_norm_l1"],
        w_peak=cfg["w_peak"],
        w_dark=cfg["w_dark"],
        peak_margin=cfg["peak_margin"],
        peak_top_fraction=cfg["peak_top_fraction"],
        dark_margin=cfg["dark_margin"],
        dark_top_fraction=cfg["dark_top_fraction"],
    )
    return losses.mean(), {name: value.mean() for name, value in components.items()}


def make_prediction(beam, proxy, sample, base_profile, physics, cfg,
                    device, require_grad, enable_z_grad=False,
                    enable_xy_grad=False):
    slm_drive = sample["slm_drive"].to(device)
    target = sample["camera"].to(device)
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if slm_drive.ndim != 2 or target.ndim != 4:
        raise ValueError(
            "Expected one 2D SLM drive and BCHW camera; got "
            f"{slm_drive.shape}, {target.shape}"
        )
    prediction = axicon_forward_proxy(
        beam=beam,
        proxy=proxy,
        slm_drive=slm_drive,
        base_profile=base_profile,
        h_asm=physics["h_asm"],
        cone_angle=physics["cone_angle"],
        upsample_factor=cfg["upsample_factor"],
        roi_size=cfg["roi_size"],
        propagation_medium_index=cfg["propagation_medium_index"],
        axicon_angle_in_medium=cfg["axicon_angle_in_medium"],
        axicon_transverse_frequency=physics["axicon_transverse_frequency"],
        axicon_profile=cfg["axicon_profile"],
        axicon_phase_depth=cfg["axicon_phase_depth_rad"],
        axicon_duty_cycle=cfg["axicon_duty_cycle"],
        axicon_radial_offset=cfg["axicon_radial_offset"],
        fixed_axicon_shift_x_m=cfg["axicon_shift_x_um"] * 1e-6,
        fixed_axicon_shift_y_m=cfg["axicon_shift_y_um"] * 1e-6,
        apply_spatial_filter=cfg["apply_spatial_filter"],
        fov_crop_size=cfg["fov_crop_size"],
        transpose_output=cfg["transpose_output"],
        require_grad=require_grad,
        enable_z_grad=enable_z_grad,
        enable_xy_grad=enable_xy_grad,
    )
    return match_target_shape(prediction, target), target


@torch.no_grad()
def initialize_camera_scale(beam, proxy, dataset, train_indices, base_profile,
                            physics, cfg, device) -> None:
    """Match mean energy on a few training samples before Adam starts."""
    ratios = []
    for index in train_indices[:min(3, len(train_indices))]:
        sample = dataset[index]
        prediction, target = make_prediction(
            beam, proxy, sample, base_profile, physics, cfg, device, False)
        pred_mean = prediction.mean().item()
        target_mean = target.mean().item()
        if pred_mean > 0 and target_mean > 0:
            ratios.append(target_mean / pred_mean)
    if ratios:
        scale = float(np.median(ratios))
        proxy.log_camera_scale.copy_(
            torch.tensor(math.log(max(scale, 1e-12)), device=device)
        )
        print(f">>> Initialized camera scale to {proxy.camera_scale().item():.6g}")


def train_epoch(beam, proxy, loader, optimizer, base_profile, physics,
                cfg, device, global_step: int) -> tuple[dict[str, float], int]:
    proxy.train()
    totals = {"loss": 0.0, "data": 0.0, "reg": 0.0, "raw_mse": 0.0}
    n_samples = 0
    for batch in tqdm(loader, desc="train", leave=False):
        enable_z_grad = (
            proxy.z_enabled
            and proxy.z_latent.requires_grad
            and global_step % cfg["z_update_every"] == 0
        )
        enable_xy_grad = (
            proxy.xy_enabled
            and proxy.xy_latent.requires_grad
            and global_step % cfg["xy_update_every"] == 0
        )
        optimizer.zero_grad(set_to_none=True)
        batch_size = len(batch["id"])
        batch_data = 0.0
        batch_mse = 0.0

        # Backward per sample so multiple upsampled FFT graphs are never retained.
        for i, sample_id in enumerate(batch["id"]):
            sample = {
                "slm_drive": batch["slm_drive"][i],
                "camera": batch["camera"][i:i + 1],
            }
            prediction, target = make_prediction(
                beam, proxy, sample, base_profile, physics, cfg, device, True,
                enable_z_grad, enable_xy_grad,
            )
            data_loss, components = visual_loss(prediction, target, cfg)
            group = sample_type_from_id(sample_id)
            weight = cfg["group_loss_weights_normalized"].get(group, 1.0)
            (weight * data_loss / batch_size).backward()
            batch_data += float((weight * data_loss).detach())
            batch_mse += float(components["raw_mse"].detach())

        reg_loss, _ = proxy_regularization(
            proxy,
            tuple(base_profile.shape),
            cfg,
            enable_z_grad=enable_z_grad,
            enable_xy_grad=enable_xy_grad,
        )
        if reg_loss.requires_grad:
            reg_loss.backward()
        if cfg["grad_clip"] is not None and cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(proxy.parameters(), cfg["grad_clip"])
        optimizer.step()
        if enable_z_grad:
            proxy.invalidate_z_kernel_cache()
        if enable_xy_grad:
            proxy.invalidate_xy_kernel_cache()
        global_step += 1

        totals["data"] += batch_data
        totals["reg"] += float(reg_loss.detach()) * batch_size
        totals["loss"] += batch_data + float(reg_loss.detach()) * batch_size
        totals["raw_mse"] += batch_mse
        n_samples += batch_size

    metrics = {name: value / max(n_samples, 1) for name, value in totals.items()}
    return metrics, global_step


@torch.no_grad()
def evaluate_indices(beam, proxy, dataset, indices, base_profile,
                     physics, cfg, device, description="val") -> dict[str, float]:
    proxy.eval()
    totals = {"loss": 0.0, "raw_mse": 0.0}
    for index in tqdm(indices, desc=description, leave=False):
        prediction, target = make_prediction(
            beam, proxy, dataset[index], base_profile, physics, cfg, device, False)
        loss, components = visual_loss(prediction, target, cfg)
        totals["loss"] += float(loss)
        totals["raw_mse"] += float(components["raw_mse"])
    return {name: value / max(len(indices), 1) for name, value in totals.items()}


def checkpoint_selection_score(train_metrics: dict[str, float],
                               val_metrics: dict[str, float],
                               has_validation: bool,
                               should_validate: bool) -> float:
    """Choose the available epoch metric used to update ``best.pt``."""
    if not has_validation:
        return float(train_metrics["loss"])
    if should_validate:
        return float(val_metrics["loss"])
    return float("inf")


def build_optimizer(proxy: AxiconProxyParameters, cfg: dict):
    groups = []
    specifications = [
        ([proxy.crosstalk_kernel_residual],
         cfg["lr_crosstalk"]
         if proxy.crosstalk_effective_kernel_size > 1 else 0.0,
         "slm_crosstalk"),
        ([proxy.zernike_coeffs], cfg["lr_zernike"], "zernike"),
        ([proxy.source_latent], cfg["lr_source"], "source"),
        ([proxy.log_camera_scale], cfg["lr_scale"], "camera_scale"),
        ([proxy.xy_latent], cfg["lr_xy"] if proxy.xy_enabled else 0.0,
         "axicon_xy"),
        ([proxy.z_latent], cfg["lr_z"] if proxy.z_enabled else 0.0,
         "propagation_z"),
        (list(proxy.transfer_parameters()),
         cfg["lr_transfer"] if proxy.transfer_enabled else 0.0,
         "polar_transfer"),
    ]
    for parameters, lr, name in specifications:
        for parameter in parameters:
            parameter.requires_grad_(lr > 0)
        if lr > 0:
            group = {"params": parameters, "lr": lr, "name": name}
            if name in {"propagation_z", "axicon_xy", "slm_crosstalk"}:
                group["weight_decay"] = 0.0
            groups.append(group)
        else:
            print(f">>> Frozen parameter group: {name}")
    if not groups:
        raise ValueError("At least one proxy learning rate must be positive")
    active_names = {group["name"] for group in groups}
    if active_names == {"propagation_z"} and cfg["z_update_every"] != 1:
        raise ValueError(
            "--z-update-every must be 1 when propagation z is the only "
            "trainable parameter group"
        )
    if active_names == {"axicon_xy"} and cfg["xy_update_every"] != 1:
        raise ValueError(
            "--xy-update-every must be 1 when axicon xy is the only "
            "trainable parameter group"
        )
    if active_names == {"propagation_z", "axicon_xy"} and (
        cfg["z_update_every"] != 1 and cfg["xy_update_every"] != 1
    ):
        raise ValueError(
            "When z and xy are the only trainable groups, at least one of "
            "--z-update-every or --xy-update-every must be 1"
        )
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def checkpoint_payload(proxy, optimizer, epoch, history, cfg, dataset,
                       train_indices, val_indices, source_shape, global_step):
    source_map = proxy.source_map(source_shape).detach().cpu()
    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "parameter_state_dict": proxy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "cfg": cfg,
        "train_ids": [dataset.samples[i]["id"] for i in train_indices],
        "val_ids": [dataset.samples[i]["id"] for i in val_indices],
        # Compatibility-friendly exported physical values.
        "zernike_coeffs": proxy.zernike_coeffs.detach().cpu(),
        "source_modulation_map": source_map,
        "slm_crosstalk_kernel": (
            proxy.slm_crosstalk_kernel(False).detach().cpu()
        ),
        "slm_crosstalk_support_pixels": proxy.crosstalk_kernel_size,
        "slm_crosstalk_subpixel_factor": proxy.crosstalk_subpixel_factor,
        "camera_scale_factor": proxy.camera_scale().detach().cpu(),
        "z_position_m": proxy.z_position_m(False).detach().cpu(),
        "delta_z_m": proxy.delta_z_m(False).detach().cpu(),
        "z_initial_m": proxy.z_initial_m.detach().cpu(),
        "z_min_m": proxy.z_min_m.detach().cpu(),
        "z_max_m": proxy.z_max_m.detach().cpu(),
        "axicon_lateral_shift_m": (
            proxy.axicon_lateral_shift_m(False).detach().cpu()
        ),
        "delta_xy_m": proxy.delta_xy_m(False).detach().cpu(),
        "xy_initial_m": proxy.xy_initial_m.detach().cpu(),
        "xy_min_m": proxy.xy_min_m.detach().cpu(),
        "xy_max_m": proxy.xy_max_m.detach().cpu(),
        "transfer_log_amp_cos": proxy.transfer_log_amp_cos.detach().cpu(),
        "transfer_log_amp_sin": proxy.transfer_log_amp_sin.detach().cpu(),
        "transfer_phase_cos": proxy.transfer_phase_cos.detach().cpu(),
        "transfer_phase_sin": proxy.transfer_phase_sin.detach().cpu(),
    }


def write_history(history: list[dict], run_dir: Path) -> None:
    if not history:
        return
    fieldnames = list(dict.fromkeys(
        key for row in history for key in row
    ))
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    with (run_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.semilogy([row["epoch"] for row in history],
                  [row["train_loss"] for row in history], label="train")
    val_x = [row["epoch"] for row in history if np.isfinite(row["val_loss"])]
    val_y = [row["val_loss"] for row in history if np.isfinite(row["val_loss"])]
    if val_x:
        axis.semilogy(val_x, val_y, marker="o", label="validation")
    axis.set(xlabel="Epoch", ylabel="Loss", title="Axicon proxy calibration")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    z_rows = [row for row in history if np.isfinite(row.get("delta_z_um", np.nan))]
    if z_rows:
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(
            [row["epoch"] for row in z_rows],
            [row["delta_z_um"] for row in z_rows],
            marker="o",
        )
        axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
        lower = z_rows[-1].get("z_min_delta_um")
        upper = z_rows[-1].get("z_max_delta_um")
        if lower is not None and upper is not None:
            axis.axhline(lower, color="tab:red", linestyle="--", alpha=0.6,
                         label="bounds")
            axis.axhline(upper, color="tab:red", linestyle="--", alpha=0.6)
            axis.legend()
        axis.set(
            xlabel="Epoch",
            ylabel="Learned delta z (um)",
            title="Bounded propagation-distance refinement",
        )
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "z_trajectory.png", dpi=150)
        plt.close(fig)

    xy_rows = [
        row for row in history
        if bool(row.get("xy_optimized", False))
        and np.isfinite(row.get("axicon_shift_x_um", np.nan))
        and np.isfinite(row.get("axicon_shift_y_um", np.nan))
    ]
    if xy_rows:
        fig, axis = plt.subplots(figsize=(8, 5))
        epochs = [row["epoch"] for row in xy_rows]
        axis.plot(
            epochs,
            [row["axicon_shift_x_um"] for row in xy_rows],
            marker="o",
            label="x (tensor axis 0)",
        )
        axis.plot(
            epochs,
            [row["axicon_shift_y_um"] for row in xy_rows],
            marker="o",
            label="y (tensor axis 1)",
        )
        final_row = xy_rows[-1]
        for key, color in (("x", "tab:blue"), ("y", "tab:orange")):
            lower = final_row.get(f"xy_{key}_min_um")
            upper = final_row.get(f"xy_{key}_max_um")
            if lower is not None and upper is not None:
                axis.axhline(lower, color=color, linestyle="--", alpha=0.35)
                axis.axhline(upper, color=color, linestyle="--", alpha=0.35)
        axis.set(
            xlabel="Epoch",
            ylabel="Axicon-centre displacement (um)",
            title="Bounded lateral axicon-alignment refinement",
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "xy_trajectory.png", dpi=150)
        plt.close(fig)


@torch.no_grad()
def save_parameter_plots(proxy: AxiconProxyParameters,
                         source_shape: tuple[int, int],
                         transfer_shape: tuple[int, int],
                         run_dir: Path) -> None:
    source = proxy.source_map(source_shape).detach().cpu().numpy()
    zernike = proxy.zernike_coeffs.detach().cpu().numpy()
    correction, log_amp, phase = proxy.transfer_correction_maps(transfer_shape)
    log_amp_np = torch.fft.fftshift(log_amp).detach().cpu().numpy()
    phase_np = torch.fft.fftshift(phase).detach().cpu().numpy()
    residual_kernel = torch.fft.fftshift(
        torch.fft.ifft2(correction - 1.0, norm="ortho")
    ).abs().square().detach().cpu().numpy()

    cos_power = (
        proxy.transfer_log_amp_cos.detach().square()
        + proxy.transfer_phase_cos.detach().square()
    )
    if proxy.transfer_azimuthal_order > 0:
        sin_power = (
            proxy.transfer_log_amp_sin.detach().square()
            + proxy.transfer_phase_sin.detach().square()
        )
        sin_power = F.pad(sin_power, (1, 0))
        coefficient_power = torch.sqrt(cos_power + sin_power)
    else:
        coefficient_power = torch.sqrt(cos_power)
    coefficient_power_np = coefficient_power.cpu().numpy()
    crosstalk_kernel = (
        proxy.slm_crosstalk_kernel(False).detach().cpu().numpy()
    )

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    image = axes[0, 0].imshow(source.T, cmap="RdBu_r", aspect="auto")
    axes[0, 0].set_title(
        f"Source modulation (min={source.min():.3f}, max={source.max():.3f})"
    )
    fig.colorbar(image, ax=axes[0, 0], label="Amplitude multiplier")

    axes[0, 1].bar(np.arange(len(zernike)), zernike)
    z_m = proxy.z_position_m(False).item()
    delta_z_um = proxy.delta_z_m(False).item() * 1e6
    xy_um = proxy.axicon_lateral_shift_m(False).detach().cpu().numpy() * 1e6
    delta_xy_um = proxy.delta_xy_m(False).detach().cpu().numpy() * 1e6
    axes[0, 1].set(
        xlabel="Zernike index", ylabel="Coefficient (rad)",
        title=(
            f"Camera scale = {proxy.camera_scale().item():.4g}\n"
            f"z = {z_m * 1e3:.6f} mm (delta={delta_z_um:+.3f} um)\n"
            f"axicon xy = ({xy_um[0]:+.3f}, {xy_um[1]:+.3f}) um "
            f"(delta=({delta_xy_um[0]:+.3f}, {delta_xy_um[1]:+.3f}) um)"
        ),
    )
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    limit = max(float(np.quantile(np.abs(log_amp_np), 0.999)), 1e-6)
    image = axes[0, 2].imshow(
        log_amp_np, cmap="RdBu_r", vmin=-limit, vmax=limit,
    )
    axes[0, 2].set_title(
        "Transfer log-amplitude" + ("" if proxy.transfer_enabled else " (disabled)")
    )
    axes[0, 2].axis("off")
    fig.colorbar(image, ax=axes[0, 2], fraction=0.046)

    phase_display_limit = max(float(np.quantile(np.abs(phase_np), 0.999)), 1e-6)
    image = axes[1, 0].imshow(
        phase_np,
        cmap="twilight",
        vmin=-phase_display_limit,
        vmax=phase_display_limit,
    )
    axes[1, 0].set_title("Transfer phase correction (rad)")
    axes[1, 0].axis("off")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].imshow(
        np.log1p(residual_kernel / max(float(np.quantile(residual_kernel, 0.999)), 1e-12)),
        cmap="magma",
    )
    axes[1, 1].set_title("Coherent residual PSF |IFFT(C-1)|²")
    axes[1, 1].axis("off")

    image = axes[1, 2].imshow(
        coefficient_power_np.T, origin="lower", aspect="auto", cmap="viridis",
    )
    axes[1, 2].set(
        xlabel="Radial control bin", ylabel="Azimuthal order m",
        title="Polar coefficient magnitude",
    )
    fig.colorbar(image, ax=axes[1, 2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(run_dir / "learned_proxy_parameters.png", dpi=150)
    plt.close(fig)

    np.savez_compressed(
        run_dir / "learned_transfer_correction.npz",
        log_amplitude=log_amp_np.astype(np.float32),
        phase_rad=phase_np.astype(np.float32),
        amplitude=np.exp(log_amp_np).astype(np.float32),
        residual_psf_intensity=residual_kernel.astype(np.float32),
        log_amp_cos=proxy.transfer_log_amp_cos.detach().cpu().numpy(),
        log_amp_sin=proxy.transfer_log_amp_sin.detach().cpu().numpy(),
        phase_cos=proxy.transfer_phase_cos.detach().cpu().numpy(),
        phase_sin=proxy.transfer_phase_sin.detach().cpu().numpy(),
        z_position_m=np.float64(z_m),
        delta_z_m=np.float64(proxy.delta_z_m(False).item()),
        z_initial_m=np.float64(proxy.z_initial_m.item()),
        z_min_m=np.float64(proxy.z_min_m.item()),
        z_max_m=np.float64(proxy.z_max_m.item()),
        axicon_lateral_shift_m=(xy_um * 1e-6).astype(np.float64),
        delta_xy_m=(delta_xy_um * 1e-6).astype(np.float64),
        xy_initial_m=proxy.xy_initial_m.detach().cpu().numpy().astype(np.float64),
        xy_min_m=proxy.xy_min_m.detach().cpu().numpy().astype(np.float64),
        xy_max_m=proxy.xy_max_m.detach().cpu().numpy().astype(np.float64),
    )

    identity_kernel = np.zeros_like(crosstalk_kernel)
    kernel_anchor = (proxy.crosstalk_effective_kernel_size - 1) // 2
    identity_kernel[kernel_anchor, kernel_anchor] = 1.0
    crosstalk_delta = crosstalk_kernel - identity_kernel
    kernel_limit = max(float(np.max(np.abs(crosstalk_kernel))), 1e-8)
    delta_limit = max(float(np.max(np.abs(crosstalk_delta))), 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, values, limit, title in [
        (axes[0], crosstalk_kernel, kernel_limit, "Learned kernel K"),
        (axes[1], crosstalk_delta, delta_limit, "Learned residual K - delta"),
    ]:
        image = axis.imshow(
            values,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        factor = proxy.crosstalk_subpixel_factor
        for boundary in range(factor, proxy.crosstalk_effective_kernel_size, factor):
            axis.axhline(boundary - 0.5, color="black", linewidth=0.5, alpha=0.4)
            axis.axvline(boundary - 0.5, color="black", linewidth=0.5, alpha=0.4)
        axis.set(
            xlabel="Sub-pixel tap",
            ylabel="Sub-pixel tap",
            title=title,
        )
        fig.colorbar(image, ax=axis, label="Coupling weight")
    fig.suptitle(
        f"SLM cross talk: {proxy.crosstalk_kernel_size}x"
        f"{proxy.crosstalk_kernel_size} pixel support, "
        f"P={proxy.crosstalk_subpixel_factor}, "
        f"K={proxy.crosstalk_effective_kernel_size}x"
        f"{proxy.crosstalk_effective_kernel_size}"
    )
    fig.tight_layout()
    fig.savefig(run_dir / "learned_slm_crosstalk_kernel.png", dpi=180)
    plt.close(fig)
    np.savez_compressed(
        run_dir / "learned_slm_crosstalk_kernel.npz",
        kernel=crosstalk_kernel.astype(np.float32),
        residual_from_identity=crosstalk_delta.astype(np.float32),
        support_size_slm_pixels=np.int64(proxy.crosstalk_kernel_size),
        subpixel_factor=np.int64(proxy.crosstalk_subpixel_factor),
        effective_kernel_size=np.int64(proxy.crosstalk_effective_kernel_size),
        parameterization=np.array("signed DC-preserving residual"),
        domain=np.array("normalized voltage-proportional SLM drive"),
        phase_stroke_rad=np.float64(2.0 * np.pi),
    )


def quantile_display(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(array, dtype=np.float32))
    low, high = np.quantile(array, [0.001, 0.999])
    return np.clip((array - low) / max(float(high - low), 1e-8), 0.0, 1.0)


@torch.no_grad()
def save_previews(beam, proxy, dataset, indices, base_profile,
                  physics, cfg, device, run_dir, split_name: str) -> None:
    preview_dir = run_dir / "samples" / split_name
    preview_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        sample = dataset[index]
        prediction, target = make_prediction(
            beam, proxy, sample, base_profile, physics, cfg, device, False)
        pred = prediction[0, 0].cpu().numpy()
        truth = target[0, 0].cpu().numpy()
        pred_d = quantile_display(pred)
        truth_d = quantile_display(truth)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for axis, image, title, cmap in [
            (axes[0], truth_d, "Aligned camera", "magma"),
            (axes[1], pred_d, "Calibrated simulation", "magma"),
            (axes[2], np.abs(pred_d - truth_d), "Display-normalized error", "inferno"),
        ]:
            axis.imshow(image, cmap=cmap, vmin=0, vmax=1)
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(f"{split_name}: {sample['id']}")
        fig.tight_layout()
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_"
                          for c in str(sample["id"]))
        fig.savefig(preview_dir / f"{index:04d}_{safe_id}.png", dpi=120)
        plt.close(fig)


def evenly_spaced(indices: list[int], count: int) -> list[int]:
    if count <= 0 or not indices:
        return []
    if len(indices) <= count:
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, count).round().astype(int)
    return [indices[position] for position in positions]


def parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "full"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return parsed


def resolve_device(requested: str, require_cuda: bool) -> torch.device:
    device = torch.device(
        "cuda" if requested == "auto" and torch.cuda.is_available()
        else "cpu" if requested == "auto"
        else requested
    )
    if require_cuda and device.type != "cuda":
        raise RuntimeError("--require-cuda was set but CUDA is unavailable/not selected")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=Path(DEFAULT_POOL_DIR))
    parser.add_argument("--phase-dir", default="0.Phase_Masks")
    parser.add_argument("--camera-dir", default="3.Aligned_Camera")
    parser.add_argument("--output-dir", type=Path, default=Path("runs_proxy_axicon"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate pairing/preprocessing without constructing the optical system.")

    parser.add_argument("--fov-crop-size", type=parse_optional_int, default=608)
    parser.add_argument("--camera-black-level", type=float, default=0.0)
    parser.add_argument("--camera-percentile", type=float, default=99.9)
    parser.add_argument("--camera-scale", type=float, default=None)
    parser.add_argument("--scale-sample-pixels", type=int, default=8192)
    parser.add_argument("--phase-level-max", type=float, default=DEFAULT_PHASE_LEVEL_MAX)
    parser.add_argument("--phase-transpose", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_TRANSPOSE_PHASE)
    parser.add_argument("--phase-flip-first-axis", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_FLIP_PHASE_FIRST_AXIS)
    parser.add_argument("--transpose-output", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_TRANSPOSE_OUTPUT_FIELD)

    parser.add_argument("--z-m", type=float, default=DEFAULT_Z_TARGET_M)
    parser.add_argument(
        "--optimize-z", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "learn a bounded ROI-domain delta-z around --z-m; the expensive "
            "full-resolution ASM remains fixed at --z-m"
        ),
    )
    parser.add_argument(
        "--z-min-m", type=float, default=None,
        help="lower z bound in metres (default: --z-m minus 0.15 mm)",
    )
    parser.add_argument(
        "--z-max-m", type=float, default=None,
        help="upper z bound in metres (default: --z-m plus 0.15 mm)",
    )
    parser.add_argument(
        "--z-update-every", type=int, default=1,
        help="enable z gradients every K optimizer steps; other groups still update",
    )
    parser.add_argument(
        "--axicon-profile",
        choices=("continuous", "binary"),
        default="continuous",
        help=(
            "axicon phase profile used by both sparse ASM construction and "
            "forward propagation; proxy calibration defaults to continuous "
            f"(simulator default: {DEFAULT_AXICON_PROFILE}) and learnable xy "
            "is continuous-only"
        ),
    )
    parser.add_argument(
        "--axicon-phase-depth-rad",
        type=float,
        default=DEFAULT_AXICON_PHASE_DEPTH_RAD,
    )
    parser.add_argument(
        "--axicon-duty-cycle", type=float, default=DEFAULT_AXICON_DUTY_CYCLE
    )
    parser.add_argument(
        "--axicon-radial-offset", type=float, default=DEFAULT_AXICON_RADIAL_OFFSET
    )
    parser.add_argument(
        "--axicon-shift-x-um",
        type=float,
        default=DEFAULT_AXICON_LATERAL_SHIFT_X_UM,
        help=(
            "fixed axicon-centre x displacement, or the initial value when "
            "--optimize-xy is enabled (x is tensor axis 0)"
        ),
    )
    parser.add_argument(
        "--axicon-shift-y-um",
        type=float,
        default=DEFAULT_AXICON_LATERAL_SHIFT_Y_UM,
        help=(
            "fixed axicon-centre y displacement, or the initial value when "
            "--optimize-xy is enabled (y is tensor axis 1)"
        ),
    )
    parser.add_argument(
        "--optimize-xy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "co-optimize bounded continuous-axicon x/y displacement via a "
            "differentiable shift-equivalent propagation path"
        ),
    )
    parser.add_argument(
        "--xy-min-x-um", type=float, default=None,
        help="lower x bound in um (default: initial x minus 25 um)",
    )
    parser.add_argument(
        "--xy-max-x-um", type=float, default=None,
        help="upper x bound in um (default: initial x plus 25 um)",
    )
    parser.add_argument(
        "--xy-min-y-um", type=float, default=None,
        help="lower y bound in um (default: initial y minus 25 um)",
    )
    parser.add_argument(
        "--xy-max-y-um", type=float, default=None,
        help="upper y bound in um (default: initial y plus 25 um)",
    )
    parser.add_argument(
        "--xy-update-every", type=int, default=1,
        help="enable x/y gradients every K optimizer steps; other groups still update",
    )
    parser.add_argument("--roi-size", type=int, default=DEFAULT_ROI_SIZE)
    parser.add_argument("--upsample-factor", type=int, default=DEFAULT_UPSAMPLE_FACTOR)
    parser.add_argument("--axicon-grating-pitch-m", type=float,
                        default=DEFAULT_AXICON_GRATING_PITCH_M)
    parser.add_argument("--propagation-medium-index", type=float,
                        default=DEFAULT_PROPAGATION_MEDIUM_INDEX)
    parser.add_argument("--axicon-angle-in-medium", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_AXICON_ANGLE_IN_MEDIUM)
    parser.add_argument("--apply-spatial-filter", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_APPLY_SPATIAL_FILTER)
    parser.add_argument("--asm-margin-factor", type=float,
                        default=DEFAULT_ASM_MARGIN_FACTOR)

    parser.add_argument("--num-zernike", type=int, default=20)
    parser.add_argument("--source-grid-x", type=int, default=64)
    parser.add_argument("--source-grid-y", type=int, default=48)
    parser.add_argument("--source-max-deviation", type=float, default=0.30)
    parser.add_argument(
        "--crosstalk-kernel-size",
        type=int,
        default=1,
        metavar="N",
        help=(
            "cross-talk support in physical SLM pixels; the learned kernel "
            "has size (N*P)x(N*P)"
        ),
    )
    parser.add_argument(
        "--crosstalk-subpixel-factor",
        type=int,
        default=1,
        metavar="P",
        help=(
            "nearest-neighbor sub-pixels per SLM pixel before cross-talk "
            "convolution; must divide --upsample-factor"
        ),
    )
    parser.add_argument(
        "--transfer-correction", action=argparse.BooleanOptionalAction,
        default=True,
        help="learn a coherent polar complex transfer correction on the propagated ROI",
    )
    parser.add_argument("--transfer-radial-bins", type=int, default=128)
    parser.add_argument("--transfer-azimuthal-order", type=int, default=96)
    parser.add_argument(
        "--transfer-angular-samples", type=int, default=256,
        help="polar synthesis samples; must be >= 2*azimuthal_order+1",
    )
    parser.add_argument("--transfer-max-log-amplitude", type=float, default=1.0)
    parser.add_argument("--transfer-max-phase-rad", type=float, default=math.pi)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-zernike", type=float, default=2e-2)
    parser.add_argument("--lr-source", type=float, default=1e-2)
    parser.add_argument("--lr-scale", type=float, default=2e-2)
    parser.add_argument("--lr-z", type=float, default=1e-2)
    parser.add_argument("--lr-xy", type=float, default=2e-2)
    parser.add_argument("--lr-transfer", type=float, default=5e-3)
    parser.add_argument("--lr-crosstalk", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--w-source-prior", type=float, default=1e-2)
    parser.add_argument("--w-source-smooth", type=float, default=1e-3)
    parser.add_argument("--w-zernike-l2", type=float, default=1e-4)
    parser.add_argument(
        "--w-z-prior", type=float, default=0.0,
        help="optional normalized quadratic prior toward the initial --z-m",
    )
    parser.add_argument(
        "--w-xy-prior", type=float, default=0.0,
        help="optional normalized quadratic prior toward the initial x/y shift",
    )
    parser.add_argument("--w-transfer-l2", type=float, default=1e-5)
    parser.add_argument("--w-transfer-radial-smooth", type=float, default=1e-5)
    parser.add_argument("--w-transfer-angular-order", type=float, default=1e-6)

    # Same visual objective family as train_fno_axicon.py.
    parser.add_argument("--w-log-display-smooth-l1", type=float, default=0.0)
    parser.add_argument("--w-ssim", type=float, default=1.0)
    parser.add_argument("--w-grad", type=float, default=0.10)
    parser.add_argument("--w-fft", type=float, default=0.0)
    parser.add_argument("--w-mean-norm-l1", type=float, default=1.0)
    parser.add_argument("--w-peak", type=float, default=0.0)
    parser.add_argument("--w-dark", type=float, default=0.0)
    parser.add_argument("--peak-margin", type=float, default=0.10)
    parser.add_argument("--peak-top-fraction", type=float, default=0.002)
    parser.add_argument("--dark-margin", type=float, default=0.10)
    parser.add_argument("--dark-top-fraction", type=float, default=0.002)

    parser.add_argument("--sys-train-ratio", type=float, default=0.90)
    parser.add_argument("--real-train-ratio", type=float, default=0.80)
    parser.add_argument("--use-group-loss-weights", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--weight-systematic", type=float, default=1.0)
    parser.add_argument("--weight-real", type=float, default=5.0)
    parser.add_argument("--weight-pert", type=float, default=5.0)
    parser.add_argument("--weight-other", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--val-max-samples", type=int, default=16)
    parser.add_argument(
        "--n-vis",
        type=int,
        default=4,
        help=(
            "legacy fallback preview count used when --n-vis-train or "
            "--n-vis-val is omitted"
        ),
    )
    parser.add_argument(
        "--n-vis-train",
        type=int,
        default=None,
        help="maximum number of final train previews; 0 disables them",
    )
    parser.add_argument(
        "--n-vis-val",
        type=int,
        default=None,
        help="maximum number of final validation previews; 0 disables them",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--initialize-camera-scale", action=argparse.BooleanOptionalAction,
                        default=True)
    return parser


def config_from_args(args, device, beam_config) -> dict:
    cfg = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    transverse_frequency = 1.0 / args.axicon_grating_pitch_m
    cfg.update({
        "device": str(device),
        "wavelength_m": float(beam_config.lambda_),
        "slm_pixel_size_m": float(beam_config.psSLM),
        "roi_pixel_size_m": float(beam_config.psSLM) / args.upsample_factor,
        "slm_shape": [int(beam_config.Nx), int(beam_config.Ny)],
        "gaussian_beam_waist_m": float(beam_config.gaussian_beam_waist),
        "axicon_transverse_frequency": transverse_frequency,
        "axicon_na_air_equiv": float(beam_config.lambda_) * transverse_frequency,
        "group_loss_weights_raw": {
            "systematic": args.weight_systematic,
            "real": args.weight_real,
            "pert": args.weight_pert,
            "other": args.weight_other,
        },
    })
    return cfg


def main() -> None:
    args = build_parser().parse_args()
    if args.z_min_m is None:
        args.z_min_m = args.z_m - 0.15e-3
    if args.z_max_m is None:
        args.z_max_m = args.z_m + 0.15e-3
    if args.xy_min_x_um is None:
        args.xy_min_x_um = args.axicon_shift_x_um - 25.0
    if args.xy_max_x_um is None:
        args.xy_max_x_um = args.axicon_shift_x_um + 25.0
    if args.xy_min_y_um is None:
        args.xy_min_y_um = args.axicon_shift_y_um - 25.0
    if args.xy_max_y_um is None:
        args.xy_max_y_um = args.axicon_shift_y_um + 25.0
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.val_every <= 0 or args.val_max_samples <= 0:
        raise ValueError("--val-every and --val-max-samples must be positive")
    if args.n_vis < 0:
        raise ValueError("--n-vis must be non-negative")
    if args.n_vis_train is None:
        args.n_vis_train = args.n_vis
    if args.n_vis_val is None:
        args.n_vis_val = args.n_vis
    if args.n_vis_train < 0 or args.n_vis_val < 0:
        raise ValueError("--n-vis-train and --n-vis-val must be non-negative")
    if args.z_m <= 0 or args.roi_size <= 0 or args.upsample_factor <= 0:
        raise ValueError("z, ROI size, and upsample factor must be positive")
    if not 0 < args.z_min_m < args.z_m < args.z_max_m:
        raise ValueError(
            "z bounds must satisfy 0 < --z-min-m < --z-m < --z-max-m"
        )
    if args.z_update_every <= 0:
        raise ValueError("--z-update-every must be positive")
    xy_values = (
        args.axicon_shift_x_um,
        args.axicon_shift_y_um,
        args.xy_min_x_um,
        args.xy_max_x_um,
        args.xy_min_y_um,
        args.xy_max_y_um,
    )
    if not all(math.isfinite(value) for value in xy_values):
        raise ValueError("axicon x/y shifts and bounds must be finite")
    if not args.xy_min_x_um < args.axicon_shift_x_um < args.xy_max_x_um:
        raise ValueError(
            "x bounds must satisfy --xy-min-x-um < --axicon-shift-x-um "
            "< --xy-max-x-um"
        )
    if not args.xy_min_y_um < args.axicon_shift_y_um < args.xy_max_y_um:
        raise ValueError(
            "y bounds must satisfy --xy-min-y-um < --axicon-shift-y-um "
            "< --xy-max-y-um"
        )
    if args.xy_update_every <= 0:
        raise ValueError("--xy-update-every must be positive")
    if args.optimize_xy and args.axicon_profile != "continuous":
        raise ValueError(
            "--optimize-xy is supported only with "
            "--axicon-profile continuous"
        )
    if not math.isfinite(args.axicon_phase_depth_rad):
        raise ValueError("--axicon-phase-depth-rad must be finite")
    if not 0.0 < args.axicon_duty_cycle < 1.0:
        raise ValueError("--axicon-duty-cycle must be strictly between 0 and 1")
    if not math.isfinite(args.axicon_radial_offset):
        raise ValueError("--axicon-radial-offset must be finite")
    if args.propagation_medium_index <= 0 or args.axicon_grating_pitch_m <= 0:
        raise ValueError("medium index and axicon grating pitch must be positive")
    if args.fov_crop_size is not None and args.fov_crop_size > args.roi_size:
        raise ValueError("--fov-crop-size cannot exceed --roi-size")
    if args.transfer_radial_bins < 2 or args.transfer_azimuthal_order < 0:
        raise ValueError("invalid polar transfer radial/order configuration")
    if args.transfer_angular_samples < 2 * args.transfer_azimuthal_order + 1:
        raise ValueError(
            "--transfer-angular-samples must be >= "
            "2*--transfer-azimuthal-order+1"
        )
    if args.transfer_max_log_amplitude <= 0 or args.transfer_max_phase_rad <= 0:
        raise ValueError("transfer correction limits must be positive")
    if args.crosstalk_kernel_size <= 0:
        raise ValueError("--crosstalk-kernel-size must be positive")
    if args.crosstalk_subpixel_factor <= 0:
        raise ValueError("--crosstalk-subpixel-factor must be positive")
    if args.upsample_factor % args.crosstalk_subpixel_factor != 0:
        raise ValueError(
            "--crosstalk-subpixel-factor must divide --upsample-factor so "
            "the final optical sampling grid remains unchanged"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device, args.require_cuda)

    beam_config = build_beam_config()
    beam_config.device = device
    expected_phase_shape = (beam_config.Nx, beam_config.Ny)
    dataset = ProxyCalibrationDataset(
        root_dir=args.root_dir,
        phase_dir=args.phase_dir,
        camera_dir=args.camera_dir,
        fov_crop_size=args.fov_crop_size,
        phase_level_max=args.phase_level_max,
        phase_transpose=args.phase_transpose,
        phase_flip_first_axis=args.phase_flip_first_axis,
        expected_phase_shape=expected_phase_shape,
        camera_black_level=args.camera_black_level,
        camera_percentile=args.camera_percentile,
        camera_scale=args.camera_scale,
        scale_sample_pixels=args.scale_sample_pixels,
        seed=args.seed,
    )
    if len(dataset) < 1:
        raise RuntimeError("At least one paired sample is required for calibration")
    first = dataset[0]
    print(f">>> First SLM drive shape: {tuple(first['slm_drive'].shape)}")
    print(f">>> First camera shape: {tuple(first['camera'].shape)}")
    if args.dry_run:
        print(">>> Dry run complete: pairing and preprocessing are valid.")
        return
    _, _, train_indices, val_indices = split_dataset(
        dataset,
        real_train_ratio=args.real_train_ratio,
        sys_train_ratio=args.sys_train_ratio,
        seed=args.seed,
    )
    if not train_indices:
        raise RuntimeError("The configured split produced an empty training set")
    has_validation = bool(val_indices)
    eval_indices = (
        evenly_spaced(val_indices, args.val_max_samples)
        if has_validation else []
    )
    train_preview_indices = evenly_spaced(train_indices, args.n_vis_train)
    val_preview_indices = evenly_spaced(val_indices, args.n_vis_val)
    checkpoint_metric = "val_loss" if has_validation else "train_loss"

    cfg = config_from_args(args, device, beam_config)
    cfg["z_refinement_model"] = (
        "ROI angular-spectrum delta-z fused with polar transfer FFT; "
        "full-resolution ASM fixed at z_m"
    )
    cfg["xy_refinement_model"] = (
        "continuous-axicon shift equivariance: sample the filtered SLM field "
        "at U(x+s), propagate through a centred axicon, then apply T_s in "
        "the existing ROI transfer FFT"
    )
    cfg["camera_scale_actual"] = dataset.camera_scale
    cfg["checkpoint_selection_metric"] = checkpoint_metric
    normalized_weights, counts = normalize_group_loss_weights(
        dataset,
        train_indices,
        cfg["group_loss_weights_raw"],
        enabled=args.use_group_loss_weights,
    )
    cfg["group_loss_weights_normalized"] = normalized_weights
    cfg["train_group_counts"] = counts

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)
    with (run_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "train": [dataset.samples[i]["id"] for i in train_indices],
            "val": [dataset.samples[i]["id"] for i in val_indices],
            "validation_evaluated": [dataset.samples[i]["id"] for i in eval_indices],
            "visualization": {
                "train": [
                    dataset.samples[i]["id"] for i in train_preview_indices
                ],
                "val": [
                    dataset.samples[i]["id"] for i in val_preview_indices
                ],
            },
        }, handle, indent=2)

    print(
        ">>> Final previews: "
        f"train={len(train_preview_indices)}/{len(train_indices)}, "
        f"val={len(val_preview_indices)}/{len(val_indices)}"
    )

    if not has_validation:
        print(
            ">>> Validation split is empty: best.pt and the learning-rate "
            "scheduler will use epoch train_loss."
        )

    na_air = cfg["axicon_na_air_equiv"]
    if not 0 < na_air < 1:
        raise ValueError(f"Axicon grating pitch gives invalid air-equivalent NA={na_air}")
    cone_angle = float(np.arcsin(na_air))
    transverse_frequency = cfg["axicon_transverse_frequency"]
    print(
        f">>> Physics: profile={args.axicon_profile}, "
        f"z={args.z_m * 1e3:.3f} mm, NA_air={na_air:.4f}, "
        f"upsample={args.upsample_factor}, ROI={args.roi_size}, "
        f"FOV={args.fov_crop_size}, n={args.propagation_medium_index:.4g}"
    )
    print(
        f">>> Bounded z refinement: enabled={args.optimize_z}, "
        f"range=[{args.z_min_m * 1e3:.3f}, {args.z_max_m * 1e3:.3f}] mm, "
        f"update_every={args.z_update_every} optimizer step(s), lr={args.lr_z:g}"
    )
    print(
        f">>> Bounded axicon xy refinement: enabled={args.optimize_xy}, "
        f"initial=({args.axicon_shift_x_um:+.3f}, "
        f"{args.axicon_shift_y_um:+.3f}) um, "
        f"x_range=[{args.xy_min_x_um:+.3f}, {args.xy_max_x_um:+.3f}] um, "
        f"y_range=[{args.xy_min_y_um:+.3f}, {args.xy_max_y_um:+.3f}] um, "
        f"update_every={args.xy_update_every} optimizer step(s), "
        f"lr={args.lr_xy:g}, continuous_only=True"
    )
    print(
        f">>> Polar transfer: enabled={args.transfer_correction}, "
        f"radial_bins={args.transfer_radial_bins}, "
        f"azimuthal_order={args.transfer_azimuthal_order}, "
        f"angular_samples={args.transfer_angular_samples}, "
        f"max_log_amp={args.transfer_max_log_amplitude:g}, "
        f"max_phase={args.transfer_max_phase_rad:g} rad"
    )
    print(
        f">>> SLM cross talk: support={args.crosstalk_kernel_size}x"
        f"{args.crosstalk_kernel_size} physical pixels, "
        f"P={args.crosstalk_subpixel_factor}, "
        f"learned_kernel={args.crosstalk_kernel_size * args.crosstalk_subpixel_factor}x"
        f"{args.crosstalk_kernel_size * args.crosstalk_subpixel_factor}, "
        f"remaining_optical_upsample="
        f"{args.upsample_factor // args.crosstalk_subpixel_factor}, "
        f"lr={args.lr_crosstalk:g}, signed_DC_preserving=True, "
        "domain=unwrapped voltage-proportional drive"
    )

    beam = HoloBeam(beam_config)
    z_query = torch.tensor([args.z_m], device=device, dtype=beam_config.fdtype)
    h_asm = build_axicon_transfer_function(
        beam,
        show_debug_plot=False,
        upsample_factor=args.upsample_factor,
        z_query=z_query,
        n_medium=args.propagation_medium_index,
        axicon_angle=cone_angle,
        axicon_angle_in_medium=args.axicon_angle_in_medium,
        axicon_transverse_frequency=transverse_frequency,
        axicon_profile=args.axicon_profile,
        axicon_phase_depth=args.axicon_phase_depth_rad,
        axicon_duty_cycle=args.axicon_duty_cycle,
        margin_factor=args.asm_margin_factor,
    )
    plt.close("all")
    physics = {
        "h_asm": h_asm,
        "cone_angle": cone_angle,
        "axicon_transverse_frequency": transverse_frequency,
    }
    base_profile = base_illumination(beam)

    proxy = AxiconProxyParameters(
        num_zernike=args.num_zernike,
        source_grid_shape=(args.source_grid_x, args.source_grid_y),
        source_max_deviation=args.source_max_deviation,
        crosstalk_kernel_size=args.crosstalk_kernel_size,
        crosstalk_subpixel_factor=args.crosstalk_subpixel_factor,
        xy_enabled=args.optimize_xy,
        xy_initial_x_m=args.axicon_shift_x_um * 1e-6,
        xy_initial_y_m=args.axicon_shift_y_um * 1e-6,
        xy_min_x_m=args.xy_min_x_um * 1e-6,
        xy_max_x_m=args.xy_max_x_um * 1e-6,
        xy_min_y_m=args.xy_min_y_um * 1e-6,
        xy_max_y_m=args.xy_max_y_um * 1e-6,
        z_enabled=args.optimize_z,
        z_initial_m=args.z_m,
        z_min_m=args.z_min_m,
        z_max_m=args.z_max_m,
        transfer_enabled=args.transfer_correction,
        transfer_radial_bins=args.transfer_radial_bins,
        transfer_azimuthal_order=args.transfer_azimuthal_order,
        transfer_angular_samples=args.transfer_angular_samples,
        transfer_max_log_amplitude=args.transfer_max_log_amplitude,
        transfer_max_phase_rad=args.transfer_max_phase_rad,
    ).to(device)
    optimizer = build_optimizer(proxy, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, args.val_every)
    )
    history: list[dict] = []
    start_epoch = 1
    global_step = 0
    if args.resume is not None:
        checkpoint = torch_load_checkpoint(args.resume, device)
        saved_state = checkpoint["parameter_state_dict"]
        current_state = proxy.state_dict()
        compatible_state = {}
        unexpected_keys = []
        shape_mismatches = {}
        for key, value in saved_state.items():
            if key not in current_state:
                unexpected_keys.append(key)
            elif current_state[key].shape != value.shape:
                shape_mismatches[key] = (
                    tuple(value.shape), tuple(current_state[key].shape)
                )
            else:
                compatible_state[key] = value

        incompatible = proxy.load_state_dict(compatible_state, strict=False)
        exact_parameter_match = not (
            incompatible.missing_keys
            or unexpected_keys
            or shape_mismatches
        )
        if not exact_parameter_match:
            print(
                ">>> Resume parameter compatibility: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={unexpected_keys}, "
                f"shape_mismatches={shape_mismatches}"
            )
        if exact_parameter_match and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (KeyError, ValueError) as error:
                print(
                    ">>> Optimizer state was not restored because parameter "
                    f"groups changed: {error}"
                )
        elif "optimizer_state_dict" in checkpoint:
            print(
                ">>> Optimizer state was not restored because the proxy "
                "parameter architecture changed."
            )
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        proxy.invalidate_z_kernel_cache()
        proxy.invalidate_xy_kernel_cache()
        print(f">>> Resumed from {args.resume} at epoch {start_epoch}")
    elif args.initialize_camera_scale:
        initialize_camera_scale(
            beam, proxy, dataset, train_indices, base_profile,
            physics, cfg, device,
        )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    best_score = min(
        (
            row[checkpoint_metric]
            for row in history
            if np.isfinite(row.get(checkpoint_metric, float("nan")))
        ),
        default=float("inf"),
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics, global_step = train_epoch(
            beam, proxy, train_loader, optimizer, base_profile,
            physics, cfg, device, global_step,
        )
        should_validate = has_validation and (
            epoch == 1 or epoch == args.epochs or epoch % args.val_every == 0
        )
        if should_validate:
            val_metrics = evaluate_indices(
                beam, proxy, dataset, eval_indices, base_profile,
                physics, cfg, device,
            )
            scheduler.step(val_metrics["loss"])
        else:
            val_metrics = {"loss": float("nan"), "raw_mse": float("nan")}
            if not has_validation:
                scheduler.step(train_metrics["loss"])

        xy_m = proxy.axicon_lateral_shift_m(False)
        delta_xy_m = proxy.delta_xy_m(False)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_data": train_metrics["data"],
            "train_reg": train_metrics["reg"],
            "train_raw_mse": train_metrics["raw_mse"],
            "val_loss": val_metrics["loss"],
            "val_raw_mse": val_metrics["raw_mse"],
            "camera_scale": proxy.camera_scale().item(),
            "z_m": proxy.z_position_m(False).item(),
            "delta_z_um": proxy.delta_z_m(False).item() * 1e6,
            "xy_optimized": int(proxy.xy_enabled),
            "axicon_shift_x_um": xy_m[0].item() * 1e6,
            "axicon_shift_y_um": xy_m[1].item() * 1e6,
            "delta_x_um": delta_xy_m[0].item() * 1e6,
            "delta_y_um": delta_xy_m[1].item() * 1e6,
            "xy_x_min_um": proxy.xy_min_m[0].item() * 1e6,
            "xy_x_max_um": proxy.xy_max_m[0].item() * 1e6,
            "xy_y_min_um": proxy.xy_min_m[1].item() * 1e6,
            "xy_y_max_um": proxy.xy_max_m[1].item() * 1e6,
            "crosstalk_center_weight": proxy.slm_crosstalk_kernel(False)[
                (proxy.crosstalk_effective_kernel_size - 1) // 2,
                (proxy.crosstalk_effective_kernel_size - 1) // 2,
            ].item(),
            "crosstalk_kernel_min": proxy.slm_crosstalk_kernel(False).min().item(),
            "crosstalk_kernel_max": proxy.slm_crosstalk_kernel(False).max().item(),
            "crosstalk_residual_l1": proxy.slm_crosstalk_residual(
                False
            ).abs().mean().item(),
            "z_min_delta_um": (
                proxy.z_min_m - proxy.z_initial_m
            ).item() * 1e6,
            "z_max_delta_um": (
                proxy.z_max_m - proxy.z_initial_m
            ).item() * 1e6,
        }
        history.append(row)
        write_history(history, run_dir)
        if has_validation:
            metric_text = f"val={row['val_loss']:.6g}"
        else:
            metric_text = "val=n/a, best_by=train"
        print(
            f"Epoch {epoch:03d}: train={row['train_loss']:.6g}, "
            f"{metric_text}, scale={row['camera_scale']:.6g}, "
            f"z={row['z_m'] * 1e3:.6f} mm "
            f"(delta={row['delta_z_um']:+.3f} um), "
            f"xy=({row['axicon_shift_x_um']:+.3f}, "
            f"{row['axicon_shift_y_um']:+.3f}) um "
            f"(delta=({row['delta_x_um']:+.3f}, "
            f"{row['delta_y_um']:+.3f}) um)"
        )

        score = checkpoint_selection_score(
            train_metrics,
            val_metrics,
            has_validation,
            should_validate,
        )
        payload = checkpoint_payload(
            proxy, optimizer, epoch, history, cfg, dataset,
            train_indices, val_indices, tuple(base_profile.shape), global_step,
        )
        payload["checkpoint_selection_metric"] = checkpoint_metric
        payload["checkpoint_selection_score"] = (
            float(score) if np.isfinite(score) else None
        )
        torch.save(payload, run_dir / "last.pt")
        if np.isfinite(score) and score < best_score:
            best_score = score
            torch.save(payload, run_dir / "best.pt")

    best_path = run_dir / "best.pt"
    if best_path.exists():
        best = torch_load_checkpoint(best_path, device)
        proxy.load_state_dict(best["parameter_state_dict"])
        proxy.invalidate_z_kernel_cache()
        proxy.invalidate_xy_kernel_cache()
        print(
            f">>> Loaded best.pt selected by {checkpoint_metric} "
            "for final visualizations."
        )
    save_parameter_plots(
        proxy,
        tuple(base_profile.shape),
        (args.roi_size, args.roi_size),
        run_dir,
    )
    save_previews(
        beam, proxy, dataset, train_preview_indices, base_profile,
        physics, cfg, device, run_dir, "train",
    )
    save_previews(
        beam, proxy, dataset, val_preview_indices, base_profile,
        physics, cfg, device, run_dir, "val",
    )
    print(f">>> Done. Outputs: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
