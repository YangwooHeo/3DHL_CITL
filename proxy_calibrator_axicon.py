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

* Zernike coefficients describe SLM-plane wavefront error.
* A bounded, coarse source map describes smooth illumination non-uniformity.
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
    DEFAULT_AXICON_GRATING_PITCH_M,
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

    def _load_phase(self, path: Path) -> torch.Tensor:
        phase = np.load(path)
        phase = np.squeeze(phase)
        if phase.ndim != 2:
            raise ValueError(f"{path} must contain a 2D phase map; got {phase.shape}")
        if self.phase_transpose:
            phase = phase.T
        if self.phase_flip_first_axis:
            phase = phase[::-1, :]
        phase = np.ascontiguousarray(phase, dtype=np.float32)
        if self.expected_phase_shape is not None and phase.shape != self.expected_phase_shape:
            raise ValueError(
                f"{path.name} has shape {phase.shape} after orientation correction; "
                f"expected {self.expected_phase_shape}"
            )
        phase *= 2.0 * np.pi / self.phase_level_max
        return torch.from_numpy(phase)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        camera = self._load_camera_raw(sample["camera"]) / self.camera_scale
        return {
            "phase": self._load_phase(sample["phase"]).float(),
            "camera": torch.from_numpy(camera).unsqueeze(0).float(),
            "id": sample["id"],
        }


class AxiconProxyParameters(nn.Module):
    """Low-dimensional, physically constrained calibration parameters."""

    def __init__(self, num_zernike: int = 20,
                 source_grid_shape: tuple[int, int] = (64, 48),
                 source_max_deviation: float = 0.30) -> None:
        super().__init__()
        if num_zernike <= 0:
            raise ValueError("num_zernike must be positive")
        if min(source_grid_shape) <= 0:
            raise ValueError("source_grid_shape values must be positive")
        if not 0 <= source_max_deviation < 1:
            raise ValueError("source_max_deviation must be in [0, 1)")
        self.zernike_coeffs = nn.Parameter(torch.zeros(num_zernike))
        self.source_latent = nn.Parameter(
            torch.zeros(1, 1, source_grid_shape[0], source_grid_shape[1])
        )
        self.log_camera_scale = nn.Parameter(torch.zeros(()))
        self.source_max_deviation = float(source_max_deviation)

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
    slm_phase: torch.Tensor,
    base_profile: torch.Tensor,
    h_asm,
    cone_angle: float,
    upsample_factor: int,
    roi_size: int,
    propagation_medium_index: float,
    axicon_angle_in_medium: bool,
    axicon_transverse_frequency: float,
    apply_spatial_filter: bool,
    fov_crop_size: int | None,
    transpose_output: bool,
    require_grad: bool,
) -> torch.Tensor:
    """Return one camera-domain prediction with simulator-matched propagation."""
    beam.zernike_coeffs = (
        proxy.zernike_coeffs if require_grad else proxy.zernike_coeffs.detach()
    )
    profile = combined_illumination(base_profile, proxy)
    if not require_grad:
        profile = profile.detach()
    beam_amp = torch.ones((), device=slm_phase.device, dtype=slm_phase.dtype)
    volume = beam.propagateToVolume_Axicon2(
        axicon_angle=cone_angle,
        upsample_factor=upsample_factor,
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
    )
    intensity = volume[:, :, 0].abs().square()
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
                         source_shape: tuple[int, int], cfg: dict) -> tuple[torch.Tensor, dict]:
    source = proxy.source_map(source_shape)
    source_prior = (source - 1.0).square().mean()
    source_smooth = (
        (source[1:, :] - source[:-1, :]).square().mean()
        + (source[:, 1:] - source[:, :-1]).square().mean()
    )
    zernike_l2 = proxy.zernike_coeffs.square().mean()
    total = (
        cfg["w_source_prior"] * source_prior
        + cfg["w_source_smooth"] * source_smooth
        + cfg["w_zernike_l2"] * zernike_l2
    )
    return total, {
        "source_prior": source_prior.detach(),
        "source_smooth": source_smooth.detach(),
        "zernike_l2": zernike_l2.detach(),
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
                    device, require_grad):
    phase = sample["phase"].to(device)
    target = sample["camera"].to(device)
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if phase.ndim != 2 or target.ndim != 4:
        raise ValueError(
            f"Expected one 2D phase and BCHW camera; got {phase.shape}, {target.shape}"
        )
    prediction = axicon_forward_proxy(
        beam=beam,
        proxy=proxy,
        slm_phase=phase,
        base_profile=base_profile,
        h_asm=physics["h_asm"],
        cone_angle=physics["cone_angle"],
        upsample_factor=cfg["upsample_factor"],
        roi_size=cfg["roi_size"],
        propagation_medium_index=cfg["propagation_medium_index"],
        axicon_angle_in_medium=cfg["axicon_angle_in_medium"],
        axicon_transverse_frequency=physics["axicon_transverse_frequency"],
        apply_spatial_filter=cfg["apply_spatial_filter"],
        fov_crop_size=cfg["fov_crop_size"],
        transpose_output=cfg["transpose_output"],
        require_grad=require_grad,
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
                cfg, device) -> dict[str, float]:
    proxy.train()
    totals = {"loss": 0.0, "data": 0.0, "reg": 0.0, "raw_mse": 0.0}
    n_samples = 0
    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        batch_size = len(batch["id"])
        batch_data = 0.0
        batch_mse = 0.0

        # Backward per sample so multiple upsampled FFT graphs are never retained.
        for i, sample_id in enumerate(batch["id"]):
            sample = {
                "phase": batch["phase"][i],
                "camera": batch["camera"][i:i + 1],
            }
            prediction, target = make_prediction(
                beam, proxy, sample, base_profile, physics, cfg, device, True)
            data_loss, components = visual_loss(prediction, target, cfg)
            group = sample_type_from_id(sample_id)
            weight = cfg["group_loss_weights_normalized"].get(group, 1.0)
            (weight * data_loss / batch_size).backward()
            batch_data += float((weight * data_loss).detach())
            batch_mse += float(components["raw_mse"].detach())

        reg_loss, _ = proxy_regularization(proxy, tuple(base_profile.shape), cfg)
        if reg_loss.requires_grad:
            reg_loss.backward()
        if cfg["grad_clip"] is not None and cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(proxy.parameters(), cfg["grad_clip"])
        optimizer.step()

        totals["data"] += batch_data
        totals["reg"] += float(reg_loss.detach()) * batch_size
        totals["loss"] += batch_data + float(reg_loss.detach()) * batch_size
        totals["raw_mse"] += batch_mse
        n_samples += batch_size

    return {name: value / max(n_samples, 1) for name, value in totals.items()}


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


def build_optimizer(proxy: AxiconProxyParameters, cfg: dict):
    groups = []
    for parameter, lr, name in [
        (proxy.zernike_coeffs, cfg["lr_zernike"], "zernike"),
        (proxy.source_latent, cfg["lr_source"], "source"),
        (proxy.log_camera_scale, cfg["lr_scale"], "camera_scale"),
    ]:
        parameter.requires_grad_(lr > 0)
        if lr > 0:
            groups.append({"params": [parameter], "lr": lr, "name": name})
        else:
            print(f">>> Frozen parameter group: {name}")
    if not groups:
        raise ValueError("At least one proxy learning rate must be positive")
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def checkpoint_payload(proxy, optimizer, epoch, history, cfg, dataset,
                       train_indices, val_indices, source_shape):
    source_map = proxy.source_map(source_shape).detach().cpu()
    return {
        "epoch": int(epoch),
        "parameter_state_dict": proxy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "cfg": cfg,
        "train_ids": [dataset.samples[i]["id"] for i in train_indices],
        "val_ids": [dataset.samples[i]["id"] for i in val_indices],
        # Compatibility-friendly exported physical values.
        "zernike_coeffs": proxy.zernike_coeffs.detach().cpu(),
        "source_modulation_map": source_map,
        "camera_scale_factor": proxy.camera_scale().detach().cpu(),
    }


def write_history(history: list[dict], run_dir: Path) -> None:
    if not history:
        return
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
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


@torch.no_grad()
def save_parameter_plots(proxy: AxiconProxyParameters,
                         source_shape: tuple[int, int], run_dir: Path) -> None:
    source = proxy.source_map(source_shape).detach().cpu().numpy()
    zernike = proxy.zernike_coeffs.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    image = axes[0].imshow(source.T, cmap="RdBu_r", aspect="auto")
    axes[0].set_title(
        f"Source modulation (min={source.min():.3f}, max={source.max():.3f})"
    )
    fig.colorbar(image, ax=axes[0], label="Amplitude multiplier")
    axes[1].bar(np.arange(len(zernike)), zernike)
    axes[1].set(xlabel="Zernike index", ylabel="Coefficient (rad)",
                title=f"Camera scale = {proxy.camera_scale().item():.4g}")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "learned_proxy_parameters.png", dpi=150)
    plt.close(fig)


def quantile_display(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(array, dtype=np.float32))
    low, high = np.quantile(array, [0.001, 0.999])
    return np.clip((array - low) / max(float(high - low), 1e-8), 0.0, 1.0)


@torch.no_grad()
def save_previews(beam, proxy, dataset, indices, base_profile,
                  physics, cfg, device, run_dir) -> None:
    preview_dir = run_dir / "samples"
    preview_dir.mkdir(exist_ok=True)
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
        fig.suptitle(str(sample["id"]))
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

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-zernike", type=float, default=2e-2)
    parser.add_argument("--lr-source", type=float, default=1e-2)
    parser.add_argument("--lr-scale", type=float, default=2e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--w-source-prior", type=float, default=1e-2)
    parser.add_argument("--w-source-smooth", type=float, default=1e-3)
    parser.add_argument("--w-zernike-l2", type=float, default=1e-4)

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
    parser.add_argument("--n-vis", type=int, default=4)
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
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.val_every <= 0 or args.val_max_samples <= 0:
        raise ValueError("--val-every and --val-max-samples must be positive")
    if args.z_m <= 0 or args.roi_size <= 0 or args.upsample_factor <= 0:
        raise ValueError("z, ROI size, and upsample factor must be positive")
    if args.propagation_medium_index <= 0 or args.axicon_grating_pitch_m <= 0:
        raise ValueError("medium index and axicon grating pitch must be positive")
    if args.fov_crop_size is not None and args.fov_crop_size > args.roi_size:
        raise ValueError("--fov-crop-size cannot exceed --roi-size")
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
    first = dataset[0]
    print(f">>> First phase shape: {tuple(first['phase'].shape)}")
    print(f">>> First camera shape: {tuple(first['camera'].shape)}")
    if args.dry_run:
        print(">>> Dry run complete: pairing and preprocessing are valid.")
        return
    if len(dataset) < 2:
        raise RuntimeError("At least two paired samples are required for train/validation")

    _, _, train_indices, val_indices = split_dataset(
        dataset,
        real_train_ratio=args.real_train_ratio,
        sys_train_ratio=args.sys_train_ratio,
        seed=args.seed,
    )
    if not train_indices:
        raise RuntimeError("The configured split produced an empty training set")
    eval_indices = evenly_spaced(
        val_indices if val_indices else train_indices,
        args.val_max_samples,
    )

    cfg = config_from_args(args, device, beam_config)
    cfg["camera_scale_actual"] = dataset.camera_scale
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
        }, handle, indent=2)

    na_air = cfg["axicon_na_air_equiv"]
    if not 0 < na_air < 1:
        raise ValueError(f"Axicon grating pitch gives invalid air-equivalent NA={na_air}")
    cone_angle = float(np.arcsin(na_air))
    transverse_frequency = cfg["axicon_transverse_frequency"]
    print(
        f">>> Physics: z={args.z_m * 1e3:.3f} mm, NA_air={na_air:.4f}, "
        f"upsample={args.upsample_factor}, ROI={args.roi_size}, "
        f"FOV={args.fov_crop_size}, n={args.propagation_medium_index:.4g}"
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
    ).to(device)
    optimizer = build_optimizer(proxy, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, args.val_every)
    )
    history: list[dict] = []
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch_load_checkpoint(args.resume, device)
        proxy.load_state_dict(checkpoint["parameter_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f">>> Resumed from {args.resume} at epoch {start_epoch}")
    elif args.initialize_camera_scale and args.lr_scale > 0:
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
    best_val = min((row["val_loss"] for row in history
                    if np.isfinite(row.get("val_loss", float("nan")))),
                   default=float("inf"))
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(
            beam, proxy, train_loader, optimizer, base_profile,
            physics, cfg, device,
        )
        should_validate = (
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

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_data": train_metrics["data"],
            "train_reg": train_metrics["reg"],
            "train_raw_mse": train_metrics["raw_mse"],
            "val_loss": val_metrics["loss"],
            "val_raw_mse": val_metrics["raw_mse"],
            "camera_scale": proxy.camera_scale().item(),
        }
        history.append(row)
        write_history(history, run_dir)
        print(
            f"Epoch {epoch:03d}: train={row['train_loss']:.6g}, "
            f"val={row['val_loss']:.6g}, scale={row['camera_scale']:.6g}"
        )

        payload = checkpoint_payload(
            proxy, optimizer, epoch, history, cfg, dataset,
            train_indices, val_indices, tuple(base_profile.shape),
        )
        torch.save(payload, run_dir / "last.pt")
        score = val_metrics["loss"] if should_validate else float("inf")
        if score < best_val:
            best_val = score
            torch.save(payload, run_dir / "best.pt")

    best_path = run_dir / "best.pt"
    if best_path.exists():
        best = torch_load_checkpoint(best_path, device)
        proxy.load_state_dict(best["parameter_state_dict"])
    save_parameter_plots(proxy, tuple(base_profile.shape), run_dir)
    preview_indices = evenly_spaced(val_indices if val_indices else train_indices,
                                    args.n_vis)
    save_previews(
        beam, proxy, dataset, preview_indices, base_profile,
        physics, cfg, device, run_dir,
    )
    print(f">>> Done. Outputs: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
