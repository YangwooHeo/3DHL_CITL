"""
FNO-assisted SLM phase optimization for the SLM + axicon system.

This script uses the trained FNO camera proxy as a differentiable camera-space
loss, while keeping the physical SLM->axicon propagation in the loop to produce
the electric-field input expected by the FNO.

Recommended workflow:
  1. Run a small number of GS adjoint iterations for a physically plausible
     phase initialization.
  2. Run a short Adam refinement against the FNO-predicted camera intensity.

Why not pure GS through the FNO?
  GS relies on replacing amplitude in a complex field and back-propagating
  through a known unitary/adjoint optical operator. The FNO proxy is a learned
  nonlinear camera mapping and does not expose that kind of exact inverse.
  It is still differentiable, so it is better used as a low-iteration gradient
  refinement on top of GS.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

from mbvam.Beam.holobeam import HoloBeam
from mbvam.Beam.holobeamconfig import HoloBeamConfig

from Phase_optimization_axicon_adam_3D import (
    _load_rescaled_target_image,
    gs_axicon_init,
)
from axicon_simulator import build_axicon_transfer_function, load_fno_proxy
from train_fno_axicon import apply_loss_roi, log_display_tensor, predict_camera, resize_batch


def load_target_image(path, target_size, device, dtype=torch.float32):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if np.iscomplexobj(arr):
            arr = np.abs(arr) ** 2
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            if arr.shape[0] in (2, 3):
                arr = arr[0]
            elif arr.shape[-1] in (2, 3):
                arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"{path} must resolve to a 2D target image; got {arr.shape}")
        arr = np.asarray(arr, dtype=np.float32)
    else:
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0

    arr = np.nan_to_num(arr.astype(np.float32))
    arr = arr - float(arr.min())
    arr = arr / (float(arr.max()) + 1e-8)
    target = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
    return resize_batch(target, target_size, mode="bilinear").clamp(0.0, 1.0)


def prepare_gs_target(path, beam_config, roi_size, use_rescaling=True,
                      original_pixel_size=2e-6):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if np.iscomplexobj(arr):
            arr = np.abs(arr) ** 2
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            if arr.shape[0] in (2, 3):
                arr = arr[0]
            elif arr.shape[-1] in (2, 3):
                arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"{path} must resolve to a 2D GS target; got {arr.shape}")
        arr = np.nan_to_num(np.asarray(arr, dtype=np.float32))
        arr = arr - float(arr.min())
        return (arr / (float(arr.max()) + 1e-8)).astype(np.float32)

    if use_rescaling:
        return _load_rescaled_target_image(
            path,
            beam_config,
            original_pixel_size=original_pixel_size,
        )

    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    arr = arr - float(arr.min())
    arr = arr / (float(arr.max()) + 1e-8)
    return arr.astype(np.float32)


def build_beam_config():
    beam_config = HoloBeamConfig()
    beam_config.lambda_ = 0.473e-6
    assert beam_config.focal_SLM is not False, "Effective focal length needs to be set."
    beam_config.binningFactor = 1
    beam_config.psSLM_physical = 8e-6 * 0.8
    beam_config.Nx_physical = 1600
    beam_config.Ny_physical = 1200
    beam_config.axis_angle = [1, 0, 0]
    beam_config.z_plane_sampling_rate = 0.5
    beam_config.amplitude_profile_type = "gaussian"
    beam_config.gaussian_beam_waist = 0.00638708 * 0.8
    return beam_config


def phase_to_slm_file_array(phase_rad):
    phase = (phase_rad.detach() % (2 * torch.pi)).cpu().numpy()
    phase_slm = np.round(phase * (1023 / (2 * np.pi))).astype(np.int16).T
    return phase_slm


def phase_to_fno_condition(phase_rad, fno_cfg):
    phase = phase_rad.transpose(0, 1)
    if fno_cfg.get("phase_flip", True):
        phase = torch.flip(phase, dims=(1,))
    return phase.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)


def scale_amp_for_fno(amp, fno_cfg, eps=1e-8):
    mode = fno_cfg.get("field_scale_mode", "global_percentile")
    if mode == "raw":
        return amp
    if mode == "global_percentile":
        scale = (fno_cfg.get("field_amp_scale_actual", None) or
                 fno_cfg.get("field_amp_scale", None))
        if scale is None:
            raise ValueError(
                "FNO cfg is missing field_amp_scale_actual. Use a checkpoint "
                "saved after dataset creation or set FIELD_AMP_SCALE during training."
            )
        return amp / (float(scale) + eps)
    if mode == "sample_norm":
        percentile = float(fno_cfg.get("field_amp_percentile", 99.9))
        scale = torch.quantile(amp.detach().reshape(-1), percentile / 100.0)
        return torch.clamp(amp / (scale + eps), 0.0, 1.0)
    raise ValueError(f"Unsupported field_scale_mode={mode!r}")


def phase_to_fno_field(phase_rad, beam, cone_angle, upsample_factor, h_asm, roi_size,
                       propagation_medium_index, axicon_angle_in_medium,
                       axicon_transverse_frequency, fno_cfg,
                       transpose_output_field=True):
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
        raise ValueError(f"FNO optimization expects a single z-plane field; got {field.shape}")
    if transpose_output_field:
        field = field.transpose(0, 1)

    amp = torch.abs(field).to(torch.float32)
    amp_scaled = scale_amp_for_fno(amp, fno_cfg)
    phase_norm = amp.clamp_min(1e-8)
    cos_p = (field.real.to(torch.float32) / phase_norm).clamp(-1.0, 1.0)
    sin_p = (field.imag.to(torch.float32) / phase_norm).clamp(-1.0, 1.0)
    fno_field = torch.stack([amp_scaled, cos_p, sin_p], dim=0).unsqueeze(0)
    return resize_batch(fno_field, int(fno_cfg["model_size"]), mode="bilinear")


def wrapped_phase_tv(phase_rad):
    phase_complex = torch.stack([torch.cos(phase_rad), torch.sin(phase_rad)], dim=0)
    dx = phase_complex[:, 1:, :] - phase_complex[:, :-1, :]
    dy = phase_complex[:, :, 1:] - phase_complex[:, :, :-1]
    return dx.abs().mean() + dy.abs().mean()


def fno_camera_loss(pred, target, mode="display_mse", roi_fraction=1.0):
    pred_roi, target_roi = apply_loss_roi(pred, target, roi_fraction)

    if mode == "display_mse":
        return F.mse_loss(log_display_tensor(pred_roi), log_display_tensor(target_roi))
    if mode == "norm_mse":
        pred_n = pred_roi / pred_roi.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        target_n = target_roi / target_roi.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return F.mse_loss(pred_n, target_n)
    if mode == "raw_mse":
        return F.mse_loss(pred_roi, target_roi)
    raise ValueError(f"Unknown FNO loss mode: {mode}")


def save_optimization_preview(target, pred, sim_intensity, phase_rad, save_path,
                              roi_fraction=1.0):
    target_roi = apply_loss_roi(target, target, roi_fraction)[0][0, 0].detach().cpu().numpy()
    pred_roi = apply_loss_roi(pred, pred, roi_fraction)[0][0, 0].detach().cpu().numpy()
    sim_roi = apply_loss_roi(sim_intensity, sim_intensity, roi_fraction)[0][0, 0].detach().cpu().numpy()
    phase_np = (phase_rad.detach() % (2 * torch.pi)).cpu().numpy()

    values = np.concatenate([target_roi.reshape(-1), pred_roi.reshape(-1), sim_roi.reshape(-1)])
    scale = max(float(np.percentile(values, 99.9)), 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(np.clip(target_roi / scale, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Target")
    axes[1].imshow(np.clip(pred_roi / scale, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("FNO camera")
    axes[2].imshow(np.clip(sim_roi / scale, 0, 1), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Axicon sim input")
    axes[3].imshow(phase_np, cmap="hsv", vmin=0, vmax=2 * np.pi)
    axes[3].set_title("SLM phase")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close(fig)


def optimize_phase_with_fno(beam, fno_model, fno_cfg, target, init_phase,
                            cone_angle, upsample_factor, h_asm, roi_size,
                            propagation_medium_index, axicon_angle_in_medium,
                            axicon_transverse_frequency, adam_iters=40, lr=0.03,
                            loss_mode="display_mse", phase_tv_weight=0.0,
                            transpose_output_field=True, print_every=5):
    for param in fno_model.parameters():
        param.requires_grad_(False)
    fno_model.eval()

    phase_var = nn.Parameter(init_phase.clone())
    optimizer = optim.Adam([phase_var], lr=lr)
    roi_fraction = fno_cfg.get("loss_roi_fraction", 1.0)

    best_loss = float("inf")
    best_phase = init_phase.detach().clone()
    loss_history = []

    for i in range(adam_iters):
        optimizer.zero_grad(set_to_none=True)
        phase = phase_var
        field = phase_to_fno_field(
            phase,
            beam,
            cone_angle,
            upsample_factor,
            h_asm,
            roi_size,
            propagation_medium_index,
            axicon_angle_in_medium,
            axicon_transverse_frequency,
            fno_cfg,
            transpose_output_field=transpose_output_field,
        )
        phase_cond = phase_to_fno_condition(phase, fno_cfg).to(field.device)
        with torch.set_grad_enabled(True):
            pred, _ = predict_camera(fno_model, field, phase_cond, fno_cfg, train_noise=False)
            loss = fno_camera_loss(pred, target, mode=loss_mode, roi_fraction=roi_fraction)
            if phase_tv_weight > 0:
                loss = loss + float(phase_tv_weight) * wrapped_phase_tv(phase)

        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        loss_history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_phase = (phase_var.detach() % (2 * torch.pi)).clone()

        if print_every and ((i + 1) % print_every == 0 or i == 0 or i + 1 == adam_iters):
            print(f"  FNO-Adam [{i + 1:4d}/{adam_iters}] | loss={loss_value:.6f}")

    return best_phase, loss_history


def build_axicon_system(beam_config, z_target, upsample_factor,
                        propagation_medium_index, axicon_grating_pitch,
                        axicon_angle_in_medium, margin_factor,
                        show_transfer_function_plot=False):
    axicon_transverse_frequency = 1.0 / axicon_grating_pitch
    axicon_na_air_equiv = beam_config.lambda_ * axicon_transverse_frequency
    if axicon_na_air_equiv >= 1.0:
        raise ValueError("Axicon grating pitch gives an invalid free-space NA >= 1.")
    cone_angle = float(np.arcsin(axicon_na_air_equiv))
    cone_angle_in_medium = float(np.arcsin(axicon_na_air_equiv / propagation_medium_index))
    print(f"Axicon air-equivalent NA={axicon_na_air_equiv:.4f}, "
          f"theta_air={cone_angle:.4f} rad, "
          f"theta_medium={cone_angle_in_medium:.4f} rad at n={propagation_medium_index:.3f}")

    beam = HoloBeam(beam_config)
    beam.phase_mask_iter = torch.zeros((beam_config.Nx, beam_config.Ny),
                                        device=beam_config.device)
    beam.slm_amplitude_profile = beam.buildSLMAmplitudeProfile()
    beam.beam_mean_amplitude_iter = torch.tensor(
        1.0, device=beam_config.device, dtype=beam_config.fdtype)

    z_eval_planes = torch.tensor([z_target], device=beam_config.device, dtype=beam_config.fdtype)
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


if __name__ == "__main__":
    FNO_CHECKPOINT = Path(
        r"C:\REVAMP\Yangwoo Heo\SLM_to_Axicon_Optimiaztion\FNO_train_59simple_patterns"
        r"\20260525_230300_mode200_width8_depth2_penalty0.1"
        r"\best.pt"
    )
    TARGET_DIRECTORY = Path(
        r"H:\Shared drives\taylorlab\3DHL\CITL\Fourier Neural Operator_Training phase masks\HollowSquare_Targets"
    )
    TARGET_IMAGE_PATH = r"H:\Shared drives\taylorlab\3DHL\CITL\Fourier Neural Operator_Training phase masks\HollowSquare_Targets\00_baseline.png"
    TARGET_PATTERNS = ("*.png", "*.PNG", "*.npy")
    OUTPUT_DIRECTORY = Path(
        r"C:\REVAMP\Yangwoo Heo\SLM_to_Axicon_Optimiaztion\FNO_train_59simple_patterns"
        r"\FNO_phase_optimization"
    )

    # Optical setup should match the forward-sim data used to train the FNO.
    Z_TARGET = 0.01149
    ROI_SIZE = 1024
    UPSAMPLE_FACTOR = 20
    AXICON_GRATING_PITCH = 1.396e-6
    PROPAGATION_MEDIUM_INDEX = 1.471
    AXICON_ANGLE_IN_MEDIUM = False
    TRANSPOSE_OUTPUT_FIELD = True
    MARGIN_FACTOR = 5000

    # Initialization/refinement. Keep FNO-Adam short; GS does the heavy lifting.
    INIT_MODE = "gs"  # "gs", "random", or "file"
    INIT_PHASE_PATH = None
    GS_ITERS = 15
    FNO_ADAM_ITERS = 35
    FNO_LR = 0.03
    FNO_LOSS_MODE = "display_mse"  # "display_mse", "norm_mse", or "raw_mse"
    PHASE_TV_WEIGHT = 0.0
    USE_RESCALING_FOR_GS_TARGET = True
    TARGET_ORIGINAL_PIXEL_SIZE = 2e-6

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    beam_config = build_beam_config()
    fno_model, fno_cfg = load_fno_proxy(FNO_CHECKPOINT, device)
    beam, cone_angle, axicon_transverse_frequency, h_asm = build_axicon_system(
        beam_config,
        z_target=Z_TARGET,
        upsample_factor=UPSAMPLE_FACTOR,
        propagation_medium_index=PROPAGATION_MEDIUM_INDEX,
        axicon_grating_pitch=AXICON_GRATING_PITCH,
        axicon_angle_in_medium=AXICON_ANGLE_IN_MEDIUM,
        margin_factor=MARGIN_FACTOR,
        show_transfer_function_plot=False,
    )

    if TARGET_IMAGE_PATH is not None:
        target_paths = [Path(TARGET_IMAGE_PATH)]
    else:
        target_paths = sorted({
            path
            for pattern in TARGET_PATTERNS
            for path in TARGET_DIRECTORY.glob(pattern)
        })
    if not target_paths:
        raise FileNotFoundError(f"No targets found in {TARGET_DIRECTORY}")

    for idx, target_path in enumerate(target_paths, start=1):
        print("=" * 80)
        print(f"[{idx}/{len(target_paths)}] FNO phase optimization: {target_path.name}")

        target = load_target_image(
            target_path,
            target_size=int(fno_cfg["model_size"]),
            device=device,
            dtype=torch.float32,
        )

        if INIT_MODE == "gs":
            gs_target = prepare_gs_target(
                target_path,
                beam_config,
                ROI_SIZE,
                use_rescaling=USE_RESCALING_FOR_GS_TARGET,
                original_pixel_size=TARGET_ORIGINAL_PIXEL_SIZE,
            )
            init_phase = gs_axicon_init(
                beam,
                gs_target,
                cone_angle,
                h_asm,
                num_iters=GS_ITERS,
                upsample_factor=UPSAMPLE_FACTOR,
                roi_size=ROI_SIZE,
                z_target_idx=0,
                verbose=True,
                n_medium=PROPAGATION_MEDIUM_INDEX,
                axicon_angle_in_medium=AXICON_ANGLE_IN_MEDIUM,
                axicon_transverse_frequency=axicon_transverse_frequency,
            )
        elif INIT_MODE == "file":
            if INIT_PHASE_PATH is None:
                raise ValueError("INIT_PHASE_PATH must be set when INIT_MODE='file'")
            init_np = np.load(INIT_PHASE_PATH).astype(np.float32).T
            init_phase = torch.from_numpy(init_np * (2 * np.pi / 1023.0)).to(
                device=beam_config.device,
                dtype=beam_config.fdtype,
            )
        elif INIT_MODE == "random":
            init_phase = torch.rand(
                (beam_config.Nx, beam_config.Ny),
                device=beam_config.device,
                dtype=beam_config.fdtype,
            ) * 2 * torch.pi
        else:
            raise ValueError(f"Unknown INIT_MODE={INIT_MODE!r}")

        best_phase, loss_history = optimize_phase_with_fno(
            beam,
            fno_model,
            fno_cfg,
            target,
            init_phase,
            cone_angle,
            UPSAMPLE_FACTOR,
            h_asm,
            ROI_SIZE,
            PROPAGATION_MEDIUM_INDEX,
            AXICON_ANGLE_IN_MEDIUM,
            axicon_transverse_frequency,
            adam_iters=FNO_ADAM_ITERS,
            lr=FNO_LR,
            loss_mode=FNO_LOSS_MODE,
            phase_tv_weight=PHASE_TV_WEIGHT,
            transpose_output_field=TRANSPOSE_OUTPUT_FIELD,
            print_every=5,
        )

        field = phase_to_fno_field(
            best_phase,
            beam,
            cone_angle,
            UPSAMPLE_FACTOR,
            h_asm,
            ROI_SIZE,
            PROPAGATION_MEDIUM_INDEX,
            AXICON_ANGLE_IN_MEDIUM,
            axicon_transverse_frequency,
            fno_cfg,
            transpose_output_field=TRANSPOSE_OUTPUT_FIELD,
        )
        phase_cond = phase_to_fno_condition(best_phase, fno_cfg).to(field.device)
        with torch.no_grad():
            pred, _ = predict_camera(fno_model, field, phase_cond, fno_cfg, train_noise=False)
        sim_intensity = field[:, 0:1].pow(2).clamp_min(0.0)

        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in target_path.stem)
        phase_path = OUTPUT_DIRECTORY / f"{safe_name}_fno_phase.npy"
        loss_path = OUTPUT_DIRECTORY / f"{safe_name}_fno_loss.npy"
        preview_path = OUTPUT_DIRECTORY / f"{safe_name}_fno_preview.png"
        np.save(phase_path, phase_to_slm_file_array(best_phase))
        np.save(loss_path, np.asarray(loss_history, dtype=np.float32))
        save_optimization_preview(
            target,
            pred,
            sim_intensity,
            best_phase,
            preview_path,
            roi_fraction=fno_cfg.get("loss_roi_fraction", 1.0),
        )
        print(f"Saved phase: {phase_path}")
        print(f"Saved preview: {preview_path}")

        del target, init_phase, best_phase, field, phase_cond, pred, sim_intensity
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
