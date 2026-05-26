import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import fft, ifft, fftshift, fftfreq

from mbvam.Beam.holobeam import HoloBeam
from mbvam.Beam.holobeamconfig import HoloBeamConfig
import mbvam.Geometry.visualize

np.math = math


def load_slm_phase_tensor(phase_path, beam_config, transpose=True,
                          flip_first_axis=True, phase_level_max=1023.0):
    phase_path = Path(phase_path)
    phase_data = np.load(phase_path)
    if phase_data.ndim != 2:
        raise ValueError(f"{phase_path} must contain a 2D SLM phase map; got shape {phase_data.shape}.")

    if transpose:
        phase_data = phase_data.T
    if flip_first_axis:
        phase_data = phase_data[::-1, :]

    phase_data = np.ascontiguousarray(phase_data, dtype=np.float32)
    expected_shape = (beam_config.Nx, beam_config.Ny)
    if phase_data.shape != expected_shape:
        raise ValueError(
            f"{phase_path} has shape {phase_data.shape} after orientation correction, "
            f"but HoloBeam expects {expected_shape}. Check TRANSPOSE_PHASE, "
            f"FLIP_PHASE_FIRST_AXIS, or the HoloBeamConfig SLM size."
        )

    phase_recon = phase_data * (2 * np.pi / phase_level_max)
    phase_tensor = torch.from_numpy(phase_recon).to(device=beam_config.device,
                                                    dtype=beam_config.fdtype)
    return phase_tensor, phase_data


def field_to_amp_cos_sin(recon_field, transpose_xy=True):
    field_np = recon_field.detach().cpu().numpy()
    field_np = np.squeeze(field_np)
    if field_np.ndim != 2:
        raise ValueError(f"Expected a single z-plane field, got shape {field_np.shape}.")
    if transpose_xy:
        field_np = field_np.T

    amplitude = np.abs(field_np).astype(np.float32)
    phase = np.angle(field_np).astype(np.float32)
    cos_phase = np.cos(phase).astype(np.float32)
    sin_phase = np.sin(phase).astype(np.float32)
    return np.stack([amplitude, cos_phase, sin_phase], axis=0)


def list_phase_files(phase_directory, pattern):
    phase_directory = Path(phase_directory)
    if not phase_directory.exists():
        raise FileNotFoundError(f"SLM phase directory does not exist: {phase_directory}")

    phase_paths = sorted(p for p in phase_directory.glob(pattern) if p.is_file())
    if not phase_paths:
        raise FileNotFoundError(f"No phase files matching {pattern!r} in {phase_directory}")
    return phase_paths


def propagate_axicon_field(beam, phase_tensor, cone_angle, upsample_factor, h_asm,
                           roi_size, propagation_medium_index,
                           axicon_angle_in_medium, axicon_transverse_frequency):
    with torch.no_grad():
        return beam.propagateToVolume_Axicon2(
            axicon_angle=cone_angle,
            upsample_factor=upsample_factor,
            phase_mask=phase_tensor,
            H_asm=h_asm,
            roi_size=roi_size,
            apply_spatial_filter=True,
            n_medium=propagation_medium_index,
            axicon_angle_in_medium=axicon_angle_in_medium,
            axicon_transverse_frequency=axicon_transverse_frequency,
        )


def _torch_load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _with_fno_inference_defaults(cfg):
    cfg = dict(cfg)
    defaults = {
        'model_size': cfg.get('sim_size', 1024),
        'prediction_mode': 'intensity',
        'intensity_activation': 'softplus',
        'field_scale_mode': 'global_percentile',
        'field_amp_percentile': 99.9,
        'field_out_scale': 4.0,
        'field_out_scale_mode': 'field_rms',
        'direct_field_activation': 'tanh',
        'direct_field_noise_std': 0.0,
        'use_slm_phase': False,
        'slm_phase_weight': 1.0,
        'slm_phase_lowpass_size': None,
        'slm_phase_ring_radius': None,
        'slm_phase_ring_width': None,
        'use_grid': True,
        'use_radial_coord': True,
        'use_axicon_phase_map': False,
        'axicon_slope_rad_per_pixel': None,
        'loss_roi_fraction': 1.0,
        'phase_flip': True,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    return cfg


def load_fno_proxy(checkpoint_path, device):
    from train_fno_axicon import AxiconFNO2d, infer_input_channels

    checkpoint_path = Path(checkpoint_path)
    ckpt = _torch_load_checkpoint(checkpoint_path, device)
    cfg = _with_fno_inference_defaults(ckpt.get('cfg', {}))
    in_ch = infer_input_channels(
        use_slm_phase=cfg['use_slm_phase'],
        use_grid=cfg['use_grid'],
        use_radial_coord=cfg['use_radial_coord'],
        use_axicon_phase_map=cfg['use_axicon_phase_map'],
        axicon_slope_rad_per_pixel=cfg['axicon_slope_rad_per_pixel'],
    )
    out_ch = 2 if cfg['prediction_mode'] == 'direct_field' else 1
    model = AxiconFNO2d(
        in_ch=in_ch,
        out_ch=out_ch,
        width=int(cfg['width']),
        modes_y=int(cfg['modes_y']),
        modes_x=int(cfg['modes_x']),
        depth=int(cfg['depth']),
        mlp_width=int(cfg['mlp_width']),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded FNO proxy from {checkpoint_path}")
    print(f"  mode={cfg['prediction_mode']}, size={cfg['model_size']}, "
          f"width={cfg['width']}, depth={cfg['depth']}, "
          f"modes=({cfg['modes_y']}, {cfg['modes_x']}), ROI={cfg['loss_roi_fraction']}")
    return model, cfg


def scale_field_channels_for_fno(field_channels, cfg, eps=1e-8):
    field = np.asarray(field_channels, dtype=np.float32).copy()
    if field.ndim != 3 or field.shape[0] != 3:
        raise ValueError(f"FNO field must be CHW amp/cos/sin with 3 channels; got {field.shape}")

    amp = np.clip(np.nan_to_num(field[0], copy=False), 0.0, None)
    scale_mode = cfg.get('field_scale_mode', 'global_percentile')
    if scale_mode == 'sample_norm':
        scale = float(np.percentile(amp, float(cfg.get('field_amp_percentile', 99.9))))
        field[0] = np.clip(amp / (scale + eps), 0.0, 1.0)
    elif scale_mode == 'global_percentile':
        scale = (cfg.get('field_amp_scale_actual', None) or
                 cfg.get('field_amp_scale', None) or
                 cfg.get('field_amp_scale_actual', None))
        if scale is None:
            raise ValueError(
                "Checkpoint cfg does not contain field_amp_scale_actual. "
                "Use a checkpoint saved by train_fno_axicon.py after dataset creation, "
                "or set FIELD_AMP_SCALE during training."
            )
        field[0] = amp / (float(scale) + eps)
    elif scale_mode == 'raw':
        field[0] = amp
    else:
        raise ValueError(f"Unsupported FNO field_scale_mode={scale_mode!r}")

    phase_norm = np.sqrt(field[1] ** 2 + field[2] ** 2).clip(min=1e-6)
    field[1] = field[1] / phase_norm
    field[2] = field[2] / phase_norm
    return field.astype(np.float32, copy=False)


def load_fno_phase_condition(phase_path, cfg, device, phase_level_max=1023.0):
    phase = np.load(phase_path).astype(np.float32)
    if cfg.get('phase_flip', True):
        phase = phase[:, ::-1].copy()
    phase = phase * (2 * np.pi / phase_level_max)
    return torch.from_numpy(phase).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)


def center_roi_numpy(arr, fraction):
    if fraction is None or float(fraction) >= 1.0:
        return arr
    if float(fraction) <= 0.0:
        raise ValueError(f"ROI fraction must be > 0; got {fraction}")
    height, width = arr.shape[-2:]
    roi_h = max(1, int(round(height * float(fraction))))
    roi_w = max(1, int(round(width * float(fraction))))
    y0 = (height - roi_h) // 2
    x0 = (width - roi_w) // 2
    return arr[..., y0:y0 + roi_h, x0:x0 + roi_w]


def common_intensity_display(sim_np, pred_np, q=99.9, eps=1e-8):
    values = np.concatenate([
        np.nan_to_num(sim_np, copy=False).reshape(-1),
        np.nan_to_num(pred_np, copy=False).reshape(-1),
    ])
    scale = max(float(np.percentile(values, q)), eps)
    return np.clip(sim_np / scale, 0.0, 1.0), np.clip(pred_np / scale, 0.0, 1.0), scale


def add_roi_rectangle(ax, image_shape, fraction):
    if fraction is None or float(fraction) >= 1.0:
        return
    height, width = image_shape[-2:]
    roi_h = max(1, int(round(height * float(fraction))))
    roi_w = max(1, int(round(width * float(fraction))))
    y0 = (height - roi_h) // 2
    x0 = (width - roi_w) // 2
    ax.add_patch(plt.Rectangle(
        (x0 - 0.5, y0 - 0.5),
        roi_w,
        roi_h,
        fill=False,
        edgecolor='deepskyblue',
        linewidth=0.8,
        linestyle=(0, (3, 3)),
        alpha=0.9,
    ))


def save_fno_inference_visualization(sim_np, pred_np, phase_name, save_path,
                                     roi_fraction=1.0, crop_to_roi=True):
    sim_view = center_roi_numpy(sim_np, roi_fraction) if crop_to_roi else sim_np
    pred_view = center_roi_numpy(pred_np, roi_fraction) if crop_to_roi else pred_np
    sim_d, pred_d, display_scale = common_intensity_display(sim_view, pred_view)
    diff_d = np.abs(pred_d - sim_d)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(sim_d, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title(f"Axicon sim - {phase_name}")
    ax[1].imshow(pred_d, cmap='gray', vmin=0, vmax=1)
    ax[1].set_title("FNO proxy")
    ax[2].imshow(diff_d, cmap='hot', vmin=0, vmax=0.5)
    ax[2].set_title("|FNO - Sim|")
    if not crop_to_roi:
        for a in ax:
            add_roi_rectangle(a, sim_d.shape, roi_fraction)
    for a in ax:
        a.axis('off')
    fig.suptitle(f"Common display scale: p99.9={display_scale:.4g}", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close(fig)


def run_fno_proxy_inference(beam, beam_config, phase_path, fno_model, fno_cfg,
                            cone_angle, upsample_factor, h_asm, roi_size,
                            propagation_medium_index, axicon_angle_in_medium,
                            axicon_transverse_frequency, transpose_phase,
                            flip_phase_first_axis, phase_level_max,
                            transpose_output_field, output_directory,
                            crop_visualization_to_roi=True, show_plot=False):
    from train_fno_axicon import predict_camera, resize_batch

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    phase_path = Path(phase_path)

    phase_tensor, _ = load_slm_phase_tensor(
        phase_path,
        beam_config,
        transpose=transpose_phase,
        flip_first_axis=flip_phase_first_axis,
        phase_level_max=phase_level_max,
    )
    recon_field = propagate_axicon_field(
        beam,
        phase_tensor,
        cone_angle,
        upsample_factor,
        h_asm,
        roi_size,
        propagation_medium_index,
        axicon_angle_in_medium,
        axicon_transverse_frequency,
    )
    field_channels_raw = field_to_amp_cos_sin(recon_field, transpose_xy=transpose_output_field)
    field_channels = scale_field_channels_for_fno(field_channels_raw, fno_cfg)

    device = next(fno_model.parameters()).device
    field = torch.from_numpy(field_channels).unsqueeze(0).to(device=device, dtype=torch.float32)
    field = resize_batch(field, int(fno_cfg['model_size']), mode='bilinear')
    phase_cond = load_fno_phase_condition(phase_path, fno_cfg, device, phase_level_max=phase_level_max)

    with torch.no_grad():
        pred, _ = predict_camera(fno_model, field, phase_cond, fno_cfg, train_noise=False)

    sim_np = field[:, 0:1].pow(2).clamp_min(0.0)[0, 0].detach().cpu().numpy()
    pred_np = pred[0, 0].detach().cpu().numpy()
    roi_fraction = fno_cfg.get('loss_roi_fraction', 1.0)

    safe_name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in phase_path.stem)
    sim_path = output_directory / f"{safe_name}_axicon_sim_fno_scale.npy"
    pred_path = output_directory / f"{safe_name}_fno_proxy.npy"
    fig_path = output_directory / f"{safe_name}_fno_proxy.png"
    np.save(sim_path, sim_np.astype(np.float32, copy=False))
    np.save(pred_path, pred_np.astype(np.float32, copy=False))
    save_fno_inference_visualization(
        sim_np,
        pred_np,
        phase_path.stem,
        fig_path,
        roi_fraction=roi_fraction,
        crop_to_roi=crop_visualization_to_roi,
    )

    if show_plot:
        sim_view = center_roi_numpy(sim_np, roi_fraction) if crop_visualization_to_roi else sim_np
        pred_view = center_roi_numpy(pred_np, roi_fraction) if crop_visualization_to_roi else pred_np
        sim_d, pred_d, _ = common_intensity_display(sim_view, pred_view)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(sim_d, cmap='gray', vmin=0, vmax=1)
        ax[0].set_title("Axicon sim")
        ax[1].imshow(pred_d, cmap='gray', vmin=0, vmax=1)
        ax[1].set_title("FNO proxy")
        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.show()

    del phase_tensor, recon_field
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Saved FNO proxy inference: {fig_path}")
    return sim_path, pred_path, fig_path


def build_axicon_transfer_function(beam, show_debug_plot, **kwargs):
    if show_debug_plot:
        return beam.build_axicon_ASM_TF(**kwargs)

    original_show = plt.show
    try:
        plt.show = lambda *args, **show_kwargs: None
        h_asm = beam.build_axicon_ASM_TF(**kwargs)
    finally:
        plt.show = original_show
        plt.close('all')
    return h_asm


def export_electric_fields(beam, beam_config, phase_paths, save_directory,
                           cone_angle, upsample_factor, h_asm, roi_size,
                           propagation_medium_index, axicon_angle_in_medium,
                           axicon_transverse_frequency, transpose_phase,
                           flip_phase_first_axis, phase_level_max,
                           overwrite_outputs, transpose_output_field):
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {len(phase_paths)} electric field file(s) to {save_directory}")
    for index, phase_path in enumerate(phase_paths, start=1):
        output_path = save_directory / phase_path.name
        if phase_path.resolve() == output_path.resolve():
            raise ValueError("SAVE_DIRECTORY must be different from SLM_PHASE_DIRECTORY "
                             "so the original phase masks are not overwritten.")
        if output_path.exists() and not overwrite_outputs:
            print(f"[{index}/{len(phase_paths)}] Skipped existing {output_path.name}")
            continue

        phase_tensor, _ = load_slm_phase_tensor(
            phase_path,
            beam_config,
            transpose=transpose_phase,
            flip_first_axis=flip_phase_first_axis,
            phase_level_max=phase_level_max,
        )
        recon_field = propagate_axicon_field(
            beam,
            phase_tensor,
            cone_angle,
            upsample_factor,
            h_asm,
            roi_size,
            propagation_medium_index,
            axicon_angle_in_medium,
            axicon_transverse_frequency,
        )
        field_channels = field_to_amp_cos_sin(recon_field, transpose_xy=transpose_output_field)
        np.save(output_path, field_channels.astype(np.float32, copy=False))
        print(f"[{index}/{len(phase_paths)}] Saved {output_path.name} with shape {field_channels.shape}")

        del phase_tensor, recon_field
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def visualize_kspace_distortion(beam, phase_tensor, cone_angle, upsample_factor,
                                n_medium=1.0, axicon_angle_in_medium=False,
                                axicon_transverse_frequency=None):
    print("\n--- [K-space Visualization Started] ---")

    slm_field = torch.exp(1j * phase_tensor)

    nx_orig, ny_orig = beam.beam_config.Nx, beam.beam_config.Ny
    ps_orig = beam.beam_config.psSLM

    slm_field_up = slm_field.repeat_interleave(upsample_factor, dim=0).repeat_interleave(upsample_factor, dim=1)
    nx_up, ny_up = slm_field_up.shape
    ps_up = ps_orig / upsample_factor

    x = torch.linspace(-(nx_up - 1) / 2, (nx_up - 1) / 2, nx_up, device=beam.beam_config.device) * ps_up
    y = torch.linspace(-(ny_up - 1) / 2, (ny_up - 1) / 2, ny_up, device=beam.beam_config.device) * ps_up
    y2 = y.view(1, -1) ** 2

    radial_frequency = beam._resolve_axicon_radial_frequency(
        axicon_angle=cone_angle,
        n_medium=n_medium,
        axicon_angle_in_medium=axicon_angle_in_medium,
        axicon_transverse_frequency=axicon_transverse_frequency,
    )
    radial_frequency_value = float(radial_frequency.detach().cpu().item())
    k_sin_alpha = 2 * torch.pi * radial_frequency

    print("  Applying Axicon Phase...")
    chunk_size = 1024
    for i in range(0, nx_up, chunk_size):
        end_idx = min(i + chunk_size, nx_up)
        x2_chunk = x[i:end_idx].view(-1, 1) ** 2
        r_chunk = torch.sqrt(x2_chunk + y2)
        slm_field_up[i:end_idx] *= torch.exp(-1j * k_sin_alpha * r_chunk)

    print("  Calculating FFT2 (GPU)...")
    k_space = torch.fft.fftshift(torch.fft.fft2(slm_field_up, norm="ortho"))

    print("  Cropping K-space to Ring region...")
    k_ring_radius = radial_frequency_value
    df_x = 1.0 / (nx_up * ps_up)
    ring_radius_px = int(k_ring_radius / df_x)

    w = int(ring_radius_px * 1.2)
    cx, cy = nx_up // 2, ny_up // 2

    _ = k_space[cx - w:cx + w, cy - w:cy + w]
    k_space_mag = torch.log1p(torch.abs(k_space)).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(k_space_mag, cmap='viridis')
    plt.title("K-Space: SLM (Hollow Rect) + Axicon", fontsize=16)
    plt.colorbar(label="Log Magnitude")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print("--- [Visualization Complete] ---\n")


def debug_aliasing_1d(slice_2d, upsample_factor, axicon_na, original_pixel_size=8e-6):
    center_y = slice_2d.shape[0] // 2
    slice_1d = slice_2d[center_y, :]
    n = len(slice_1d)

    current_dx = original_pixel_size / upsample_factor
    x_discrete = np.arange(-n // 2, n // 2) * current_dx * 1e6

    interp_factor = 10
    n_interp = n * interp_factor

    spectrum = fftshift(fft(slice_1d))
    spectrum_padded = np.zeros(n_interp, dtype=complex)

    pad_start = (n_interp - n) // 2
    spectrum_padded[pad_start:pad_start + n] = spectrum

    slice_1d_interp = np.abs(ifft(fftshift(spectrum_padded))) * interp_factor
    x_interp = np.arange(-n_interp // 2, n_interp // 2) * (current_dx / interp_factor) * 1e6

    f_max = 1.0 / (2 * current_dx)
    f_theory = 2 * axicon_na / 473e-9
    freqs = fftshift(fftfreq(n, d=current_dx)) / 1000
    spectrum_mag = np.abs(spectrum)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    zoom_range = 25

    ax1.plot(x_interp, slice_1d_interp, 'b-', alpha=0.8, label='Sinc Interpolated (Continuous)')
    ax1.plot(x_discrete, slice_1d, 'ro', markersize=4, label='Simulated Discrete Points')
    ax1.set_xlim([-zoom_range, zoom_range])
    ax1.set_title(f"Spatial Domain (Upsample: {upsample_factor}x)\nPixel size: {current_dx * 1e6:.2f} um")
    ax1.set_xlabel("Position (um)")
    ax1.set_ylabel("Intensity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(freqs, spectrum_mag, 'k-', linewidth=1.5)
    ax2.fill_between(freqs, spectrum_mag, color='gray', alpha=0.3)

    ax2.axvline(-f_max / 1000, color='r', linestyle='--', label='Nyquist Limit')
    ax2.axvline(f_max / 1000, color='r', linestyle='--')
    ax2.axvline(-f_theory / 1000, color='g', linestyle='--', label='NA/lambda')
    ax2.axvline(f_theory / 1000, color='g', linestyle='--')

    ax2.set_title("Frequency Spectrum (Aliasing Check)")
    ax2.set_xlabel("Spatial Frequency (1/mm)")
    ax2.set_ylabel("Magnitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def build_beam_config():
    beam_config = HoloBeamConfig()
    beam_config.psSLM_physical = 8e-6 * 0.8

    # UV
    #beam_config.lambda_ = 0.365e-6
    #beam_config.focal_SLM = 0.020

    # BLUE
    beam_config.lambda_ = 0.473e-6
    #beam_config.focal_SLM = 0.12625
    assert beam_config.focal_SLM is not False

    beam_config.binningFactor = 1
    beam_config.Nx_physical = 1600
    beam_config.Ny_physical = 1200
    beam_config.axis_angle = [1, 0, 00]
    beam_config.z_plane_sampling_rate = 0.5
    beam_config.amplitude_profile_type = 'gaussian'
    beam_config.gaussian_beam_waist = 0.00638708 * 0.8
    return beam_config


if __name__ == "__main__":

    RUN_MODE = "viewer"  # "export_field", "viewer", "fno_viewer", or "fno_batch"

    #PHASE_MASK = r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Proxy_calibration_AltBeam_1image\Epoch_500\Proxy_train_pool\HollowRectangle\slm_phase.npy"
    PHASE_MASK = r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Checkerboard_generation\files\phase_mask_pitch2x.npy"
    #PHASE_MASK = r"G:\shared drive\taylorlab\3DHL\CITL\Fourier Neural Operator_Training phase masks\test_hollowsquares\00_baseline.npy"

    FNO_CHECKPOINT = r""  # e.g. r"C:\...\05_18_2026_FNO_training\20260525_123456\best.pt"
    FNO_OUTPUT_DIRECTORY = Path(PHASE_MASK).parent / "fno_proxy_outputs"
    FNO_CROP_VISUALIZATION_TO_ROI = True
    FNO_SHOW_SINGLE_PLOT = True

    SLM_PHASE_DIRECTORY = Path(PHASE_MASK).parent
    PHASE_GLOB = "*.npy"
    Z_TARGET = 0.01149  # Physical propagation distance in the selected medium [m], 11.49mm for cuvette condition
    SAVE_DIRECTORY = SLM_PHASE_DIRECTORY / f"electric_fields_z{Z_TARGET * 1000:.1f}mm"

    TRANSPOSE_PHASE = True
    FLIP_PHASE_FIRST_AXIS = True
    PHASE_LEVEL_MAX = 1023.0
    EXPORT_ROI_SIZE = 1024
    VIEWER_ROI_SIZE = 1024
    OVERWRITE_OUTPUTS = True
    MATCH_VIEWER_ORIENTATION = True
    SHOW_TRANSFER_FUNCTION_PLOT = RUN_MODE.lower() in {"viewer", "fno_viewer"}

    beam_config = build_beam_config()

    axicon_grating_pitch = 1.396e-6
    upsample_factor = 20
    propagation_medium_index = 1.471
    axicon_angle_in_medium = False
    axicon_transverse_frequency = 1.0 / axicon_grating_pitch
    axicon_na_air_equiv = beam_config.lambda_ * axicon_transverse_frequency
    if axicon_na_air_equiv >= 1.0:
        raise ValueError("Axicon grating pitch gives an invalid free-space NA >= 1.")
    cone_angle = float(np.arcsin(axicon_na_air_equiv))
    cone_angle_in_medium = float(np.arcsin(axicon_na_air_equiv / propagation_medium_index))
    print(f"Axicon air-equivalent NA={axicon_na_air_equiv:.4f}, "
          f"theta_air={cone_angle:.4f} rad, "
          f"theta_medium={cone_angle_in_medium:.4f} rad at n={propagation_medium_index:.3f}")
    print(f"Single physical z target: {Z_TARGET * 1000:.3f} mm")

    print('1. Initializing beam')
    beam = HoloBeam(beam_config)
    z_eval_planes = torch.tensor([Z_TARGET],
                                 device=beam_config.device,
                                 dtype=beam_config.fdtype)

    h_asm = build_axicon_transfer_function(
        beam,
        SHOW_TRANSFER_FUNCTION_PLOT,
        upsample_factor=upsample_factor,
        z_query=z_eval_planes,
        n_medium=propagation_medium_index,
        axicon_angle=cone_angle,
        axicon_angle_in_medium=axicon_angle_in_medium,
        axicon_transverse_frequency=axicon_transverse_frequency,
        margin_factor=5000,
    )
    print('2. Transfer function computation has been completed')

    beam.slm_amplitude_profile = beam.buildSLMAmplitudeProfile()
    beam.beam_mean_amplitude_iter = torch.tensor(1.0, device=beam_config.device, dtype=beam_config.fdtype)

    run_mode = RUN_MODE.lower()
    if run_mode == "export_field":
        phase_paths = list_phase_files(SLM_PHASE_DIRECTORY, PHASE_GLOB)
        export_electric_fields(
            beam=beam,
            beam_config=beam_config,
            phase_paths=phase_paths,
            save_directory=SAVE_DIRECTORY,
            cone_angle=cone_angle,
            upsample_factor=upsample_factor,
            h_asm=h_asm,
            roi_size=EXPORT_ROI_SIZE,
            propagation_medium_index=propagation_medium_index,
            axicon_angle_in_medium=axicon_angle_in_medium,
            axicon_transverse_frequency=axicon_transverse_frequency,
            transpose_phase=TRANSPOSE_PHASE,
            flip_phase_first_axis=FLIP_PHASE_FIRST_AXIS,
            phase_level_max=PHASE_LEVEL_MAX,
            overwrite_outputs=OVERWRITE_OUTPUTS,
            transpose_output_field=MATCH_VIEWER_ORIENTATION,
        )
    elif run_mode == "viewer":
        phase_tensor, phase_data = load_slm_phase_tensor(
            PHASE_MASK,
            beam_config,
            transpose=TRANSPOSE_PHASE,
            flip_first_axis=FLIP_PHASE_FIRST_AXIS,
            phase_level_max=PHASE_LEVEL_MAX,
        )
        plt.imshow(phase_data)
        plt.show()
        #plt.hist(phase_data.flatten(), bins=1024);plt.show()

        # Plane wave debug: Activate the line below.
        #phase_tensor = torch.ones_like(phase_tensor) * 1023

        #visualize_kspace_distortion(beam, phase_tensor, cone_angle, upsample_factor,
        #                            n_medium=propagation_medium_index,
        #                            axicon_angle_in_medium=axicon_angle_in_medium,
        #                            axicon_transverse_frequency=axicon_transverse_frequency)
        recon = propagate_axicon_field(
            beam,
            phase_tensor,
            cone_angle,
            upsample_factor,
            h_asm,
            VIEWER_ROI_SIZE,
            propagation_medium_index,
            axicon_angle_in_medium,
            axicon_transverse_frequency,
        )
        print('3. Axicon propagation is successfully computed')
        recon_np = recon.detach().cpu().numpy()
        recon_intensity_np = np.abs(recon_np) ** 2
        recon_intensity_np = np.transpose(recon_intensity_np, (1, 0, 2))

        for i in range(recon_intensity_np.shape[2]):
            slice_2d = recon_intensity_np[:, :, i]
            plt.figure(figsize=(10, 10))
            plt.imshow(slice_2d, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Intensity')
            plt.title(f"Bessel Beam Intensity - Slice at z = {z_eval_planes[i].item() * 1000:.1f}mm")
            plt.show()

        #debug_aliasing_1d(slice_2d=slice_2d, upsample_factor=upsample_factor,
        #                  axicon_na=axicon_na_air_equiv, original_pixel_size=8e-6)
        mbvam.Geometry.visualize.openCVSliceViewer(recon_intensity_np) #Press q to quit the app.
    elif run_mode in {"fno_viewer", "fno_batch"}:
        if not FNO_CHECKPOINT:
            raise ValueError("Set FNO_CHECKPOINT to a trained FNO best.pt before using FNO inference modes.")

        fno_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fno_model, fno_cfg = load_fno_proxy(FNO_CHECKPOINT, fno_device)
        phase_paths = [Path(PHASE_MASK)] if run_mode == "fno_viewer" else list_phase_files(SLM_PHASE_DIRECTORY, PHASE_GLOB)

        print(f"Running FNO proxy inference for {len(phase_paths)} phase mask(s)")
        for index, phase_path in enumerate(phase_paths, start=1):
            print(f"[{index}/{len(phase_paths)}] {phase_path.name}")
            run_fno_proxy_inference(
                beam=beam,
                beam_config=beam_config,
                phase_path=phase_path,
                fno_model=fno_model,
                fno_cfg=fno_cfg,
                cone_angle=cone_angle,
                upsample_factor=upsample_factor,
                h_asm=h_asm,
                roi_size=VIEWER_ROI_SIZE,
                propagation_medium_index=propagation_medium_index,
                axicon_angle_in_medium=axicon_angle_in_medium,
                axicon_transverse_frequency=axicon_transverse_frequency,
                transpose_phase=TRANSPOSE_PHASE,
                flip_phase_first_axis=FLIP_PHASE_FIRST_AXIS,
                phase_level_max=PHASE_LEVEL_MAX,
                transpose_output_field=MATCH_VIEWER_ORIENTATION,
                output_directory=FNO_OUTPUT_DIRECTORY,
                crop_visualization_to_roi=FNO_CROP_VISUALIZATION_TO_ROI,
                show_plot=FNO_SHOW_SINGLE_PLOT and run_mode == "fno_viewer",
            )
    else:
        raise ValueError(f"Unsupported RUN_MODE={RUN_MODE!r}. "
                         "Use 'export_field', 'viewer', 'fno_viewer', or 'fno_batch'.")
