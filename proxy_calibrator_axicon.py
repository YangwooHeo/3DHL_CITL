"""
Proxy model calibrator for SLM + Axicon system.

Key differences from the lens-only version
------------------------------------------
1. Forward model uses propagateToVolume_Axicon2 (axicon + ASM), not propagateToVolume.
2. A single z-plane is selected from the axicon volume output and compared to the
   camera image. The z-plane is chosen at the physical distance between the axicon
   and the imaging plane.
3. H_asm is pre-built once at training start (not per-iteration) since it is
   deterministic given z_query / axicon_angle and does not require gradient.
4. The forward no longer re-applies |·|^2 to an already-intensity tensor.
5. Per-sample scale + offset normalization is applied so the proxy's RMS matches
   the camera image RMS within each sample — this is critical for axicon because
   intensity across samples can differ by orders of magnitude, unlike lens proxy
   where focal peak is roughly constant.

What is learned
---------------
- zernike_coeffs           : wavefront aberration at SLM plane (20 modes by default)
- source_modulation_map    : per-pixel amplitude correction of the SLM illumination
- camera_scale_factor      : global intensity scale (camera gain × optical throughput)
"""

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import math
if not hasattr(np, 'math'):
    np.math = math
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_func

from mbvam.Beam.holobeam import HoloBeam
from mbvam.Beam.holobeamconfig import HoloBeamConfig
from mbvam.Beam.zernike import combine_zernike_basis, compute_zernike_basis


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (unchanged from lens version)
# ─────────────────────────────────────────────────────────────────────────────

class ProxyCalibrationDataset(Dataset):
    def __init__(self, root_dir, phase_filename='slm_phase.npy',
                 camera_filename='Final_Aligned_Camera.png'):
        self.samples = []
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        subdirs = sorted(d for d in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, d)))
        for d in subdirs:
            dir_path = os.path.join(root_dir, d)
            ph = os.path.join(dir_path, phase_filename)
            cm = os.path.join(dir_path, camera_filename)
            if os.path.exists(ph) and os.path.exists(cm):
                self.samples.append({'phase_path': ph, 'camera_path': cm, 'id': d})
            else:
                print(f"[Skip] Missing files in {d}")
        print(f">>> Found {len(self.samples)} valid samples in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        phase_data = np.load(s['phase_path']).T.astype(np.float32)
        phase_data = phase_data[::-1, :] ## 04_22_2026 Fix: Image flip compensation
        phase_rad  = phase_data * (2 * np.pi / 1023)
        phase_tensor = torch.from_numpy(phase_rad).float()

        H, W = phase_tensor.shape[-2], phase_tensor.shape[-1]

        cam = Image.open(s['camera_path']).convert('L')
        cam = cam.transpose(Image.TRANSPOSE)
        if cam.size != (W, H):
            cam = cam.resize((W, H), Image.BILINEAR)
        cam_tensor = transforms.ToTensor()(cam).squeeze()   # [H, W] in [0, 1]

        return phase_tensor, cam_tensor, s['id']


# ─────────────────────────────────────────────────────────────────────────────
# Forward helper for axicon proxy
# ─────────────────────────────────────────────────────────────────────────────
def get_center_crop_slice(shape, crop_ratio):
    """
    Return (y_slice, x_slice, (y0, y1, x0, x1)) for a centered crop.
    crop_ratio in (0, 1]; 1.0 = full image, 0.5 = central 50%.
    """
    H, W = shape
    ch = int(round(H * crop_ratio))
    cw = int(round(W * crop_ratio))
    y0 = (H - ch) // 2
    x0 = (W - cw) // 2
    y1 = y0 + ch
    x1 = x0 + cw
    return slice(y0, y1), slice(x0, x1), (y0, y1, x0, x1)
 
 
def draw_roi_overlay(ax, bounds, color='cyan', linewidth=1.2, linestyle='--'):
    """Draw a thin rectangle on an imshow axis given (y0, y1, x0, x1)."""
    y0, y1, x0, x1 = bounds
    # imshow uses (col=x, row=y). Rectangle takes xy=(x, y).
    from matplotlib.patches import Rectangle
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                     fill=False, edgecolor=color,
                     linewidth=linewidth, linestyle=linestyle)
    ax.add_patch(rect)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Forward helper for axicon proxy
# ─────────────────────────────────────────────────────────────────────────────
 
def axicon_forward_proxy(beam, slm_phase, H_asm, cone_angle,
                          upsample_factor, roi_size, z_target_idx,
                          use_profile):
    """
    Run the axicon forward model and return intensity at a single z plane,
    scaled by camera_scale_factor.
 
    This is the axicon analogue of `beam.forward_proxy_2D` from the lens version.
    Written as a standalone function (rather than a method of HoloBeam) so that
    it's self-contained and easy to debug — the in-class forward_proxy_2D_Axicon
    has a few gradient-path bugs we want to avoid.
    """
    beam_amp = torch.tensor(1.0, device=beam.beam_config.device,
                            dtype=beam.beam_config.fdtype)
 
    # NOTE: convert_to_intensity=False — we want complex field so we can pick
    # a slice and take |·|^2 once. Calling with True and then |·|^2 again
    # (as the existing forward_proxy_2D_Axicon does) squares the intensity
    # which is wrong.
    vol_field = beam.propagateToVolume_Axicon2(
        axicon_angle=cone_angle,
        upsample_factor=upsample_factor,
        phase_mask=slm_phase,
        beam_mean_amplitude=beam_amp,
        slm_amplitude_profile=use_profile,
        H_asm=H_asm,
        roi_size=roi_size,
        convert_to_intensity=False,
    )
    slice_cplx = vol_field[:, :, z_target_idx]
    intensity  = torch.abs(slice_cplx) ** 2                 # [roi, roi]
 
    # Global intensity scale (abs to keep it positive)
    if beam.camera_scale_factor is not None:
        intensity = intensity * torch.abs(beam.camera_scale_factor)
    return intensity
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
 
def train_proxy_model_axicon(beam, dataloader, H_asm, cone_angle,
                              upsample_factor, roi_size, z_target_idx,
                              epochs=150, device='cuda',
                              loss_normalize='per_sample_rms',
                              source_clamp=(1e-6, 5.0),
                              center_crop_ratio=1.0):
    """
    Calibrate proxy parameters (Zernike, source map, scale) for the axicon system.
 
    loss_normalize
    --------------
    'per_sample_rms'  (default, recommended for axicon):
        Normalize both sim and target to unit RMS within each sample before MSE.
        Removes sample-to-sample intensity scale variation — axicon patterns can
        differ by orders of magnitude, and plain MSE lets bright samples dominate
        the gradient. RMS (not max) normalization is also less sensitive to a few
        hot pixels that are common in axicon speckle.
 
    'per_sample_max':
        Normalize by max of each sample. Simpler but more sensitive to outlier
        pixels.
 
    'none':
        Raw MSE on un-normalized outputs — proxy must learn the absolute scale
        via camera_scale_factor. This is what the lens version effectively did
        but is usually unstable for axicon.
 
    source_clamp
    ------------
    (min, max) range for source_modulation_map. The default 5.0 upper bound
    prevents non-physical runaway that the original code allowed.
 
    center_crop_ratio
    -----------------
    Fraction of image (centered) used for loss computation. 1.0 = full image,
    0.7 = central 70% region (recommended when camera frame has mirrored-pad
    artifacts at the edges that you don't want the proxy to fit). Applied
    AFTER per-sample normalization — normalization uses the full image so the
    RMS reference is stable, only the MSE is restricted to the crop.
    """
    print(f"\n>>> Start Axicon Proxy Calibration on {device}...")
 
    beam.init_proxy_params(num_zernike=20, device=device)
    beam.beam_config.device = device
 
    optimizer = optim.Adam([
        {'params': beam.zernike_coeffs,        'lr': 0.05},
        {'params': beam.source_modulation_map, 'lr': 0.05},
        {'params': beam.camera_scale_factor,   'lr': 0.02},
    ])
 
    loss_history = []
    progress_bar = tqdm(range(epochs), desc="Axicon Calibration")
 
    for epoch in progress_bar:
        epoch_loss = 0.0
        grad_src_mean = 0.0
        grad_znk_mean = 0.0
        n_batches = 0
 
        for phases, targets, _ in dataloader:
            phases  = phases.to(device)
            targets = targets.to(device)
 
            optimizer.zero_grad()
 
            batch_loss = 0.0
            B = phases.shape[0]
 
            # Per-sample forward + loss (axicon forward is too large to batch
            # in the tensor dim — upsampled FFTs would OOM). We accumulate loss,
            # then call backward once per batch so Adam step sees the sum.
            for i in range(B):
                single_phase = phases[i]
 
                sim = axicon_forward_proxy(
                    beam, single_phase, H_asm, cone_angle,
                    upsample_factor, roi_size, z_target_idx,
                    use_profile=beam.buildSLMAmplitudeProfile(),
                )
                tgt = targets[i]
 
                # Resize sim to match camera resolution if needed
                if sim.shape != tgt.shape:
                    sim_r = F.interpolate(
                        sim.unsqueeze(0).unsqueeze(0),
                        size=tgt.shape, mode='area').squeeze()
                else:
                    sim_r = sim
 
                # Per-sample normalization (full image — RMS reference stable)
                if loss_normalize == 'per_sample_rms':
                    sim_n = sim_r / (torch.sqrt((sim_r**2).mean()) + 1e-8)
                    tgt_n = tgt   / (torch.sqrt((tgt**2).mean())   + 1e-8)
                elif loss_normalize == 'per_sample_max':
                    sim_n = sim_r / (sim_r.max() + 1e-8)
                    tgt_n = tgt   / (tgt.max()   + 1e-8)
                else:  # 'none'
                    sim_n, tgt_n = sim_r, tgt
 
                # Optional center-crop restriction for the MSE
                # (useful when camera frame has mirror-padded regions at edges)
                if center_crop_ratio < 1.0:
                    ys, xs, _ = get_center_crop_slice(sim_n.shape,
                                                      center_crop_ratio)
                    sim_n = sim_n[ys, xs]
                    tgt_n = tgt_n[ys, xs]
 
                batch_loss = batch_loss + F.mse_loss(sim_n, tgt_n)
 
            # Average across batch so lr is consistent w.r.t. batch_size
            batch_loss = batch_loss / B
            batch_loss.backward()
 
            if beam.source_modulation_map.grad is not None:
                grad_src_mean += beam.source_modulation_map.grad.abs().mean().item()
            if beam.zernike_coeffs.grad is not None:
                grad_znk_mean += beam.zernike_coeffs.grad.abs().mean().item()
 
            optimizer.step()
 
            # Bounded constraint on source map
            with torch.no_grad():
                beam.source_modulation_map.clamp_(min=source_clamp[0],
                                                   max=source_clamp[1])
 
            epoch_loss += batch_loss.item()
            n_batches  += 1
 
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
 
        progress_bar.set_postfix({
            'Loss':     f"{avg_loss:.5f}",
            'Scale':    f"{beam.camera_scale_factor.item():.2e}",
            'SrcGrad':  f"{grad_src_mean / max(n_batches,1):.2e}",
            'ZnkGrad':  f"{grad_znk_mean / max(n_batches,1):.2e}",
        })
 
    print(">>> Calibration Finished.")
    return loss_history

# ─────────────────────────────────────────────────────────────────────────────
# Utility / metrics (unchanged, with some cleanup)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(target, prediction):
    pred_c = np.clip(prediction, 0, 1)
    targ_c = np.clip(target, 0, 1)
    mse    = np.mean((targ_c - pred_c) ** 2)
    ssim   = ssim_func(targ_c, pred_c, data_range=1.0, win_size=3)
    return mse, ssim


def force_resize(tensor_img, target_res=(500, 500)):
    if tensor_img.dim() == 2:
        img_in = tensor_img.unsqueeze(0).unsqueeze(0)
    elif tensor_img.dim() == 3:
        img_in = tensor_img.unsqueeze(0)
    else:
        img_in = tensor_img
    return F.interpolate(img_in, size=target_res, mode='area').squeeze()


def quantile_normalize(img_input, q_min=0.0, q_max=0.999):
    if isinstance(img_input, Image.Image):
        img = transforms.ToTensor()(img_input)
    elif isinstance(img_input, np.ndarray):
        img = torch.from_numpy(img_input).float()
        if img_input.dtype == np.uint8:
            img = img / 255.0
    elif isinstance(img_input, torch.Tensor):
        img = img_input.clone()
    else:
        raise TypeError(f"Unsupported image type: {type(img_input)}")

    img_flat = img.flatten()
    v_min = torch.quantile(img_flat, q_min)
    v_max = torch.quantile(img_flat, q_max)

    img = torch.clamp(img, min=v_min, max=v_max)
    return (img - v_min) / (v_max - v_min + 1e-8)


def analyze_zernike_results(beam, save_path='zernike_analysis.png'):
    print("Analyzing Zernike Aberrations...")
    coeffs = beam.zernike_coeffs.detach().cpu().numpy()

    zernike_names = [
        "0: Piston",
        "1: Tilt V",   "2: Tilt H",
        "3: Defocus",  "4: Astig. V",
        "5: Astig. H", "6: Coma V",
        "7: Coma H",   "8: Spherical",
        "9: Trefoil V","10: Trefoil H",
        "11: Sec. Astig V", "12: Sec. Astig H",
        "13: Quadrafoil V", "14: Quadrafoil H",
    ]
    if len(coeffs) > len(zernike_names):
        zernike_names += [f"{i}: Order {i}"
                          for i in range(len(zernike_names), len(coeffs))]
    else:
        zernike_names = zernike_names[:len(coeffs)]

    device = beam.beam_config.device
    if hasattr(beam, 'zernike_basis') and beam.zernike_basis is not None:
        basis = beam.zernike_basis
    else:
        res_y, res_x = beam.beam_config.Ny_physical, beam.beam_config.Nx_physical
        basis = compute_zernike_basis(len(coeffs), [res_y, res_x],
                                       dtype=torch.float32).to(device)

    aberration_complex = combine_zernike_basis(beam.zernike_coeffs, basis)
    aberration_map     = aberration_complex.angle().detach().cpu().squeeze().numpy()

    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    bars = ax1.barh(zernike_names, coeffs, color='skyblue', edgecolor='navy')
    ax1.set_xlabel("Coefficient Magnitude (rad)")
    ax1.set_title("Dominant Zernike Modes")
    ax1.grid(axis='x', linestyle='--', alpha=0.7)

    max_idx = np.argmax(np.abs(coeffs))
    bars[max_idx].set_color('salmon')
    bars[max_idx].set_edgecolor('red')

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(aberration_map, cmap='jet')
    ax2.set_title(f"Reconstructed Wavefront Error\n(Dominant: {zernike_names[max_idx]})")
    plt.colorbar(im, ax=ax2, label='Phase Error (rad)')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Zernike analysis saved to {save_path}")


def save_proxy_params(beam, filepath):
    state = {
        'zernike_coeffs':        beam.zernike_coeffs.detach().cpu(),
        'source_modulation_map': beam.source_modulation_map.detach().cpu(),
        'camera_scale_factor':   beam.camera_scale_factor.detach().cpu(),
    }
    torch.save(state, filepath)
    print(f"Proxy parameters saved to {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Paths / device ───────────────────────────────────────────────────────
    DATA_ROOT = r'H:\Shared drives\taylorlab\3DHL\CITL\Proxy_Train_Pool_1'
    DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Beam config ──────────────────────────────────────────────────────────
    beam_config = HoloBeamConfig()
    beam_config.lambda_              = 0.473e-6
    beam_config.binningFactor        = 1
    beam_config.psSLM_physical       = 8e-6 * 0.8
    beam_config.Nx_physical          = 1600
    beam_config.Ny_physical          = 1200
    beam_config.axis_angle           = [1, 0, 0]
    beam_config.z_plane_sampling_rate = 0.5
    beam_config.amplitude_profile_type = 'gaussian'
    beam_config.gaussian_beam_waist  = 0.00638708 * 0.8   # waist at axicon plane

    assert beam_config.focal_SLM is not False

    # ── Axicon config ────────────────────────────────────────────────────────
    Axicon_grating_pitch = 1.396e-6
    upsample_factor      = 24
    Axicon_NA            = beam_config.lambda_ / Axicon_grating_pitch
    Cone_angle           = float(np.arcsin(Axicon_NA))

    roi_size = 1600

    # z-plane where the camera was positioned during data collection [m]
    # Use a *single* plane that matches the physical measurement distance.
    # z_eval_planes covers a small window around it so ASM is consistent.
    z_camera      = 0.009
    z_min, z_max  = z_camera - 0.002, z_camera + 0.002
    z_steps       = 1
    z_eval_planes = torch.linspace(z_min, z_max, steps=z_steps,
                                    device=beam_config.device)
    z_target_idx  = z_steps // 2    # center plane = z_camera

    # ── Build beam ───────────────────────────────────────────────────────────
    print('1. Initializing beam')
    beam = HoloBeam(beam_config)
    beam.slm_amplitude_profile = beam.buildSLMAmplitudeProfile()

    # H_asm is deterministic & detached from autograd — build once
    print('2. Building axicon ASM transfer function')
    H_asm = beam.build_axicon_ASM_TF(
        upsample_factor = upsample_factor,
        z_query         = z_eval_planes,
        axicon_angle    = Cone_angle,
        margin_factor   = 8000,
    )

    # ── Dataset / loader ─────────────────────────────────────────────────────
    print(f'3. Loading training data from {DATA_ROOT}')
    dataset   = ProxyCalibrationDataset(
        root_dir        = DATA_ROOT,
        phase_filename  = 'slm_phase.npy',
        camera_filename = 'Final_Aligned_Camera.png',
    )
    # batch_size=1 is recommended for axicon: each forward already uses large
    # memory due to upsample. Increase only if GPU allows.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # ── Baseline snapshot (before training) ──────────────────────────────────
    fixed_batch  = next(iter(dataloader))
    fixed_phase  = fixed_batch[0][0].to(DEVICE)
    fixed_target = fixed_batch[1][0].to(DEVICE)

    print(">>> Snapshotting baseline before training...")
    beam.init_proxy_params(num_zernike=20, device=DEVICE)  # so scale factor exists
    with torch.no_grad():
        baseline_raw = axicon_forward_proxy(
            beam, fixed_phase, H_asm, Cone_angle,
            upsample_factor, roi_size, z_target_idx,
            use_profile=beam.slm_amplitude_profile,
        )
        print(f"Baseline max = {baseline_raw.max().item():.3e}")
        baseline_norm     = quantile_normalize(baseline_raw, q_max=0.999)
        baseline_resized  = force_resize(baseline_norm, target_res=(500, 500))
        baseline_snapshot = baseline_resized.detach().cpu().numpy()

    # ── Train ────────────────────────────────────────────────────────────────
    print("4. Training proxy model...")
    CENTER_CROP_RATIO = 0.7     # 1.0 = full frame, 0.7 = central 70%
    loss_history = train_proxy_model_axicon(
        beam              = beam,
        dataloader        = dataloader,
        H_asm             = H_asm,
        cone_angle        = Cone_angle,
        upsample_factor   = upsample_factor,
        roi_size          = roi_size,
        z_target_idx      = z_target_idx,
        epochs            = 25,
        device            = DEVICE,
        loss_normalize    = 'per_sample_rms',
        center_crop_ratio = CENTER_CROP_RATIO,
    )
    save_proxy_params(beam, 'proxy_model_params_axicon.pt')
 
    # ── Proxy snapshot (after training) ──────────────────────────────────────
    print(">>> Snapshotting proxy after training...")
    with torch.no_grad():
        proxy_raw = axicon_forward_proxy(
            beam, fixed_phase, H_asm, Cone_angle,
            upsample_factor, roi_size, z_target_idx,
            use_profile=beam.buildSLMAmplitudeProfile(),
        )
        proxy_norm     = quantile_normalize(proxy_raw, q_max=0.999)
        proxy_resized  = force_resize(proxy_norm, target_res=(500, 500))
        target_resized = force_resize(fixed_target, target_res=(500, 500))
 
        proxy_snapshot  = proxy_resized.detach().cpu().numpy()
        target_snapshot = target_resized.detach().cpu().numpy()
 
    # ── Plots ────────────────────────────────────────────────────────────────
    print("5. Saving plots...")
    plt.figure()
    plt.semilogy(loss_history, color='red')
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training Loss (Axicon Proxy)")
    plt.grid(True)
    plt.savefig('Loss_curve_axicon.png')
    plt.close()
 
    mse_base  = np.mean((target_snapshot - baseline_snapshot)**2)
    mse_proxy = np.mean((target_snapshot - proxy_snapshot)**2)
 
    # Dataset transposes camera images to match (Nx, Ny) phase convention.
    # Revert .T so plots show the ORIGINAL camera orientation (row=y, col=x),
    # matching how the PNG looks in a file viewer.
    target_disp   = target_snapshot.T
    baseline_disp = baseline_snapshot.T
    proxy_disp    = proxy_snapshot.T
 
    # ROI bounds (on display-space array, same shape as *_disp)
    _, _, roi_bounds = get_center_crop_slice(target_disp.shape, CENTER_CROP_RATIO)
 
    fig, ax = plt.subplots(1, 3, figsize=(18, 7))
    ax[0].imshow(target_disp,   cmap='gray', vmin=0, vmax=1)
    ax[0].set_title("Target (Camera)")
    draw_roi_overlay(ax[0], roi_bounds)
 
    ax[1].imshow(baseline_disp, cmap='gray', vmin=0, vmax=1)
    ax[1].set_title(f"Baseline (Before)\nMSE: {mse_base:.4f}")
    draw_roi_overlay(ax[1], roi_bounds)
 
    ax[2].imshow(proxy_disp,    cmap='gray', vmin=0, vmax=1)
    ax[2].set_title(f"Proxy (After)\nMSE: {mse_proxy:.4f}")
    draw_roi_overlay(ax[2], roi_bounds)
 
    plt.tight_layout()
    plt.savefig('before_after_comparison_axicon.png', dpi=150)
    plt.close()
 
    analyze_zernike_results(beam, save_path='zernike_analysis_axicon.png')
 
    source_map = beam.source_modulation_map.detach().cpu().squeeze().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(source_map, cmap='inferno', aspect='auto',
               vmin=0, vmax=min(source_map.max(), 3))
    plt.colorbar(label='Amplitude')
    plt.title("Learned Source Amplitude (SLM Plane)")
    plt.tight_layout()
    plt.savefig('trained_source_map_axicon.png')
    plt.close()
 
    # ── Per-sample MSE diagnostic ────────────────────────────────────────────
    print("Calculating per-sample MSE...")
    names, mses = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            phase, real, name = dataset[i]
            phase = phase.to(DEVICE)
            real  = real.to(DEVICE)
 
            pred = axicon_forward_proxy(
                beam, phase, H_asm, Cone_angle,
                upsample_factor, roi_size, z_target_idx,
                use_profile=beam.buildSLMAmplitudeProfile(),
            )
            if pred.shape != real.shape:
                pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0),
                                      size=real.shape, mode='area').squeeze()
 
            # Per-sample RMS normalize (consistent with training loss)
            pred_n = pred / (torch.sqrt((pred**2).mean()) + 1e-8)
            real_n = real / (torch.sqrt((real**2).mean()) + 1e-8)
 
            names.append(name)
            mses.append(F.mse_loss(pred_n, real_n).item())
 
    plt.figure(figsize=(12, 6))
    plt.plot(names, mses, marker='o', linestyle='None', color='blue', markersize=8)
    plt.xlabel("Sample"); plt.ylabel("MSE (RMS-normalized)")
    plt.title("Per-sample MSE after training")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('MSE_per_sample_axicon.png')
    plt.close()
 
    print("Done.")

