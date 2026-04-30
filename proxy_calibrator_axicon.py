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
 
def get_combined_amplitude(beam, base_profile_cache):
    """
    Compute the combined SLM amplitude profile (base × modulation) using a
    cached base Gaussian to avoid per-iteration recomputation.
 
    `base_profile_cache` is the Gaussian profile (or whatever non-trainable
    illumination) computed ONCE at training start. This function only does:
        1. base × source_modulation_map  (autograd-tracked)
        2. mean-normalize to keep total energy stable
 
    This is mathematically equivalent to beam.buildSLMAmplitudeProfile() in the
    proxy training context, but skips the redundant Gaussian rebuild every step.
    """
    if beam.source_modulation_map is None:
        return base_profile_cache
    combined = base_profile_cache * beam.source_modulation_map
    mean_val = torch.mean(torch.abs(combined)) + 1e-12
    return combined / mean_val
 
 
def axicon_forward_proxy(beam, slm_phase, H_asm, cone_angle,
                          upsample_factor, roi_size, z_target_idx,
                          use_profile):
    """
    Run the axicon forward model and return intensity at a single z plane,
    scaled by camera_scale_factor.
 
    `use_profile` is the pre-computed SLM amplitude (base × modulation) — caller
    is responsible for keeping it consistent with current source_modulation_map.
    Pass `get_combined_amplitude(beam, base_profile_cache)` per training iter.
    """
    beam_amp = torch.tensor(1.0, device=beam.beam_config.device,
                            dtype=beam.beam_config.fdtype)
 
    # convert_to_intensity=False — we want complex field so we can take |·|^2
    # exactly once. Calling with True and then |·|^2 again squares the intensity,
    # which is the bug the original forward_proxy_2D_Axicon had.
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
    intensity  = torch.abs(slice_cplx) ** 2
 
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
                              # Per-parameter learning rates
                              lr_zernike=0.05, lr_source=0.05, lr_scale=0.02,
                              # Source map constraints / regularization
                              source_clamp=(0.7, 1.3),
                              source_l2_lambda=1e-2,
                              source_smooth_lambda=1e-3):
    """
    Calibrate proxy parameters (Zernike, source map, scale) for the axicon system.
 
    Parameter freezing
    ------------------
    Setting any of `lr_zernike`, `lr_source`, `lr_scale` to 0 effectively freezes
    that parameter group. Adam still has those parameters in its state, but
    with lr=0 the update is exactly zero. (We also call requires_grad_(False)
    when lr=0 to skip gradient computation entirely — saves time and memory.)
 
    Source map regularization
    -------------------------
    The source modulation map represents per-pixel deviation of the actual
    illumination from the ideal Gaussian beam (=1). Without regularization, the
    optimizer abuses its high DOF to fit camera artifacts (mirror seams, model
    mismatch, etc.), producing ring-shaped or zero-valued patterns that have
    no physical meaning. Three layers of constraint are applied:
 
      1. Hard clamp: `source_clamp = (lo, hi)` — values outside [lo, hi] are
         clipped after every step. Default (0.7, 1.3) = ±30% deviation, which
         is realistic for laser illumination non-uniformity.
 
      2. L2 penalty toward 1: `source_l2_lambda * mean((s - 1)^2)` keeps the
         map close to ideal unless data demands otherwise. λ = 1e-2 is a
         moderate prior; raise it if overfitting persists.
 
      3. Smoothness penalty: `source_smooth_lambda * mean(|∇s|^2)` discourages
         high-frequency speckle that the laser physically cannot produce. λ =
         1e-3 keeps Gaussian-scale variations free while killing pixel-level noise.
 
    Disable any layer by setting its λ to 0.
    """
    print(f"\n>>> Start Axicon Proxy Calibration on {device}...")
 
    beam.init_proxy_params(num_zernike=20, device=device)
    beam.beam_config.device = device
 
    # ── Freeze parameters with lr=0 ──────────────────────────────────────────
    if lr_zernike == 0.0:
        beam.zernike_coeffs.requires_grad_(False)
        print(">>> [Frozen] zernike_coeffs (lr=0)")
    if lr_source == 0.0:
        beam.source_modulation_map.requires_grad_(False)
        print(">>> [Frozen] source_modulation_map (lr=0)")
    if lr_scale == 0.0:
        beam.camera_scale_factor.requires_grad_(False)
        print(">>> [Frozen] camera_scale_factor (lr=0)")
 
    # Build optimizer ONLY with parameters that have requires_grad=True
    optim_groups = []
    if beam.zernike_coeffs.requires_grad:
        optim_groups.append({'params': beam.zernike_coeffs,        'lr': lr_zernike})
    if beam.source_modulation_map.requires_grad:
        optim_groups.append({'params': beam.source_modulation_map, 'lr': lr_source})
    if beam.camera_scale_factor.requires_grad:
        optim_groups.append({'params': beam.camera_scale_factor,   'lr': lr_scale})
 
    if len(optim_groups) == 0:
        print(">>> WARNING: All parameters frozen. Nothing to train.")
        return []
 
    optimizer = optim.Adam(optim_groups)
 
    # ── Pre-compute the base Gaussian once (it does not require grad) ────────
    # This avoids rebuilding the Gaussian profile on every forward pass.
    print(">>> Pre-computing base illumination profile...")
    if beam.beam_config.amplitude_profile_type == 'gaussian':
        base_profile = beam.buildGaussianSourceProfile(
            beam.beam_config.gaussian_beam_waist).detach()
    else:
        base_profile = beam.buildFlatTopSourceProfile().detach()
    base_profile = base_profile.to(device)
 
    loss_history = []
    progress_bar = tqdm(range(epochs), desc="Axicon Calibration")
 
    for epoch in progress_bar:
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_reg_loss  = 0.0
        grad_src_mean = 0.0
        grad_znk_mean = 0.0
        n_batches = 0
 
        for phases, targets, _ in dataloader:
            phases  = phases.to(device)
            targets = targets.to(device)
 
            optimizer.zero_grad()
 
            # Build combined amplitude ONCE per batch (not per sample) — it
            # depends only on source_modulation_map, which is the same for all
            # samples in a batch.
            combined_amp = get_combined_amplitude(beam, base_profile)
 
            data_loss = 0.0
            B = phases.shape[0]
 
            for i in range(B):
                single_phase = phases[i]
 
                sim = axicon_forward_proxy(
                    beam, single_phase, H_asm, cone_angle,
                    upsample_factor, roi_size, z_target_idx,
                    use_profile=combined_amp,
                )
                tgt = targets[i]
 
                # Resize sim to match camera resolution if needed
                if sim.shape != tgt.shape:
                    sim_r = F.interpolate(
                        sim.unsqueeze(0).unsqueeze(0),
                        size=tgt.shape, mode='area').squeeze()
                else:
                    sim_r = sim
 
                # Per-sample normalization
                if loss_normalize == 'per_sample_rms':
                    sim_n = sim_r / (torch.sqrt((sim_r**2).mean()) + 1e-8)
                    tgt_n = tgt   / (torch.sqrt((tgt**2).mean())   + 1e-8)
                elif loss_normalize == 'per_sample_max':
                    sim_n = sim_r / (sim_r.max() + 1e-8)
                    tgt_n = tgt   / (tgt.max()   + 1e-8)
                else:
                    sim_n, tgt_n = sim_r, tgt
 
                data_loss = data_loss + F.mse_loss(sim_n, tgt_n)
 
            data_loss = data_loss / B
 
            # ── Regularization on source_modulation_map ──────────────────────
            reg_loss = torch.tensor(0.0, device=device)
            if beam.source_modulation_map.requires_grad:
                s = beam.source_modulation_map
                if source_l2_lambda > 0:
                    reg_loss = reg_loss + source_l2_lambda * ((s - 1.0) ** 2).mean()
                if source_smooth_lambda > 0:
                    # Total-variation-like L2 of finite differences
                    dx = s[1:, :] - s[:-1, :]
                    dy = s[:, 1:] - s[:, :-1]
                    reg_loss = reg_loss + source_smooth_lambda * (
                        (dx ** 2).mean() + (dy ** 2).mean())
 
            total_loss = data_loss + reg_loss
            total_loss.backward()
 
            if (beam.source_modulation_map.requires_grad
                    and beam.source_modulation_map.grad is not None):
                grad_src_mean += beam.source_modulation_map.grad.abs().mean().item()
            if (beam.zernike_coeffs.requires_grad
                    and beam.zernike_coeffs.grad is not None):
                grad_znk_mean += beam.zernike_coeffs.grad.abs().mean().item()
 
            optimizer.step()
 
            # Hard clamp on source map after each step
            if beam.source_modulation_map.requires_grad:
                with torch.no_grad():
                    beam.source_modulation_map.clamp_(min=source_clamp[0],
                                                       max=source_clamp[1])
 
            epoch_loss      += total_loss.item()
            epoch_data_loss += data_loss.item()
            epoch_reg_loss  += float(reg_loss.item()) if torch.is_tensor(reg_loss) else 0.0
            n_batches       += 1
 
        avg_loss      = epoch_loss      / max(n_batches, 1)
        avg_data_loss = epoch_data_loss / max(n_batches, 1)
        avg_reg_loss  = epoch_reg_loss  / max(n_batches, 1)
        loss_history.append(avg_loss)
 
        progress_bar.set_postfix({
            'L_total':  f"{avg_loss:.5f}",
            'L_data':   f"{avg_data_loss:.5f}",
            'L_reg':    f"{avg_reg_loss:.5f}",
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
    DATA_ROOT = r'G:\공유 드라이브\taylorlab\3DHL\CITL\Proxy_Train_Pool_2'
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
    upsample_factor      = 12
    Axicon_NA            = beam_config.lambda_ / Axicon_grating_pitch
    Cone_angle           = float(np.arcsin(Axicon_NA))
 
    roi_size = 1600
 
    # z-plane where the camera was positioned during data collection [m]
    # Use a *single* plane that matches the physical measurement distance.
    # z_eval_planes covers a small window around it so ASM is consistent.
    z_camera      = 0.005
    z_min, z_max  = z_camera , z_camera 
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
        margin_factor   = 2000,
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
 
    # ── Pre-build base profile (cached, no autograd) ─────────────────────────
    # Used by both baseline snapshot (before training) and the all-sample
    # visualization loop (after training). buildGaussianSourceProfile is a pure
    # function of beam_config so this is fully deterministic.
    if beam_config.amplitude_profile_type == 'gaussian':
        base_profile = beam.buildGaussianSourceProfile(
            beam_config.gaussian_beam_waist).detach().to(DEVICE)
    else:
        base_profile = beam.buildFlatTopSourceProfile().detach().to(DEVICE)
 
    # ── Initialize proxy params BEFORE baseline (for camera_scale_factor) ────
    print(">>> Initializing proxy parameters for baseline snapshot...")
    beam.init_proxy_params(num_zernike=20, device=DEVICE)
 
    # ── Baseline forwards (BEFORE training, all samples) ─────────────────────
    # Cache baseline forwards now so we can compare against post-training later
    # without re-running them. We need to do this BEFORE training mutates the
    # proxy params.
    print(">>> Computing baseline forwards for all samples...")
    baseline_combined = get_combined_amplitude(beam, base_profile)
    baseline_forwards = {}
    with torch.no_grad():
        for i in range(len(dataset)):
            ph, tgt, name = dataset[i]
            ph  = ph.to(DEVICE)
            tgt = tgt.to(DEVICE)
            sim = axicon_forward_proxy(
                beam, ph, H_asm, Cone_angle,
                upsample_factor, roi_size, z_target_idx,
                use_profile=baseline_combined,
            )
            if sim.shape != tgt.shape:
                sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0),
                                     size=tgt.shape, mode='area').squeeze()
            baseline_forwards[name] = sim.detach().cpu()
 
    # ── Train ────────────────────────────────────────────────────────────────
    print("4. Training proxy model...")
    # Set any lr to 0.0 to FREEZE that parameter (e.g. lr_source=0 for samples
    # where you only want Zernike to be learned).
    loss_history = train_proxy_model_axicon(
        beam               = beam,
        dataloader         = dataloader,
        H_asm              = H_asm,
        cone_angle         = Cone_angle,
        upsample_factor    = upsample_factor,
        roi_size           = roi_size,
        z_target_idx       = z_target_idx,
        epochs             = 20,
        device             = DEVICE,
        loss_normalize     = 'per_sample_rms',
        # Per-parameter learning rates (0.0 = freeze)
        lr_zernike         = 0.05,
        lr_source          = 0.05,
        lr_scale           = 0.02,
        # Source-map constraints
        source_clamp         = (0.7, 1.3),   # ±30% deviation, hard clipped
        source_l2_lambda     = 1e-2,         # pull toward 1
        source_smooth_lambda = 1e-3,         # discourage high-frequency speckle
    )
    save_proxy_params(beam, 'proxy_model_params_axicon.pt')
 
    # ── Post-training visualization: ALL SAMPLES ─────────────────────────────
    print(">>> Generating per-sample comparison plots...")
    output_dir = 'sample_comparisons'
    os.makedirs(output_dir, exist_ok=True)
 
    # Trained profile is constant across samples — compute once
    with torch.no_grad():
        trained_combined = get_combined_amplitude(beam, base_profile)
 
    per_sample_summary = []   # (name, mse_baseline, mse_proxy)
 
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Sample plots"):
            phase, real, name = dataset[idx]
            phase = phase.to(DEVICE)
            real  = real.to(DEVICE)
 
            # Forward with trained proxy
            proxy_sim = axicon_forward_proxy(
                beam, phase, H_asm, Cone_angle,
                upsample_factor, roi_size, z_target_idx,
                use_profile=trained_combined,
            )
            if proxy_sim.shape != real.shape:
                proxy_sim = F.interpolate(
                    proxy_sim.unsqueeze(0).unsqueeze(0),
                    size=real.shape, mode='area').squeeze()
 
            # Retrieve baseline forward (computed before training)
            baseline_sim = baseline_forwards[name].to(DEVICE)
 
            # Normalize for display (quantile to handle outlier hot pixels)
            tgt_n = quantile_normalize(real,         q_max=0.999)
            base_n = quantile_normalize(baseline_sim, q_max=0.999)
            prox_n = quantile_normalize(proxy_sim,    q_max=0.999)
 
            # Resize to common display size (500×500)
            tgt_d  = force_resize(tgt_n,  target_res=(500, 500)).cpu().numpy()
            base_d = force_resize(base_n, target_res=(500, 500)).cpu().numpy()
            prox_d = force_resize(prox_n, target_res=(500, 500)).cpu().numpy()
 
            # Revert dataset transpose for display (so plot orientation matches
            # the original camera PNG when viewed in a file viewer)
            tgt_d  = tgt_d.T
            base_d = base_d.T
            prox_d = prox_d.T
 
            mse_base  = float(np.mean((tgt_d - base_d) ** 2))
            mse_proxy = float(np.mean((tgt_d - prox_d) ** 2))
            per_sample_summary.append((name, mse_base, mse_proxy))
 
            fig, ax = plt.subplots(1, 3, figsize=(18, 7))
            ax[0].imshow(tgt_d,  cmap='gray', vmin=0, vmax=1)
            ax[0].set_title(f"Target — {name}")
            ax[1].imshow(base_d, cmap='gray', vmin=0, vmax=1)
            ax[1].set_title(f"Baseline (Before)\nMSE: {mse_base:.4f}")
            ax[2].imshow(prox_d, cmap='gray', vmin=0, vmax=1)
            ax[2].set_title(f"Proxy (After)\nMSE: {mse_proxy:.4f}")
            for a in ax:
                a.axis('off')
            plt.tight_layout()
            # Sanitize sample name for filename (Windows-safe)
            safe = "".join(c if c.isalnum() or c in '-_.' else '_' for c in name)
            plt.savefig(os.path.join(output_dir,
                f'compare_{idx:02d}_{safe}.png'), dpi=120)
            plt.close()
 
    # Print summary table
    print("\n=== Per-sample MSE summary ===")
    print(f"{'Sample':<35} {'Baseline':>10} {'Proxy':>10}  Δ")
    for name, mb, mp in per_sample_summary:
        delta = "↓" if mp < mb else "↑"
        print(f"{name:<35} {mb:>10.4f} {mp:>10.4f}  {delta}")
    print(f"Saved {len(per_sample_summary)} comparison plots to {output_dir}/")
 
    # ── Loss curve ───────────────────────────────────────────────────────────
    print(">>> Saving loss curve...")
    plt.figure()
    plt.semilogy(loss_history, color='red')
    plt.xlabel("Epoch"); plt.ylabel("Total Loss (data + reg)")
    plt.title("Training Loss (Axicon Proxy)")
    plt.grid(True)
    plt.savefig('Loss_curve_axicon.png', dpi=120)
    plt.close()
 
    # ── Zernike analysis ─────────────────────────────────────────────────────
    analyze_zernike_results(beam, save_path='zernike_analysis_axicon.png')
 
    # ── Source map ───────────────────────────────────────────────────────────
    source_map = beam.source_modulation_map.detach().cpu().squeeze().numpy()
    plt.figure(figsize=(7, 6))
    # Range matched to clamp window so deviations from 1 are visible
    src_min, src_max = source_map.min(), source_map.max()
    plt.imshow(source_map.T, cmap='RdBu_r', aspect='auto',
               vmin=min(src_min, 0.6), vmax=max(src_max, 1.4))
    plt.colorbar(label='Modulation factor (1 = no modulation)')
    plt.title(f"Learned Source Modulation Map  "
              f"(min={src_min:.3f}, max={src_max:.3f})")
    plt.tight_layout()
    plt.savefig('trained_source_map_axicon.png', dpi=120)
    plt.close()
 
    # ── Per-sample MSE bar chart (reuse cached values) ───────────────────────
    print(">>> Saving per-sample MSE bar chart...")
    names      = [s[0] for s in per_sample_summary]
    mses_base  = [s[1] for s in per_sample_summary]
    mses_proxy = [s[2] for s in per_sample_summary]
 
    x = np.arange(len(names))
    plt.figure(figsize=(max(8, 0.6 * len(names)), 6))
    plt.bar(x - 0.2, mses_base,  width=0.4, label='Baseline', color='lightgray')
    plt.bar(x + 0.2, mses_proxy, width=0.4, label='Proxy',     color='steelblue')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.xlabel("Sample"); plt.ylabel("MSE (display-normalized)")
    plt.title("Per-sample MSE: baseline vs proxy")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('MSE_per_sample_axicon.png', dpi=120)
    plt.close()
 
    print("Done.")