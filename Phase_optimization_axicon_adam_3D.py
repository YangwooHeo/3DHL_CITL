from mbvam.Beam.holobeam import HoloBeam
from mbvam.Beam.holobeamconfig import HoloBeamConfig
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import zoom
import os


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities  (unchanged from 2D version)
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_target(target_np, roi_size, device, fdtype):
    t = torch.tensor(target_np, dtype=fdtype, device=device)
    if t.shape != (roi_size, roi_size):
        t = TF.resize(t.unsqueeze(0), [roi_size, roi_size]).squeeze()
    return t / (t.max() + 1e-8)


def _axicon_phase_map(Nx_up, Ny_up, ps_up, cone_angle, lambda_, device, fdtype):
    x = torch.linspace(-(Nx_up-1)/2, (Nx_up-1)/2, Nx_up, device=device, dtype=fdtype) * ps_up
    y = torch.linspace(-(Ny_up-1)/2, (Ny_up-1)/2, Ny_up, device=device, dtype=fdtype) * ps_up
    X, Y = torch.meshgrid(x, y, indexing='ij')
    R = torch.sqrt(X**2 + Y**2)
    k_sin_a = (2 * torch.pi / lambda_) * torch.sin(
        torch.tensor(cone_angle, dtype=fdtype, device=device))
    return -k_sin_a * R


@torch.no_grad()
def _forward_complex_volume(phase, beam_obj, cone_angle, upsample_factor, H_asm, roi_size):
    """Forward pass → complex volume [roi, roi, Nz]. No gradient."""
    return beam_obj.propagateToVolume_Axicon2(
        axicon_angle=cone_angle,
        upsample_factor=upsample_factor,
        phase_mask=phase,
        roi_size=roi_size,
        H_asm=H_asm,
        convert_to_intensity=False,
    )


@torch.no_grad()
def _backward_adjoint_single(E_out_z, z_idx, beam_obj, cone_angle,
                              upsample_factor, H_asm, roi_size):
    """
    Adjoint back-projection for a single z-plane.
    Returns E_slm contribution [Nx, Ny] complex.
    """
    cfg = beam_obj.beam_config
    device, fdtype, cdtype = cfg.device, cfg.fdtype, cfg.cdtype
    Nx, Ny = cfg.Nx, cfg.Ny
    Nx_up = Nx * upsample_factor
    Ny_up = Ny * upsample_factor
    ps_up = cfg.psSLM / upsample_factor

    is_sparse = isinstance(H_asm, dict) and H_asm.get('type') == 'sparse'

    # Step 1: adjoint of ROI crop → zero-pad
    m0 = Nx_up // 2 - roi_size // 2
    n0 = Ny_up // 2 - roi_size // 2
    E_pad = torch.zeros((Nx_up, Ny_up), dtype=cdtype, device=device)
    E_pad[m0:m0 + roi_size, n0:n0 + roi_size] = E_out_z

    # Step 2: adjoint of IFFT → FFT
    E_k = torch.fft.fft2(E_pad, norm="ortho")

    # Step 3: adjoint of (gather ring + ×H) → ×conj(H) + scatter
    if is_sparse:
        ring_mask = H_asm['ring_mask']
        H_sp      = H_asm['H_asm_sparse']
        E_k_back  = torch.zeros((Nx_up, Ny_up), dtype=cdtype, device=device)
        E_k_back[ring_mask] = E_k[ring_mask] * torch.conj(H_sp[:, z_idx])
    else:
        E_k_back = E_k * torch.conj(H_asm[:, :, z_idx])

    # Step 4: adjoint of FFT → IFFT
    E_ax_back = torch.fft.ifft2(E_k_back, norm="ortho")

    # Step 5: adjoint of ×axicon_phase → ×conj(axicon_phase)
    phi_ax = _axicon_phase_map(Nx_up, Ny_up, ps_up, cone_angle,
                                cfg.lambda_, device, fdtype)
    E_slm_up = E_ax_back * torch.exp(-1j * phi_ax)

    # Step 6: adjoint of repeat_interleave → reshape + sum + normalize
    E_slm = (E_slm_up.reshape(Nx, upsample_factor, Ny, upsample_factor)
                      .sum(dim=(1, 3)) / (upsample_factor ** 2))
    return E_slm


# ─────────────────────────────────────────────────────────────────────────────
# 3D GS adjoint warm-up
# ─────────────────────────────────────────────────────────────────────────────

def gs_axicon_init_3d(beam_obj, target_image_np, cone_angle, H_asm,
                      num_iters=15, upsample_factor=6, roi_size=1600,
                      verbose=True):
    """
    GS-adjoint warm-up for multiple z-planes.

    Each iteration:
      1. Forward → complex volume [roi, roi, Nz]
      2. Per-plane amplitude constraint with energy matching
      3. Accumulate adjoint back-projections from ALL z-planes (sum)
      4. SLM amplitude constraint → extract phase

    The multi-plane adjoint sum is the Wirtinger gradient of a sum-of-plane
    losses, so it gives a descent direction for the full 3D objective.

    Why fewer iters than 2D GS:
      With Nz planes contributing to the adjoint sum, each step already
      carries more information. 10–20 iters is typically sufficient.
    """
    cfg = beam_obj.beam_config
    device, fdtype, cdtype = cfg.device, cfg.fdtype, cfg.cdtype

    target_I = _prepare_target(target_image_np, roi_size, device, fdtype)
    target_A = torch.sqrt(target_I)
    amp_prof = beam_obj.slm_amplitude_profile.to(cdtype)

    is_sparse = isinstance(H_asm, dict) and H_asm.get('type') == 'sparse'
    Nz = H_asm['H_asm_sparse'].shape[1] if is_sparse else H_asm.shape[2]

    phase = torch.rand((cfg.Nx, cfg.Ny), device=device, dtype=fdtype) * 2 * torch.pi

    for i in range(num_iters):

        # ── Forward: full volume ─────────────────────────────────────────────
        E_vol = _forward_complex_volume(phase, beam_obj, cone_angle,
                                         upsample_factor, H_asm, roi_size)
        # E_vol: [roi, roi, Nz] complex

        # ── Accumulate adjoint contributions from all z-planes ───────────────
        E_slm_accum = torch.zeros((cfg.Nx, cfg.Ny), dtype=cdtype, device=device)

        for z in range(Nz):
            E_z = E_vol[:, :, z]

            # Energy-matched amplitude constraint
            E_rms      = torch.sqrt((torch.abs(E_z) ** 2).mean() + 1e-30)
            tgt_rms    = torch.sqrt((target_A ** 2).mean() + 1e-30)
            scale      = E_rms / (tgt_rms + 1e-30)
            E_out_c    = (target_A * scale) * torch.exp(1j * torch.angle(E_z))

            # Adjoint back-projection for this plane
            E_slm_z = _backward_adjoint_single(E_out_c, z, beam_obj, cone_angle,
                                                upsample_factor, H_asm, roi_size)
            E_slm_accum += E_slm_z

        # ── SLM amplitude constraint (normalize accumulation by Nz) ──────────
        # Dividing by Nz keeps the SLM-plane energy scale consistent regardless
        # of how many z-planes are used.
        E_slm_accum /= Nz
        phase = torch.angle(amp_prof * torch.exp(1j * torch.angle(E_slm_accum)))

        # ── Verbose: evaluate per-slice normalized MSE ───────────────────────
        if verbose and (i + 1) % 5 == 0:
            E_check = _forward_complex_volume(phase, beam_obj, cone_angle,
                                               upsample_factor, H_asm, roi_size)
            mse_per_z = []
            for z in range(Nz):
                I_z    = torch.abs(E_check[:, :, z]) ** 2
                I_norm = I_z / (I_z.max() + 1e-8)
                mse_per_z.append(
                    torch.nn.functional.mse_loss(I_norm, target_I).item())
            print(f"  GS3D [{i+1:3d}/{num_iters}] | "
                  f"per-slice norm-MSE  mean={np.mean(mse_per_z):.5f}  "
                  f"max={np.max(mse_per_z):.5f}  "
                  f"min={np.min(mse_per_z):.5f}")

    return phase.detach()


# ─────────────────────────────────────────────────────────────────────────────
# 3D Adam refinement loss
# ─────────────────────────────────────────────────────────────────────────────

def _loss_3d(recon_vol, target_I, mode='per_slice_norm'):
    """
    Compute 3D loss from volume intensity [roi, roi, Nz] and 2D target [roi, roi].

    mode options
    ------------
    'per_slice_norm'  (default, recommended)
        Normalize each z-slice to [0,1] before MSE, then average over Nz.
        → Each z-plane contributes equally regardless of absolute intensity.
        → Prevents high-intensity planes from dominating the gradient.

    'raw_mean'
        Mean of raw MSE across slices (same as old optimize_slm_phase_3d).
        Simple but gradients are dominated by high-intensity z-planes.

    'worst_slice'
        Loss = max-over-z of per-slice normalized MSE.
        Optimizes the worst plane first; useful if some z-planes are
        persistently bad. Slower convergence on average but better uniformity.
    """
    Nz = recon_vol.shape[2]
    target_3d = target_I.unsqueeze(2)   # [roi, roi, 1]  broadcast-ready

    if mode == 'raw_mean':
        return torch.nn.functional.mse_loss(recon_vol, target_3d.expand_as(recon_vol))

    # Per-slice normalization
    # recon_vol: [roi, roi, Nz]  →  max over spatial dims for each z
    slice_max = recon_vol.amax(dim=(0, 1), keepdim=True) + 1e-8   # [1, 1, Nz]
    recon_norm = recon_vol / slice_max                              # [roi, roi, Nz]

    if mode == 'per_slice_norm':
        return torch.nn.functional.mse_loss(
            recon_norm, target_3d.expand_as(recon_norm))

    if mode == 'worst_slice':
        slice_losses = torch.stack([
            torch.nn.functional.mse_loss(recon_norm[:, :, z], target_I)
            for z in range(Nz)
        ])
        return slice_losses.max()

    raise ValueError(f"Unknown loss mode: {mode}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point: 3D GS-Adam hybrid
# ─────────────────────────────────────────────────────────────────────────────

def optimize_hybrid_gs_adam_3d(beam_obj, target_image_np, cone_angle, H_asm,
                                gs_iters=15, adam_iters=300, lr=0.05,
                                upsample_factor=6, roi_size=1600,
                                loss_mode='per_slice_norm'):
    """
    Two-stage 3D phase optimizer for SLM + Axicon system.

    Stage 1 — GS adjoint warm-up (3D, no autograd)
      Accumulates adjoint back-projections from ALL z-planes each iteration.

    Stage 2 — Adam refinement (3D, autograd)
      Loss: per-slice normalized MSE averaged over Nz  (or 'raw_mean' / 'worst_slice')

    Parameters
    ----------
    gs_iters    : GS warm-up iterations. 10–20 is usually enough.
    adam_iters  : Adam refinement iterations.
    lr          : Adam learning rate. Start with 0.02–0.05 for 3D
                  (lower than 2D because Nz planes × gradient can be noisy).
    loss_mode   : '3d loss mode. See _loss_3d docstring.

    Returns
    -------
    final_phase     : ndarray [Nx, Ny], float, in [0, 2π]
    recon_vol_np    : ndarray [roi, roi, Nz], raw intensity
    loss_adam       : list[float]
    phase_slm       : ndarray [Ny, Nx], int16, ready for SLM (0–1023)
    """
    device = beam_obj.beam_config.device
    fdtype = beam_obj.beam_config.fdtype

    is_sparse = isinstance(H_asm, dict) and H_asm.get('type') == 'sparse'
    Nz = H_asm['H_asm_sparse'].shape[1] if is_sparse else H_asm.shape[2]

    target_I = _prepare_target(target_image_np, roi_size, device, fdtype)

    # ══ Stage 1: GS adjoint warm-up (3D) ════════════════════════════════════
    print("=" * 60)
    print(f"Stage 1 — 3D GS Adjoint Warm-up  ({gs_iters} iters, Nz={Nz})")
    print("=" * 60)
    init_phase = gs_axicon_init_3d(
        beam_obj, target_image_np, cone_angle, H_asm,
        num_iters=gs_iters, upsample_factor=upsample_factor,
        roi_size=roi_size, verbose=True,
    )

    # Snapshot after GS for visualization
    with torch.no_grad():
        E_gs_vol = _forward_complex_volume(init_phase, beam_obj, cone_angle,
                                            upsample_factor, H_asm, roi_size)
        I_gs_vol = torch.abs(E_gs_vol) ** 2   # [roi, roi, Nz]

    # ══ Stage 2: Adam refinement (3D) ════════════════════════════════════════
    print()
    print("=" * 60)
    print(f"Stage 2 — 3D Adam Refinement  ({adam_iters} iters, loss='{loss_mode}')")
    print("=" * 60)

    slm_phase_var = nn.Parameter(init_phase.clone())
    optimizer     = optim.Adam([slm_phase_var], lr=lr)
    loss_adam     = []

    for i in range(adam_iters):
        optimizer.zero_grad()

        recon_vol = beam_obj.propagateToVolume_Axicon2(
            axicon_angle=cone_angle,
            upsample_factor=upsample_factor,
            phase_mask=slm_phase_var,
            roi_size=roi_size,
            H_asm=H_asm,
            convert_to_intensity=False,
        )
        recon_I_vol = torch.abs(recon_vol) ** 2   # [roi, roi, Nz]

        loss = _loss_3d(recon_I_vol, target_I, mode=loss_mode)
        loss.backward()
        optimizer.step()

        loss_adam.append(loss.item())

        if (i + 1) % 10 == 0:
            # Per-slice min/max for monitoring uniformity
            with torch.no_grad():
                slice_max_vals = recon_I_vol.amax(dim=(0, 1))
            print(f"  Adam [{i+1:4d}/{adam_iters}] | Loss: {loss.item():.6f} | "
                  f"slice peak  min={slice_max_vals.min().item():.3e}  "
                  f"max={slice_max_vals.max().item():.3e}")

    print("3D Optimization complete!")

    # ── Post-processing ───────────────────────────────────────────────────────
    final_phase  = (slm_phase_var.detach() % (2 * torch.pi)).cpu().numpy()
    recon_vol_np = recon_I_vol.detach().cpu().numpy()
    I_gs_np      = I_gs_vol.cpu().numpy()
    target_np    = target_I.cpu().numpy()

    # ── Visualization ─────────────────────────────────────────────────────────
    # Row 0: loss curve / target / final phase
    # Row 1: GS result at z[0], z[Nz//2], z[-1]
    # Row 2: Adam result at z[0], z[Nz//2], z[-1]
    z_show = [0, Nz // 2, Nz - 1]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    axes[0, 0].plot(loss_adam, color='steelblue')
    axes[0, 0].set_title(f'Adam Loss ({loss_mode})')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_yscale('log')

    axes[0, 1].imshow(target_np, cmap='hot')
    axes[0, 1].set_title('Target')

    im_ph = axes[0, 2].imshow(final_phase, cmap='hsv', vmin=0, vmax=2*np.pi)
    axes[0, 2].set_title('Final SLM Phase')
    fig.colorbar(im_ph, ax=axes[0, 2])

    for col, z in enumerate(z_show):
        # GS snapshot (normalized for display)
        I_gs_z = I_gs_np[:, :, z]
        axes[1, col].imshow(I_gs_z / (I_gs_z.max() + 1e-8), cmap='hot')
        axes[1, col].set_title(f'GS warm-up | z idx {z} (norm)')

        # Adam final (normalized for display)
        I_ad_z = recon_vol_np[:, :, z]
        axes[2, col].imshow(I_ad_z / (I_ad_z.max() + 1e-8), cmap='hot')
        axes[2, col].set_title(f'Adam final | z idx {z} (norm)')

    plt.tight_layout()
    plt.show()

    # ── SLM integer format ────────────────────────────────────────────────────
    phase_slm = final_phase * (1023 / (2 * np.pi))
    phase_slm = np.round(phase_slm).astype(np.int16).T

    return final_phase, recon_vol_np, loss_adam, phase_slm


# ─────────────────────────────────────────────────────────────────────────────
# Keep original 2D functions intact
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_target(target_np, roi_size, device, fdtype):
    t = torch.tensor(target_np, dtype=fdtype, device=device)
    if t.shape != (roi_size, roi_size):
        t = TF.resize(t.unsqueeze(0), [roi_size, roi_size]).squeeze()
    return t / (t.max() + 1e-8)


def _forward_complex(phase, beam_obj, cone_angle, upsample_factor, H_asm, roi_size, z_idx):
    vol = beam_obj.propagateToVolume_Axicon2(
        axicon_angle=cone_angle, upsample_factor=upsample_factor,
        phase_mask=phase, roi_size=roi_size, H_asm=H_asm, convert_to_intensity=False,
    )
    return vol[:, :, z_idx]


def _backward_adjoint(E_out, beam_obj, cone_angle, upsample_factor,
                       H_asm, roi_size, z_idx):
    return _backward_adjoint_single(E_out, z_idx, beam_obj, cone_angle,
                                    upsample_factor, H_asm, roi_size)


def gs_axicon_init(beam_obj, target_image_np, cone_angle, H_asm,
                   num_iters=10, upsample_factor=6, roi_size=1600,
                   z_target_idx=0, verbose=True):
    cfg = beam_obj.beam_config
    device, fdtype, cdtype = cfg.device, cfg.fdtype, cfg.cdtype
    target_I = _prepare_target(target_image_np, roi_size, device, fdtype)
    target_A = torch.sqrt(target_I)
    amp_prof = beam_obj.slm_amplitude_profile.to(cdtype)
    phase = torch.rand((cfg.Nx, cfg.Ny), device=device, dtype=fdtype) * 2 * torch.pi

    for i in range(num_iters):
        E_out = _forward_complex(phase, beam_obj, cone_angle,
                                  upsample_factor, H_asm, roi_size, z_target_idx)
        E_rms   = torch.sqrt((torch.abs(E_out)**2).mean() + 1e-30)
        tgt_rms = torch.sqrt((target_A**2).mean() + 1e-30)
        scale   = E_rms / (tgt_rms + 1e-30)
        E_out_c = target_A * scale * torch.exp(1j * torch.angle(E_out))
        E_slm   = _backward_adjoint(E_out_c, beam_obj, cone_angle,
                                     upsample_factor, H_asm, roi_size, z_target_idx)
        phase   = torch.angle(amp_prof * torch.exp(1j * torch.angle(E_slm)))

        if verbose and (i + 1) % 5 == 0:
            E_c  = _forward_complex(phase, beam_obj, cone_angle,
                                     upsample_factor, H_asm, roi_size, z_target_idx)
            I_n  = torch.abs(E_c)**2
            I_n  = I_n / (I_n.max() + 1e-8)
            mse  = torch.nn.functional.mse_loss(I_n, target_I).item()
            print(f"  GS [{i+1:3d}/{num_iters}] | norm-MSE: {mse:.6f}")
    return phase.detach()


def optimize_hybrid_gs_adam(beam_obj, target_image_np, cone_angle, H_asm,
                             gs_iters=10, adam_iters=100, lr=0.02,
                             upsample_factor=6, roi_size=1600, z_target_idx=0):
    device = beam_obj.beam_config.device
    fdtype = beam_obj.beam_config.fdtype
    target_I = _prepare_target(target_image_np, roi_size, device, fdtype)

    print("=" * 55)
    print(f"Stage 1 — GS Adjoint Warm-up ({gs_iters} iters)")
    print("=" * 55)
    init_phase = gs_axicon_init(beam_obj, target_image_np, cone_angle, H_asm,
                                 num_iters=gs_iters, upsample_factor=upsample_factor,
                                 roi_size=roi_size, z_target_idx=z_target_idx)
    with torch.no_grad():
        E_gs = _forward_complex(init_phase, beam_obj, cone_angle,
                                  upsample_factor, H_asm, roi_size, z_target_idx)
        I_gs = torch.abs(E_gs)**2

    print()
    print("=" * 55)
    print(f"Stage 2 — Adam Refinement ({adam_iters} iters)")
    print("=" * 55)
    slm_phase_var = nn.Parameter(init_phase.clone())
    optimizer = optim.Adam([slm_phase_var], lr=lr)
    loss_adam = []

    for i in range(adam_iters):
        optimizer.zero_grad()
        recon_field = beam_obj.propagateToVolume_Axicon2(
            axicon_angle=cone_angle, upsample_factor=upsample_factor,
            phase_mask=slm_phase_var, roi_size=roi_size, H_asm=H_asm)
        recon_I      = torch.abs(recon_field[:, :, z_target_idx])**2
        recon_I_norm = recon_I / (recon_I.max() + 1e-8)
        loss = torch.nn.functional.mse_loss(recon_I_norm, target_I)
        loss.backward()
        optimizer.step()
        loss_adam.append(loss.item())
        if (i + 1) % 5 == 0:
            print(f"  Adam [{i+1:4d}/{adam_iters}] | Loss: {loss.item():.6f}")

    print("Optimization complete!")
    final_phase = (slm_phase_var.detach() % (2 * torch.pi)).cpu().numpy()
    final_I     = recon_I.detach().cpu().numpy()
    I_gs_np     = I_gs.cpu().numpy()
    target_np_  = target_I.cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0,0].plot(loss_adam, color='steelblue'); axes[0,0].set_yscale('log')
    axes[0,0].set_title('Adam Loss')
    axes[0,1].imshow(target_np_, cmap='hot'); axes[0,1].set_title('Target')
    im_ph = axes[0,2].imshow(final_phase, cmap='hsv', vmin=0, vmax=2*np.pi)
    axes[0,2].set_title('Final Phase'); fig.colorbar(im_ph, ax=axes[0,2])
    axes[1,0].imshow(I_gs_np/(I_gs_np.max()+1e-8), cmap='hot')
    axes[1,0].set_title(f'After GS ({gs_iters} iters)')
    axes[1,1].imshow(final_I/(final_I.max()+1e-8), cmap='hot')
    axes[1,1].set_title(f'After Adam ({adam_iters} iters, norm)')
    axes[1,2].imshow(final_I, cmap='hot'); axes[1,2].set_title('Final raw')
    plt.tight_layout(); plt.show()

    phase_slm = np.round(final_phase*(1023/(2*np.pi))).astype(np.int16).T
    return final_phase, final_I, loss_adam, phase_slm


################
##### MAIN #####
################
if __name__ == '__main__':
    save_directory = r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\03_24_2026_Axicon_phase_opt\HollowRect100um"

    beam_config = HoloBeamConfig()
    beam_config.psSLM_physical = 8e-6 * 0.8 
    beam_config.lambda_ = 0.473e-6
    assert beam_config.focal_SLM is not False, 'Effective Focal length needs to be set.'
    beam_config.binningFactor = 1
    beam_config.Nx_physical = 1600
    beam_config.Ny_physical = 1200
    beam_config.axis_angle = [1, 0, 0]
    beam_config.z_plane_sampling_rate = 0.5
    beam_config.amplitude_profile_type = 'gaussian'
    beam_config.gaussian_beam_waist = 0.00638708

    beam = HoloBeam(beam_config)
    beam.phase_mask_iter = torch.zeros((beam_config.Nx, beam_config.Ny),
                                        device=beam_config.device)
    beam.slm_amplitude_profile = beam.buildSLMAmplitudeProfile()
    beam.beam_mean_amplitude_iter = torch.tensor(
        1.0, device=beam_config.device, dtype=beam_config.fdtype)

    # ── Target image ──────────────────────────────────────────────────────────
    image = Image.open(
        r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment"
        r"\Proxy_calibration_AltBeam_1image\Proxy_train_pool\HollowRectangle\Raw_target.png"
    ).convert("L")

    original_pixel_size = 2e-6
    desired_pixel_size  = (beam_config.abbe_res_x, beam_config.abbe_res_y)
    desired_resolution  = (beam.beam_config.Nx, beam.beam_config.Ny)

    image_array  = np.array(image)
    rescaled_img = zoom(image_array,
                        (original_pixel_size / desired_pixel_size[1],
                         original_pixel_size / desired_pixel_size[0]), order=1)

    target_w, target_h = desired_resolution
    cur_h, cur_w = rescaled_img.shape
    final_image  = np.zeros((target_h, target_w), dtype=rescaled_img.dtype)

    off_y = (target_h - cur_h) // 2
    off_x = (target_w - cur_w) // 2
    dy0, dx0 = max(0, off_y),  max(0, off_x)
    sy0, sx0 = max(0, -off_y), max(0, -off_x)
    ch = min(cur_h - sy0, target_h - dy0)
    cw = min(cur_w - sx0, target_w - dx0)
    if ch > 0 and cw > 0:
        final_image[dy0:dy0+ch, dx0:dx0+cw] = rescaled_img[sy0:sy0+ch, sx0:sx0+cw]

    rescaled_image = (final_image.T / 255.0).astype(np.float32)

    # ── Axicon / z config ─────────────────────────────────────────────────────
    Axicon_grating_pitch = 1.396e-6 # for testing
    upsample_factor = 20
    Axicon_NA       = beam.lambda_ /Axicon_grating_pitch  #0.08
    Cone_angle      = np.arcsin(Axicon_NA)

    z_min   = 0.001
    z_max   = 0.015
    z_steps = 15
    z_eval_planes = torch.linspace(z_min, z_max, steps=z_steps,
                                    device=beam_config.device)

    H_asm = beam.build_axicon_ASM_TF(
        upsample_factor=upsample_factor,
        z_query=z_eval_planes,
        axicon_angle=Cone_angle,
        margin_factor=8000,
    )

    # ── 2D optimization (single z-plane) ─────────────────────────────────────
    # optimized_phase, recon_img, loss, phase_slm = optimize_hybrid_gs_adam(
    #     beam_obj=beam, target_image_np=rescaled_image,
    #     cone_angle=Cone_angle, H_asm=H_asm,
    #     gs_iters=10, adam_iters=100, lr=0.02,
    #     upsample_factor=upsample_factor, roi_size=1600, z_target_idx=0,
    # )

    # ── 3D optimization (all z-planes) ───────────────────────────────────────
    optimized_phase, recon_vol, loss, phase_slm = optimize_hybrid_gs_adam_3d(
        beam_obj=beam, target_image_np=rescaled_image,
        cone_angle=Cone_angle, H_asm=H_asm,
        gs_iters=50,          
        adam_iters=200,       
        lr=0.01,              
        upsample_factor=upsample_factor,
        roi_size=1600,
        loss_mode='per_slice_norm',   # 'per_slice_norm' | 'raw_mean' | 'worst_slice'
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(save_directory, 'phase_mask_3d_gs_adam_moreGS.npy'), phase_slm)
    np.save(os.path.join(save_directory, 'recon_vol_3d.npy'), recon_vol)
    print(f"Saved to {save_directory}")