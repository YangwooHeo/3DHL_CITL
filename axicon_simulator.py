import numpy as np
import math
np.math = math 
import torch
#from torchvision import transforms
import matplotlib.pyplot as plt
#from PIL import Image
#from pathlib import Path
from scipy.fft import fft, ifft, fftshift, fftfreq

from mbvam.Beam.holobeam import HoloBeam
from mbvam.Beam.holobeamconfig import HoloBeamConfig
import mbvam.Geometry.visualize

import torch.nn.functional as F

def visualize_kspace_distortion(beam, phase_tensor, Cone_angle, upsample_factor,
                                n_medium=1.0, axicon_angle_in_medium=False,
                                axicon_transverse_frequency=None):
    print("\n--- [K-space Visualization Started] ---")
    
    # 1. SLM Field
    slm_field = torch.exp(1j * phase_tensor)
    
    # 2. Upsampling
    Nx_orig, Ny_orig = beam.beam_config.Nx, beam.beam_config.Ny
    ps_orig = beam.beam_config.psSLM
    
    slm_field_up = slm_field.repeat_interleave(upsample_factor, dim=0).repeat_interleave(upsample_factor, dim=1)
    Nx_up, Ny_up = slm_field_up.shape
    ps_up = ps_orig / upsample_factor
    
    x = torch.linspace(-(Nx_up-1)/2, (Nx_up-1)/2, Nx_up, device=beam.beam_config.device) * ps_up
    y = torch.linspace(-(Ny_up-1)/2, (Ny_up-1)/2, Ny_up, device=beam.beam_config.device) * ps_up
    Y2 = y.view(1, -1) ** 2
    
    radial_frequency = beam._resolve_axicon_radial_frequency(
        axicon_angle=Cone_angle,
        n_medium=n_medium,
        axicon_angle_in_medium=axicon_angle_in_medium,
        axicon_transverse_frequency=axicon_transverse_frequency,
    )
    radial_frequency_value = float(radial_frequency.detach().cpu().item())
    k_sin_alpha = 2 * torch.pi * radial_frequency
    
    print("  Applying Axicon Phase...")
    chunk_size = 1024
    for i in range(0, Nx_up, chunk_size):
        end_idx = min(i + chunk_size, Nx_up)
        X2_chunk = x[i:end_idx].view(-1, 1) ** 2
        R_chunk = torch.sqrt(X2_chunk + Y2)
        slm_field_up[i:end_idx] *= torch.exp(-1j * k_sin_alpha * R_chunk)
        
    print("  Calculating FFT2 (GPU)...")
    k_space = torch.fft.fftshift(torch.fft.fft2(slm_field_up, norm="ortho"))
    
    print("  Cropping K-space to Ring region...")
    k_ring_radius = radial_frequency_value
    df_x = 1.0 / (Nx_up * ps_up)
    ring_radius_px = int(k_ring_radius / df_x)
    
    w = int(ring_radius_px * 1.2)
    cx, cy = Nx_up // 2, Ny_up // 2
    
    k_space_cropped = k_space[cx-w:cx+w, cy-w:cy+w]
    k_space_mag = torch.log1p(torch.abs(k_space)).cpu().numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(k_space_mag, cmap='viridis')
    plt.title("K-Space: SLM (Hollow Rect) + Axicon", fontsize=16)
    plt.colorbar(label="Log Magnitude")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print("--- [Visualization Complete] ---\n")

def debug_aliasing_1d(slice_2d, upsample_factor, axicon_NA, original_pixel_size=8e-6):
    """
    Aliasing analysis
    """
    center_y = slice_2d.shape[0] // 2
    slice_1d = slice_2d[center_y, :]
    N = len(slice_1d)
    
    current_dx = original_pixel_size / upsample_factor
    x_discrete = np.arange(-N//2, N//2) * current_dx * 1e6 # um 단위
    
    # Sinc Interpolation (Fourier Zero-padding)
    interp_factor = 10
    N_interp = N * interp_factor
    
    # FFT and Zero-padding
    spectrum = fftshift(fft(slice_1d))
    spectrum_padded = np.zeros(N_interp, dtype=complex)
    
    pad_start = (N_interp - N) // 2
    spectrum_padded[pad_start : pad_start + N] = spectrum
    
    slice_1d_interp = np.abs(ifft(fftshift(spectrum_padded))) * interp_factor
    x_interp = np.arange(-N_interp//2, N_interp//2) * (current_dx / interp_factor) * 1e6
    
    # Frequency spectrum analysis
    f_max = 1.0 / (2 * current_dx) 
    f_theory = 2 * axicon_NA / 473e-9
    freqs = fftshift(fftfreq(N, d=current_dx)) / 1000 # 1/mm unit
    spectrum_mag = np.abs(spectrum)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Plot 1: (Spatial Domain) ---
    zoom_range = 25 
    
    ax1.plot(x_interp, slice_1d_interp, 'b-', alpha=0.8, label='Sinc Interpolated (Continuous)')
    ax1.plot(x_discrete, slice_1d, 'ro', markersize=4, label='Simulated Discrete Points')
    ax1.set_xlim([-zoom_range, zoom_range])
    ax1.set_title(f"Spatial Domain (Upsample: {upsample_factor}x)\nPixel size: {current_dx*1e6:.2f} um")
    ax1.set_xlabel("Position (um)")
    ax1.set_ylabel("Intensity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: (Frequency Domain) ---
    ax2.plot(freqs, spectrum_mag, 'k-', linewidth=1.5)
    ax2.fill_between(freqs, spectrum_mag, color='gray', alpha=0.3)
    
    # Nyquist Limit
    ax2.axvline(-f_max/1000, color='r', linestyle='--', label='Nyquist Limit')
    ax2.axvline(f_max/1000, color='r', linestyle='--')
    ax2.axvline(-f_theory/1000, color='g', linestyle='--', label='NA/lambda')
    ax2.axvline(f_theory/1000, color='g', linestyle='--')
    
    ax2.set_title("Frequency Spectrum (Aliasing Check)")
    ax2.set_xlabel("Spatial Frequency (1/mm)")
    ax2.set_ylabel("Magnitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    #PHASE_MASK = r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Proxy_calibration_AltBeam_1image\Epoch_500\Proxy_train_pool\HollowRectangle\slm_phase.npy"
    #PHASE_MASK = r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\03_24_2026_Axicon_phase_opt\HollowRect100um\phase_mask_3d_gs_adam_moreGS.npy"
    #PHASE_MASK = r"G:\공유 드라이브\taylorlab\3DHL\Phase masks\optimized_with_mbvam\2026_04_09_AxiconNA0.34_demag_spatial_1to15mm\phase_mask_blue.npy"
    PHASE_MASK = r"G:\공유 드라이브\taylorlab\3DHL\CITL\Proxy_Train_Pool_2\checker_20\slm_phase.npy"
    phase_data = np.load(PHASE_MASK)
    phase_data = phase_data.T
    phase_data = phase_data[::-1, :]
    plt.imshow(phase_data);plt.show()
    #plt.hist(phase_data.flatten(), bins=1024);plt.show()
    phase_data = phase_data.astype(np.float32)
    phase_recon = phase_data * (2*np.pi/1023)
    phase_tensor = torch.from_numpy(phase_recon).float()
    
    ## Plane wave debug: Activate the line below.
    phase_tensor = torch.ones_like(phase_tensor) * 1023
    
    beam_config = HoloBeamConfig()
    beam_config.psSLM_physical = 8e-6 * 0.8 ### Current 4f system has 0.8 M (f1 = 250, f2 = 200)
    ### UV
    #beam_config.lambda_ = 0.365e-6
    #beam_config.focal_SLM = 0.020

    ### BLUE
    beam_config.lambda_ = 0.473e-6
    #beam_config.focal_SLM = 0.12625
    assert beam_config.focal_SLM is not False

    beam_config.binningFactor = 1
    beam_config.Nx_physical = 1600
    beam_config.Ny_physical = 1200
    beam_config.axis_angle = [1,0,00]
    beam_config.z_plane_sampling_rate=0.5
    beam_config.amplitude_profile_type = 'gaussian'
    beam_config.gaussian_beam_waist = 0.00638708 * 0.8 #[m], measured beam waist of the Gaussian beam. Blue: 0.0063188. UV: 0.0038708. Ignored for flat top.
    phase_tensor = phase_tensor.to(device=beam_config.device, dtype=beam_config.fdtype)
    
    ### Axicon config
    Axicon_grating_pitch = 1.396e-6 # for testing
    upsample_factor = 10 # axicon spatial frequency is 0.339um (lambda/4NA)
    propagation_medium_index = 1.48  # resin. Use 1.0 for free-space propagation.
    axicon_angle_in_medium = False
    Axicon_transverse_frequency = 1.0 / Axicon_grating_pitch
    Axicon_NA_air_equiv = beam_config.lambda_ * Axicon_transverse_frequency
    if Axicon_NA_air_equiv >= 1.0:
        raise ValueError("Axicon grating pitch gives an invalid free-space NA >= 1.")
    Cone_angle = float(np.arcsin(Axicon_NA_air_equiv))
    Cone_angle_in_medium = float(np.arcsin(Axicon_NA_air_equiv / propagation_medium_index))
    print(f"Axicon air-equivalent NA={Axicon_NA_air_equiv:.4f}, "
          f"theta_air={Cone_angle:.4f} rad, "
          f"theta_medium={Cone_angle_in_medium:.4f} rad at n={propagation_medium_index:.3f}")

    # 2. Beam initialize
    print('1. Initializing beam')
    beam = HoloBeam(beam_config)
    #z_eval_planes = beam.local_coord_vec[2] 
    z_min = 0.004
    z_max = 0.006
    z_steps = 3
    # Physical propagation distances in the selected medium [m].
    # These are not optical path lengths and are not free-space-equivalent z.
    z_eval_planes = torch.linspace(z_min, z_max, steps=z_steps,
                                    device=beam_config.device,
                                    dtype=beam_config.fdtype)
    #H_asm = beam.build_true_ASM_TF(upsample_factor=upsample_factor, z_query=z_eval_planes)
    H_asm = beam.build_axicon_ASM_TF(
        upsample_factor=upsample_factor,
        z_query=z_eval_planes,
        n_medium=propagation_medium_index,
        axicon_angle=Cone_angle,
        axicon_angle_in_medium=axicon_angle_in_medium,
        axicon_transverse_frequency=Axicon_transverse_frequency,
        margin_factor=5000,
    )
    print('2. Transfer function computation has been completed')

    beam.slm_amplitude_profile = beam.buildSLMAmplitudeProfile()
    beam.beam_mean_amplitude_iter = torch.tensor(1.0, device=beam_config.device, dtype=beam_config.fdtype)
    #visualize_kspace_distortion(beam, phase_tensor, Cone_angle, upsample_factor,
    #                            n_medium=propagation_medium_index,
    #                            axicon_angle_in_medium=axicon_angle_in_medium,
    #                            axicon_transverse_frequency=Axicon_transverse_frequency)
    recon = beam.propagateToVolume_Axicon2(
        axicon_angle=Cone_angle,
        upsample_factor=upsample_factor,
        phase_mask=phase_tensor,
        H_asm=H_asm,
        roi_size=600,
        apply_spatial_filter=False,
        n_medium=propagation_medium_index,
        axicon_angle_in_medium=axicon_angle_in_medium,
        axicon_transverse_frequency=Axicon_transverse_frequency,
    )
    print('3. Axicon propagation is successfully computed')
    recon_np = recon.to('cpu').numpy()
    recon_intensity_np = np.abs(recon_np)**2
    recon_intensity_np = np.transpose(recon_intensity_np, (1, 0, 2))

    z_index = recon_intensity_np.shape[2] // 2

    for i in range(0, z_steps, 1):
        slice_2d = recon_intensity_np[:, :, i]
        #np.save(r'C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Axicon_upsample_test\Accuracy_Comparison\WO_Spatial_Filter_Demagnified.npy', slice_2d)
        plt.figure(figsize=(10, 10))
        centX=1600*upsample_factor/2
        centY=1200*upsample_factor/2
        #slice_2d = slice_2d[int(centY-400): int(centY+400), int(centX-400): int(centX+400)]
        plt.imshow(slice_2d, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Intensity')
        plt.title(f"Bessel Beam Intensity - Slice at z = {z_eval_planes[i].item()*1000:.1f}mm")

        plt.show()
        #plt.savefig(r'C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Axicon_upsample_test\Accuracy_Comparison\New_MF100_HollowRect_NA'+str(round(Axicon_NA_air_equiv,3))+r'_Z'+str(i)+r'_PS'+str(round(8/upsample_factor, 3))+r'um.png', dpi=150)
    #debug_aliasing_1d(slice_2d=slice_2d, upsample_factor=upsample_factor, axicon_NA = Axicon_NA_air_equiv, original_pixel_size=8e-6)
    mbvam.Geometry.visualize.openCVSliceViewer(recon_intensity_np) #Press q to quit the app.
