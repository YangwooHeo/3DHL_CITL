
import numpy as np
import math
np.math = math 
import sys
import os
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from mbvam.Beam.holobeam import HoloBeam
from mbvam.Beam.holobeamconfig import HoloBeamConfig

def grad_hook(name):
    def hook(grad):
        print(f"[GRAD] {name}: mean={grad.mean().item()}, std={grad.std().item()}, max={grad.max().item()}")
    return hook

# =============================================================================
# [2] Data Loader & Preprocessing
# =============================================================================
def load_and_prep_data(phase_path, captured_path, target_path, target_shape, max_phase=1023):
    """
    Load data from .npy and .png files, resize, and normalize.
    
    Args:
        target_shape: (H, W) tuple
    Returns:
        current_phase (np.int16), captured_img (float32), target_img (float32)
    """
    H, W = target_shape
    print(f"[Data] Loading and preprocessing data to shape ({H}, {W})...")

    # 1. Load Phase (.npy)
    if os.path.exists(phase_path):
        phase_data = np.load(phase_path)
        # 02_19_2026 fix to have transposed integer input
        phase_data = phase_data.T
        phase_data = phase_data.astype(np.float32)
        phase_recon = phase_data * (2*np.pi/max_phase)
        current_phase = torch.from_numpy(phase_recon).float()
    else:
        raise FileNotFoundError(f"Phase file not found: {phase_path}")

    # 2. Load Captured Image (.png)
    if os.path.exists(captured_path):
        cap_img = Image.open(captured_path).convert('L') # Grayscale
        cap_img = cap_img.transpose(Image.TRANSPOSE)
        cap_img = cap_img.resize((H, W), resample=Image.BILINEAR)
        captured_img = np.array(cap_img).astype(np.float32) / 255.0 # Normalize 0~1
    else:
        raise FileNotFoundError(f"Captured image not found: {captured_path}")

    # 3. Load Target Image (.png)
    if os.path.exists(target_path):
        tgt_img = Image.open(target_path).convert('L')
        tgt_img = tgt_img.transpose(Image.TRANSPOSE)
        tgt_img = tgt_img.resize((H, W), resample=Image.BILINEAR)
        target_img = np.array(tgt_img).astype(np.float32) / 255.0 # Normalize 0~1
    else:
        raise FileNotFoundError(f"Target image not found: {target_path}")

    print("[Data] Preparation Complete.")
    return current_phase, captured_img, target_img

def quantile_normalize(img_input, q_min=0.0, q_max=0.999):
    """
    Normalization with outlier elimination by quantile normalization
    Args:
        img_tensor: input image tensor
        q_min: lower bound
        q_max: higher bound (0.995 ~ 0.999)
    """
    img = img_input.astype(np.float32)
    
    v_min = np.quantile(img, q_min)
    v_max = np.quantile(img, q_max)
    
    img = np.clip(img, a_min=v_min, a_max=v_max)
    
    img_norm = (img - v_min) / (v_max - v_min + 1e-8)
    
    return img_norm

# =============================================================================
# [3] CITL Solver (HoloBeam Proxy)
# =============================================================================
class CITLSolver:
    def __init__(self, config: HoloBeamConfig, proxy_param_path: str, max_phase_level=1023):
        self.device = config.device
        self.max_phase_level = max_phase_level
        
        print(f"[CITL] Initializing HoloBeam on {self.device}...")
        self.beam = HoloBeam(config)
        self.beam.H = self.beam.buildEffectiveTF() # Kernel init
        
        # Load Proxy Params
        self.load_proxy_params(proxy_param_path)
        
    def load_proxy_params(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Proxy parameter file not found: {path}")
            
        print(f"[CITL] Loading proxy parameters from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        


        # Helper function to load safely
        def safe_load(attr_name, key_name):
            if key_name in checkpoint:
                val = checkpoint[key_name].to(self.device)
                curr_attr = getattr(self.beam, attr_name, None)
                if curr_attr is None:
                    setattr(self.beam, attr_name, val.requires_grad_(False))
                else:
                    curr_attr.data = val
                return True
            return False

        safe_load('zernike_coeffs', 'zernike_coeffs')
        safe_load('source_modulation_map', 'source_modulation_map')
        
        # Camera Scale special handling
        if 'camera_scale_factor' in checkpoint:
            val = checkpoint['camera_scale_factor'].to(self.device)
            if not hasattr(self.beam, 'camera_scale_factor') or self.beam.camera_scale_factor is None:
                self.beam.camera_scale_factor = val.requires_grad_(False)
            else:
                self.beam.camera_scale_factor.data = val
            print(f'camera_scale_factor is {val}')

        print("[CITL] Parameters loaded and frozen.")

    def step(self, current_phase_np, captured_img_np, target_img_np, lr=0.5):
        # 1. Prep
        current_phase_np.requires_grad_(True)
        
        captured_t = torch.tensor(captured_img_np, dtype=torch.float32, device=self.device)
        target_t = torch.tensor(target_img_np, dtype=torch.float32, device=self.device)

        # 2. Forward Proxy
        simulated_intensity = self.beam.forward_proxy_2D(phase_tensor)
        
        ### debugging forward proxy
        print(f"Debug: Sim Shape: {simulated_intensity.shape}")
        print(f"Debug: Phase Requires Grad: {phase_tensor.requires_grad}")
        #sim = simulated_intensity.detach().cpu().numpy().astype(float)
        #plt.imshow(sim, cmap='gray', vmin=0, vmax=1)
        #plt.show()

        # 3. Calc Gradient from Reality
        diff = captured_t - target_t
        loss_val = torch.mean(diff ** 2).item()
        #grad_L_wrt_I = (2.0 / diff.numel()) * diff
        grad_L_wrt_I = diff
        
        sim = grad_L_wrt_I.detach().cpu().numpy().astype(float)
        #plt.imshow(sim, cmap='gray')
        #plt.show()
        
        # 4. Backward Injection
        #if phase_tensor.grad is not None: phase_tensor.grad.zero_()
        simulated_intensity.backward(gradient=grad_L_wrt_I)

        grad_phi = phase_tensor.grad.detach()
        
        # 5. Update
        with torch.no_grad():
            grad_phi = phase_tensor.grad
            print(f'max grad_phi value is {grad_phi.max()}')
            print(f'avg grad_phi value is {grad_phi.mean()}')
            updated_phase_rad = phase_tensor - lr * grad_phi
            print(f'current phase rad has mean of {phase_tensor.mean()} and max of {phase_tensor.max()} and std of {phase_tensor.std()}')
            print(f'updated phase rad has mean of {updated_phase_rad.mean()} and max of {updated_phase_rad.max()} and std of {updated_phase_rad.std()}')
            updated_phase_val = (updated_phase_rad / (2 * np.pi)) * self.max_phase_level
            updated_phase_val = torch.remainder(updated_phase_val, self.max_phase_level + 1)
            updated_phase_np = updated_phase_val.detach().cpu().numpy().astype(np.int16)

        sim_img_np = simulated_intensity.detach().cpu().squeeze().numpy()
        
        return updated_phase_np, loss_val, sim_img_np, grad_phi, updated_phase_rad

# =============================================================================
# [4] Visualization Function (Paper Style)
# =============================================================================
def resize_proxy_output(img, target_size=500, interp=cv2.INTER_CUBIC):
    """
    img: (H, W) or (H, W, C) numpy array
    return: resized image (500, 500, C)
    """
    resized = cv2.resize(img, (target_size, target_size), interpolation=interp)
    return resized

def visualize_results(target, captured, init_phase, new_phase, simulated, updated_sim, grad_map, loss, learning_rate, save_path="citl_result.png"):
    """
    Generates a comprehensive comparison plot.
    """
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x

    target = to_numpy(target)
    captured = to_numpy(captured)
    init_phase = to_numpy(init_phase)
    new_phase = to_numpy(new_phase)
    simulated = to_numpy(simulated)
    updated_sim = to_numpy(updated_sim)

    simulated = quantile_normalize(simulated)
    updated_sim = quantile_normalize(updated_sim)
    target = cv2.resize(target, (500, 500), interpolation=cv2.INTER_LINEAR)
    captured = cv2.resize(captured, (500, 500), interpolation=cv2.INTER_LINEAR)
    simulated = cv2.resize(simulated, (500, 500), interpolation=cv2.INTER_LINEAR)
    updated_sim = cv2.resize(updated_sim, (500, 500), interpolation=cv2.INTER_LINEAR)

    plt.figure(figsize=(18, 10))
    plt.suptitle(f"CITL Optimization Step Result (Loss: {loss:.6f}), (Learning Rate: {learning_rate})", fontsize=20)
    
    # 1. Target Image
    plt.subplot(2, 4, 1)
    plt.title("Ground Truth (Target)", fontsize=14)
    plt.imshow(target, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    # 2. Captured Image (Reality)
    plt.subplot(2, 4, 2)
    plt.title("Captured Image", fontsize=14)
    plt.imshow(captured, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    # 3. Simulated Image (Proxy Prediction)
    plt.subplot(2, 4, 3)
    plt.title("Proxy Simulated (Initial)", fontsize=14)
    plt.imshow(simulated, cmap='gray') # Scale might vary, let it auto-scale or fix if known
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    # 4. Initial Phase
    plt.subplot(2, 4, 7)
    plt.title("Initial Phase Mask", fontsize=14)
    plt.imshow(init_phase, cmap='twilight', vmin=0, vmax=1023)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04, label='SLM Level')

    # 5. Updated Phase
    plt.subplot(2, 4, 8)
    plt.title("Updated Phase Mask (New SLM Input)", fontsize=14)
    plt.imshow(new_phase, cmap='twilight', vmin=0, vmax=1023)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04, label='SLM Level')

    # 6. Gradient
    plt.subplot(2, 4, 5)
    plt.title("Gradient", fontsize=14)
    #diff = np.abs(target - captured)
    #plt.imshow(diff, cmap='inferno', vmin=0, vmax=1)
    plt.imshow(grad_map, cmap='seismic', vmin=-0.02, vmax=0.02)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    # 7. Phase Mask diff
    plt.subplot(2, 4, 6)
    plt.title("Phase Mask Diff", fontsize=14)
    plt.imshow(new_phase-init_phase, cmap='twilight', vmin=-10, vmax=10)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04, label='SLM Level')
    
    # 8. Proxy Prediction for updated phase mask
    plt.subplot(2, 4, 4)
    plt.title("Proxy simulated (updated)", fontsize=14)
    plt.imshow(updated_sim, cmap='gray')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04, label='SLM Level')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    print(f"[Vis] Result saved to {save_path}")
    plt.show()

# =========================================================================
# main
# =========================================================================
if __name__ == "__main__":
    
    ## first update
    HOME_PATH = Path(r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Proxy_calibration_AltBeam_1image\Epoch_500_zer_5order")
    SAMPLE_NAME = Path(r"HollowRectangle")
    PROXY_PATH = HOME_PATH / "proxy_model_params.pt"
    PHASE_PATH = HOME_PATH / "Proxy_train_pool" / SAMPLE_NAME / "slm_phase.npy"
    CAPTURED_PATH = HOME_PATH / "Proxy_train_pool" / SAMPLE_NAME / "Final_Aligned_Camera.png"
    TARGET_PATH = HOME_PATH / "Proxy_train_pool" / SAMPLE_NAME / "Raw_target.png"

    ## Nth update 
    #PROXY_PATH =    r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Proxy_calibration_AltBeam_1image\Epoch_500\proxy_model_params.pt"
    #PHASE_PATH =    r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Experiment_Data\Proxy_Train_pool_AltBeam_02_25_2026\HollowRectangle\slm_phase.npy"
    #PHASE_PATH =    r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Proxy_calibration_AltBeam_1image\Epoch_500\Phase_Update\new_phase_lr100_1.npy"
    #CAPTURED_PATH = r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Experiment_Data\Proxy_Train_pool_AltBeam_02_25_2026\HollowRectangle\Final_Aligned_Camera.png" 
    #CAPTURED_PATH = r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Proxy_calibration_AltBeam_1image\Epoch_500\Phase_Update\Final_Aligned_Camera.png" 
    #TARGET_PATH =   r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Experiment_Data\Proxy_Train_pool_AltBeam_02_25_2026\HollowRectangle\Raw_target.png"
    
    # 1. Beam Config 
    beam_config = HoloBeamConfig()
    #config.lambda_ = 0.365e-6
    #config.focal_SLM = 0.020
    beam_config.lambda_ = 0.473e-6
    beam_config.focal_SLM = 0.12625
    beam_config.binningFactor = 1
    beam_config.Nx_physical = 1600
    beam_config.Ny_physical = 1200
    beam_config.axis_angle = [1,0,00]
    beam_config.z_plane_sampling_rate=0.5
    beam_config.amplitude_profile_type = 'gaussian' #'gaussian' #'flat_top'
    beam_config.gaussian_beam_waist = 0.0038708 #[m], measured beam waist of the Gaussian beam. Blue: 0.0063188. UV: 0.0038708. Ignored for flat top.
    beam_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. Beam initialize
    beam = HoloBeam(beam_config)
    H = beam.buildEffectiveTF()
    beam.H = H
    beam.slm_amplitude_profile = beam.buildSLMAmplitudeProfile()
    beam_mean_amplitude_iter = torch.tensor(1.0, device=beam_config.device, dtype=beam_config.fdtype)

    # 3. Init Solver
    solver = CITLSolver(beam_config, PROXY_PATH)

    # 4. Load Data
    curr_phase, cap_img, tgt_img = load_and_prep_data(
        PHASE_PATH, CAPTURED_PATH, TARGET_PATH, 
        target_shape=(beam_config.Ny_physical, beam_config.Nx_physical)
    )
    
    phase_tensor = torch.tensor(curr_phase, dtype=torch.float32, requires_grad=True)
    phase_tensor.register_hook(grad_hook("phase_tensor"))
    #with torch.no_grad():
    #    baseline_raw = beam.forward_propagate_2D(
    #        phase_tensor, 
    #        defocus=0.0, 
    #        source_profile=beam.slm_amplitude_profile 
    #    )

    #    baseline_raw = baseline_raw / (baseline_raw.max() + 1e-8) # 0~1 Normalize
    #    baseline_snapshot = baseline_raw.detach().cpu().numpy()
    #plt.imshow(baseline_snapshot, cmap='gray', vmin=0, vmax=1)
    #plt.show()

    # 5. Update Step
    print("Running CITL Step...")
    learning_rate = 100
    new_phase, loss, sim_img, grad_map, new_phase_rad = solver.step(curr_phase, cap_img, tgt_img, lr=learning_rate)
    
    # 5.1. Validate if updated phase makes reasonable output
    if not torch.is_tensor(new_phase_rad):
        phase_input = torch.tensor(new_phase_rad, dtype=torch.float32, device=beam_config.device)
    else:
        phase_input = new_phase_rad.to(beam_config.device)
    updated_simulated_intensity = solver.beam.forward_proxy_2D(phase_input)

    curr_phase = (curr_phase / (2 * np.pi)) * 1023

    # 6. Visualize & Save
    visualize_results(tgt_img, cap_img, curr_phase, new_phase, sim_img, updated_simulated_intensity, grad_map, loss, learning_rate)
    # 7. Save New Phase
    print(f'maximum of new phase is {new_phase.max()}')
    new_phase = new_phase.squeeze()
    np.save("new_phase.npy", new_phase.T)
    print("Done.")