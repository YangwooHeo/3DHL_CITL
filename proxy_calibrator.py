
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

# ==============================================================================
# 1. Dataset: Load SLM Phase(.npy) & Camera Image(.png)
# ==============================================================================
class ProxyCalibrationDataset(Dataset):
    def __init__(self, root_dir, phase_filename='slm_phase.npy', camera_filename='Final_Aligned_Camera.png'):
        """
        Args:
            root_dir (str): ex) : './data/calibration_set'
            phase_filename (str): SLM file (.npy)
            camera_filename (str): camera image file (.png)
        """
        self.samples = []
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        subdirs = sorted(subdirs)

        for d in subdirs:
            dir_path = os.path.join(root_dir, d)
            phase_path = os.path.join(dir_path, phase_filename)
            cam_path = os.path.join(dir_path, camera_filename)
            
            if os.path.exists(phase_path) and os.path.exists(cam_path):
                self.samples.append({
                    'phase_path': phase_path,
                    'camera_path': cam_path,
                    'id': d
                })
            else:
                print(f"[Skip] Missing files in {d}")

        print(f">>> Found {len(self.samples)} valid samples in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        phase_data = np.load(sample['phase_path'])
        # hotfix for slm phase dimension matching
        phase_data = phase_data.T
        phase_data = phase_data.astype(np.float32)
        phase_recon = phase_data * (2*np.pi/1023)
        phase_tensor = torch.from_numpy(phase_recon).float()
        
        H, W = phase_tensor.shape[-2], phase_tensor.shape[-1]

        cam_img = Image.open(sample['camera_path']).convert('L')
        cam_img = cam_img.transpose(Image.TRANSPOSE)
        
        if cam_img.size != (W, H):
            cam_img = cam_img.resize((W, H), Image.BILINEAR)
            
        cam_tensor = transforms.ToTensor()(cam_img).squeeze() # [H, W]
        
        sample_name = sample['id']
        return phase_tensor, cam_tensor, sample_name

# ==============================================================================
# 2. Training Function
# ==============================================================================
def train_proxy_model(beam: HoloBeam, dataloader, epochs=150, device='cuda'):
    """
    Proxy Parameter (Aberration, Source, Scale) optimization
    """
    print(f"\n>>> Start Proxy Calibration on {device}...")
    
    beam.init_proxy_params(num_zernike=20, device = device) 
    beam.beam_config.device = device
    
    optimizer = optim.Adam([
        {'params': beam.zernike_coeffs,      'lr': 0.05}, # Aberration (Zernike)
        {'params': beam.source_modulation_map,'lr': 0.2}, # Source distribution (Source Map)
        {'params': beam.camera_scale_factor, 'lr': 0.05}  # Normalization (Scale)
    ])
    
    loss_fn = torch.nn.MSELoss()
    loss_history = []
    
    progress_bar = tqdm(range(epochs), desc="Calibration")

    for epoch in progress_bar:
        epoch_loss = 0.0
        grad_source_mean = 0.0
        grad_zernike_mean = 0.0

        for batch_idx, (phases, targets, _) in enumerate(dataloader):
            phases = phases.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            batch_outputs = []
            
            for i in range(len(phases)):
                single_phase = phases[i] # (H, W)
                
                #output = beam.forward_proxy(single_phase)
                output = beam.forward_propagate_2D(single_phase)
                batch_outputs.append(output)
            
            simulated_batch = torch.stack(batch_outputs) 
            
            # Loss Calculation
            loss = loss_fn(simulated_batch, targets)
            
            loss.backward()
            
            # debugging for grad computation for each hardware defect
            if beam.source_modulation_map.grad is not None:
                grad_source_mean += beam.source_modulation_map.grad.abs().mean().item()
            if beam.zernike_coeffs.grad is not None:
                grad_zernike_mean += beam.zernike_coeffs.grad.abs().mean().item()

            optimizer.step()

            # --- Non-negative Constraints ---
            with torch.no_grad():
                beam.source_modulation_map.clamp_(min=1e-6) 
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        avg_src_grad = grad_source_mean / len(dataloader) 
        avg_zernike_grad = grad_zernike_mean / len(dataloader) 
        loss_history.append(avg_loss)
        
        progress_bar.set_postfix({
            'Loss': f"{avg_loss:.5f}", 
            'Scale': f"{beam.camera_scale_factor.item():.2f}",
            'SrcGrad': f"{avg_src_grad:.2e}", 
            'ZnkGrad': f"{avg_zernike_grad:.2e}" 
        })

    print(">>> Calibration Finished.")

    return loss_history

def calculate_metrics(target, prediction):
    """
    Get MSE and SSIM. (Target, Prediction: Numpy Array, 0~1 Range)
    """
    # 0~1 Clipping 
    pred_clipped = np.clip(prediction, 0, 1)
    targ_clipped = np.clip(target, 0, 1)
    
    mse = np.mean((targ_clipped - pred_clipped) ** 2)
    
    # SSIM computation (win_size need adjustment based on image size)
    ssim_val = ssim_func(targ_clipped, pred_clipped, data_range=1.0, win_size=3)
    
    return mse, ssim_val

def force_resize(tensor_img, target_res=(500, 500)):
    """
    Make tensor to the size of certain target
    """
    # 1. Dimension match: (1, 1, H, W)
    if tensor_img.dim() == 2:   # (H, W)
        img_in = tensor_img.unsqueeze(0).unsqueeze(0)
    elif tensor_img.dim() == 3: # (C, H, W)
        img_in = tensor_img.unsqueeze(0)
    else:                       # (B, C, H, W)
        img_in = tensor_img

    # 2. Interpolation
    resized = F.interpolate(img_in, size=target_res, mode='area') 
    
    return resized.squeeze()

def quantile_normalize(img_input, q_min=0.0, q_max=0.999):
    """
    Normalization with outlier elimination by quantile normalization
    Args:
        img_tensor: input image tensor
        q_min: lower bound
        q_max: higher bound (0.995 ~ 0.999)
    """
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
    
    v_min = torch.quantile(img, q_min)
    v_max = torch.quantile(img, q_max)
    print(f'max of img is {torch.max(img)}, v_max of quantile is {v_max}')
    
    img = torch.clamp(img, min=v_min, max=v_max)
    
    img_norm = (img - v_min) / (v_max - v_min + 1e-8)
    
    return img_norm
# ==============================================================================
# 3. Visualization & Saving
# ==============================================================================
def analyze_zernike_results(beam, save_path='zernike_analysis.png'):
    """
    Analyze zernike coefficient and visualize
    """
    print("Analyzing Zernike Aberrations...")
    
    # 1. Data extraction
    coeffs = beam.zernike_coeffs.detach().cpu().numpy()
    
    # 2. Zernike index mapping
    zernike_names = [
        "0: Piston", 
        "1: Tilt V", "2: Tilt H", 
        "3: Defocus", "4: Astig. V", 
        "5: Astig. H", "6: Coma V", 
        "7: Coma H", "8: Spherical", 
        "9: Trefoil V", "10: Trefoil H",
        "11: Sec. Astig V", "12: Sec. Astig H",
        "13: Quadrafoil V", "14: Quadrafoil H"
    ]
    
    # for the higher order
    if len(coeffs) > len(zernike_names):
        zernike_names += [f"{i}: Order {i}" for i in range(len(zernike_names), len(coeffs))]
    else:
        zernike_names = zernike_names[:len(coeffs)]

    # 3. 2D Wavefront Error Map reconstruction
    device = beam.beam_config.device
    if hasattr(beam, 'zernike_basis') and beam.zernike_basis is not None:
        basis = beam.zernike_basis
    else:
        res_y, res_x = beam.beam_config.Ny_physical, beam.beam_config.Nx_physical
        basis = compute_zernike_basis(len(coeffs), [res_y, res_x], dtype=torch.float32).to(device)
    
    # 4. Combine basis and coefficients
    # combine_zernike_basis returns Complex Tensor so that angle should be extracted from that.
    aberration_complex = combine_zernike_basis(beam.zernike_coeffs, basis)
    aberration_map = aberration_complex.angle().detach().cpu().squeeze().numpy()
    
    fig = plt.figure(figsize=(18, 8))
    
    ax1 = fig.add_subplot(1, 2, 1)
    bars = ax1.barh(zernike_names, coeffs, color='skyblue', edgecolor='navy')
    ax1.set_xlabel("Coefficient Magnitude (Radians approx.)")
    ax1.set_title("Dominant Zernike Modes")
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    max_idx = np.argmax(np.abs(coeffs))
    bars[max_idx].set_color('salmon')
    bars[max_idx].set_edgecolor('red')
    
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(aberration_map, cmap='jet') # jet or twilight_shifted
    ax2.set_title(f"Reconstructed Wavefront Error\n(Dominant: {zernike_names[max_idx]})")
    plt.colorbar(im, ax=ax2, label='Phase Error (Radians)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Zernike analysis saved to {save_path}")

# analyze_zernike_results(beam)

def save_proxy_params(beam, filepath):
    state = {
        'zernike_coeffs': beam.zernike_coeffs.detach().cpu(),
        'source_modulation_map': beam.source_modulation_map.detach().cpu(),
        'camera_scale_factor': beam.camera_scale_factor.detach().cpu()
    }
    torch.save(state, filepath)
    print(f"Proxy parameters saved to {filepath}")

# ==============================================================================
# 4. Main Execution
# ==============================================================================
if __name__ == "__main__":
    # --- Configuration ---
    DATA_ROOT = r'C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Experiment_Data\Proxy_Train_pool_AltBeam_02_25_2026' 
    #DATA_ROOT = r'C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Experiment_Data\Proxy_Train_pool_1' 
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Beam Config 
    beam_config = HoloBeamConfig()

    ### UV
    #beam_config.lambda_ = 0.365e-6
    #beam_config.focal_SLM = 0.020

    ### BLUE
    beam_config.lambda_ = 0.473e-6
    #beam_config.focal_SLM = 0.016
    beam_config.focal_SLM = 0.12625

    beam_config.binningFactor = 1
    beam_config.Nx_physical = 1600
    beam_config.Ny_physical = 1200
    beam_config.axis_angle = [1,0,00]
    beam_config.z_plane_sampling_rate=0.5
    beam_config.amplitude_profile_type = 'gaussian'
    beam_config.gaussian_beam_waist = 0.0038708 #[m], measured beam waist of the Gaussian beam. Blue: 0.0063188. UV: 0.0038708. Ignored for flat top.
    assert beam_config.focal_SLM is not False
    
    # 2. Beam initialize
    print('1. Initializing beam')
    beam = HoloBeam(beam_config)
    H = beam.buildEffectiveTF()
    beam.H = H
    beam.slm_amplitude_profile = beam.buildSLMAmplitudeProfile()
    beam_mean_amplitude_iter = torch.tensor(1.0, device=beam_config.device, dtype=beam_config.fdtype)
    
    # 3. Load training dataset
    print('2. Loading training data set from ' + str(DATA_ROOT))
    dataset = ProxyCalibrationDataset(root_dir=DATA_ROOT, phase_filename='slm_phase.npy', camera_filename='Final_Aligned_Camera.png')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True) # batch size control based on the memory
    
    
    fixed_batch = next(iter(dataloader))
    fixed_phase = fixed_batch[0][0].unsqueeze(0).to(beam.beam_config.device) # (1, H, W)
    fixed_target = fixed_batch[1][0].to(beam.beam_config.device)             # (500, 500)

    print(">>> Snapshotting Baseline for the first batch before braining)...")
    print(fixed_phase[0].shape)
    with torch.no_grad():
        baseline_raw = beam.forward_proxy_2D(
            fixed_phase[0], 
            defocus=0.0, 
            source_profile=beam.slm_amplitude_profile 
        )

        print(f'baseline forward maximum is {baseline_raw.max()}')
        baseline_raw = quantile_normalize(baseline_raw, q_max=0.999) 
        baseline_raw = force_resize(tensor_img=baseline_raw, target_res=(500,500))
        baseline_snapshot = baseline_raw.detach().cpu().numpy()

    # 4. Train proxy model
    print("3. Start Training...")
    loss_history = train_proxy_model(beam, dataloader, epochs=500, device=DEVICE)
    save_proxy_params(beam, 'proxy_model_params.pt')

    # 5. Inferece trained proxy model
    print(">>> Snapshotting Proxy for the first batch after training)...")
    with torch.no_grad():
        #proxy_img = beam.forward_proxy(fixed_phase[0])
        proxy_img = beam.forward_proxy_2D(fixed_phase[0])

        #proxy_img = torch.clamp(proxy_img, 0, 1) 
        #proxy_img = quantile_normalize(proxy_img, q_max=0.999) 

        proxy_img = force_resize(tensor_img=proxy_img,target_res=(500,500))
        fixed_target = force_resize(tensor_img=fixed_target,target_res=(500,500))
        proxy_snapshot = proxy_img.detach().cpu().numpy()
        target_snapshot = fixed_target.detach().cpu().numpy()

    # Visualization
    print("4. Saving Plots...")
    plt.semilogy(loss_history, label='Train Loss', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss History")
    plt.grid(True)
    plt.savefig('Loss_curve.png')

    mse_base = np.mean((target_snapshot - baseline_snapshot)**2)
    mse_proxy = np.mean((target_snapshot - proxy_snapshot)**2)

    fig, ax = plt.subplots(1, 3, figsize=(18, 7))

    ax[0].imshow(target_snapshot, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title("Target (Camera)")

    ax[1].imshow(baseline_snapshot, cmap='gray', vmin=0, vmax=1)
    ax[1].set_title(f"Baseline (Before Train)\nMSE: {mse_base:.4f}")

    ax[2].imshow(proxy_snapshot, cmap='gray', vmin=0, vmax=1)
    ax[2].set_title(f"Proxy (After Train)\nMSE: {mse_proxy:.4f}")

    plt.tight_layout()
    plt.savefig('before_after_comparison.png')
    print("Saved to before_after_comparison.png")

    analyze_zernike_results(beam)
    
    source_map_np = beam.source_modulation_map.detach().cpu().squeeze().numpy()    
    plt.figure(figsize=(6, 6))
    plt.imshow(source_map_np, cmap='inferno', aspect='auto', vmin=0, vmax=3) 
    plt.colorbar(label='Amplitude')
    plt.title("Optimized Source Amplitude Profile (Physical SLM Plane)")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('trained_source_map.png')
    print(" -> Saved 'learned_source_map.png'")

    print("Calculating MSE per sample for final analysis...")
    sample_names = []
    sample_mses = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            phase, real_img, sample_name = dataset[i]
            
            #pred_img = beam.forward_proxy(phase)
            pred_img = beam.forward_proxy_2D(phase)
            #pred_img = quantile_normalize(pred_img,q_max=0.999)
            
            mse_val = F.mse_loss(pred_img, real_img).item()
            
            sample_names.append(sample_name)
            sample_mses.append(mse_val)
            
    plt.figure(figsize=(12, 6))
    
    plt.plot(sample_names, sample_mses, marker='o', linestyle='None', color='blue', markersize=8)
    
    plt.xlabel("Sample Name (Directory)")
    plt.ylabel("MSE Loss")
    plt.title("Final MSE Loss per Sample")
    
    plt.xticks(rotation=45, ha='right') 
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout() 
    plt.savefig('MSE_per_sample.png')
    print("Saved to MSE_per_sample.png")