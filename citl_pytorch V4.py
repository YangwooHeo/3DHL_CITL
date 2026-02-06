"""
CITL (Camera-in-the-Loop) Phase Mask Optimization using PyTorch

Based on: "Neural Holography with Camera-in-the-loop Training" (SIGGRAPH Asia 2020)

The CITL update rule:
    φ(k) = φ(k-1) - η * (∂L/∂I_captured) * (∂f/∂φ)

Where:
    - φ(k-1) : Current phase mask (input)
    - ∂L/∂I_captured : Gradient of loss w.r.t. captured intensity (from camera)
    - ∂f/∂φ : Gradient of proxy model (ASM) w.r.t. phase (computed via autodiff)
    - η : Learning rate
    - φ(k) : New phase mask (output)

Usage:
    from citl_pytorch import CITLOptimizer
    
    optimizer = CITLOptimizer(
        wavelength=520e-9,
        slm_pixel_pitch=8e-6,
        propagation_distance=0.2
    )
    
    # Load phase mask and captured image
    phi_k = optimizer.citl_update(
        phase_mask_path="phase_mask_uv.npy",
        captured_image_path="captured.png",
        ground_truth_path="ground_truth.png",
        learning_rate=0.1
    )
    
    # Save new phase mask
    np.save("phase_mask_new.npy", phi_k)
"""

import torch
import torch.nn as nn
import torch.fft
import numpy as np
from PIL import Image
import os
from scipy import ndimage
from scipy.ndimage import gaussian_filter, shift as nd_shift
from scipy.signal import correlate2d
import matplotlib.pyplot as plt


# ============================================================================
# IMAGE PREPROCESSING (from your working v4 code - NumPy based)
# ============================================================================

def load_image(filepath):
    """Load an image and convert to grayscale numpy array."""
    if filepath.endswith('.npy'):
        return np.load(filepath).astype(np.float64)
    img = Image.open(filepath)
    if img.mode != 'L':
        img = img.convert('L')
    return np.array(img, dtype=np.float64)


def find_bright_circle(image, sigma=5):
    """Find the bright circular region in the captured hologram."""
    smoothed = gaussian_filter(image, sigma=sigma)
    threshold = smoothed.mean() + 0.5 * smoothed.std()
    binary = smoothed > threshold
    
    labeled, num_features = ndimage.label(binary)
    
    if num_features == 0:
        h, w = image.shape
        return w // 2, h // 2, min(w, h) // 4
    
    component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    largest_component = np.argmax(component_sizes) + 1
    
    component_mask = labeled == largest_component
    y_indices, x_indices = np.where(component_mask)
    
    center_x = int(np.mean(x_indices))
    center_y = int(np.mean(y_indices))
    
    radius_x = (x_indices.max() - x_indices.min()) // 2
    radius_y = (y_indices.max() - y_indices.min()) // 2
    radius = max(radius_x, radius_y)
    radius = int(radius * 1.2)
    
    return center_x, center_y, radius


def find_dc_term(image, center_x, center_y, search_radius=50):
    """Find the DC term (brightest spot) near the center."""
    h, w = image.shape
    
    y_start = max(0, center_y - search_radius)
    y_end = min(h, center_y + search_radius)
    x_start = max(0, center_x - search_radius)
    x_end = min(w, center_x + search_radius)
    
    search_region = image[y_start:y_end, x_start:x_end]
    local_max_idx = np.unravel_index(np.argmax(search_region), search_region.shape)
    dc_y = y_start + local_max_idx[0]
    dc_x = x_start + local_max_idx[1]
    
    dc_intensity = image[dc_y, dc_x]
    threshold = dc_intensity * 0.5
    
    dc_radius = 5
    for r in range(5, 50):
        angles = np.linspace(0, 2*np.pi, 16)
        ring_intensities = []
        for angle in angles:
            check_x = int(dc_x + r * np.cos(angle))
            check_y = int(dc_y + r * np.sin(angle))
            if 0 <= check_x < w and 0 <= check_y < h:
                ring_intensities.append(image[check_y, check_x])
        if ring_intensities and np.mean(ring_intensities) < threshold:
            dc_radius = r
            break
    
    return dc_x, dc_y, dc_radius


def remove_dc_term(image, dc_x, dc_y, dc_radius, method='inpaint', aggressive=True):
    """Remove the DC term from the image."""
    result = image.copy()
    h, w = image.shape
    
    if aggressive:
        remove_radius = int(dc_radius * 3)
    else:
        remove_radius = int(dc_radius * 1.5)
    
    y, x = np.ogrid[:h, :w]
    dc_mask = ((x - dc_x)**2 + (y - dc_y)**2) <= remove_radius**2
    
    if method == 'inpaint':
        boundary_radius = remove_radius + 2
        angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
        boundary_values = []
        
        for angle in angles:
            bx = int(dc_x + boundary_radius * np.cos(angle))
            by = int(dc_y + boundary_radius * np.sin(angle))
            if 0 <= bx < w and 0 <= by < h:
                boundary_values.append(image[by, bx])
        
        boundary_mean = np.mean(boundary_values) if boundary_values else np.mean(image)
        
        for yi in range(max(0, dc_y - remove_radius), min(h, dc_y + remove_radius + 1)):
            for xi in range(max(0, dc_x - remove_radius), min(w, dc_x + remove_radius + 1)):
                dist = np.sqrt((xi - dc_x)**2 + (yi - dc_y)**2)
                if dist <= remove_radius:
                    angle = np.arctan2(yi - dc_y, xi - dc_x)
                    bx = int(dc_x + boundary_radius * np.cos(angle))
                    by = int(dc_y + boundary_radius * np.sin(angle))
                    
                    if 0 <= bx < w and 0 <= by < h:
                        edge_val = image[by, bx]
                    else:
                        edge_val = boundary_mean
                    result[yi, xi] = edge_val
        
        smoothed = gaussian_filter(result, sigma=2)
        blend_radius = remove_radius + 5
        
        for yi in range(max(0, dc_y - blend_radius), min(h, dc_y + blend_radius + 1)):
            for xi in range(max(0, dc_x - blend_radius), min(w, dc_x + blend_radius + 1)):
                dist = np.sqrt((xi - dc_x)**2 + (yi - dc_y)**2)
                if remove_radius - 3 < dist < remove_radius + 3:
                    alpha = (dist - (remove_radius - 3)) / 6
                    result[yi, xi] = alpha * result[yi, xi] + (1 - alpha) * smoothed[yi, xi]
    
    elif method == 'gaussian':
        outer_radius = remove_radius * 2
        annulus_mask = (((x - dc_x)**2 + (y - dc_y)**2) > remove_radius**2) & \
                       (((x - dc_x)**2 + (y - dc_y)**2) <= outer_radius**2)
        
        if np.sum(annulus_mask) > 0:
            local_mean = np.mean(image[annulus_mask])
            local_std = np.std(image[annulus_mask])
        else:
            local_mean = np.mean(image)
            local_std = np.std(image)
        
        noise = np.random.normal(0, local_std * 0.1, size=np.sum(dc_mask))
        result[dc_mask] = local_mean + noise
    
    return result


def crop_circular_region(image, center_x, center_y, radius, target_fill_ratio=0.5):
    """Crop region with correct fill ratio."""
    h, w = image.shape
    
    circle_diameter = 2 * radius
    crop_size = int(circle_diameter / target_fill_ratio)
    crop_half = crop_size // 2
    
    x_start = center_x - crop_half
    x_end = center_x + crop_half
    y_start = center_y - crop_half
    y_end = center_y + crop_half
    
    pad_left = max(0, -x_start)
    pad_right = max(0, x_end - w)
    pad_top = max(0, -y_start)
    pad_bottom = max(0, y_end - h)
    
    x_start = max(0, x_start)
    x_end = min(w, x_end)
    y_start = max(0, y_start)
    y_end = min(h, y_end)
    
    cropped = image[y_start:y_end, x_start:x_end]
    
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        cropped = np.pad(cropped, 
                        ((pad_top, pad_bottom), (pad_left, pad_right)), 
                        mode='constant', constant_values=0)
    
    return cropped


def scale_to_target(image, target_shape):
    """Scale image to target dimensions."""
    img_min, img_max = image.min(), image.max()
    
    if img_max != img_min:
        normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image, dtype=np.uint8)
    
    pil_img = Image.fromarray(normalized)
    pil_img = pil_img.resize((target_shape[1], target_shape[0]), Image.Resampling.LANCZOS)
    
    scaled = np.array(pil_img, dtype=np.float64)
    if img_max != img_min:
        scaled = scaled / 255 * (img_max - img_min) + img_min
    
    return scaled


def normalize_intensity(image):
    """Normalize to [0, 1] range."""
    img_min, img_max = image.min(), image.max()
    if img_max != img_min:
        return (image - img_min) / (img_max - img_min)
    return np.zeros_like(image)


def align_images_cross_correlation(image, reference):
    """Align image to reference using cross-correlation."""
    img_norm = (image - image.mean()) / (image.std() + 1e-10)
    ref_norm = (reference - reference.mean()) / (reference.std() + 1e-10)
    
    correlation = correlate2d(ref_norm, img_norm, mode='same', boundary='fill')
    
    max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    center_y, center_x = correlation.shape[0] // 2, correlation.shape[1] // 2
    shift_y = max_idx[0] - center_y
    shift_x = max_idx[1] - center_x
    
    aligned = nd_shift(image, (shift_y, shift_x), mode='constant', cval=0)
    return aligned, (shift_y, shift_x)


def preprocess_captured(captured, ground_truth_shape, target_fill_ratio=0.5,
                        remove_dc=True, dc_method='inpaint', dc_aggressive=True):
    """Full preprocessing pipeline for captured image."""
    center_x, center_y, radius = find_bright_circle(captured)
    dc_x, dc_y, dc_radius = find_dc_term(captured, center_x, center_y)
    
    if remove_dc:
        processed = remove_dc_term(captured, dc_x, dc_y, dc_radius, 
                                   method=dc_method, aggressive=dc_aggressive)
    else:
        processed = captured
    
    cropped = crop_circular_region(processed, center_x, center_y, radius, target_fill_ratio)
    scaled = scale_to_target(cropped, ground_truth_shape)
    normalized = normalize_intensity(scaled)
    
    return normalized, {'center': (center_x, center_y), 'radius': radius,
                        'dc_location': (dc_x, dc_y), 'dc_radius': dc_radius}


# ============================================================================
# PYTORCH ASM PROXY MODEL (for autodiff)
# ============================================================================

class ASMPropagation(nn.Module):
    """
    Angular Spectrum Method propagation in PyTorch.
    
    This allows automatic differentiation through the propagation.
    """
    
    def __init__(self, wavelength, pixel_pitch, distance, slm_shape):
        """
        Parameters:
        -----------
        wavelength : float
            Wavelength in meters (e.g., 520e-9)
        pixel_pitch : float
            SLM pixel pitch in meters (e.g., 8e-6)
        distance : float
            Propagation distance in meters
        slm_shape : tuple
            (height, width) of SLM
        """
        super().__init__()
        
        self.wavelength = wavelength
        self.pixel_pitch = pixel_pitch
        self.distance = distance
        self.slm_shape = slm_shape
        
        # Precompute the transfer function H (this is constant)
        Ny, Nx = slm_shape
        
        # Frequency coordinates
        fx = torch.fft.fftfreq(Nx, pixel_pitch)
        fy = torch.fft.fftfreq(Ny, pixel_pitch)
        FY, FX = torch.meshgrid(fy, fx, indexing='ij')
        
        # Transfer function
        k_squared = (1 / wavelength) ** 2
        freq_squared = FX**2 + FY**2
        
        # Propagating waves only
        kz = torch.sqrt(torch.clamp(k_squared - freq_squared, min=0))
        
        # H = exp(i * 2π * kz * z)
        H = torch.exp(1j * 2 * np.pi * kz * distance)
        
        # Zero out evanescent waves
        H[freq_squared > k_squared] = 0
        
        # Register as buffer (not a parameter, but moves with model to GPU)
        self.register_buffer('H', H)
    
    def forward(self, phase):
        """
        Forward propagation.
        
        Parameters:
        -----------
        phase : torch.Tensor
            Phase values in radians, shape (H, W)
        
        Returns:
        --------
        intensity : torch.Tensor
            Intensity at target plane, shape (H, W)
        field : torch.Tensor
            Complex field at target plane (for debugging)
        """
        # Phase to complex field: E = exp(i * phase)
        field_slm = torch.exp(1j * phase)
        
        # ASM propagation: E_out = IFFT{ FFT{E_in} * H }
        field_fft = torch.fft.fft2(field_slm)
        propagated_fft = field_fft * self.H
        field_out = torch.fft.ifft2(propagated_fft)
        
        # Intensity: I = |E|^2
        intensity = torch.abs(field_out) ** 2
        
        return intensity, field_out


# ============================================================================
# CITL OPTIMIZER
# ============================================================================

class CITLOptimizer:
    """
    Camera-in-the-Loop Optimizer.
    
    Implements the CITL update rule:
        φ(k) = φ(k-1) - η * (∂L/∂I_captured) * (∂f/∂φ)
    """
    
    def __init__(self, wavelength, slm_pixel_pitch, propagation_distance, 
                 slm_shape=(1200, 1600), max_phase=1023,
                 target_fill_ratio=0.5, remove_dc=False,
                 device='cpu'):
        """
        Parameters:
        -----------
        wavelength : float
            Laser wavelength in meters
        slm_pixel_pitch : float
            SLM pixel pitch in meters
        propagation_distance : float
            Distance from SLM to target plane in meters
        slm_shape : tuple
            SLM resolution (height, width)
        max_phase : int
            Maximum SLM phase value (1023 for 10-bit)
        target_fill_ratio : float
            Fill ratio for preprocessing
        remove_dc : bool
            Whether to remove DC term from captured images
        device : str
            'cpu' or 'cuda'
        """
        self.wavelength = wavelength
        self.slm_pixel_pitch = slm_pixel_pitch
        self.propagation_distance = propagation_distance
        self.slm_shape = slm_shape
        self.max_phase = max_phase
        self.target_fill_ratio = target_fill_ratio
        self.remove_dc = remove_dc
        self.device = torch.device(device)
        
        # Create ASM propagation model
        self.asm = ASMPropagation(
            wavelength, slm_pixel_pitch, propagation_distance, slm_shape
        ).to(self.device)
        
        print(f"CITL Optimizer initialized:")
        print(f"  Wavelength: {wavelength*1e9:.1f} nm")
        print(f"  Pixel pitch: {slm_pixel_pitch*1e6:.2f} µm")
        print(f"  Distance: {propagation_distance*100:.1f} cm")
        print(f"  SLM shape: {slm_shape}")
        print(f"  Device: {device}")
    
    def load_phase_mask(self, path):
        """Load phase mask from .npy file."""
        phase_np = np.load(path).astype(np.float64)
        print(f"Phase mask loaded: {phase_np.shape}, range [{phase_np.min()}, {phase_np.max()}]")
        return phase_np
    
    def load_ground_truth(self, path):
        """Load and normalize ground truth image."""
        gt = load_image(path)
        gt_norm = normalize_intensity(gt)
        print(f"Ground truth loaded: {gt.shape}")
        return gt_norm
    
    def preprocess_captured_image(self, captured_path, ground_truth_shape):
        """Preprocess captured camera image."""
        captured = load_image(captured_path)
        print(f"Captured image loaded: {captured.shape}")
        
        preprocessed, info = preprocess_captured(
            captured, ground_truth_shape, self.target_fill_ratio,
            remove_dc=self.remove_dc
        )
        
        return preprocessed, info
    
    def compute_captured_loss(self, captured_path, ground_truth_path):
        """
        Compute loss between captured image and ground truth.
        
        Returns:
        --------
        loss_value : float
            MSE loss value
        grad_L_wrt_I : np.ndarray
            Gradient ∂L/∂I_captured
        aligned_captured : np.ndarray
            Preprocessed and aligned captured image
        ground_truth : np.ndarray
            Normalized ground truth
        """
        # Load ground truth
        ground_truth = self.load_ground_truth(ground_truth_path)
        
        # Preprocess captured image
        preprocessed, info = self.preprocess_captured_image(captured_path, ground_truth.shape)
        
        # Align
        aligned, shift = align_images_cross_correlation(preprocessed, ground_truth)
        print(f"Alignment shift: {shift}")
        
        # Compute MSE loss: L = (1/N) * Σ(I - T)²
        diff = aligned - ground_truth
        loss_value = np.mean(diff ** 2)
        
        # Gradient of MSE w.r.t. captured intensity: ∂L/∂I = (2/N) * (I - T)
        N = ground_truth.size
        grad_L_wrt_I = (2.0 / N) * diff
        
        print(f"Captured Loss (MSE): {loss_value:.6f}")
        
        return loss_value, grad_L_wrt_I, aligned, ground_truth
    
    def compute_citl_gradient(self, phase_mask_np, captured_aligned, ground_truth):
        """
        Compute CITL gradient: uses captured image loss but backprops through proxy.
        
        CITL formula: ∂L/∂φ = (∂L/∂I_captured) * (∂I_proxy/∂φ)
        
        We can't backprop through the physical system, so we:
        1. Compute ∂L/∂I from the CAPTURED image
        2. Use the PROXY model (ASM) to backpropagate this gradient to φ
        
        Parameters:
        -----------
        phase_mask_np : np.ndarray
            Phase mask in SLM units (0 to max_phase)
        captured_aligned : np.ndarray
            Preprocessed and aligned captured image (from camera)
        ground_truth : np.ndarray
            Target intensity pattern
        
        Returns:
        --------
        captured_loss : float
            MSE loss of captured image
        grad_wrt_phi : np.ndarray
            Gradient ∂L/∂φ (same shape as phase mask)
        simulated : np.ndarray
            Simulated intensity from proxy model (for visualization)
        """
        # Step 1: Compute loss gradient from CAPTURED image
        # L = (1/N) * Σ(I_captured - I_target)²
        # ∂L/∂I = (2/N) * (I_captured - I_target)
        diff = captured_aligned - ground_truth
        captured_loss = np.mean(diff ** 2)
        N = ground_truth.size
        grad_L_wrt_I_captured = (2.0 / N) * diff  # Shape: (500, 500)
        
        print(f"  Captured MSE: {captured_loss:.6f}")
        print(f"  ∂L/∂I_captured norm: {np.linalg.norm(grad_L_wrt_I_captured):.6f}")
        
        # Step 2: Convert phase mask to radians and create PyTorch tensor
        phase_rad = phase_mask_np.astype(np.float64) / self.max_phase * 2 * np.pi
        phase_tensor = torch.tensor(phase_rad, dtype=torch.float32, 
                                    device=self.device, requires_grad=True)
        
        # Step 3: Forward pass through ASM proxy
        intensity_full, field = self.asm(phase_tensor)
        
        # Step 4: Extract center region to match captured image size
        cy, cx = intensity_full.shape[0] // 2, intensity_full.shape[1] // 2
        ty, tx = ground_truth.shape[0] // 2, ground_truth.shape[1] // 2
        
        y_start = cy - ty
        x_start = cx - tx
        
        intensity_crop = intensity_full[y_start:y_start + ground_truth.shape[0],
                                        x_start:x_start + ground_truth.shape[1]]
        
        # Normalize simulated intensity to [0, 1] for fair comparison
        intensity_min = intensity_crop.min()
        intensity_max = intensity_crop.max()
        if (intensity_max - intensity_min) > 1e-10:
            intensity_norm = (intensity_crop - intensity_min) / (intensity_max - intensity_min)
        else:
            intensity_norm = intensity_crop - intensity_min
        
        # Step 5: Instead of computing proxy loss, we INJECT the captured gradient
        # We use .backward() with a gradient argument
        # This computes: ∂L/∂φ = ∂L/∂I_captured * ∂I_proxy/∂φ
        
        # Convert captured gradient to tensor
        grad_L_tensor = torch.tensor(grad_L_wrt_I_captured, dtype=torch.float32, 
                                     device=self.device)
        
        # Account for normalization: if I_norm = (I - min) / (max - min)
        # then ∂I_norm/∂I = 1 / (max - min)
        # So ∂L/∂I_raw = ∂L/∂I_norm * ∂I_norm/∂I = ∂L/∂I_norm / (max - min)
        scale = (intensity_max - intensity_min).detach()
        if scale > 1e-10:
            grad_for_backward = grad_L_tensor / scale
        else:
            grad_for_backward = grad_L_tensor
        
        # Backward pass: inject the captured gradient
        # This computes ∂L/∂φ using chain rule through ASM
        intensity_crop.backward(grad_for_backward)
        
        # Step 6: Get gradient w.r.t. phase (in radians)
        grad_wrt_phi_rad = phase_tensor.grad.cpu().numpy()
        
        # Convert gradient from radians to SLM units
        grad_wrt_phi = grad_wrt_phi_rad * (self.max_phase / (2 * np.pi))
        
        print(f"  ∂L/∂φ norm: {np.linalg.norm(grad_wrt_phi):.6f}")
        print(f"  ∂L/∂φ min/max: {grad_wrt_phi.min():.4f} / {grad_wrt_phi.max():.4f}")
        
        # Get simulated intensity for visualization
        simulated = intensity_norm.detach().cpu().numpy()
        
        return captured_loss, grad_wrt_phi, simulated
    
    def citl_update(self, phase_mask_path, captured_path, ground_truth_path, 
                    learning_rate=1, save_path=None):
        """
        Perform one CITL update step.
        
        CITL Update Rule:
            φ(k) = φ(k-1) - η * ∂L/∂φ
        
        Where ∂L/∂φ is computed using:
            - Loss L from CAPTURED image (physical measurement)
            - Gradient backpropagated through PROXY model (ASM)
        
        This is the key insight of CITL: we use the real captured image's loss,
        but since we can't backprop through physics, we use the proxy model's
        gradients as an approximation.
        
        Parameters:
        -----------
        phase_mask_path : str
            Path to current phase mask φ(k-1) (.npy file)
        captured_path : str
            Path to captured camera image
        ground_truth_path : str
            Path to ground truth image
        learning_rate : float
            Learning rate η
        save_path : str, optional
            Path to save new phase mask φ(k)
        
        Returns:
        --------
        phi_k : np.ndarray
            New phase mask φ(k)
        loss : float
            Captured image loss
        """
        print("\n" + "=" * 70)
        print("CITL UPDATE STEP")
        print("=" * 70)
        
        # Step 1: Load current phase mask φ(k-1)
        print("\n[1] Loading phase mask φ(k-1)...")
        phi_k_minus_1 = self.load_phase_mask(phase_mask_path)
        
        # Step 2: Load ground truth
        print("\n[2] Loading ground truth...")
        ground_truth = self.load_ground_truth(ground_truth_path)
        
        # Step 3: Preprocess captured image
        print("\n[3] Preprocessing captured image...")
        captured = load_image(captured_path)
        print(f"  Captured image shape: {captured.shape}")
        
        preprocessed, info = preprocess_captured(
            captured, ground_truth.shape, self.target_fill_ratio,
            remove_dc=self.remove_dc
        )
        
        aligned_captured, shift = align_images_cross_correlation(preprocessed, ground_truth)
        print(f"  Alignment shift: {shift}")
        
        # Step 4: Compute CITL gradient
        # This uses CAPTURED image loss but backprops through PROXY (ASM)
        print("\n[4] Computing CITL gradient...")
        print("  (Using captured loss, backprop through ASM proxy)")
        
        captured_loss, gradient, simulated = self.compute_citl_gradient(
            phi_k_minus_1, aligned_captured, ground_truth
        )
        
        # Step 5: Update phase mask
        # φ(k) = φ(k-1) - η * ∂L/∂φ
        print("\n[5] Updating phase mask...")
        print(f"  Learning rate: {learning_rate}")
        
        phi_k = phi_k_minus_1 - learning_rate * gradient
        
        # Clip to valid range [0, max_phase]
        phi_k = np.clip(phi_k, 0, self.max_phase)
        
        # Statistics on the update
        phase_change = phi_k - phi_k_minus_1
        print(f"  Phase change - mean: {phase_change.mean():.4f}, std: {phase_change.std():.4f}")
        print(f"  Phase change - min: {phase_change.min():.2f}, max: {phase_change.max():.2f}")
        
        # Convert to int16 for SLM
        phi_k_int = phi_k.astype(np.int16)
        
        # Save if requested
        if save_path:
            np.save(save_path, phi_k_int)
            print(f"\n[6] New phase mask saved to: {save_path}")
        
        print("\n" + "=" * 70)
        print("CITL UPDATE COMPLETE")
        print(f"  Captured Loss (MSE): {captured_loss:.6f}")
        print("=" * 70)
        
        # Store for visualization
        self._last_update = {
            'phi_k_minus_1': phi_k_minus_1,
            'phi_k': phi_k_int,
            'aligned_captured': aligned_captured,
            'ground_truth': ground_truth,
            'simulated': simulated,
            'gradient': gradient
        }
        
        return phi_k_int, captured_loss
    
    def visualize_update(self, phi_k_minus_1=None, phi_k=None, aligned_captured=None, 
                         ground_truth=None, simulated_intensity=None, gradient=None,
                         save_path=None):
        """Visualize the CITL update."""
        
        # Use stored values if not provided
        if hasattr(self, '_last_update'):
            if phi_k_minus_1 is None:
                phi_k_minus_1 = self._last_update.get('phi_k_minus_1')
            if phi_k is None:
                phi_k = self._last_update.get('phi_k')
            if aligned_captured is None:
                aligned_captured = self._last_update.get('aligned_captured')
            if ground_truth is None:
                ground_truth = self._last_update.get('ground_truth')
            if simulated_intensity is None:
                simulated_intensity = self._last_update.get('simulated')
            if gradient is None:
                gradient = self._last_update.get('gradient')
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Row 1: Phase masks
        axes[0, 0].imshow(phi_k_minus_1, cmap='twilight')
        axes[0, 0].set_title('φ(k-1) - Previous Phase')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(phi_k, cmap='twilight')
        axes[0, 1].set_title('φ(k) - New Phase')
        axes[0, 1].axis('off')
        
        # Phase difference
        phase_diff = phi_k.astype(float) - phi_k_minus_1.astype(float)
        vmax = max(abs(phase_diff.min()), abs(phase_diff.max()), 1)
        im = axes[0, 2].imshow(phase_diff, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[0, 2].set_title(f'Phase Change\nMean: {phase_diff.mean():.2f}, Std: {phase_diff.std():.2f}')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2])
        
        # Gradient
        if gradient is not None:
            grad_vmax = max(abs(gradient.min()), abs(gradient.max()), 1e-6)
            im2 = axes[0, 3].imshow(gradient, cmap='RdBu', vmin=-grad_vmax, vmax=grad_vmax)
            axes[0, 3].set_title(f'Gradient ∂L/∂φ\nNorm: {np.linalg.norm(gradient):.4f}')
            axes[0, 3].axis('off')
            plt.colorbar(im2, ax=axes[0, 3])
        else:
            axes[0, 3].axis('off')
        
        # Row 2: Images
        axes[1, 0].imshow(aligned_captured, cmap='gray')
        axes[1, 0].set_title('Captured (aligned)')
        axes[1, 0].axis('off')
        
        if simulated_intensity is not None:
            axes[1, 1].imshow(simulated_intensity, cmap='gray')
            axes[1, 1].set_title('Simulated (proxy)')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].axis('off')
        
        axes[1, 2].imshow(ground_truth, cmap='gray')
        axes[1, 2].set_title('Ground Truth')
        axes[1, 2].axis('off')
        
        # Difference (captured vs GT)
        diff = np.abs(aligned_captured - ground_truth)
        axes[1, 3].imshow(diff, cmap='hot')
        axes[1, 3].set_title(f'|Captured - GT|\nMSE: {np.mean(diff**2):.4f}')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("CITL Phase Mask Optimization (PyTorch)")
    print("=" * 70)
    
    # Default paths - UPDATE THESE
    phase_mask_path = r'C:\Users\abrar79\OneDrive - BUET\Desktop\Neural 3DHL\phase_mask_uv.npy'
    captured_path = r'C:\Users\abrar79\OneDrive - BUET\Desktop\Neural 3DHL\fg2.png'
    ground_truth_path = r'C:\Users\abrar79\OneDrive - BUET\Desktop\Neural 3DHL\dot2.png'
    output_path = r'C:\Users\abrar79\OneDrive - BUET\Desktop\Neural 3DHL\optimized_phase_mask.npy'
    
    # Parse command line arguments
    if len(sys.argv) >= 4:
        phase_mask_path = sys.argv[1]
        captured_path = sys.argv[2]
        ground_truth_path = sys.argv[3]
    if len(sys.argv) >= 5:
        output_path = sys.argv[4]
    
    # Check files exist
    if not all(os.path.exists(p) for p in [phase_mask_path, captured_path, ground_truth_path]):
        print("\nUsage: python citl_pytorch.py <phase_mask.npy> <captured.png> <ground_truth.png> [output.npy]")
        print("\nPlease provide valid file paths.")
        sys.exit(1)
    
    # Create optimizer
    # PHYSICAL PARAMETERS - MATCHED TO YOUR SETUP
    optimizer = CITLOptimizer(
        wavelength=365e-9,          # 365 nm UV laser
        slm_pixel_pitch=8e-6,       # 8 µm pixels
        propagation_distance=0.52,   # UPDATE THIS! Distance from SLM to target in meters
        slm_shape=(1200, 1600),     # Ny=1200, Nx=1600
        max_phase=1023,             # 10-bit SLM
        target_fill_ratio=0.5,
        remove_dc=True,            # Set True to remove DC term
        device='cpu'                # Use 'cuda' if you have GPU
    )
    
    # Perform CITL update
    phi_k, captured_loss = optimizer.citl_update(
        phase_mask_path=phase_mask_path,
        captured_path=captured_path,
        ground_truth_path=ground_truth_path,
        learning_rate=10,      # Very high LR to compensate for tiny gradients
        save_path=output_path
    )
    
    # Visualize the update (uses stored values from citl_update)
    print("\nGenerating visualization...")
    optimizer.visualize_update(save_path="citl_update_visualization.png")
    
    print(f"\nNew phase mask saved to: {output_path}")
    print(f"You can now load this on the SLM and capture a new image.")
