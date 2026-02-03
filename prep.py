
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_crop(path, target_shape=None):
    """Load, Center Crop (Aspect Ratio Preserved), and Resize."""
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError(f"Image not found at {path}")
    if len(img.shape) == 3: img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: img_gray = img
    
    if target_shape is None: return img_gray
    
    h, w = img_gray.shape
    target_h, target_w = target_shape
    crop_size = min(h, w)
    
    center_y, center_x = h // 2, w // 2
    start_x = center_x - (crop_size // 2)
    start_y = center_y - (crop_size // 2)
    
    cropped_img = img_gray[start_y:start_y+crop_size, start_x:start_x+crop_size]
    resized_img = cv2.resize(cropped_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized_img

def get_centroid_and_area(img, sigma=0):
    """Calculate Centroid and Area with optional blur for stability."""
    proc_img = img.copy()
    if sigma > 0: proc_img = cv2.GaussianBlur(proc_img, (0, 0), sigma)
    _, thresh = cv2.threshold(proc_img, 30, 255, cv2.THRESH_TOZERO)
    M = cv2.moments(thresh)
    if M["m00"] == 0: return (img.shape[1]/2, img.shape[0]/2), 1.0
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    area = M["m00"]
    return (cX, cY), area

def align_and_scale(source, target, sigma=0, manual_shift=(0,0), manual_scale_mult=1.0):
    """Aligns source to target (Auto + Manual Offset)."""
    h, w = target.shape
    
    # Auto Calc
    (cX_src, cY_src), area_src = get_centroid_and_area(source, sigma=sigma)
    (cX_tgt, cY_tgt), area_tgt = get_centroid_and_area(target, sigma=0)
    
    # Shift
    dx = (cX_tgt - cX_src) + manual_shift[0]
    dy = (cY_tgt - cY_src) + manual_shift[1]
    M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    translated_img = cv2.warpAffine(source, M_trans, (w, h))
    
    # Scale
    auto_scale = np.sqrt(area_tgt / area_src)
    final_scale = auto_scale * manual_scale_mult
    M_scale = cv2.getRotationMatrix2D((cX_tgt, cY_tgt), 0, final_scale)
    scaled_img = cv2.warpAffine(translated_img, M_scale, (w, h))
    
    return scaled_img, (dx, dy), final_scale

def normalize_intensity(img):
    """Normalize to 0.0 ~ 1.0"""
    img_float = img.astype(float)
    min_val, max_val = np.min(img_float), np.max(img_float)
    if max_val - min_val == 0: return img_float
    return (img_float - min_val) / (max_val - min_val)

def calculate_psnr(img1, img2):
    """Calculate PSNR between two normalized images (0.0 ~ 1.0)."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf') # Perfect match
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# --- Main Execution ---

# 1. Path Setup (Change these paths if needed)
camera_img_path = '01222026/Image__2026-01-20__17-34-22.jpg'
target_img_path = '01222026/dot2.png'

# 2. Check & Load
if not os.path.exists(target_img_path):
    print(f"Warning: Target file not found at {os.getcwd()}")
    # For demo purposes, creating a dummy target if file is missing (Remove this in real use)
    target_img = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(target_img, (256, 256), 100, 255, -1)
else:
    target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)

if not os.path.exists(camera_img_path):
    print(f"Warning: Camera file not found. Using dummy.")
    camera_img = np.zeros((720, 1080), dtype=np.uint8) # Dummy
else:
    camera_img = load_and_crop(camera_img_path, target_shape=target_img.shape)

# ==========================================
# Manual Tuning Section
# ==========================================
manual_shift_x, manual_shift_y = -1.5, 2.5
manual_scale_adj = 0.908
# ==========================================

# 3. Process
blur_sigma = 3
final_img, shift, scale = align_and_scale(
    camera_img, target_img, 
    sigma=blur_sigma, 
    manual_shift=(manual_shift_x, manual_shift_y), 
    manual_scale_mult=manual_scale_adj
)

# 4. Normalize and save intermediate images
norm_camera = normalize_intensity(final_img)
norm_target = normalize_intensity(target_img)

plt.imsave('Norm_camera.png', norm_camera, cmap='gray')
plt.imsave('Norm_target.png', norm_target, cmap='gray')

# 5. Analysis (Diff & PSNR)
diff_map = norm_camera - norm_target
psnr_val = calculate_psnr(norm_camera, norm_target)

# --- Visualization ---
plt.figure(figsize=(15, 6)) # Width increased for 5 plots

# Plot 1: Cropped Input
plt.subplot(1, 4, 1)
plt.title("1. Camera raw")
plt.imshow(camera_img, cmap='gray')
plt.axis('off')

# Plot 2: Aligned Result (+ PSNR)
plt.subplot(1, 4, 2)
plt.title(f"2. Camera mag & aligned \nPSNR: {psnr_val:.2f} dB")
plt.imshow(norm_camera, cmap='gray')
plt.axis('off')

# Plot 3: Target
plt.subplot(1, 4, 3)
plt.title("3. Target")
plt.imshow(target_img, cmap='gray')
plt.axis('off')

# Plot 4: Difference Map
plt.subplot(1, 4, 4)
plt.title("4. Difference Map \n (Camera - Target)")
im = plt.imshow(diff_map, cmap='seismic', vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.axis('off')
plt.tight_layout()
plt.show()

# Plot 5: Difference Histogram
plt.title("5. Difference Histogram")
# Flatten the 2D array to 1D and plot
plt.hist(diff_map.ravel(), bins=60, color='gray', alpha=0.7, range=(-1, 1))
plt.axvline(0, color='red', linestyle='--', linewidth=1) # Zero line
plt.xlabel("Difference Value")
plt.ylabel("Pixel Count")
plt.grid(True, alpha=0.3)
plt.xlim(-1, 1)
plt.show()

print(f"Final PSNR: {psnr_val:.4f} dB")