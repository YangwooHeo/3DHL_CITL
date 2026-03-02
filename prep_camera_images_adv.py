
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ==========================================
# 1. Core Logic 
# ==========================================

def load_and_crop(path, target_shape=None):
    img = cv2.imread(str(path))
    if img is None: return None
    if len(img.shape) == 3: img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: img_gray = img
    
    if target_shape is None: return img_gray
    
    h, w = img_gray.shape
    target_h, target_w = target_shape
    crop_size = min(h, w)
    
    center_y, center_x = h // 2, w // 2
    start_x = center_x #- (crop_size // 2)
    start_y = center_y - (crop_size // 2)
    
    cropped_img = img_gray[start_y:start_y+crop_size, start_x:start_x+crop_size]
    resized_img = cv2.resize(cropped_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized_img

def get_centroid_and_area(img, sigma=0):
    proc_img = img.copy()
    if sigma > 0: proc_img = cv2.GaussianBlur(proc_img, (0, 0), sigma)
    _, thresh = cv2.threshold(proc_img, 30, 255, cv2.THRESH_TOZERO)
    M = cv2.moments(thresh)
    if M["m00"] == 0: return (img.shape[1]/2, img.shape[0]/2), 1.0
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    area = M["m00"]
    return (cX, cY), area

def normalize_intensity(img):
    img_float = img.astype(float)
    min_val, max_val = np.min(img_float), np.max(img_float)
    if max_val - min_val == 0: return img_float
    return (img_float - min_val) / (max_val - min_val)

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))

# ==========================================
# 2. GUI Application Class
# ==========================================

class AlignmentApp:
    def __init__(self, root, camera_img, target_img, save_path):
        self.root = root
        self.root.title("Fine-Tune Alignment Tool")
        self.save_path = save_path
        
        # Image Data
        self.camera_raw = camera_img
        self.target_raw = target_img
        self.h, self.w = target_img.shape
        self.norm_target = normalize_intensity(target_img)
        
        # Auto Alignment Pre-calculation
        (cX_src, cY_src), area_src = get_centroid_and_area(self.camera_raw, sigma=3)
        (cX_tgt, cY_tgt), area_tgt = get_centroid_and_area(self.target_raw, sigma=0)
        #self.auto_scale = np.sqrt(area_tgt / area_src)
        self.auto_scale = 1.5789435554535411
        print(f'scale = {self.auto_scale}')
        #self.auto_dx = cX_tgt - cX_src
        self.auto_dx = 103
        print(f'dx = {self.auto_dx}')
        #self.auto_dy = cY_tgt - cY_src
        self.auto_dy = -55
        print(f'dy = {self.auto_dy}')
        #self.src_center = (cX_src, cY_src)
        self.src_center = (144, 271)
        print(f'src_center = {self.src_center}')
        
        # --- UI Variables (Tkinter vars) ---
        self.var_scale_pct = tk.DoubleVar(value=11.4) # 100%
        self.var_rotation = tk.DoubleVar(value=0.1)    # 0 deg
        self.var_shift_x = tk.DoubleVar(value=0.0)     # 0 px manual offset
        self.var_shift_y = tk.DoubleVar(value=4.0)     # 0 px manual offset
        self.var_view_mode = tk.IntVar(value=1)        # 0: Overlay, 1: Diff
        self.var_psnr = tk.StringVar(value="PSNR: -- dB")

        # --- Layout Setup ---
        self.setup_ui()
        self.update_image() # Initial render

    def setup_ui(self):
        # 1. Image Canvas (Left Side)
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.canvas_frame, width=self.w, height=self.h, bg="black")
        self.canvas.pack()
        
        # 2. Controls (Right Side)
        self.controls_frame = ttk.Frame(self.root, width=300)
        self.controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Title
        ttk.Label(self.controls_frame, text="Alignment Controls", font=("Arial", 12, "bold")).pack(pady=10)
        
        # PSNR Display
        ttk.Label(self.controls_frame, textvariable=self.var_psnr, font=("Arial", 12, "bold"), foreground="blue").pack(pady=5)

        # Control Rows
        self.create_control_row("Scale (%)", self.var_scale_pct, 50.0, 150.0, 0.1)
        self.create_control_row("Rotation (deg)", self.var_rotation, -180.0, 180.0, 0.1)
        self.create_control_row("Shift X (px)", self.var_shift_x, -100.0, 100.0, 0.5)
        self.create_control_row("Shift Y (px)", self.var_shift_y, -100.0, 100.0, 0.5)

        # View Mode
        ttk.Label(self.controls_frame, text="View Mode").pack(pady=(20, 5))
        frame_mode = ttk.Frame(self.controls_frame)
        frame_mode.pack()
        ttk.Radiobutton(frame_mode, text="Overlay", variable=self.var_view_mode, value=0, command=self.update_image).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(frame_mode, text="Diff Map", variable=self.var_view_mode, value=1, command=self.update_image).pack(side=tk.LEFT, padx=5)

        # Export Button
        btn_export = ttk.Button(self.controls_frame, text="💾 EXPORT RESULT", command=self.export_result)
        btn_export.pack(pady=30, fill=tk.X, ipady=10)

    def create_control_row(self, label_text, tk_var, min_val, max_val, step):
        """Helper to create Label, [-], Slider, [+], Entry row"""
        frame = ttk.Frame(self.controls_frame)
        frame.pack(fill=tk.X, pady=5)
        
        # Label
        ttk.Label(frame, text=label_text, width=12).pack(side=tk.LEFT)
        
        # Minus Button
        btn_minus = ttk.Button(frame, text="-", width=2, 
                               command=lambda: self.adjust_val(tk_var, -step))
        btn_minus.pack(side=tk.LEFT)
        
        # Slider (Scale)
        scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=tk_var, orient=tk.HORIZONTAL,
                          command=lambda x: self.update_image())
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Plus Button
        btn_plus = ttk.Button(frame, text="+", width=2, 
                              command=lambda: self.adjust_val(tk_var, step))
        btn_plus.pack(side=tk.LEFT)
        
        # Entry (Direct Input)
        entry = ttk.Entry(frame, textvariable=tk_var, width=6)
        entry.pack(side=tk.LEFT, padx=5)
        entry.bind('<Return>', lambda event: self.update_image()) # Enter key triggers update

    def adjust_val(self, tk_var, step):
        val = tk_var.get()
        tk_var.set(round(val + step, 2))
        self.update_image()

    def update_image(self):
        # 1. Get Params
        s_pct = self.var_scale_pct.get()
        angle = self.var_rotation.get()
        dx = self.var_shift_x.get()
        dy = self.var_shift_y.get()
        
        final_scale = self.auto_scale * (s_pct / 100.0)
        total_dx = self.auto_dx + dx
        total_dy = self.auto_dy + dy

        # 2. Warp Affine
        M = cv2.getRotationMatrix2D(self.src_center, angle, final_scale)
        M[0, 2] += total_dx
        M[1, 2] += total_dy
        
        self.aligned_img = cv2.warpAffine(self.camera_raw, M, (self.w, self.h), flags=cv2.INTER_LINEAR)
        
        # 3. PSNR & Normalize
        norm_aligned = normalize_intensity(self.aligned_img)
        psnr = calculate_psnr(norm_aligned, self.norm_target)
        self.var_psnr.set(f"PSNR: {psnr:.2f} dB")
        
        # 4. Display Logic
        if self.var_view_mode.get() == 0: # Overlay
            disp = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            # Green = Target, Magenta = Camera
            tgt_uint = (self.norm_target * 255).astype(np.uint8)
            cam_uint = (norm_aligned * 255).astype(np.uint8)
            
            disp[:, :, 1] = tgt_uint # G
            disp[:, :, 0] = cam_uint # B
            disp[:, :, 2] = cam_uint # R
        else: # Diff Map
            diff = np.abs(norm_aligned - self.norm_target)
            diff_uint = (np.clip(diff * 255, 0, 255)).astype(np.uint8)
            # Heatmap style visualization
            disp = cv2.applyColorMap(diff_uint, cv2.COLORMAP_JET)

        # Convert to ImageTk
        img_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Update Canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk # Keep reference!

    def export_result(self):
        # Save Images
        norm_aligned = normalize_intensity(self.aligned_img)
        plt.imsave(self.save_path / "Final_Aligned_Camera.png", norm_aligned, cmap='gray')
        plt.imsave(self.save_path / "Final_Diff_Map.png", np.abs(norm_aligned - self.norm_target), cmap='gray')
        
        # Save Params
        with open(self.save_path / "final_params.txt", "w") as f:
            f.write(f"Scale_Pct: {self.var_scale_pct.get()}\n")
            f.write(f"Rotation: {self.var_rotation.get()}\n")
            f.write(f"Shift_X: {self.var_shift_x.get()}\n")
            f.write(f"Shift_Y: {self.var_shift_y.get()}\n")
            f.write(f"{self.var_psnr.get()}\n")
        
        print(f"Saved results to {self.save_path}")
        tk.messagebox.showinfo("Export", "Saved Successfully!")

# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    # --- Path Setup ---C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Proxy_calibration_AltBeam_1image\Epoch_500\Phase_Update
    #homepath = Path(r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Experiment_Data\Proxy_Train_pool_AltBeam_02_25_2026\HollowRectangle")
    homepath = Path(r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Proxy_calibration_AltBeam_1image\Epoch_500\Phase_Update")
    if not homepath.exists(): homepath = Path(".")

    camera_path = homepath / "Raw_camera.jpg"
    target_path = homepath / "Raw_target.png"

    # --- Load ---
    if target_path.exists():
        target_img = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    else:
        target_img = np.zeros((512, 512), dtype=np.uint8)
        cv2.circle(target_img, (256, 256), 100, 255, -1)

    if camera_path.exists():
        camera_img = load_and_crop(camera_path, target_shape=target_img.shape)
    else:
        camera_img = np.zeros_like(target_img)

    # --- Run App ---
    root = tk.Tk()
    app = AlignmentApp(root, camera_img, target_img, homepath)
    root.mainloop()