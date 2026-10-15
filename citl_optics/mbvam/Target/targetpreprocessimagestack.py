import os
import numpy as np
from PIL import Image  
import skimage

#IMPLEMENTATION 1
def targetPreprocessingImageStack(image_folder, beam_config, right_reading='reflect', 
                                     interp_method='bilinear'):
    
    # Get image files
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    
    # Get z positions from filenames
    image_pos = [float(f.replace('_', '.')) for f in image_files]
    
    # Sort files
    sort_idx = np.argsort(image_pos)
    image_pos = [image_pos[i] for i in sort_idx]
    image_files = [image_files[i] for i in sort_idx]

    # Load images
    images = []
    for f in image_files:
        img = skimage.io.imread(os.path.join(image_folder, f))
        
        # Handle color images
        if img.ndim == 3:
            img = img.mean(axis=-1)
            
        # Get right reading direction
        if right_reading == 'reflect':
            img = img.T
        elif right_reading == 'transmit':
            img = np.rot90(img)
            
        images.append(img)
        
    images = np.array(images)
    
    # Resize/pad images to match beam AR
    ar_images = images.shape[1] / images.shape[2]
    ar_beam = beam_config.Ny / beam_config.Nx
    
    if ar_images > ar_beam:
        # Fit X, pad Y 
        x_out = beam_config.Nx
        y_out = int(np.rint(x_out * ar_images))
        y_pad = [(y_out - images.shape[1]) // 2, 
                 y_out - images.shape[1] - (y_out - images.shape[1]) // 2]
        images = np.pad(images, [(0,0), (0,0), y_pad], mode='constant')
        
    else:
        # Fit Y, pad X
        y_out = beam_config.Ny
        x_out = int(np.rint(y_out / ar_images))
        x_pad = [(x_out - images.shape[2]) // 2,
                 x_out - images.shape[2] - (x_out - images.shape[2]) // 2]
        images = np.pad(images, [(0,0), x_pad, (0,0)], mode='constant')
        
    # Resize to beam grid        
    images = scipy.ndimage.zoom(images, (beam_config.Nx / images.shape[1],
                           beam_config.Ny / images.shape[2]))
                           
    # Map z positions to beam grid                    
    z_out = np.linspace(-1, 1, beam_config.Nz)
                           
    # Interpolate                
    x, y, z = np.meshgrid(np.linspace(-1,1,beam_config.Nx),
                          np.linspace(-1,1,beam_config.Ny),
                          z_out, indexing='ij')
                          
    positive = scipy.interpolate.interpn((np.linspace(-1, 1, images.shape[1]),
                        np.linspace(-1, 1, images.shape[2]),
                        image_pos),
                       images, x, y, z, method=interp_method)
                       
    zero = ~positive
    
    return positive, zero





#IMPLEMENTATION 2
def targetPreprocessingImageStack(image_folder, Nx, Ny, full_FOV_z, 
                                    fit_method='fit_sq_stretch', interp_order=1, 
                                    equalize=True, right_reading='reflect'):

    # (0) Parse input arguments
    
    # (1) Recognize image files in folder
    image_files = [f for f in Path(image_folder).iterdir() if f.is_file()]
    
    # Format filenames and get z positions
    image_names = [f.stem for f in image_files]
    image_positions = [float(n.replace('_', '.')) for n in image_names]
    
    # Sort images by z position
    sorted_indices = np.argsort(image_positions)
    image_positions_sorted = np.array(image_positions)[sorted_indices]

    # (2) Read images in order
    image_count = len(image_files)
    positive_target = np.zeros((Nx, Ny, image_count), dtype=np.float32)
    
    for i, idx in enumerate(sorted_indices):
        img_path = image_files[idx]
        img = np.array(Image.open(img_path))
        
        # Determine the right reading
        if right_reading == 'reflect':
            img = np.transpose(img)
        elif right_reading == 'transmit':
            img = scipy.ndimage.rotate(img, 90)
            
        # Determine the aspect ratio of input and target size
        
        # Resize image 
        Ny_img, Nx_img = img.shape
        AR_img = Ny_img / Nx_img
        AR_target = Ny / Nx
        
        if fit_method == 'fit': 
            # Fit based on aspect ratio
            if AR_img > AR_target:
                Ny_out = Ny
                Nx_out = int(np.round(Ny / AR_img))
            else:
                Nx_out = Nx
                Ny_out = int(np.round(Nx * AR_img))
                
        elif fit_method == 'fill':
            # Fill target region
            if AR_img > AR_target:
                Nx_out = Nx
                Ny_out = int(np.round(Nx * AR_img))
                img = skimage.transform.resize(img, (Nx_out, Ny_out), order=interp_order)
                img = img[:, (Ny_out//2 - Ny//2):(Ny_out//2 + Ny//2)]
            else:
                Ny_out = Ny
                Nx_out = int(np.round(Ny / AR_img))
                img = skimage.transform.resize(img, (Nx_out, Ny_out), order=interp_order)
                img = img[(Nx_out//2 - Nx//2):(Nx_out//2 + Nx//2), :]
        
        elif fit_method in ['fit_sq_stretch', 'fill_sq_stretch']:
            # Square then stretch
            AR_img_sq = min(AR_img, 1/AR_img)
            Nx_sq = int(np.round(Ny_img / AR_img_sq))
            Ny_sq = Nx_sq
            if fit_method == 'fill_sq_stretch':
                if AR_img > 1:
                    img = img[:, (Ny_img//2 - Nx_sq//2):(Ny_img//2 + Nx_sq//2)] 
                else:
                    img = img[(Nx_img//2 - Ny_sq//2):(Nx_img//2 + Ny_sq//2), :]
            else:
                img = skimage.transform.resize(img, (Nx_sq, Ny_sq), order=interp_order)
            Nx_out = Nx
            Ny_out = Ny
                        
        img = skimage.transform.resize(img, (Nx_out, Ny_out), order=interp_order)
            
        # Insert into the target plane
        x_mid = Nx//2
        y_mid = Ny//2
        x_span = Nx_out//2
        y_span = Ny_out//2
        x_range = range(x_mid-x_span, x_mid+x_span)
        y_range = range(y_mid-y_span, y_mid+y_span)
        positive_target[x_range, y_range, i] = img
        
    # (3) Equalize the target planes  
    if equalize:
        plane_strengths = np.sum(positive_target, axis=(0,1))
        min_strength = np.min(plane_strengths)
        weights = min_strength / plane_strengths
        for i in range(image_count):
            positive_target[:,:,i] *= weights[i]
            
    # (4) Get zero_target
    zero_target = ~positive_target
    
    # (5) Get z positions
    Nz = image_count
    Zv = image_positions_sorted * (full_FOV_z / 2)
    
    return positive_target, zero_target