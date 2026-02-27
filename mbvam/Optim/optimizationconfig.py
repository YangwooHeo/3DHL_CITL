import numpy as np
from datetime import datetime
import logging
import torch
import os

class OptimizationConfig:

    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.fdtype = torch.float32 #this determine the precision of target, w, eps and coord_vec, and the response. Everything in the object space grid.

        # Target setup
        self.target_file_path = 'resources/loft_cross.stl' #'input\3DCross.stl'; %Input file path

        # Mesh operations

        # Voxel operations
        self.voxelize_by_opengl = True
        self.target_num_voxel_z = 128 # number of voxel along z direction. The number of voxels along x,y are determined by the aspect ratios of the STL.
        self.target_padding_mode = None #None, 'equal_voxel', 'equal_length'
        self.target_sizing_mode = 'fit' # None, 'fit', 'fill' or 'stretch'. None means no sizing. 'fit' and 'fill' scale the target domain while maintaining aspect ratios. 'stretch' scaling each dimension independently.
        self.target_domain_size = [1e-3, 1e-3, 1e-3] #[m], Size of the simulation volume in xyz for fitting or stretching. Ignored if target_sizing_mode is None.
        self.target_rotation = [1, 0, 0, np.pi*0] #axis-angle representation. The created rotation matrix maps the original coordinate to final coordinate. G_final = M*G_init
        self.target_offset = [0, 0, 0] #[m], shift target inside the medium. The translation is applied after rotation.
        self.target_scale_in_domain = [0.8, 0.8, 0.8] #scale the target within the simulation volume. The scaling is applied after rotation and translation.

        # self.use_prebuilt_resampling_coord = False #If true, use prebuilt resampling coordinates and store in beam during optimization. Otherwise, generate new coordinates every time.

        self.auto_lr_fractional_change_per_iter = 0.001 #For optimizer.autotuneLR learning rate
        self.lr_phase_mask = 1e16 # default learning rate for phase mask if 'auto' lr is not used.
        self.lr_beam_mean_amplitude = 1e9 # default learning rate for beam_meanamplitude if 'auto' lr is not used.
        self.debug_mode = True #This setting retains gradient in most tensors for debugging purpose. It is not needed for normal operation.
        self.grad_checkpoint = False #If true, use gradient checkpointing to save memory. This is useful for large problem size.
                
        # Output 
        self.log_level = logging.DEBUG
        self.save_optimization = False
        self.run_name = f'Run_{datetime.now().strftime("%Y_%m_%d__%H_%M_")}' #datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.outer_save_directory = '../optim_output/'
        self.save_directory = os.path.join(os.path.abspath(self.outer_save_directory), self.run_name,'')
    
    def resetSaveDirectory(self):
        self.save_directory = os.path.join(os.path.abspath(self.outer_save_directory), self.run_name,'')

    def __repr__(self): #print all attributes, each in a new line
        attributes = [f'{attr}: {getattr(self, attr)}' for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self, attr))]
        return '\n'.join(attributes)
        