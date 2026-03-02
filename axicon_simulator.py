import numpy as np
import math
np.math = math 
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from mbvam.Beam.holobeam import HoloBeam
from mbvam.Beam.holobeamconfig import HoloBeamConfig
import mbvam.Geometry.visualize


if __name__ == "__main__":
    
    PHASE_MASK = r"C:\Users\cowgr\Documents\PhD\Research\REVAMP\Holographic\3DHL\CITL_Experiment\Proxy_calibration_AltBeam_1image\Epoch_500\Proxy_train_pool\HollowRectangle\slm_phase.npy"
    phase_data = np.load(PHASE_MASK)
    phase_data = phase_data.T
    phase_data = phase_data.astype(np.float32)
    phase_recon = phase_data * (2*np.pi/1023)
    phase_tensor = torch.from_numpy(phase_recon).float()
    
    beam_config = HoloBeamConfig()

    ### UV
    #beam_config.lambda_ = 0.365e-6
    #beam_config.focal_SLM = 0.020

    ### BLUE
    beam_config.lambda_ = 0.473e-6
    beam_config.focal_SLM = 0.12625

    beam_config.binningFactor = 1
    beam_config.Nx_physical = 1600
    beam_config.Ny_physical = 1200
    beam_config.axis_angle = [1,0,00]
    beam_config.z_plane_sampling_rate=0.5
    beam_config.amplitude_profile_type = 'gaussian'
    beam_config.gaussian_beam_waist = 0.0038708 #[m], measured beam waist of the Gaussian beam. Blue: 0.0063188. UV: 0.0038708. Ignored for flat top.
    assert beam_config.focal_SLM is not False
    Axicon_grating_pitch = 1.396e-6
    Axicon_NA = beam_config.lambda_/Axicon_grating_pitch
    Cone_angle = np.arcsin(Axicon_NA)
    print(f'Axicon angle is {Cone_angle}')

    # 2. Beam initialize
    print('1. Initializing beam')
    beam = HoloBeam(beam_config)
    #z_eval_planes = beam.local_coord_vec[2] 
    z_eval_planes = torch.linspace(0, 0.1, steps=51, device=beam_config.device) #unit: m
    print(z_eval_planes*1000)
    H_asm = beam.build_true_ASM_TF(z_query=z_eval_planes)

    beam.slm_amplitude_profile = beam.buildSLMAmplitudeProfile()
    beam.beam_mean_amplitude_iter = torch.tensor(1.0, device=beam_config.device, dtype=beam_config.fdtype)

    recon = beam.propagateToVolume_Axicon(axicon_angle=Cone_angle, axicon_n=1.5, phase_mask=phase_tensor, H_asm=H_asm)
    recon_np = recon.to('cpu').numpy()
    recon_intensity_np = np.abs(recon_np)**2

    mbvam.Geometry.visualize.openCVSliceViewer(recon_intensity_np) #Press q to quit the app.