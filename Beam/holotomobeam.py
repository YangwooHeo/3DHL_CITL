import torch
from torch.utils.checkpoint import checkpoint
from mbvam.Beam.holobeam import HoloBeam
from mbvam.Beam.holotomobeamconfig import HoloTomoBeamConfig
from tqdm import tqdm


class HoloTomoBeam(HoloBeam):
    '''
    Instances of this class represent a holo-tomographic beam.
    It is written as an abstraction of this time-multiplexed configuration to enable memory saving and cleaner programming.
    Therefore instead of brute-force storing all copies of the same hologram beam, it assume rotation invariance and store only essetial information.
    
    Assumptions:
    - All configuration variables of the beam are rotationally invariant. 
    These includes wavelength, pixel size, and beam size, material stacks..etc.
    The only difference between the beams is the rotation angle.

    - Response (degree of polymerization) only depends on accumulated dose. Also called reciprocity.
    The rate and order of the dose delivery does not matter.
    The dose delivered by each beam angle carries the same influence to the final response.

    - The beam amplitude and profile are constant.
    
    This class overrides the methods of the base class such that the aggregated results are delivered.
    Since GPU memory is likely to be the bottleneck on the resolution of the beam, we do not force vectorizing over beam index.
    Instead, we use a for loop to iterate over the beams.
    '''
    def __init__(self, beam_config): 

        if not isinstance(beam_config, HoloTomoBeamConfig):
            raise Exception("beam_config is not an instance of HoloTomoBeamConfig")

        super().__init__(beam_config)

    @torch.no_grad()
    def initializePhaseMaskAndEnergy(self, global_grid_energy_req, global_coord_vec):
        '''
        This function overrides the base class method and extend its results to multiple sub-beams.
        It initialize one phase masks for each sub-beams and one common beam mean amplitude for all sub-beams.
        '''
        # (0) Initialize the variables
        global_grid_energy_req_per_sb = global_grid_energy_req/self.beam_config.num_sub_beam
        self.phase_mask_init = torch.zeros(self.beam_config.Nx, self.beam_config.Ny, self.beam_config.num_sub_beam, device=self.beam_config.device, dtype=self.beam_config.fdtype)
        # beam_mean_amplitude_sum = 0.0 #running sum of the mean beam field amplitude.
        beam_mean_intensity_sum = 0.0 #running sum of the mean beam intensity.

        #Operation common to all sub-beams
        # aberration_compensation = -self.buildEffectiveTF(z_query=0.0).squeeze() #the natural focal point is always at z=0.0 in local grid.

        # (2) Assert volumetric phase on the dose (or intensity field)
        # local_vol_field = self.propagateToVolume(phase_mask = torch.zeros(self.beam_config.Nx, self.beam_config.Ny, device=self.beam_config.device, dtype=self.beam_config.fdtype),
        #                     beam_mean_amplitude=torch.tensor(1.0, device=self.beam_config.device, dtype=self.beam_config.fdtype),
        #                     )
        # local_vol_field = self.propagateToVolume(phase_mask = aberration_compensation.angle(),
        #                     beam_mean_amplitude=torch.tensor(1.0, device=self.beam_config.device, dtype=self.beam_config.fdtype),
        #                     )
            
        # local_vol_phase = torch.angle(local_vol_field) #the volumetric phase obtained by propagating the phase mask from SLM plane 
    
        # For each sub-beam
        for sb_idx in tqdm(range(self.beam_config.num_sub_beam), desc='Initialization progress', unit='sub-beam'):

            # (1) Map energy_required back to local grids
            local_grid_energy_req = self.mapGlobalToLocal(global_grid_energy_req_per_sb,
                                                        global_coord_vec,
                                                        use_precomputed_sampling_coord=False,
                                                        override_axis_angle=self.beam_config.scan_axis_angle[sb_idx])
        
            # (2) Assert volumetric phase on the energy required
            # local_vol_field = torch.sqrt(local_grid_energy_req)*torch.exp(local_vol_phase) #complex amplitude of the volumetric light field
            local_vol_field = torch.sqrt(local_grid_energy_req)*torch.exp(1j*2*torch.pi*torch.rand_like(local_grid_energy_req)) #complex amplitude of the volumetric light field

            # (3) Backpropagate the field to SLM
            mask_field = self.propagateToMask(local_vol_field)

            # (4) Extract the phase and use it as initial phase mask
            mask_field_2D = torch.mean(mask_field, dim=2) #average over z to get 2D field
            self.phase_mask_init[:,:,sb_idx] = torch.angle(mask_field_2D)

            # (5) Extract the power to set the total power
            # beam_mean_amplitude_sum += mask_field.abs().mean() #the mean beam field amplitude. Initialized as the mean over the SLM plane.
            beam_mean_intensity_sum += (mask_field.abs()**2).mean() #the running sum of mean beam intensity
        

        self.phase_mask_iter = self.phase_mask_init.clone().detach()
        # self.beam_mean_amplitude_init = torch.tensor(beam_mean_amplitude_sum/self.beam_config.num_sub_beam,
        #                                         device=self.beam_config.device,
        #                                         dtype=self.beam_config.fdtype) #average the amplitude over all sub-beams
        self.beam_mean_amplitude_init = torch.tensor((beam_mean_intensity_sum/self.beam_config.num_sub_beam).sqrt(),
                                        device=self.beam_config.device,
                                        dtype=self.beam_config.fdtype) #average the intensity over all sub-beams, then take square root to get amplitude
        self.beam_mean_amplitude_iter = self.beam_mean_amplitude_init.detach().clone()

        # (6) Set the optimization variables to require gradient
        self.phase_mask_iter.requires_grad = True
        self.beam_mean_amplitude_iter.requires_grad = True


    def propagateAllSubBeamsToGlobalVolumeIntensity(self, global_coord_vec):
        '''
        This function overrides the base class method and extend its results to multiple sub-beams.
        It propagate each sub-beams and interpolate them to the global grid.
        To lower memory requirement, this function loop over all sub-beams to avoid simultaneously storing the all local intensity fields.
        '''
        global_grid_shape = tuple([gcv.size()[0] for gcv in global_coord_vec])
        global_vol_intensity = torch.zeros(global_grid_shape, device=self.beam_config.device, dtype=self.beam_config.fdtype)

        for sb_idx in range(self.beam_config.num_sub_beam):
            if self.beam_config.grad_checkpoint:
                # Forward propagate phase mask
                local_vol_intensity = checkpoint(self.propagateToVolume, self.phase_mask_iter[:,:,sb_idx],
                                    self.beam_mean_amplitude_iter, convert_to_intensity=True, use_reentrant=False)
                # Interpolate on global grid
                global_vol_intensity += checkpoint(self.mapLocalToGlobal, local_vol_intensity, global_coord_vec, use_precomputed_sampling_coord=False,
                                                            override_axis_angle=self.beam_config.scan_axis_angle[sb_idx], use_reentrant=False)            
            else:
                # Forward propagate phase mask
                local_vol_intensity = self.propagateToVolume(phase_mask = self.phase_mask_iter[:,:,sb_idx],
                                    beam_mean_amplitude=self.beam_mean_amplitude_iter,
                                    convert_to_intensity=True)
                # Interpolate on global grid
                global_vol_intensity += self.mapLocalToGlobal(local_vol_intensity,
                                                            global_coord_vec,
                                                            use_precomputed_sampling_coord=False,
                                                            override_axis_angle=self.beam_config.scan_axis_angle[sb_idx])
        return global_vol_intensity


    

