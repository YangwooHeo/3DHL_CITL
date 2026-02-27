from mbvam.Beam.beam import Beam
from mbvam.Beam.holobeamconfig import HoloBeamConfig, MaterialLayer
from mbvam.Geometry.coordinate import CoordinateArray
import torch
from torch.utils.checkpoint import checkpoint
from typing import Union
from tqdm import tqdm
from .zernike import combine_zernike_basis, compute_zernike_basis 

class HoloBeam(Beam):
    '''
    This subclass of Beam implements the coherent propagation of the holographic beam.
    It has methods to construct propagation kernels, store them as attributes, and propagate the beam with angular spectrum method.
    #TODO: Use alpha (absorption of the resin) to convert to volumetric dose
    '''
    def __init__(self, beam_config):
        if not isinstance(beam_config, HoloBeamConfig):
            raise Exception("beam_config is not an instance of HoloBeamConfig")
        
        super().__init__(beam_config)
        self.phase_mask_init = None # Initial guess of the phase mask
        self.phase_mask_iter = None # This is the phase mask for the iterative optimization. This is where the grad will be stored.
        self.beam_mean_amplitude_init = None #Initial guess of the mean amplitude of the beam.
        self.beam_mean_amplitude_iter = None #0D tensor describing the mean amplitude of the beam. This will be updated during optimization.
        self.slm_amplitude_profile = None #2D tensor describing the profile of the beam at the SLM plane. Normalized to 1. Could be Gaussian or flat top.
        self.H = None # The propagation kernel

        ### New addition for CITL proxy model ###
        self.zernike_coeffs = None
        self.zernike_basis = None
        self.source_modulation_map = None
        self.camera_scale_factor = None

    def _get_zernike_phase(self, shape):
        """
        return zernike phase from zernike coefficients
        
        """
        if self.zernike_coeffs is None:
            return 0
        if self.zernike_basis is None or self.zernike_basis.shape[-2:] != shape[-2:]:
            self.zernike_basis = compute_zernike_basis(num_polynomials=len(self.zernike_coeffs),
                                                       field_res=shape[-2:],
                                                       dtype=self.beam_config.fdtype).to(self.beam_config.device)

        coeffs = self.zernike_coeffs.view(-1, 1, 1)
        zernike_phase = (coeffs*self.zernike_basis).sum(dim=0)

        return zernike_phase

    @torch.no_grad()
    def buildASPropagationTF(self, z_prop:Union[CoordinateArray, torch.Tensor]=None, material_layer:MaterialLayer=None, delta:bool=False):
        '''
        This function build the full/incremental transfer function for angular spectrum propagation in a homogeneous medium.
        The transfer function implemented here is written as equation 6.20 in Fourier Optics Notes.

        CAUTION: This function does not take refractions at material interface into account. Use buildEffectiveTF() for that.
        This function is used to build full transfer function for free space propagation, and incremental transfer function for propagation in a homogeneous medium.
        This function cannot be used to build full transfer function in a layered medium.
        When computing the full transfer function in free space, the'z_prop' is relative to free space focus.
        For delta transfer function, 'z_prop' is distance is only the net distance from the previous propagation.
        To compute delta transfer function, remember to enter 'z_prop' to be the delta_z values, and set 'delta' flag to be true.
        
        Parameters
        ----------
        z_prop: float tensor, z propagation distance. Positive for forward and negative for backward.
        
        material_layer: MaterialLayer object, the material layer the beam is propagating in.

        delta: bool, whether to compute the delta transfer function. Default is False, which computes the full transfer function.

        '''
        # Defaults
        if z_prop is None:
            z_prop = self.local_coord_vec[2] # this returns a local z coordinate vector in torch.Tensor, already in the correct device and dtype
        else:
            z_prop = torch.atleast_1d(torch.as_tensor(z_prop, device=self.beam_config.device, dtype=self.beam_config.fdtype))
            
        if material_layer is None: #by default use the first layer of the material stack
            n = self.beam_config.material_stack[0].n
            alpha = self.beam_config.material_stack[0].alpha
        else:
            n = material_layer.n
            alpha = material_layer.alpha
            
        if delta: # if delta transfer function is to be computed
            const_prefactor = 1 #set const_prefactor to 1 because the prefactor is already included in the full transfer function it builds on.
        else: 
            if override_for_debug := True:
                const_prefactor = 1 #set const_prefactor to 1 because the prefactor is already included in the full transfer function it builds on.
            else:
                k0 = 2*torch.pi/self.beam_config.lambda_ #wavevector in the first medium
                k0 = torch.tensor(k0, device=self.beam_config.device, dtype=self.beam_config.fdtype) 
                const_prefactor = torch.exp(1j*k0*2*self.beam_config.focal_SLM)/(1j*self.beam_config.lambda_*self.beam_config.focal_SLM)


        Nx = self.beam_config.Nx
        Ny = self.beam_config.Ny
        X_slm, Y_slm = torch.meshgrid(
            torch.linspace(-(Nx-1)/2, (Nx-1)/2, Nx, device=self.beam_config.device, dtype=self.beam_config.fdtype)*self.beam_config.psSLM,
            torch.linspace(-(Ny-1)/2, (Ny-1)/2, Ny, device=self.beam_config.device, dtype=self.beam_config.fdtype)*self.beam_config.psSLM,
            indexing='ij')

        X_slm = X_slm.to(self.beam_config.fdtype)
        Y_slm = Y_slm.to(self.beam_config.fdtype)

        gamma = torch.sqrt(1 - (X_slm/(n*self.beam_config.focal_SLM))**2 - (Y_slm/(n*self.beam_config.focal_SLM))**2) # directional cosine relative to z (optical axis), sqrt of a real number
        gamma = torch.real(gamma) # truncate all spatial frequencies that are mapped to evanescent waves. sqrt(-|real|)= purely complex. 

        z_prop = z_prop[None,None,:] # add two dimensions to z_prop to make it 1 x 1 x len(z_prop)
        gamma = gamma[:,:,None] # add one dimension to gamma to make it Nx x Ny x 1

        km = 2*torch.pi*n/self.beam_config.lambda_ #wavevector in the current medium
        exponent = 1j*km*z_prop*gamma - (alpha/2.0)*z_prop/gamma #first term is phase curvature, second term is attenuation.
        #Bug fix: Note that alpha is attenuation coef of intensity, so the above line now use alpha/2 for decay of complex amplitude.
        
        H = const_prefactor*torch.exp(exponent)

        if torch.any(torch.isinf(exponent) | torch.isnan(exponent)): #This check is redundant if input is valid.
            self.logger.warning('Propagation transfer function has NaN or Inf values. Setting the TF elements to be zero. Check input.')
            H[torch.isinf(exponent) | torch.isnan(exponent)] = 0 #remove the NaN and Inf resulted from 0/0 and 1/0 operations because of gamma=0

        H = H.to(self.beam_config.cdtype)
        H.requires_grad_(False) #set requires_grad to be false so that gradient is not computed w.r.t. H
        return H

    @torch.no_grad()
    def buildEffectiveTF(self, z_query:Union[CoordinateArray, torch.Tensor]=None) -> torch.Tensor:
        '''
        This function build the effective transfer function for angular spectrum propagation in a layered medium.
        This function takes the focal plane shift due to refraction into account.

        (1) It uses an absolute coordinate system that is consistent to how the interface_z in MaterialLayer is defined.
        (2) It build the transfer function incrementally for each material layer, starting from the free space focus.

        Parameters
        ----------
        z_query: float tensor, z position in the absolute coordinate system. Consistent with how the interface_z in MaterialLayer is defined.
        If None, the transfer function is built using the Zv attribute of the beam config object.

        '''
        # Defaults
        if z_query is None:
            z_query = self.local_coord_vec[2] # this returns a local z coordinate vector in torch.Tensor, already in the correct device and dtype
        else:
            z_query = torch.atleast_1d(torch.as_tensor(z_query, device=self.beam_config.device, dtype=self.beam_config.fdtype))
            
        # Check if z_query is strictly increasing
        if torch.any(z_query[1:] <= z_query[:-1]): # if any element is smaller than or equal to the previous element
            raise ValueError('z_query must be strictly increasing.')
        
        material_stack = self.beam_config.material_stack #get the material stack from beam config
        
        # Bin z_query into material layers
        edges = torch.as_tensor([layer.interface_z for layer in material_stack], device=self.beam_config.device) #Edge of bins is the z position of material interfaces
        material_indices = torch.bucketize(z_query, edges, right=False) - 1 # get material indices which start from 0. "right=False" means the bins are defined with right edge closed (a,b].
        #It put values to the left bin if possible and minimize redundant computation.

        if torch.all(material_indices == 0): # beam doesn't cross any material interface
            H_eff = self.buildASPropagationTF(z_query-float(self.beam_config.free_space_focus_z), material_stack[0], False) #relative to free space focus. Subtract python float instead of numpy array.
        else:
            # Double pass algorithm. First evaluate the kernel at the interface. Then evaluate the effective kernel after respective interfaces.

            # First get transfer function to interfaces
            # num_crossed_interface = material_indices.max() #This is how many interfaces the beam crosses
            unique_material_indices = material_indices.unique()
            #unique_material_indices.numel() is the number of interfaces subsequent transfer functions need to base upon
            H_interface = torch.zeros(self.beam_config.Nx, self.beam_config.Ny, unique_material_indices.numel(), dtype=self.beam_config.cdtype, device=self.beam_config.device)

            H_iter = self.buildASPropagationTF(z_prop=0, material_layer=material_stack[0], delta=False) # initialize
            H_iter_progress_inter = 0 #This tracks the progress of H_iter. The left face of the material layer i is called interface i.
            H_iter_progress_z = self.beam_config.free_space_focus_z #This tracks the z position of H_iter. 

            for array_ind, interface in enumerate(unique_material_indices): 
                #Build up the transfer function progressively for a particular interface. 
                while H_iter_progress_inter < interface: #Iterate until the desired interface is reached
                    dz = material_stack[H_iter_progress_inter+1].interface_z - H_iter_progress_z #distance to the next interface
                    H_iter *= self.buildASPropagationTF(dz, material_stack[H_iter_progress_inter], True) #propagate in current medium to reach next interface
                    H_iter_progress_inter += 1
                    H_iter_progress_z += dz

                H_interface[...,array_ind] = H_iter.squeeze() #squeeze to remove the singleton dimension
        
            # Build kernel at z_query 
            dist = z_query - edges[material_indices]  # distance from the last interface. Note z_query residing in the 0th medium would have infinite distance.
            H_eff = torch.zeros(self.beam_config.Nx, self.beam_config.Ny, z_query.numel(), dtype=self.beam_config.cdtype, device=self.beam_config.device, requires_grad=False)

            for array_ind, mat in enumerate(unique_material_indices): #iterate for each material
                z_query_idx = (material_indices == mat)  #select the indices of z_query that are in the current material
                if mat == 0: # Exception for the first material. TF of these planes are not build upon any interface.
                    H_eff[...,z_query_idx] = self.buildASPropagationTF(z_query[z_query_idx]-float(self.beam_config.free_space_focus_z), material_stack[0], False) #Subtract python float instead of numpy array.
                else: # For other materials, build upon the interface
                    H_eff[...,z_query_idx] = H_interface[...,array_ind][:,:,None] * self.buildASPropagationTF(dist[z_query_idx], material_stack[mat], True)
                    # [:,:,None] is to add a dimension to H_interface so that it can broadcast to the same shape as H_eff[...,z_query_idx]
            H_eff.requires_grad_(False) #set requires_grad to be false so that gradient is not computed w.r.t. H

        return H_eff
    
    def build_true_ASM_TF(self, z_query: torch.Tensor, n_medium=1.0):
        '''
        Real spatial frequency based ASM transfer function, whereas current buildEffectiveTF was based on pupil/focal plane of objective lens.
        '''
        Nx, Ny = self.beam_config.Nx, self.beam_config.Ny
        ps = self.beam_config.psSLM
        lambda_ = self.beam_config.lambda_
        
        fx = torch.fft.fftfreq(Nx, d=ps, device=self.beam_config.device, dtype=self.beam_config.fdtype)
        fy = torch.fft.fftfreq(Ny, d=ps, device=self.beam_config.device, dtype=self.beam_config.fdtype)
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        
        gamma_sq = 1 - (lambda_ * FX / n_medium)**2 - (lambda_ * FY / n_medium)**2
        gamma = torch.sqrt(torch.clamp(gamma_sq, min=0)) # Clamp to prevent Evanescent wave
        
        km = 2 * torch.pi * n_medium / lambda_
        z_query = z_query[None, None, :]
        gamma = gamma[:, :, None]
        
        # ASM Transfer Function: H(fx, fy, z) = exp(i * k_z * z)
        exponent = 1j * km * z_query * gamma
        H_asm = torch.exp(exponent)
        
        # Masking evanescent wave 
        H_asm[gamma_sq[:, :, None] < 0] = 0
        
        return H_asm.to(self.beam_config.cdtype)

    # def storeTransferFunction(self, H:torch.Tensor): #This function stores the transfer function in the beam object.
    #     self.H = H

    # def clearTransferFunction(self): #This function clears the transfer function in the beam object.
    #     self.H = None

    def propagateToVolume(self, phase_mask:torch.Tensor=None, beam_mean_amplitude:torch.Tensor=None, slm_amplitude_profile:torch.Tensor=None, H:torch.Tensor=None, convert_to_intensity=False):
        '''
        Propagate using the stored transfer function

        FFT convention: 
        As derived in AS spectrum notes, forward FFT is used for forward propagation (taking phase mask to volume E-field). 
        The inverse FFT is used for backward propagation (taking volume E-field back to phase mask).
        With torch.fft.fft2(norm='ortho'), the FFT operation is self-adjoint (ifft2 is adjoint of fft2).
        
        fftshift convention:
        All quantities are saved in physical format (center-oriented) where the middle of the array represent the lowest frequencies.
        fftshift is applied after fft2 during forward propagation.
        Then ifftshift is applied before ifft2 during backward propagation.
        '''

        # Defaults
        if phase_mask is None:
            phase_mask = self.phase_mask_iter

        if beam_mean_amplitude is None:
            beam_mean_amplitude = self.beam_mean_amplitude_iter

        if slm_amplitude_profile is None:
            slm_amplitude_profile = self.slm_amplitude_profile

        if H is None:
            H = self.H

        #Check inputs
        if phase_mask is None:
            raise ValueError('Phase mask is not provided nor stored.')

        if phase_mask.dim() != 2:
            raise ValueError('phase_mask must be a 2D tensor. Current dimension is ' +str(phase_mask.dim()))
        
        if (beam_mean_amplitude is None):
            raise ValueError('beam_mean_amplitude is not provided nor stored.')
        
        if (slm_amplitude_profile is None):
            raise ValueError('slm_amplitude_profile is not provided nor stored.')

        if slm_amplitude_profile.dim() != 2:
            raise ValueError('slm_amplitude_profile must be a 2D tensor.')

        if H is None:
            raise ValueError('Transfer function is not provided nor stored. Use buildTransferFunction() to build the transfer function first.')
        
        # Add aberrated phase map onto the phase mask
        aberration_phase = self._get_zernike_phase(phase_mask.shape[-2:])
        phase_mask = phase_mask + aberration_phase

        #Computation
        # slm_field = torch.sqrt(beam_total_energy[None,None]*slm_energy_profile)*torch.exp(1j*phase_mask) #float tensor becomes complex
        slm_field = beam_mean_amplitude[None,None]*slm_amplitude_profile*torch.exp(1j*phase_mask) #float tensor becomes complex
        vol_field = torch.fft.fft2(slm_field[:,:,None]*H, dim=(0,1), norm="ortho")

        if convert_to_intensity:
            vol_field = torch.abs(vol_field)**2

        return torch.fft.fftshift(vol_field, dim=(0,1))

    @torch.no_grad()
    def propagateToVolume_Axicon(self, axicon_angle: float, axicon_n: float, phase_mask=None, beam_mean_amplitude=None, 
                                 slm_amplitude_profile=None, H_asm=None, convert_to_intensity=False):
        '''
        Axicon lens forward propagation
        '''
        if phase_mask is None: phase_mask = self.phase_mask_iter
        if beam_mean_amplitude is None: beam_mean_amplitude = self.beam_mean_amplitude_iter
        if slm_amplitude_profile is None: slm_amplitude_profile = self.slm_amplitude_profile

        # 1. SLM plane phase with Zernike aberration
        aberration_phase = self._get_zernike_phase(phase_mask.shape[-2:])
        total_phase = phase_mask + aberration_phase
        slm_field = beam_mean_amplitude[None,None] * slm_amplitude_profile * torch.exp(1j * total_phase)

        # 2. Axicon phase mask
        Nx, Ny = self.beam_config.Nx, self.beam_config.Ny
        ps = self.beam_config.psSLM
        x = torch.linspace(-(Nx-1)/2, (Nx-1)/2, Nx, device=self.beam_config.device, dtype=self.beam_config.fdtype) * ps
        y = torch.linspace(-(Ny-1)/2, (Ny-1)/2, Ny, device=self.beam_config.device, dtype=self.beam_config.fdtype) * ps
        X, Y = torch.meshgrid(x, y, indexing='ij')
        R = torch.sqrt(X**2 + Y**2)
        
        k = 2 * torch.pi / self.beam_config.lambda_
        
        # Axicon phase: exp(-i * k * (n-1) * gamma * R)
        axicon_phase_mask = -k * (axicon_n - 1) * axicon_angle * R
        field_after_axicon = slm_field * torch.exp(1j * axicon_phase_mask)

        # 3. Space to Space True ASM propagation 
        E_fourier = torch.fft.fft2(field_after_axicon, dim=(0,1), norm="ortho")
        vol_fourier = E_fourier[:,:,None] * H_asm
        vol_field = torch.fft.ifft2(vol_fourier, dim=(0,1), norm="ortho")

        if convert_to_intensity:
            vol_field = torch.abs(vol_field)**2

        return vol_field


    def propagateToMask(self, volume_field, H:torch.Tensor=None):
        '''
        Propagate the volume field back to the phase mask using the inverse transfer function.
        volume field is the complex amplitude of the electric field in the volume.
        Output the complex light field at phase modulation plane for every input plane.
        Output is 3D, same shape as input.

        FFT convention:
        As derived in AS spectrum notes, forward FFT is used for forward propagation (taking phase mask to volume E-field).
        The inverse FFT is used for backward propagation (taking volume E-field back to phase mask).
        With torch.fft.fft2(norm='ortho'), the FFT operation is self-adjoint (ifft2 is adjoint of fft2).

        fftshift convention:
        All quantities are saved in physical format (center-oriented) where the middle of the array represent the lowest frequencies.
        fftshift is applied after fft2 during forward propagation.
        Then ifftshift is applied before ifft2 during backward propagation.
        '''

        # Check inputs

        if volume_field.dim() != 3:
            raise ValueError('volume_field must be a 3D tensor.')
        
        if H is None:
            H = self.H
        if H is None:
            raise ValueError('Transfer function is not provided nor stored. Use buildTransferFunction() to build the transfer function first.')

        # Computation
        mask_field = torch.fft.ifft2(torch.fft.ifftshift(volume_field, dim=(0, 1)), dim=(0, 1), norm="ortho") #still 3D
        mask_field /= H #divide by the transfer function
        mask_field[torch.isinf(mask_field) | torch.isnan(mask_field)] = 0 #remove inf and nan resulted from region where H is zero. These frequencies are evanescent waves.
        
        # Aberration compensation
        if self.zernike_coeffs is not None:
            aberration_phase = self._get_zernike_phase(mask_field.shape[-2:])
            correction_phasor = torch.exp(-1j * aberration_phase)
            mask_field *= correction_phasor

        return mask_field

    #@torch.no_grad()
    def buildSLMAmplitudeProfile(self):
        '''
        This function builds the source profile at the SLM plane.
        The amplitude profile should have mean value equal to 1.
        If the source profile is Gaussian, the kwargs should contain the gaussian_beam_waist in [m].
        '''

        # If computed profile exist and not in the training step, return cached value
        if self.slm_amplitude_profile is not None and self.source_modulation_map is None:
             return self.slm_amplitude_profile
        
        if self.beam_config.amplitude_profile_type == 'flat_top':
            slm_amplitude_profile = self.buildFlatTopSourceProfile()
        elif self.beam_config.amplitude_profile_type == 'gaussian':
            slm_amplitude_profile = self.buildGaussianSourceProfile(self.beam_config.gaussian_beam_waist)
        else:
            raise ValueError('Invalid source profile type.')
        
        if self.source_modulation_map is not None:
            combined_profile = slm_amplitude_profile * self.source_modulation_map
        else:
            combined_profile = slm_amplitude_profile

        # Re-normalization 
        mean_val = torch.mean(torch.abs(combined_profile)) + 1e-12
        final_profile = combined_profile / mean_val
        
        if self.source_modulation_map is None or not self.source_modulation_map.requires_grad:
            self.slm_amplitude_profile = final_profile
            
        return final_profile

    @torch.no_grad()
    def buildFlatTopSourceProfile(self): # flat top with mean value of 1.
        return torch.ones(self.beam_config.Nx, self.beam_config.Ny, device=self.beam_config.device, dtype=self.beam_config.fdtype)

    @torch.no_grad()
    def buildGaussianSourceProfile(self, gaussian_beam_waist): # gaussian with mean value of 1.
        '''
        gaussian_beam_waist is in [m]
        w_0 as defined in https://en.wikipedia.org/wiki/Gaussian_beam#Beam_waist
        It is the radius at which the field amplitudes fall to 1/e of their axial values.
        It is where the intensity values fall to 1/e^2 of their axial values.
        Also denoted as sigma.
        '''
        Xv = torch.linspace(-(self.beam_config.Nx-1)/2.0, (self.beam_config.Nx-1)/2.0, self.beam_config.Nx)*self.beam_config.psSLM #Center coordinates of the pixels
        Yv = torch.linspace(-(self.beam_config.Ny-1)/2.0, (self.beam_config.Ny-1)/2.0, self.beam_config.Ny)*self.beam_config.psSLM
        X, Y = torch.meshgrid(Xv, Yv, indexing='ij')
        R = torch.sqrt(X**2 + Y**2)

        slm_amplitude_profile = torch.exp(-(R**2)/(gaussian_beam_waist**2))
        slm_amplitude_profile = slm_amplitude_profile/slm_amplitude_profile.mean() #normalize to have mean amplitude value of 1.

        return slm_amplitude_profile.detach().to(self.beam_config.device)

    def compensateSpatialFilter(self):
        # Copy holobeamconfig
        # create another beam object with different material layer and focal length

        # build the effective transfer function for the new beam
        # Forward through the material stack of the spatial filter and backward through free space
        # This is the additional transfer function to be multiplied to the effective transfer function of the original beam.
        ...

    @torch.no_grad()
    def initializePhaseMaskAndEnergy(self, global_grid_energy_req, global_coord_vec):
        '''
        This function initialize the value for the phase mask and beam mean amplitude. It also set these tensors to require gradient.
        The implementation differ for time-multiplexed beams.

        Parameters
        ----------
        global_grid_energy_req: FloatTensor, the energy required for this beam at each voxel of the global grid.
        global_coord_vec: tuple[FloatTensor], the coordinate vectors of the global grid.
        '''
        # (1) Map energy_required back to local grids
        local_grid_energy_req = self.mapGlobalToLocal(global_grid_energy_req, global_coord_vec, use_precomputed_sampling_coord=False).to(self.beam_config.fdtype)
        
        local_vol_field = torch.sqrt(local_grid_energy_req)*torch.exp(1j*2*torch.pi*torch.rand_like(local_grid_energy_req)) #complex amplitude of the volumetric light field

        # (3) Backpropagate the field to SLM
        mask_field = self.propagateToMask(local_vol_field)

        # (4) Extract the phase and use it as initial phase mask
        mask_field_2D = torch.mean(mask_field, dim=2) #average over z to get 2D field
        self.phase_mask_init = torch.angle(mask_field_2D)
        self.phase_mask_iter = self.phase_mask_init.detach().clone()
        self.logger.debug(f'Mean amplitude of every plane: {torch.mean(mask_field.abs(), dim=(0,1))}')


        # (5) Extract the power to set the total power
        # self.beam_mean_amplitude_init = mask_field.abs().mean() #the mean beam field amplitude. Initialized as the mean over the SLM plane.
        self.beam_mean_amplitude_init = ((mask_field.abs())**2).mean().sqrt() #the mean beam field amplitude, initialized using sqrt of mean intensities of all planes and all pixels.
        self.beam_mean_amplitude_iter = self.beam_mean_amplitude_init.detach().clone()
        self.logger.debug(f'Mean amp of every plane and every pixel: {torch.mean(mask_field.abs())}')

        # (6) Set the optimization variables to require gradient
        self.phase_mask_iter.requires_grad = True
        self.beam_mean_amplitude_iter.requires_grad = True

    def propagateAllSubBeamsToGlobalVolumeIntensity(self, global_coord_vec):
        '''
        This method propagate each sub-beams and interpolate them to the global grid.
        In the case of HoloBeam, there is only one sub-beam.
        The implementation differ for time-multiplexed beams.
        '''
        
        if self.beam_config.grad_checkpoint:
            # Forward propagate phase mask
            local_vol_intensity = checkpoint(self.propagateToVolume,convert_to_intensity=True, use_reentrant=False)
            # Interpolate on global grid
            global_vol_intensity = checkpoint(self.mapLocalToGlobal,local_vol_intensity, global_coord_vec, use_precomputed_sampling_coord=False, use_reentrant=False)
        else:
            # Forward propagate phase mask
            local_vol_intensity = self.propagateToVolume(convert_to_intensity=True)

            # Interpolate on global grid
            global_vol_intensity = self.mapLocalToGlobal(local_vol_intensity, global_coord_vec, use_precomputed_sampling_coord=False)

        return global_vol_intensity

    @torch.no_grad()
    def gerchbergsaxton(self, target_intensity:torch.Tensor, H:torch.Tensor, iterations:int=50):
        '''
        Optimize a phase mask for the specified target intensity distribution using Gerchberg-Saxton algorithm.
        target_intensity is ordered in x,y,z.
        zv is the z positions of the planes in the local coordinate system.

        This implementation is isolated from the multi-beam optimization. It is used for debugging and testing SLM display only.
        '''
        #Convert inputs
        target_intensity = target_intensity.to(self.beam_config.device).to(self.beam_config.fdtype)
        target_amplitude = torch.sqrt(target_intensity)

        #Build required components for this function only. These do not affect the stored attributes.
        beam_mean_amplitude = torch.as_tensor(1.0, device=self.beam_config.device, dtype=self.beam_config.fdtype) #beam mean amplitude
        slm_amplitude_profile = self.buildSLMAmplitudeProfile()

        #Initial guess
        local_vol_field = target_amplitude*torch.exp(1j*2*torch.pi*torch.rand_like(target_amplitude)) #complex amplitude of the volumetric light field
        mask_field = self.propagateToMask(local_vol_field, H=H)
        phase_mask = torch.angle(torch.mean(mask_field, dim=2)) #average over z to get 2D field

        #Gerchberg-Saxton iterations
        for iter in tqdm(range(iterations)):
            # Forward propagate phase mask, enforce amplitude constraint in the Fourier space (modulation plane)
            local_vol_field = self.propagateToVolume(phase_mask=phase_mask,
                                                    beam_mean_amplitude=beam_mean_amplitude,
                                                    slm_amplitude_profile=slm_amplitude_profile,
                                                    H=H,
                                                    convert_to_intensity=False)

            local_vol_field = target_amplitude*torch.exp(1j*torch.angle(local_vol_field)) #enforce amplitude constraint in the real space

            # Backward propagate to get the new phase mask
            mask_field = self.propagateToMask(local_vol_field, H=H)
            phase_mask = torch.angle(torch.mean(mask_field, dim=2))

        return phase_mask

    def init_proxy_params(self, num_zernike=20, init_source_map=None, device=None):
        """
        Activate gradient and initialize the parameters of proxy model
        """
        
        target_device = device if device is not None else self.beam_config.device
        self.beam_config.device = target_device # Config update

        print(f"[HoloBeam] Initializing Proxy Params on {target_device}...")

        # 1. Zernike Coefficients to zero
        self.zernike_coeffs = torch.zeros(
            num_zernike, 
            device=target_device, 
            dtype=self.beam_config.fdtype
        ).requires_grad_(True)

        # 2. Source Map to 1
        if init_source_map is None:
            #current_profile = self.buildSLMAmplitudeProfile().to(target_device).detach()
            #self.source_modulation_map = current_profile.clone().requires_grad_(True)
            self.source_modulation_map = torch.ones(
                (self.beam_config.Nx, self.beam_config.Ny),
                device=target_device,
                dtype=self.beam_config.fdtype
            ).requires_grad_(True)
        else:
            self.source_modulation_map = init_source_map.to(target_device).detach().requires_grad_(True)

        # 3. Global Scale Factor
        self.camera_scale_factor = torch.tensor(1.0, device=target_device, dtype=self.beam_config.fdtype).requires_grad_(True)

    def get_proxy_parameters(self):
        """
        Return parameter list to the Optimizer
        """
        params = [self.zernike_coeffs, self.source_modulation_map, self.camera_scale_factor]
        return params

    #def forward_propagate_2D(self, slm_phase, defocus=0.0):
    def forward_proxy_2D(self, slm_phase, source_profile=None, defocus=0.0):
        """
        Test code for the Camera in the Loop system. (written on 01_28_2026)
        Return intensity at a specific focal plane.
        
        Input: SLM phase
        Output: reconstructed amplitutde
        """
        if not slm_phase.requires_grad:
            slm_phase.requires_grad_(True)

        # beam initialization
        if source_profile is not None:
            use_profile = source_profile
        elif self.source_modulation_map is not None:
            use_profile = self.source_modulation_map
        elif self.slm_amplitude_profile is not None:
            use_profile = self.slm_amplitude_profile
        else:
            use_profile = self.buildSLMAmplitudeProfile()
        beam_amp = torch.tensor(1, device=self.beam_config.device)
        #use_profile = self.buildSLMAmplitudeProfile()

        # propagation
        vol_field = self.propagateToVolume(phase_mask=slm_phase, 
                                           beam_mean_amplitude=beam_amp, 
                                           slm_amplitude_profile=use_profile,
                                           H=self.H,
                                           convert_to_intensity=False)
        
        # get focal plane
        zv = self.local_coord_vec[2]
        defocus_m = defocus * 1e-9

        # check if defocus exceeds the volume
        if defocus_m < zv.min() or defocus_m > zv.max():
             print(f"Warning: Defocus {defocus}nm is out of volume range [{zv.min().item()*1e9:.1f}, {zv.max().item()*1e9:.1f}]nm.")

        dz = zv[1] - zv[0]
        z_start = zv[0]
        
        # Index = (Target_Position - Start_Position) / Step_Size
        exact_idx = (defocus_m - z_start) / dz
        
        # index clamping
        exact_idx = torch.clamp(exact_idx, 0, len(zv) - 1.001)

        # Linear Interpolation (Lerp)
        idx_low = torch.floor(exact_idx).long()
        idx_high = idx_low + 1
        alpha = exact_idx - idx_low  

        slice_low = vol_field[:, :, idx_low]
        slice_high = vol_field[:, :, idx_high]
        
        # Complex Interpolation
        target_field_slice = (1 - alpha) * slice_low + alpha * slice_high
        reconstructed_intensity = torch.abs(target_field_slice) ** 2
        
        if self.camera_scale_factor is not None:
            return reconstructed_intensity * torch.abs(self.camera_scale_factor)
        else:
            return reconstructed_intensity

    def forward_proxy_2D_Axicon(self, slm_phase, axicon_angle=0.0174, axicon_n=1.5, source_profile=None, defocus=0):
        """
        Return intensity at a specific focal plane.
        
        Input: SLM phase
        Output: reconstructed amplitutde
        """
        if not slm_phase.requires_grad:
            slm_phase.requires_grad_(True)

        if source_profile is not None:
            use_profile = source_profile
        elif self.source_modulation_map is not None:
            use_profile = self.source_modulation_map
        elif self.slm_amplitude_profile is not None:
            use_profile = self.slm_amplitude_profile
        else:
            use_profile = self.buildSLMAmplitudeProfile()
        beam_amp = torch.tensor(1.0, device=self.beam_config.device, dtype=self.beam_config.fdtype)

        vol_field = self.propagateToVolume_Axicon(
                                                axicon_angle=axicon_angle, 
                                                axicon_n=axicon_n,
                                                phase_mask=slm_phase, 
                                                beam_mean_amplitude=beam_amp, 
                                                slm_amplitude_profile=use_profile,
                                                H_asm=self.H_asm,
                                                convert_to_intensity=True
        )

        # get focal plane
        zv = self.local_coord_vec[2]
        defocus_m = defocus * 1e-9

        # check if defocus exceeds the volume
        if defocus_m < zv.min() or defocus_m > zv.max():
             print(f"Warning: Defocus {defocus}nm is out of volume range [{zv.min().item()*1e9:.1f}, {zv.max().item()*1e9:.1f}]nm.")

        dz = zv[1] - zv[0]
        z_start = zv[0]
        
        # Index = (Target_Position - Start_Position) / Step_Size
        exact_idx = (defocus_m - z_start) / dz
        
        # index clamping
        exact_idx = torch.clamp(exact_idx, 0, len(zv) - 1.001)

        # Linear Interpolation (Lerp)
        idx_low = torch.floor(exact_idx).long()
        idx_high = idx_low + 1
        alpha = exact_idx - idx_low  

        slice_low = vol_field[:, :, idx_low]
        slice_high = vol_field[:, :, idx_high]
        
        # Complex Interpolation
        target_field_slice = (1 - alpha) * slice_low + alpha * slice_high
        reconstructed_intensity = torch.abs(target_field_slice) ** 2

        final_image = reconstructed_intensity * torch.abs(self.camera_scale_factor)
        
        return final_image
