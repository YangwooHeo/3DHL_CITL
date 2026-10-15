import torch
from torch.utils.checkpoint import checkpoint
import logging
import copy
from mbvam.Beam.holobeam import HoloBeam
from mbvam.Optim.optimizationconfig import OptimizationConfig
from mbvam.Target.targetobject import TargetObject
from mbvam.Optim.metric import Metric
from mbvam.Response.effectivedose import EffectiveDose
from mbvam.Response.responsemodel import ResponseModel
from tqdm import tqdm # from tqdm.autonotebook import tqdm

class MaskOptimizer():
    def __init__(self,
                beams:tuple[HoloBeam],
                optim_config:OptimizationConfig,
                target_object:TargetObject,
                eff_dose_model:EffectiveDose,
                response_model:ResponseModel,
                metric:Metric):
        self.logger = logging.getLogger(__name__)
        self.beams = beams
        self.optim_config = optim_config
        self.target_object = target_object
        self.eff_dose_model = eff_dose_model
        self.response_model = response_model
        self.metric = metric

        for beam in beams: #initalize the beams with non-learnable parameters
            beam.H = beam.buildEffectiveTF() #build transfer functions. Use beam.H = beam.buildASPropagationTF() for creating free-space hologram.
            beam.H.requires_grad = False #set the transfer function to not require gradient
            beam.slm_amplitude_profile = beam.buildSLMAmplitudeProfile() #set amplitude profile
            beam.slm_amplitude_profile.requires_grad = False #set the amplitude profile to not require gradient

        self.auto_lr_phase_mask = None #auto-tuned learning rate for phase mask
        self.auto_lr_beam_mean_amplitude = None #auto-tuned learning rate for beam mean amplitude
        self.last_optimize_kwargs = None #store the last kwargs used in optimize() for __repr__

    @torch.no_grad()
    def generateInitialGuess(self):
        '''
        Initialize phase and amplitude variables. Generate a first guess solution.
        '''
        # (1) Invert TargetObject.array with response to get dose (or intensity field)
        energy_required = self.response_model.map_inv(self.target_object.array)

        for beam_idx, beam in enumerate(self.beams):
            energy_required_per_beam = self.eff_dose_model.inv(energy_required, beam_idx) # This function calculates the energy required for each beam to achieve the desired dose.

            # This function initialize the value for the phase mask and beam mean amplitude. It also set these tensors to require gradient.
            beam.initializePhaseMaskAndEnergy(global_grid_energy_req = energy_required_per_beam, global_coord_vec = self.target_object.coord_vec)

    
    def forward(self):

        # Declare global intensity
        dose_list = []

        # (1) For each of the beams
        for beam in self.beams:
            if self.optim_config.grad_checkpoint:
                intensity_global = checkpoint(beam.propagateAllSubBeamsToGlobalVolumeIntensity,(self.target_object.coord_vec), use_reentrant=False)
            else:
                intensity_global = beam.propagateAllSubBeamsToGlobalVolumeIntensity(self.target_object.coord_vec)

            dose_list.append(intensity_global.to(self.optim_config.device).to(self.optim_config.fdtype))
        
        dose_tuple = tuple(dose_list)


        if self.optim_config.grad_checkpoint:
            # (2) Get effective dose from dose of all beams
            eff_dose = checkpoint(self.eff_dose_model, dose_tuple, use_reentrant=False)
            # (3) Apply response model
            response = checkpoint(self.response_model.map, eff_dose, use_reentrant=False)
        else:
            eff_dose = self.eff_dose_model(dose_tuple)
            response = self.response_model.map(eff_dose)

        if self.optim_config.debug_mode: #debug cases where the gradient becomes nan or inf
            for dose in dose_tuple:
                dose.retain_grad() #retain the gradient for the dose for debugging
            self.dose_tuple_debug = dose_tuple # save dose_tuple for debugging

            eff_dose.retain_grad() #retain the gradient for the effective dose for debugging
            self.eff_dose_debug = eff_dose # save eff_dose for debugging
            response.retain_grad() #retain the gradient for the response for debugging
            self.response_debug = response # save response for debugging

        # (4) Calculate metric
        return checkpoint(self.metric, response, use_reentrant=False) if self.optim_config.grad_checkpoint else self.metric(response)
    

    def optimize(self, optimizer = 'SGD', num_iter = 100, lr_phase_mask = None, lr_beam_mean_amplitude = None, **kwargs):
        '''
        optimizer: str, optional. 'SGD' (default) or 'Adam'

        num_iter: int, optional. Number of iterations. Default 100.

        lr_phase_mask: float, optional. 'auto' or None (default). Learning rate for phase mask.
        If 'auto', use the value previously autotuned by optimizer.autotuneLR().
        If None, use the default value in the config file.

        lr_beam_mean_amplitude: float, optional. 'auto' or None (default). Learning rate for beam mean amplitude. 
        If 'auto', use the value previously autotuned by optimizer.autotuneLR().
        If None, use the default value in the config file.
        '''
        #save the last kwargs used in optimize() for __repr__
        self.last_optimize_kwargs = {'optimizer': optimizer, 'num_iter': num_iter, 'lr_phase_mask': lr_phase_mask, 'lr_beam_mean_amplitude': lr_beam_mean_amplitude, 'kwargs': kwargs}

        if lr_phase_mask == 'auto':
            lr_phase_mask = self.auto_lr_phase_mask
            self.logger.info(f'Using auto_lr_phase_mask: {self.auto_lr_phase_mask}')

        if lr_phase_mask is None: #This None could come from default kwarg or from auto_lr_phase_mask
            lr_phase_mask = self.optim_config.lr_phase_mask #fall back to config value
            self.logger.warning(f'No lr_phase_mask provided, falling back to value in optimization config: {self.optim_config.lr_phase_mask}')

        if lr_beam_mean_amplitude == 'auto':
            lr_beam_mean_amplitude = self.auto_lr_beam_mean_amplitude
            self.logger.info(f'Using auto_lr_beam_mean_amplitude: {self.auto_lr_beam_mean_amplitude}')

        if lr_beam_mean_amplitude is None: #This None could come from default kwarg or from auto_lr_phase_mask
            lr_beam_mean_amplitude = self.optim_config.lr_beam_mean_amplitude #fall back to config value
            self.logger.warning(f'No lr_beam_mean_amplitude provided, falling back to value in optimization config: {self.optim_config.lr_beam_mean_amplitude}')
        
        # Use one parameter group to include all optimization variables.
        # optim_var = [ beam.phase_mask_iter for beam in self.beams ] + [ beam.beam_mean_amplitude_iter for beam in self.beams ]

        # Create two parameter groups: one for phase_mask_iter and one for beam_mean_amplitude
        optim_var = [{'params': [beam.phase_mask_iter for beam in self.beams], 'lr': lr_phase_mask},
                    {'params': [beam.beam_mean_amplitude_iter for beam in self.beams], 'lr': lr_beam_mean_amplitude}]

        # Create a parameter group for each optimization variable
        # optim_var = [{'params': beam.phase_mask_iter, 'lr': lr[i]} for i, beam in enumerate(self.beams)]
        # optim_var += [{'params': beam.beam_mean_amplitude_iter, 'lr': lr[i + len(self.beams)]} for i, beam in enumerate(self.beams)]

        # Setup optimizer
        if optimizer == 'SGD':
            optimizer = torch.optim.SGD(optim_var, **kwargs)
        elif optimizer == 'Adam':
            optimizer = torch.optim.Adam(optim_var, **kwargs)
        else:
            raise ValueError('Optimizer not supported.')

        # Start loop
        for iter in tqdm(range(num_iter), desc='Optimization iteration'):
            optimizer.zero_grad(set_to_none=True) #zero the gradient buffers
            loss = self.forward() #Forward pass

            self.logger.info(f'Iteration {iter+1}, loss: {float(loss)}')
            if self.exit(loss, iter):
                break

            loss.backward() #Backward pass
            optimizer.step()

    def exit(self, loss, iter):
        exit_flag = False

        if torch.isinf(loss): #infinite loss
            self.logger.critical(f'Loss reaches infinity at iteration {iter}. Check prior steps.')
            exit_flag = True

        if not torch.is_nonzero(loss): #reaches zero
            self.logger.warning(f'Loss reaches zero at iteration {iter}. Check results.')
            exit_flag = True
        return exit_flag

    def autotuneLR(self, *args, fractional_change_per_iter=None, **kwargs):
        '''
        Autotune the learning rate for phase_mask_iter and beam_mean_amplitude.

        Parameters
        ----------
        fractional_change_per_iter: float, optional. The distance that the variables will move in one step. Default 0.05.

        *args, **kwargs: arguments and keyword arguments to generateInitialGuess()
        '''
        if fractional_change_per_iter is None:
            fractional_change_per_iter = self.optim_config.auto_lr_fractional_change_per_iter

        # Save a copy of loss history
        original_metric_history = copy.deepcopy(self.metric.history)

        self.generateInitialGuess(*args, **kwargs)

        # Get the mean of the absolute values of the variables
        mean_phase_mask = torch.as_tensor([beam.phase_mask_iter.abs().mean() for beam in self.beams])
        mean_beam_mean_amplitude = torch.as_tensor([beam.beam_mean_amplitude_iter.abs().mean() for beam in self.beams])

        # Average over all beams
        mean_phase_mask = mean_phase_mask.mean() #Or override it as 2pi
        mean_beam_mean_amplitude = mean_beam_mean_amplitude.mean()

        # Set the gradient to zero or None
        for beam in self.beams:
            beam.phase_mask_iter.grad = None
            beam.beam_mean_amplitude_iter.grad = None

        # Calculate the gradient
        loss_init = self.forward()
        loss_init.backward()
        self.metric.history = original_metric_history # Put back the loss history (avoid adding an entry)

        with torch.no_grad():
            # Get the mean of the absolute values of the gradient
            mean_phase_mask_grad = torch.as_tensor([beam.phase_mask_iter.grad.abs().mean() for beam in self.beams])
            mean_beam_mean_amplitude_grad = torch.as_tensor([beam.beam_mean_amplitude_iter.grad.abs().mean() for beam in self.beams])

            # Average over all beams
            mean_phase_mask_grad = mean_phase_mask_grad.mean()
            mean_beam_mean_amplitude_grad = mean_beam_mean_amplitude_grad.mean()

            # Calculate the learning rate that will move the variables by fractional_change_per_iter
            self.auto_lr_phase_mask = fractional_change_per_iter * mean_phase_mask / mean_phase_mask_grad
            self.auto_lr_beam_mean_amplitude = fractional_change_per_iter * mean_beam_mean_amplitude / mean_beam_mean_amplitude_grad

        return self.auto_lr_phase_mask, self.auto_lr_beam_mean_amplitude
    
    def __repr__(self): #print actual value of auto_lr_phase_mask and auto_lr_beam_mean_amplitude
        return f'''
            auto_lr_phase_mask: {self.auto_lr_phase_mask}, 
            auto_lr_beam_mean_amplitude: {self.auto_lr_beam_mean_amplitude},
            kwargs: {self.last_optimize_kwargs.__repr__() if self.last_optimize_kwargs else None}
            '''