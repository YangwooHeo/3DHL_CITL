import torch
from mbvam.Target.targetobject import TargetObject
import matplotlib.pyplot as plt

class Metric(): 
    '''
    This superclass implements the structure of a general metric.
    Its subclasses evaluate specific metrics.

    #TODO: evantually subclass torcheval.metrics.Metric[torch.Tensor]
    # Main: https://pytorch.org/torcheval/main/metric_example.html
    # Notebook example: https://github.com/pytorch/torcheval/blob/main/examples/Introducing_TorchEval.ipynb
    # https://pytorch.org/torcheval/main/torcheval.metrics.html
    # Another way is to implement as loss functions: https://pytorch.org/docs/stable/nn.html#loss-functions
    '''
    def __init__(self):
        self.history = []
    
    def clear(self):
        self.history.clear()

    def plotHistory(self, plot_range=None):

        if plot_range is None:
            plot_values = self.history
        else:
            plot_values = self.history[plot_range]

        fig, ax = plt.subplots()
        ax.plot(plot_values, linewidth=2)
        ax.set_xlabel('Iteration')  # Updated line
        ax.set_ylabel('Loss')
        ax.grid(True)
        plt.show()
        
        return fig, ax


class BCLPLoss(Metric):
    '''
    This class store the BCLP parameter and evaluate the loss.
    Make sure all tensors have requires_grad=False except for phase.
    '''

    def __init__(self, target_obj:TargetObject, **kwargs):
        super().__init__()
        self.name = 'BCLP'
        self.target_obj = target_obj

        kwargs.setdefault('p', 2)
        kwargs.setdefault('q', 2)
        self.__dict__.update(kwargs)
        self.checkInputs()
        self.disableGrad()


    def checkInputs(self):
        # Make sure all tensor inputs are on the same device as the target object
        if isinstance(self.target_obj.w, torch.Tensor):
            self.target_obj.w = self.target_obj.w.to(self.target_obj.array.device)
        if isinstance(self.target_obj.eps, torch.Tensor):
            self.target_obj.eps = self.target_obj.eps.to(self.target_obj.array.device)


    def disableGrad(self):
        # Make sure all tensor inputs from TargetObject have requires_grad=False
        self.target_obj.array.requires_grad = False
        if isinstance(self.target_obj.w, torch.Tensor):
            self.target_obj.w.requires_grad = False
        if isinstance(self.target_obj.eps, torch.Tensor):
            self.target_obj.eps.requires_grad = False
        if isinstance(self.target_obj.dvol, torch.Tensor):
            self.target_obj.dvol.requires_grad = False

    def __call__(self, response:torch.Tensor):
        '''This function calculates the BCLP metric.'''
        latest_loss =  self.evaluate(response)
        self.history.append(float(latest_loss))
        return latest_loss
    
    def evaluate(self, response:torch.Tensor):
        '''This function calculates the BCLP metric.'''
        error = response - self.target_obj.array
        error_from_tolerance_band = torch.abs(error)-self.target_obj.eps
        v = error_from_tolerance_band > 0
        integral = torch.sum((self.target_obj.w*torch.abs(error_from_tolerance_band)**self.p)[v])*self.target_obj.dvol #TAG: MIXED_PRECISION
        return integral**(self.q/self.p)
    
    def __repr__(self):
        return f'p={self.p}, q={self.q}'
    
if __name__ == '__main__':
    # test if backward works for BCLPLoss
    import mbvam
    from math import pi
    import matplotlib.pyplot as plt
    # %matplotlib qt
    import torch

    beam_config = mbvam.Beam.holobeamconfig.HoloBeamConfig()
    beam_config.binningFactor = 8

    optim_config = mbvam.Optim.optimizationconfig.OptimizationConfig()
    optim_config.target_num_voxel_z = 128
    optim_config.target_file_path = r'G:\Shared drives\taylorlab\CAL Projects\3DHL\Codebase\3d-holographic-lithography-py\inputs\3d_cross.stl' 
    optim_config.target_domain_size = tuple(ds/beam_config.binningFactor for ds in optim_config.target_domain_size)

    target_obj = mbvam.Target.targetpreprocess.targetPreprocess(optim_config)
    
    loss_fn = BCLPLoss(target_obj, p=2, q=2)
    test_tensor = torch.rand_like(target_obj.array)
    test_tensor.requires_grad = True
    loss_val = loss_fn(test_tensor)
    print(loss_val)
    loss_val.backward()
    print(test_tensor.grad)
    if test_tensor.grad is not None:
        print('Test passed.')
