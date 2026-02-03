import torch
import logging
import matplotlib.pyplot as plt

class ResponseModel():
    _default_gen_log_fun = {'A': 0, 'K': 1, 'B': 25, 'M': 0.5, 'nu': 1, 'clamp_f_before_map': (-1.0, None), 'clamp_mapped_before_map_inv': (0.0, 1.0), 'clamp_f_after_map_inv': (0.0, 1.0)}
    _default_linear = {'M':1, 'C':0} 
    # _default_interpolation = {'interp_min': 0, 'interp_max':1, 'n_pts': 512}

    def __init__(self, form :str = 'gen_log_fun', device=None, **kwargs):
        '''
        This class stores the material response parameter and compute material response based on effective dose.
        
        Parameters
        ----------
        form : str ('gen_log_fun', 'linear', 'identity', 'freeform')

        A : float, optional
            parameter in generalized logistic function (Richard's curve)
            Left asymptote

        K : float, optional
            parameter in generalized logistic function (Richard's curve)
            Right asymptote

        B : float, optional
            parameter in generalized logistic function (Richard's curve)
            Steepness of the curve

        M : float, optional
            parameter in generalized logistic function (Richard's curve)
            M shifts the curve left or right. It is the location of inflextion point when nu = 1. 

        nu : float, optional
            parameter in generalized logistic function (Richard's curve)
            Influence location of maximum slope relative to the two asymptotes. 'Skew' the curve towards either end.

        clamp_f_before_map : tuple[float, None] or None, optional
            Only used for generalized logistic function.
            Clamp the input f to a range before mapping.
            This avoid producing -inf or inf in the exponential terms in the generalized logistic function.
            These infinite exponential terms will causes nan in gradient evaluation.
        
        clamp_mapped_before_map_inv : tuple[float, None] or None, optional
            Only used for generalized logistic function.
            clamp the mapped dose to a range before inverse mapping. This is useful when the inverse map is not defined for the entire range of mapped dose.
            For example, the inverse map of generalized logistic function is not defined for mapped dose < A or > K.

        clamp_f_after_map_inv : tuple[float, None] or None, optional
            Only used for generalized logistic function.
            clamp the inverse map to a range. This prevents producing extremely large or small dose after inverse mapping.

        M : float, optional
            parameter in linear (affine) function
            M is the slope of the curve: map = M*f + C

        C : float, optional
            parameter in linear (affine) function
            M is the y-intercept of the curve: map = M*f + C        

        '''

        #default parameters
        if device is None: #Need to determine device first
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device
        self.logger = logging.getLogger(__name__)
        self.form = form

        if self.form == 'gen_log_fun':
            self.map = self._map_glf
            # self.dmapdf = self._dmapdf_glf
            self.map_inv = self._map_inv_glf
            self.params = self._default_gen_log_fun.copy() #Shallow copy avoid editing dict '_default_gen_log_fun' in place 
            self.params.update(kwargs) #up-to-date parameters. Default dict is not updated
            
        elif self.form == 'linear':  
            self.map = self._map_lin
            # self.dmapdf = self._dmapdf_lin
            self.map_inv = self._map_inv_lin
            self.params = self._default_linear.copy() #Shallow copy avoid editing dict '_default_linear' in place 
            self.params.update(kwargs) #up-to-date parameters. Default dict is not updated

        elif self.form == 'identity':
            self.map = self._map_id
            # self.dmapdf = self._dmapdf_id
            self.map_inv = self._map_inv_id
            self.params = {} #empty parameter dict
        else:
            raise Exception('Form: Other analytical functions are not supported yet.')

    #=================================Analytic: Generalized logistic function================================================

    #Definition of generalized logistic function: https://en.wikipedia.org/wiki/Generalised_logistic_function
    def _map_glf(self, f : torch.Tensor):
        '''
        This function maps the effective dose to material response using generalized logistic function (Richard's curve).

        Debugging notes:
            if f is not clamped, the exponential term in the generalized logistic function can produce -inf or inf, which will cause nan in gradient evaluation.
            The following clamping solutions are tested but they still do not solve the problem.
      
            Solution 1: exp_term = torch.clamp(exp_term, min=torch.finfo(exp_term.dtype).min, max=torch.finfo(exp_term.dtype).max) #do not produce -inf or inf which could create nan in gradient evaluation
            Solution 2: mapped = torch.clamp(mapped, min=self.params['A']+torch.finfo(mapped.dtype).eps) #do not produce -inf or inf which could create nan in gradient evaluation

            In solution 1, regardless of the min and max that we set, even when exp_term is some extremely large values, the nan still occurs.
            Combination of solution 1 and 2 are also tested, but the nan still occurs.
            The only solution we found is to clamp f before mapping.
        '''
        if self.params['clamp_f_before_map'] is not None:
            f = torch.clamp(f, min=self.params['clamp_f_before_map'][0], max=self.params['clamp_f_before_map'][1]) #do not produce -inf or inf which could create nan in gradient evaluation
        numerator = self.params['K'] - self.params['A']
        exp_term = torch.exp(-self.params['B']*(f-self.params['M'])) #cache result for later computation of derivative
        denominator = (1+exp_term)**(1/self.params['nu'])
        mapped = self.params['A'] + (numerator/denominator)
        return mapped

    def _map_inv_glf(self, mapped : torch.Tensor):
        if self.params['clamp_mapped_before_map_inv'] is not None:
            mapped = torch.clamp(mapped, min=self.params['clamp_mapped_before_map_inv'][0], max=self.params['clamp_mapped_before_map_inv'][1])

        numerator = -torch.log(((self.params['K'] - self.params['A'])/(mapped - self.params['A']))**self.params['nu'] - 1)
        f = (numerator/self.params['B']) + self.params['M']

        if self.params['clamp_f_after_map_inv'] is not None:
            f = torch.clamp(f, min=self.params['clamp_f_after_map_inv'][0], max=self.params['clamp_f_after_map_inv'][1])

        return f

    #=================================Analytic: Linear (affine) function=====================================================
    #Definition of linear function: mapped = M*f + C
    def _map_lin(self, f : torch.Tensor):
        return self.params['M']*f + self.params['C']

    def _map_inv_lin(self, mapped : torch.Tensor):
        return (mapped-self.params['C'])/self.params['M']

    #=================================Analytic: Identity function============================================================
    #Definition of identity: mapped = f
    def _map_id(self, f : torch.Tensor):
        return f

    def _map_inv_id(self, mapped : torch.Tensor):
        return mapped
    
    #=================================Utilities==========================================================================
    def plotMap(self, fig = None, ax = None, lb = 0, ub = 1, n_pts=512, block=True, **plot_kwargs):

        f_test = torch.linspace(lb,ub,n_pts)
        mapped_f_test = self.map(f_test)

        if ax == None:
            fig, ax =  plt.subplots()

        f_test = f_test.to('cpu').numpy()
        mapped_f_test = mapped_f_test.to('cpu').numpy()

        ax.plot(f_test, mapped_f_test,**plot_kwargs)
        ax.set_xlabel('Optical dose')
        ax.set_ylabel('Material response (mapped dose)')

        if block == False:
            fig.show() #does not block. This function does not accept block argument.
        else:
            if 'label' in plot_kwargs:
                ax.legend()
            plt.show(block=True)

        return fig, ax


    def plotMapInv(self, fig = None, ax = None, lb = 0, ub = 1, n_pts=512, block=True, **plot_kwargs):

        map_test = torch.linspace(lb,ub,n_pts)
        f_test = self.map_inv(map_test)

        map_test = map_test.to('cpu').numpy()
        f_test = f_test.to('cpu').numpy()
        
        if ax == None:
            fig, ax =  plt.subplots()

        ax.plot(map_test, f_test, **plot_kwargs)
        ax.set_xlabel('Material response (mapped dose)')
        ax.set_ylabel('Optical dose')

        if block == False:
            fig.show() #does not block. This function does not accept block argument.
        else:
            if 'label' in plot_kwargs:
                ax.legend()
            plt.show(block=True)

        return fig, ax

    def __repr__(self):
        return (f'form: {self.form}, params: {str(self.params)}')