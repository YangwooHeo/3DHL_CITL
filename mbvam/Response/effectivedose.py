import torch
import logging

class EffectiveDose():
    '''
    This effective dose model compute the effective photoexcitation from photodosage of multiple beams which can be at different wavelengths.
    This model combines photodosage from multiple beams linearly and quadratically.
    Higher order terms are not currently supported.

    Parameters
    ----------
    num_beams : int, number of beams to combine when evaluating effective dose

    coef_lin : list, 1D numpy array, or 1D tensor. Optional. Coefficient of the linear terms.

    coef_quad : list of list, 2D numpy array, or 2D tensor. Optional. Coefficients for quadratic terms.

    '''
    def __init__(self, num_beams, coef_lin=None, coef_quad=None, device=None) -> None:
        self.logger = logging.getLogger(__name__)
        self.num_beams = num_beams

        if device is None: #Need to determine device first
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device

        if coef_lin is None:
            self.coef_lin = torch.zeros(self.num_beams, device=self.device, requires_grad=False) #_coef_lin is the coefficient of the linear term in the material response function
        else:
            self.coef_lin = coef_lin

        if coef_quad is None:
            self.coef_quad = torch.zeros((self.num_beams,self.num_beams), device=self.device, requires_grad=False) #_coef_quad is the coefficient of the quadratic term in the material response function
        else:
            self.coef_quad = coef_quad

    #=================================Setting coefficients================================================
    @property
    def coef_lin(self):
        return self._coef_lin
    
    @coef_lin.setter
    def coef_lin(self, input):
        self._coef_lin = torch.as_tensor(input, device=self.device)
        self._coef_lin = self._coef_lin.broadcast_to((self.num_beams)) #if input is scalar, broadcast to proper shape
        self._coef_lin.requires_grad = False
        self.logger.info(f'Stored coef_lin: {self._coef_lin}')

    @property
    def coef_quad(self):
        return self._coef_quad
    
    @coef_quad.setter
    def coef_quad(self, input):
        self._coef_quad = torch.as_tensor(input, device=self.device)
        self._coef_quad = self._coef_quad.broadcast_to((self.num_beams,self.num_beams)) #if input is scalar, broadcast to proper shape
        self._coef_quad.requires_grad = False
        self._coef_quad = (self._coef_quad + self._coef_quad.T)/2 # Enforce symmetry
        self.logger.info(f'Stored coef_quad: {self._coef_quad}')

    #=================================Computation of effective dose================================================

    def __call__(self, dose_tuple:tuple):
        '''This function calculates the effective dose based on the material response parameters.'''
        return self.linear(dose_tuple) + self.quadratic(dose_tuple)
    
    def linear(self, dose_tuple:tuple):
        '''This function calculates the linear term of the material response.'''
        lin_term = torch.zeros_like(dose_tuple[0])
        
        if torch.all(self.coef_lin == 0.0):
            pass
        else:
            for ind,c in enumerate(self.coef_lin):
                self.logger.debug(f'Evaluating linear term with coefficient({ind}): {c}')
                if c != 0.0:
                    lin_term += c*dose_tuple[ind]

        return lin_term
    
    def quadratic(self, dose_tuple:tuple):
        '''This function calculates the quadratic term of the material response.'''
        quad_term = torch.zeros_like(dose_tuple[0])
        
        if torch.all(self.coef_quad == 0.0):
            pass
        else:
            for row_ind,row in enumerate(self.coef_quad):
                for col_ind,c in enumerate(row):
                    # Simplest implementation to evaluate every term
                    # if c != 0.0:
                    #     self.logger.debug(f'Evaluating quadratic term with coefficient({row_ind},{col_ind}): {c}')
                    #     quad_term += c*dose_tuple[row_ind]*dose_tuple[col_ind]

                    # The following implementation only evaluate the off-diagonal terms once because the coefficient matrix should be symmetric.
                    if (col_ind == row_ind) and (c != 0.0): #diagonal non-zero coefficients
                        self.logger.debug(f'Evaluating quadratic term with coefficient({row_ind},{col_ind}): {c}')
                        quad_term += c*dose_tuple[row_ind]**2
                    elif (col_ind > row_ind) and (c != 0.0): #upper triangular non-zero coefficients
                        self.logger.debug(f'Evaluating quadratic term with coefficient({row_ind},{col_ind}): {c}')
                        quad_term += 2.0*c*dose_tuple[row_ind]*dose_tuple[col_ind]
        return quad_term

    #=================================Computation of inverse of effective dose================================================
    def specialCaseClassification(self) -> str:
        '''This function identifies the special cases of the effective dose model.'''
        if torch.all(self.coef_lin == 0.0) and torch.all(self.coef_quad == 0.0):
            raise Exception('Both linear and quadratic coefficients are zero. This is not a valid effective dose model.')
        elif torch.any(self.coef_lin != 0.0) and torch.all(self.coef_quad == 0.0):
            # Linear case
            if torch.all(self.coef_lin >= 0.0):
                case = 'linear_additive'
            elif torch.any(self.coef_lin > 0.0) and torch.any(self.coef_lin < 0.0):
                case = 'linear_subtractive'
            else:
                raise Exception('This linear case is not supported. Maybe all linear coefficients are negative.')

        elif torch.all(self.coef_lin == 0.0) and torch.any(self.coef_quad != 0.0):
            if torch.all(torch.diagonal(self.coef_quad) > 0.0): #Diagonal full of positive terms
                case = 'quadratic_additive'
            elif torch.all(torch.abs(torch.diagonal(self.coef_quad)) > 0.0): #Diagonal full but mix of positive and negative terms
                case = 'quadratic_subtractive'
            elif torch.all(torch.diagonal(self.coef_quad) == 0.0): #no diagonal term
                case = 'cross_quadratic_only'
            else: #non-full diagonal
                case = 'realistic_photoswitch_photoinitiator'
        elif torch.all(torch.diagonal(self.coef_quad) == 0.0) and torch.any(self.coef_quad != 0.0) and torch.any(self.coef_lin != 0.0):
            case = 'realistic_photoswitch_photoinitiator'
        else:
            raise Exception('This mixed case is not supported.')
        return case
            


    def inv(self, energy_required, beam_idx):
        '''This function calculates the inverse of the effective dose with respect to the beam_idx-th beam.'''
        case = self.specialCaseClassification()        

        if (case == 'linear_additive') or (case == 'linear_subtractive'):
            # num_positive_beams = torch.sum(self.coef_lin > 0.0)
            # num_negative_beams = torch.sum(self.coef_lin < 0.0)

            sum_of_lin_positive_coef = torch.sum(self.coef_lin[self.coef_lin > 0.0]).abs()
            sum_of_lin_negative_coef = torch.sum(self.coef_lin[self.coef_lin < 0.0]).abs()

            if self.coef_lin[beam_idx] > 0.0:
                energy_required_per_beam = energy_required*(self.coef_lin[beam_idx].abs()/sum_of_lin_positive_coef)
            elif self.coef_lin[beam_idx] < 0.0:
                energy_required_per_beam = (-1.0*energy_required) #Invert the target, energy_required_per_beam is now non-positive
                energy_required_per_beam -= energy_required_per_beam.min() #then offset it until it is non-negative.
                energy_required_per_beam *= (self.coef_lin[beam_idx].abs()/sum_of_lin_negative_coef)
            else:
                energy_required_per_beam = torch.zeros_like(energy_required)

        elif (case == 'quadratic_additive') or (case == 'quadratic_subtractive'):

            # num_positive_beams = torch.sum(torch.diagonal(self.coef_quad) > 0.0)
            # num_negative_beams = torch.sum(torch.diagonal(self.coef_quad) < 0.0)
            diagonal_of_quad = torch.diagonal(self.coef_quad)

            sum_of_quad_positive_coef = torch.sum(diagonal_of_quad[diagonal_of_quad > 0.0]).abs()
            sum_of_quad_negative_coef = torch.sum(diagonal_of_quad[diagonal_of_quad < 0.0]).abs()

            if self.coef_quad[beam_idx,beam_idx] > 0.0:
                #The beam after squaring, will need to contribute its share. This ignore the cross-term coupling.
                #This assumed the beams are independent but contributing in the same direction.
                energy_required_per_beam = torch.sqrt(energy_required)*(self.coef_quad[beam_idx,beam_idx].abs()/sum_of_quad_positive_coef)
            elif self.coef_quad[beam_idx,beam_idx] < 0.0:
                energy_required_per_beam = (-1.0*energy_required) #Invert the target, energy_required_per_beam is now non-positive
                energy_required_per_beam -= energy_required_per_beam.min() #then offset it until it is non-negative.
                energy_required_per_beam = torch.sqrt(energy_required_per_beam)
                energy_required_per_beam *= (self.coef_quad[beam_idx,beam_idx].abs()/sum_of_quad_negative_coef)
            else:
                energy_required_per_beam = torch.zeros_like(energy_required)
        elif ((case == 'cross_quadratic_only') or (case == 'realistic_photoswitch_photoinitiator')) and self.num_beams == 2:
            #case where effective excitation is proportional to I_1*I_2
            energy_required_per_beam = torch.sqrt(energy_required)
        else:
            raise Exception(f'Case {case} is not supported.')

        self.logger.info(f'Inverting effective dose for beam {beam_idx}')
        self.logger.info(f'energy_required has min: {energy_required.min()} and max: {energy_required.max()}')
        self.logger.info(f'energy_required_per_beam has min: {energy_required_per_beam.min()} and max: {energy_required_per_beam.max()}')

        return energy_required_per_beam


    def __repr__(self):
        return (f'num_beams: {self.num_beams}, Linear coef: {self._coef_lin}, Quad coef: {self._coef_quad}')

if __name__ == '__main__':
    #Test with only 2 beams
    model = EffectiveDose(2)
    print(model)
    model.coef_lin = torch.tensor([1,-1])
    model.coef_quad = torch.zeros((2,2), dtype=torch.float32)

    dose_tuple = (torch.tensor([1,1,1], device = model.device, dtype=torch.float32),
                     torch.tensor([0,5,0], device = model.device, dtype=torch.float32),
                     )
    lin_term = model.linear(dose_tuple)
    quad_term = model.quadratic(dose_tuple)
    print(f'linear term: {lin_term}')
    print(f'quad term: {quad_term}')


    #Test with 3 beams
    model = EffectiveDose(3)
    model.coef_lin = torch.tensor([1,2,3])
    model.coef_quad = torch.tensor(
                        [[1,2,3],
                        [4,5,6],
                        [7,8,9]],
                        dtype=torch.float32)

    dose_tuple = (torch.tensor([1,1,1], device = model.device, dtype=torch.float32),
                     torch.tensor([0,5,0], device = model.device, dtype=torch.float32),
                     torch.tensor([3,0,0], device = model.device, dtype=torch.float32)
                     )
    lin_term = model.linear(dose_tuple)
    quad_term = model.quadratic(dose_tuple)
    print(f'linear term: {lin_term}')
    print(f'quad term: {quad_term}')


    #Test with 3 beams, with modified terms
    model.coef_lin = torch.tensor([0,0,0])
    model.coef_quad = torch.tensor(
                        [[0,0,0],
                        [0,2,0],
                        [0.5,0,0]],
                        dtype=torch.float32)

    lin_term = model.linear(dose_tuple)
    quad_term = model.quadratic(dose_tuple)
    print(f'linear term: {lin_term}')
    print(f'quad term: {quad_term}')