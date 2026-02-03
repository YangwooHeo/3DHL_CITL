import torch
import mbvam.Geometry.resample
import vedo
import numpy as np

class TargetObject():
    def __init__(self, device:torch.device, dtype:torch.dtype, coord_vec:tuple[torch.Tensor], array:torch.Tensor, eps:torch.Tensor=0, w:torch.Tensor=1, store_coord_grid:bool=False) -> None:
        '''
        This class store the target array (real-valued), its tolerance, weighting and coordinate system.
        
        #TODO: make all variable private and add getter and setter. This helps to ensure variables are always tensor and on the right device.
        '''
        self.device = device
        self.dtype = dtype
        self.coord_vec = tuple(torch.as_tensor(cv, device=self.device, dtype=self.dtype) for cv in coord_vec)
        
        self.array = torch.as_tensor(array,device=self.device, dtype=self.dtype)
        self.eps = torch.as_tensor(eps,device=self.device, dtype=self.dtype)
        self.w = torch.as_tensor(w,device=self.device, dtype=self.dtype)
        self.store_coord_grid = store_coord_grid

        if store_coord_grid:
            self.storeCoordGrid()
            print('coord_grid is built and stored in target.')

    @property
    def coord_center(self):
        return torch.as_tensor(
                ((self.coord_vec[0].max()+self.coord_vec[0].min())/2.0,
                (self.coord_vec[1].max()+self.coord_vec[1].min())/2.0,
                (self.coord_vec[2].max()+self.coord_vec[2].min())/2.0),
                device=self.device)
    
    @property
    def coord_half_span(self):
        return torch.as_tensor(((self.coord_vec[0].max()-self.coord_vec[0].min())/2.0,
                (self.coord_vec[1].max()-self.coord_vec[1].min())/2.0,
                (self.coord_vec[2].max()-self.coord_vec[2].min())/2.0),
                device=self.device)
    
    @property
    def dvol(self): # This function returns the volume of each voxel in the target array.
        return torch.prod(2*self.coord_half_span)/self.array.numel() #TAG: MIXED_PRECISION
    
    @property
    def coord_grid(self): #getter
        #If grid is stored, return it. Otherwise, build it as-needed.
        if self.store_coord_grid:
            return self._coord_grid
        else:
            return self.buildCoordGrid()
    
    @torch.inference_mode()
    def buildCoordGrid(self):
        return torch.meshgrid(*self.coord_vec, indexing='ij')

    def storeCoordGrid(self):
        self._coord_grid = self.buildCoordGrid()

    def clearCoordGrid(self):
        self._coord_grid = None

    @torch.inference_mode()
    def centerCoord(self):
        '''
        Center the coordinate system to the center of the array
        '''
        center = self.coord_center #evaluate coordinate center once.
        self.coord_vec = (self.coord_vec[0]-center[0], self.coord_vec[1]-center[1], self.coord_vec[2]-center[2])

        if self.store_coord_grid: # Rebuild coord_grid since it is updated
            self.storeCoordGrid()

    def pad(self, padding_mode='equal_voxel'):
        '''
        Pad zeros to the target array.
        padding_mode:
            'equal_voxel' pad the array has equal size in all dimensions. Unless voxel size is isotropic, enforcing equal number of voxel may gives a grid with different physical size in each dimension.
            'equal_length' pad the array such that coord_vec half span is equal in all dimensions. Note that number of voxels may not be equal in all dimensions afterwards.
        '''
        if padding_mode is None:
            pass
        elif padding_mode == 'equal_voxel':
            #Determine the dimension with maximum size and pad zeros to make all dimensions equal.
            largest_dim = torch.argmax(torch.as_tensor(self.array.shape))
            largest_dim_size = self.array.shape[largest_dim]
            pad_size = (largest_dim_size - torch.as_tensor(self.array.shape, dtype=torch.float32))/2 #ordered as x,y,z, without rounding
            pad_size_before = torch.floor(pad_size).to(torch.int32) #round down
            pad_size_after = torch.ceil(pad_size).to(torch.int32) #round up
            
            #Pad zeros such that the grid has equal size in all dimensions. The pad size should be listed from last dimension to first.
            self.array = torch.nn.functional.pad(self.array,
                                                (pad_size_before[2], pad_size_after[2], pad_size_before[1], pad_size_after[1], pad_size_before[0], pad_size_after[0]),
                                                mode='constant',
                                                value=0)
            
            #Check if all dimensions are now equal, if not raise error.
            if not torch.all(torch.as_tensor(self.array.shape) == largest_dim_size):
                raise RuntimeError('Padding of target array does not result in equal dimensions.')
            
            if torch.as_tensor(self.w).dim() == 3:
                raise NotImplementedError('Padding of weight is not well defined.'
                                            'Please use scalar weight in preprocessing and assign weight array afterwards explicitly.')
            if torch.as_tensor(self.eps).dim() == 3:
                raise NotImplementedError('Padding of tolerance is not well defined.'
                                            'Please use scalar tolerance in preprocessing and assign tolerance array afterwards explicitly.')

            #Update coord_vec to reflect the new size of the array.
            coord_center = self.coord_center

            # Implementation assuming isotropic voxel
            # largest_half_span = self.coord_half_span[largest_dim]
            # self.coord_vec = (torch.linspace(coord_center[0]-largest_half_span, coord_center[0]+largest_half_span, largest_dim_size, device=self.device),
            #                 torch.linspace(coord_center[1]-largest_half_span, coord_center[1]+largest_half_span, largest_dim_size, device=self.device),
            #                 torch.linspace(coord_center[2]-largest_half_span, coord_center[2]+largest_half_span, largest_dim_size, device=self.device))

            # Implementation that do not assume isotropic voxel
            original_half_span = self.coord_half_span
            discretization = torch.as_tensor((self.coord_vec[0][1]-self.coord_vec[0][0], self.coord_vec[1][1]-self.coord_vec[1][0], self.coord_vec[2][1]-self.coord_vec[2][0])).abs()
            self.coord_vec = [
                torch.linspace(coord_center[ind]-original_half_span[ind]-pad_size_before[ind]*discretization[ind],
                            coord_center[ind]+original_half_span[ind]+pad_size_after[ind]*discretization[ind],
                            largest_dim_size,
                            device=self.device)
                              for ind in range(3)]
            self.coord_vec = tuple(self.coord_vec)
                              
        elif padding_mode == 'equal_length':
            raise NotImplementedError('Padding to equal length is not implemented yet.')
        else:
            raise ValueError('padding_mode should be None, "equal_voxel" or "equal_length".')

    @torch.inference_mode()
    def scaleCoordSpan(self, target_sizing_mode:str=None, target_domain_size:tuple[float, float, float]=None):
        '''
        Scale the coordinate to the prescribed size.

        Parameters
        ----------
        target_sizing_mode: None, 'fit' or 'stretch'

        target_domain_size: [x,y,z] size of the target domain in [m]
        
        '''
        if target_sizing_mode is None or target_domain_size is None:
            pass
        else:
            full_span = 2*self.coord_half_span #evalutate span once.
            # scale_factor = torch.tensor((target_domain_size[0]/full_span[0], target_domain_size[1]/full_span[1], target_domain_size[2]/full_span[2]))
            scale_factor = torch.tensor(target_domain_size, device=self.device)/full_span
            
            if target_sizing_mode == 'fit':
                scale_factor = scale_factor.min() #Scale by the minimum factor to fit the domain size. Maintain aspect ratio.
                self.coord_vec = (self.coord_vec[0]*scale_factor, self.coord_vec[1]*scale_factor, self.coord_vec[2]*scale_factor)
            elif target_sizing_mode == 'fill': #Scale by the maximum factor to fill the domain. Some dimensions may exceed target_domain_size. Maintain aspect ratio. 
                scale_factor = scale_factor.max()
                self.coord_vec = (self.coord_vec[0]*scale_factor, self.coord_vec[1]*scale_factor, self.coord_vec[2]*scale_factor)
            elif target_sizing_mode == 'stretch':
                self.coord_vec = (self.coord_vec[0]*scale_factor[0], self.coord_vec[1]*scale_factor[1], self.coord_vec[2]*scale_factor[2])
            else:
                raise ValueError('target_sizing_mode should be None, "fit" or "stretch".')
            
            if self.store_coord_grid: # Rebuild coord_grid since it is updated
                self.storeCoordGrid()


    @torch.inference_mode()
    def rotateTranslateScaleDataInGrid(self,
                                    axis_angle:tuple[float, float, float, float]=(1.0, 1.0, 1.0, 0.0),
                                    target_offset:tuple[float, float, float]=(0.0, 0.0, 0.0),
                                    target_scale_in_domain:tuple[float, float, float]=(1.0, 1.0, 1.0)
                                    ):
        '''
        Rotate the target and translate the data inside the array, tolerance and weight in one resampling step.
        This function modify the target array, tolereance and weight but not the coordinate vector and coordinate grid.
        Implemented in a similar way as the mapGlobalToLocal and mapLocalToGlobal in Beam.py.

        Parameters
        ----------
        target_rotation: [x,y,z,w] axis-angle representation of the rotation.
        target_offset: [x,y,z] offset of the target in the global coordinate system.

        Since the rotation and offset are applied on the object itself, the sampling grid is transformed in the opposite direction.
        '''
        if axis_angle[3] == 0.0 and target_offset==(0.0, 0.0, 0.0) and target_scale_in_domain==(1.0, 1.0, 1.0): #If no rotation, translation, nor scaling, then skip.
            pass
        else:
            output_shape = self.array.shape
            pre_grid_scale = self.coord_half_span #This construct the sampling grid with a physical size of the original grid.            
            pre_translation = 0.0
            axis_angle = (axis_angle[0],
                        axis_angle[1],
                        axis_angle[2],
                        -axis_angle[3]) #The angle is negated because the rotation is applied on the object itself.
            post_translation = tuple(-offset_axis for offset_axis in target_offset)
            post_grid_scale = 1/(pre_grid_scale*torch.as_tensor(target_scale_in_domain, device=self.device)) #This allow the sampling to be done in -1 to +1 range.

            sampling_coord_5d = mbvam.Geometry.resample.constructResamplingGrid(
                                                output_shape=output_shape,
                                                pre_grid_scale=pre_grid_scale,
                                                pre_translation=pre_translation,
                                                axis_angle=axis_angle,
                                                post_translation=post_translation,
                                                post_grid_scale=post_grid_scale,
                                                device = torch.device('cpu') #device=self.device
                                                ).detach().to(self.device) #This construction process of this grid leave huge memory use even after completion. Perform this process in CPU.
            
            #TODO: Establish a larger grid to accommodate the entire transformed object to avoid clipping. 
            # Another solution to clipping is to transform the mesh before voxelization, but the z-axis is defined on the transformed mesh.
            # if rebound_coord: 
            #     ...
                #get the vertices of the bounding box of the object. 
                #back transform the vertices (translate + rotate, as in reverse of rotate + translate), determine the maximum x,y,z values of these vertices
                #establish a new coord_vec (through output_shape and pre_grid_scale) that can accommodate the transformed object
                #resample the object to the new coord_vec

            self.array = mbvam.Geometry.resample.resampleVolume(self.array, sampling_coord_5d, output_shape=output_shape, device=self.device)
            if torch.as_tensor(self.w).dim() == 3:
                # self.w = mbvam.Geometry.resample.resampleVolume(self.w, sampling_coord_5d, output_shape=output_shape, device=self.device)
                raise NotImplementedError('Resampling of weight is not well defined because of potential out-of-bound region.'
                                        'Please use scalar weight in preprocessing and assign weight array afterwards explicitly.')
            if torch.as_tensor(self.eps).dim() == 3:
                # self.eps = mbvam.Geometry.resample.resampleVolume(self.eps, sampling_coord_5d, output_shape=output_shape, device=self.device)
                raise NotImplementedError('Resampling of tolerance is not well defined because of potential out-of-bound region.'
                                           'Please use scalar tolerance in preprocessing and assign tolerance array afterwards explicitly.')

    def show(self, **kwargs): #show the target array
        target_array_np = self.array.to('cpu').numpy()
        if target_array_np.dtype == np.float16:
            target_array_np = target_array_np.astype(np.float32) #VTK volume does not support float16.
        vol = vedo.Volume(target_array_np,mode=0)

        kwargs.setdefault('axes', 9) #default with bounding box drawn: 9. https://vedo.embl.es/docs/vedo/plotter.html#Plotter
        kwargs.setdefault('bg', 'black') #default black background
        vedo.applications.RayCastPlotter(vol, **kwargs).show(viewup="x")    


    
if __name__ == '__main__':
    ...