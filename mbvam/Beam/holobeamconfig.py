import numpy as np
import torch
import warnings as warning
from dataclasses import dataclass
import mbvam.Geometry.coordinate
from copy import deepcopy

class HoloBeamConfig:

    def __init__(self):
        '''
        This class stores the configuration of the holographic beam.
        Configuration files should be a python file that modifies the attributes of this class.
        This class is all implemented in numpy, with the only exception of defining pyTorch device and data type for creating tensors in Beam objects.
        '''
        # pytorch params used in Beam class
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.fdtype = torch.float32 #float16 is not supported yet in transfer function calculation. float32 is the default for pytorch.
        self.cdtype = torch.complex64
        self.grad_checkpoint = False #This apply gradient checkpoint in the sub-beam level. The checkpoint at optimization level is in OptimConfig.

        # Independent params
        self.binningFactor = 8
        self.croppingFactor = 1
        
        # System config
        # self.verbose = True #Deprecated, now use log level.
        self.lambda_ = 0.473e-6  #[m], Wavelength of the light
        self.focal_SLM = 0.016 #[m], Effective focal length of the projection system after SLM
        self.psSLM_physical = 8e-6 #[m], Physical pixel size of the SLM
        self.Nx_physical = 1920 #[px], Number of pixels in the x-direction of the SLM
        self.Ny_physical = 1200 #[px], Number of pixels in the y-direction of the SLM
        self.axial_fov_aspect_ratio = 1 #Aspect ratio of the FOV. This is the ratio of the axial FOV (z) to the lateral FOV (x).
        self.z_plane_sampling_rate = 1 #[px], Set number of voxel per PSF in the z-direction. When sampling rate is 1, Nz = FOV_z/paraxial_res_z
        self.amplitude_profile_type = 'flat_top' #'flat_top' or 'gaussian'
        self.gaussian_beam_waist = 0.0063188 #[m], measured beam waist of the Gaussian beam. Blue: 0.0063188. UV: 0.0038708. Ignored for flat top.
        # self.totalPower = 1 #[W], Total power of the light. This should be an optimization variable
        
        # Rotation and translation
        '''Specified as axis-angle representation. 
        First 3 elements is the rotation axis (Euler axis), and the last element is rotation angle in rad.
        The rotation matrix describe the transformation of global coordinates to local coordinates.
        This also means the columns of rotation matrix is the local unit vectors in global coordinates.'''
        self.axis_angle = [1, 0, 0, 0*np.pi/4]
        self.origin_glo = [0, 0, 0] #origin of this beam relative to the global reference grid. Expressed in global grid coordinates.
        
        # Propagation kernels
        # self.kernel_selection = 'H_as' #'H_fresnel', 'H_as', 'H_eff'. This select the default kernel to be used in propagation steps.

        self.patterning_depth = 1.5e-3 #[m], desired patterning depth beneath the quartz-resin interface
        self.cuvette_thickness = 1.25e-3 #[m], thickness of the cuvette

        self.material_stack = MaterialStack([MaterialLayer(name='air', interface_z=-np.inf, n=1, alpha=0),  #Alpha of air should be zero.#
                                            MaterialLayer(name='quartz', interface_z=-self.patterning_depth-self.cuvette_thickness, n=1, alpha=0), #at 473nm
                                            MaterialLayer(name='resin', interface_z=-self.patterning_depth, n=1, alpha=0)]) #index at 473nm for TEGDMA/BPAGDMA 50/50wt%, alpha is arbitrary
        
    @property
    def psSLM(self): #[m], logical SLM pixel dimensions
        return self.psSLM_physical * self.binningFactor
    
    @property
    def Nx(self):   #int, Number of pixels in X direction, set to be power of 2 for efficient FFT
        value = round(self.Nx_physical / self.binningFactor / self.croppingFactor)
        if np.mod(value,2):
            warning.warn('Nx is not even. Mask centering is not guaranteed!')
        return value
   
    @property
    def Ny(self):   #int, Number of pixels in Y direction, set to be power of 2 for efficient FF;
        value = round(self.Ny_physical / self.binningFactor / self.croppingFactor)
        if np.mod(value,2):
            warning.warn('Nx is not even. Mask centering is not guaranteed!')
        return value
    
    @property
    def Nz(self):   #int, Number of pixels in Z direction, set such that the aspect ratio of of the FOV_x:FOV_z is 1:1
        return round(self.fov[2]*self.z_plane_sampling_rate/self.getAbbeResZInMedium(-1)) #use z resolution of the last medium

    #================= Paraxial parameters =================
    @property
    def paraxial_NA_x(self): #paraxial NA
        return self.psSLM*self.Nx/(2*self.focal_SLM)
    
    @property
    def paraxial_NA_y(self): #paraxial NA
        return self.psSLM*self.Ny/(2*self.focal_SLM)

    @property
    def paraxial_res_x(self): #[m], Resolution at target volume along X direction, at paraxial limit
        return self.lambda_/(2*self.paraxial_NA_x)

    @property
    def paraxial_res_y(self): #[m], Resolution at target volume along Y direction, at paraxial limit
       return self.lambda_/(2*self.paraxial_NA_y)
    
    @property
    def paraxial_res_z(self): #[m], Resolution at target volume along Z direction, at paraxial limit
        return self.getParaxialResZInMedium(0) #use paraxial NA
        # return self.abbe_res_z #use definition of NA

    def getParaxialResZInMedium(self, layer_index=0): #A function to query the voxel size at a specific layer, indicated by the index of the material layer.
        return 2*self.material_stack[layer_index].n*self.lambda_/(self.paraxial_NA_x**2) #use paraxial NA
    
    @property
    def max_fov_z(self): #[m] scalar, full FOV (including both sides) in z-direction in air. ONLY RELEVANT IN SINGLE BEAM CONFIG.
       #This use paraxial assumption
       return 2*self.lambda_*(self.focal_SLM**2)/((self.psSLM**2)*self.Nx)
    
    #================= Non-paraxial parameters =================
    @property
    def NA_x(self): #Definition of NA
        return np.sin(np.arctan(self.paraxial_NA_x))

    @property
    def NA_y(self): #Definition of NA
        return np.sin(np.arctan(self.paraxial_NA_y))

    @property
    def abbe_res_x(self): #[m] scalar, full size PSF in x-direction as in Abbe resolution.
        return self.lambda_/(2*self.NA_x)

    @property
    def abbe_res_y(self): #[m] scalar, full size PSF in y-direction as in Abbe resolution.
        return self.lambda_/(2*self.NA_y)

    @property
    def abbe_res_z(self): #[m] scalar, full size PSF in z-direction as in Abbe resolution. Depends on the refractive index of the medium, which is assumed to be air.
        return self.getAbbeResZInMedium(layer_index=0) #using refractive index of first material (air)
    
    def getAbbeResZInMedium(self, layer_index=0):
        return 2*self.material_stack[layer_index].n*self.lambda_/(self.NA_x**2) # expressed as 2*n*lambda/(NA_x**2)

    
    @property
    def fov(self): #Physical dimension of the beam in the first material (air)
        # The aspect ratio of fov_z:fov_x is set by the setting self.axial_fov_aspect_ratio
        return mbvam.Geometry.coordinate.CoordinateArray([self.abbe_res_x*self.Nx, self.abbe_res_y*self.Ny, self.abbe_res_x*self.Nx*self.axial_fov_aspect_ratio],
                                                        in_free_space=False,
                                                        physical_unit='m',
                                                        frame='local')
        
    @property
    def Xv(self):
        return mbvam.Geometry.coordinate.CoordinateArray(np.linspace(-self.fov[0]/2, self.fov[0]/2, self.Nx),
                                                        in_free_space=False,
                                                        physical_unit='m',
                                                        frame='local')

    @property
    def Yv(self):
        return mbvam.Geometry.coordinate.CoordinateArray(np.linspace(-self.fov[1]/2, self.fov[1]/2, self.Ny),
                                                        in_free_space=False,
                                                        physical_unit='m',
                                                        frame='local')
    @property
    def Zv(self):
        return mbvam.Geometry.coordinate.CoordinateArray(np.linspace(-self.fov[2]/2, self.fov[2]/2, self.Nz),
                                                        in_free_space=False,
                                                        physical_unit='m',
                                                        frame='local')

    @property
    def free_space_focus_z(self): #free space focus relative to current z=0 which is inside medium
        medium_focus = mbvam.Geometry.coordinate.CoordinateArray(np.array(0.0), in_free_space=False, physical_unit='m', frame='local')
        return self.material_stack.getFreeSpaceZPositions(medium_focus) #This return CoordinateArray with in_free_space=True
    
    #================= Other parameters =================
    # @property
    # def source(self): #E-field amplitude of each pixel, [sqrt(W/m^2)]=[W^(1/2)/m]
    #     return np.sqrt(self.totalPower/(self.Nx*self.Ny*self.psSLM**2)) * np.ones((self.Nx, self.Ny))
    
    @property
    def voxelization_res(self): #number of samples used in voxelization
        return max(self.Nx, self.Ny)
    
    
    def __repr__(self):
        return f'''
                device={self.device},
                fdtype={self.fdtype},
                cdtype={self.cdtype},
                binningFactor={self.binningFactor},
                croppingFactor={self.croppingFactor},
                lambda_={self.lambda_}, 
                focal_SLM={self.focal_SLM}, 
                psSLM_physical={self.psSLM_physical},
                Nx_physical={self.Nx_physical},
                Ny_physical={self.Ny_physical},
                axial_fov_aspect_ratio={self.axial_fov_aspect_ratio},
                z_plane_sampling_rate={self.z_plane_sampling_rate},
                axis_angle={self.axis_angle}, 
                origin_glo={self.origin_glo}, 
                patterning_depth={self.patterning_depth}, 
                cuvette_thickness={self.cuvette_thickness}, 
                material_stack={self.material_stack}, 
                psSLM={self.psSLM}, 
                Nx={self.Nx}, 
                Ny={self.Ny}, 
                Nz={self.Nz}, 
                paraxial_NA_x={self.paraxial_NA_x},
                paraxial_NA_y={self.paraxial_NA_y},
                paraxial_res_x={self.paraxial_res_x},
                paraxial_res_y={self.paraxial_res_y},
                paraxial_res_z={self.paraxial_res_z},
                max_fov_z={self.max_fov_z},
                NA_x={self.NA_x},
                NA_y={self.NA_y},
                abbe_res_x={self.abbe_res_x},
                abbe_res_y={self.abbe_res_y},
                abbe_res_z={self.abbe_res_z},
                fov={self.fov},
                free_space_focus_z={self.free_space_focus_z},
                voxelization_res={self.voxelization_res}
                '''
    
@dataclass
class MaterialLayer:
    name: str
    interface_z: float
    n: float
    alpha: float

    
class MaterialStack(list):
    '''
    This class is a list of MaterialLayer objects. It is used to tabulate the material stack.
    The coordinate system is always fixed. z=0 at set at the focus of the beam in the last medium.

    CoordinateArray.in_free_space is used to indicate whether material is inserted into the beam.
    True means no material is inserted. The coordinate represent the corresponding beam position in air.
    False means the stack of material is inserted. The coordinate represent the corresponding beam position in respective media.
    '''
    def __init__(self, material_layer_list:list):
        super().__init__(material_layer_list)
        self.check_params()

    @property
    def num_interfaces(self): #Number of interfaces in the material stack. This is the number of layers minus 1.
        return len(self) - 1 #0 for no interface 

    def check_params(self):
        interface_z_array = np.asarray([layer.interface_z for layer in self])
        if not np.all(np.diff(interface_z_array) > 0):
            raise ValueError("Material stack interface positions not in ascending order!")
            
        if self[0].n != 1:
            raise ValueError("First material does not have unit refractive index!")
            
        if self[0].alpha != 0:
            raise ValueError("First material has non-zero attenuation!")
            

    def getRefractedZPositions(self, z_query):
        '''
        Convert free-space z-position to corresponding refracted focal z-position in media.
        For each index interface, push/pull the points located after the
        interface depending on index ratio. Push for going into increased index medium
        Pull for going into decreased-index medium.
        Iterate for different medium to reach final location. Similar idea to recursion.

        Input: ndarray or mbvam.Geometry.coordinate.CoordinateArray of free space z-positions
        Output: ndarray or mbvam.Geometry.coordinate.CoordinateArray of refracted z-positions
        '''
        
        z_query = np.atleast_1d(deepcopy(z_query)).astype(float) #Make a copy of the input to avoid modifying the input in-place. Guard against integer input (e.g. 0)

        for i in range(self.num_interfaces):
            interface_z = self[i+1].interface_z #the location of the interface
            n_ratio = self[i+1].n / self[i].n
            set_affected = z_query > interface_z
            z_query[set_affected] = interface_z + (z_query[set_affected] - interface_z)*n_ratio            

        if isinstance(z_query, mbvam.Geometry.coordinate.CoordinateArray): #Add additional handling if the input is a CoordinateArray object.
            if z_query.in_free_space == False:
                raise ValueError('z_query is already in refracted CoordinateArray. No need to convert.')
            else:
                z_query.in_free_space = False

        return z_query
    
    
    def getFreeSpaceZPositions(self, z_query):
        '''
        Convert refracted focal z-position in media back to corresponding free-space z-position.
        It is the reverse process of getRefractedZPositions.
        The only difference is that the iterator now counts back, and that the n_ratio is applied reciprocally.

        Input: ndarray or CoordinateArray of refracted z-positions
        Output: ndarray or CoordinateArray of free space z-positions
        '''
        z_query = np.atleast_1d(deepcopy(z_query)).astype(float) #Make a copy of the input to avoid modifying the input in-place. Guard against integer input (e.g. 0)

        for i in range(self.num_interfaces, 0, -1):
            interface_z = self[i].interface_z
            n_ratio = self[i].n / self[i-1].n
            set_affected = z_query > interface_z
            z_query[set_affected] = interface_z + (z_query[set_affected] - interface_z)/n_ratio

        if isinstance(z_query, mbvam.Geometry.coordinate.CoordinateArray):
            if z_query.in_free_space == True:
                raise ValueError('z_query is already in free space CoordinateArray. No need to convert.')
            else:
                z_query.in_free_space = True

        return z_query


if __name__ == "__main__":
    config = HoloBeamConfig()
    print(f"Number of pixels in X: {config.Nx}")
    print(f"Number of pixels in Y: {config.Ny}")
    print(f"Paraxial resolution in X: {config.paraxial_res_x}")
    print(f"Paraxial resolution in Y: {config.paraxial_res_y}")
    print(f'Abbe resolution in Z: {config.abbe_res_z}')
    print(f'Abbe resolution in Z layer 0: {config.getAbbeResZInMedium(0)}')
    print(f'Abbe resolution in Z layer 1: {config.getAbbeResZInMedium(1)}')
    print(f'Abbe resolution in Z layer 2: {config.getAbbeResZInMedium(2)}')
    print(f"Maximum field of view in Z: {config.max_fov_z}")
    print(f"Paraxial numerical aperture in X: {config.paraxial_NA_x}")
    print(f"Numerical aperture in X: {config.NA_x}")
    print(f"Source: {config.source}")
    print(f"Voxelization resolution: {config.voxelization_res}")
    print(f"Material stack: {config.material_stack}")
    print(f"Free space focus z: {config.free_space_focus_z}")
    print(f"Free space focus z (CoordinateArray input): {config.material_stack.getFreeSpaceZPositions(mbvam.Geometry.coordinate.CoordinateArray(0, in_free_space=False))}")
    print(f"Free space focus z (integer input): {config.material_stack.getFreeSpaceZPositions(0)}")
    print(f"Patterning depth: {config.patterning_depth}")
    print(f"Cuvette thickness: {config.cuvette_thickness}")

    # Test manipulation of CoordinateArray
    x = mbvam.Geometry.coordinate.CoordinateArray(np.array([0, 1, 2, 3, 4, 5]), in_free_space=True)
    print(x.size)
    print(x.shape)
    print(x*2)

    # Test MaterialStack
    print(config.material_stack[1].name)
    print(config.material_stack[1].interface_z)
    print(config.material_stack[1].n)
    print(config.material_stack[1].alpha)
    print(config.material_stack.num_interfaces)
    print(config.material_stack.getRefractedZPositions(np.array([0, 1, 2, 3, 4, 5])))
    print(config.material_stack.getFreeSpaceZPositions(np.array([0, 1, 2, 3, 4, 5])))
    print(config.material_stack.getRefractedZPositions(mbvam.Geometry.coordinate.CoordinateArray(np.array([0, 1, 2, 3, 4, 5]), in_free_space=True)))
    print(config.material_stack.getFreeSpaceZPositions(mbvam.Geometry.coordinate.CoordinateArray(np.array([0, 1, 2, 3, 4, 5]), in_free_space=False)))

    if test_for_error := False:
        print(config.material_stack.getRefractedZPositions(mbvam.Geometry.coordinate.CoordinateArray(np.array([0, 1, 2, 3, 4, 5]), in_free_space=False))) #These test will intentionally cause errors.
        print(config.material_stack.getFreeSpaceZPositions(mbvam.Geometry.coordinate.CoordinateArray(np.array([0, 1, 2, 3, 4, 5]), in_free_space=True))) #These test will intentionally cause errors.
    
    if stack_not_in_order := False:
        config.material_stack = MaterialStack([MaterialLayer(name='air', interface_z=1, n=1, alpha=0), #Alpha of air should be zero.
                                            MaterialLayer(name='quartz', interface_z=2, n=1.4639, alpha=0), #at 473nm
                                            MaterialLayer(name='resin', interface_z=-3, n=1.515979, alpha=0)])
    else:
        config.material_stack = MaterialStack([MaterialLayer(name='air', interface_z=1, n=1, alpha=0), #Alpha of air should be zero.
                                            MaterialLayer(name='quartz', interface_z=2, n=1.4639, alpha=0), #at 473nm
                                            MaterialLayer(name='resin', interface_z=3, n=1.515979, alpha=0)])



