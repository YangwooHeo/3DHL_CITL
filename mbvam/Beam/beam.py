import torch
import logging
import mbvam.Geometry.resample

class Beam:
    def __init__(self, beam_config):    
        self.beam_config = beam_config #short
        self.logger = logging.getLogger(__name__)

        self._global_to_local_sampling_coord = None #This is the sampling coordinate grid that maps global grid to local grid. It is computed once and stored here.
        self._local_to_global_sampling_coord = None #This is the sampling coordinate grid that maps local grid to global grid. It is computed once and stored here.

    @property
    def local_coord_vec(self): #Get the local grid vector in local coordinates Xv, Yv, Zv. It does not contain information about its orientation relative to global grid.
        # This coordinate is centered around the focus in the medium, which is consistent with the absolute coordinate system defined in CoordinateArray and HoloBeamConfig.
        return [torch.as_tensor(self.beam_config.Xv, device=self.beam_config.device, dtype=self.beam_config.fdtype),
                torch.as_tensor(self.beam_config.Yv, device=self.beam_config.device, dtype=self.beam_config.fdtype),
                torch.as_tensor(self.beam_config.Zv, device=self.beam_config.device, dtype=self.beam_config.fdtype)] #may need to use .view(np.ndarray) so that torch.as_tensor properly accepts it
    
    @property
    def local_coord_grid(self): #Output the local ndgrid from Xv, Yv, Zv to serve as a reference grid
        X,Y,Z = torch.meshgrid(*self.local_coord_vec(), indexing='ij') #unpack the list
        return X,Y,Z
    
    
    def mapGlobalToLocal(self, vol_3d_glo, global_grid_vec, use_precomputed_sampling_coord=False, override_axis_angle=None):
        '''
        Map the volume voxel data from the provided global grid (in global coordinates) to local grid (in local coordinates).
        The sampling grid is an expression of local grid in global coordinates.
        The sampling grid is constructed according to the axis_angle and origin_glo parameter stored in holobeamconfig object.

        The original volume data is described in origin grid and is to be mapped to destination grid.
        Origin: global reference grid -->  Desintation: local grid.

        When use_precomputed_sampling_coord is True, the sampling coordinate is computed and stored in self._global_to_local_sampling_coord for later use.
        Subsequent calls to this function with use_precomputed_sampling_coord stay True will use the precomputed sampling coordinate.
        If the parameters of the grid transformation is changed, the precomputed sampling coordinate needs to be manually cleared by calling self.clearSamplingCoord().

        override_axis_angle: None or list of 4 floats, optional. If provided, local to global mapping will use this provided axis_angle instead of that in beam_config.
        This is to enable sub-beam to use a different axis_angle than the parent beam.
        This parameter is ignored if ((use_precomputed_sampling_coord is True) AND (if there exist a sampling coordinate grid stored)).
        '''

        output_shape = [self.beam_config.Nx, self.beam_config.Ny, self.beam_config.Nz]

        if (use_precomputed_sampling_coord==False) or (self._global_to_local_sampling_coord is None): #build sampling coordinate

            #When constructing the sampling coordinate grid, we construct a grid with a physical size of local grid, rotate it, then offset it by origin_glo.
            #Finally, we normalize the sampling coordinate with the half span of the global grid (because the grid_sample function expects the coordinate to be in [-1,1] range).

            #pre-scale and pre-translate
            pre_grid_scale = torch.as_tensor(self.beam_config.fov, device=self.beam_config.device)/2 #This scale the created grid to the size of the bounding box, as expressed in global coordinates.
            pre_translation = torch.as_tensor((0,0,0), device=self.beam_config.device, dtype=self.beam_config.fdtype) #pre_translation is zero
            
            #rotation
            axis_angle = self.beam_config.axis_angle if override_axis_angle is None else override_axis_angle #use the axis_angle in beam_config unless overridden

            #post-translation is expressed in unit that assumes the boundary of the volume is [-1,1] in all dimensions.
            post_translation = torch.as_tensor(self.beam_config.origin_glo, device=self.beam_config.device, dtype=self.beam_config.fdtype)
            
            #post-scale (normalization)
            global_grid_half_span_x = (torch.max(global_grid_vec[0])-torch.min(global_grid_vec[0]))/2
            global_grid_half_span_y = (torch.max(global_grid_vec[1])-torch.min(global_grid_vec[1]))/2
            global_grid_half_span_z = (torch.max(global_grid_vec[2])-torch.min(global_grid_vec[2]))/2

            normalization_factor = torch.as_tensor((global_grid_half_span_x, global_grid_half_span_y, global_grid_half_span_z), device=self.beam_config.device, dtype=self.beam_config.fdtype)

            post_grid_scale = 1/normalization_factor #post_grid_scale is the inverse of the normalization factor

            sampling_coord_5d = mbvam.Geometry.resample.constructResamplingGrid(
                                                            output_shape=output_shape,
                                                            pre_grid_scale=pre_grid_scale,
                                                            pre_translation=pre_translation,
                                                            axis_angle=axis_angle,
                                                            post_translation=post_translation,
                                                            post_grid_scale=post_grid_scale,
                                                            device=self.beam_config.device
                                                            )
            if use_precomputed_sampling_coord==True:
                self._global_to_local_sampling_coord = sampling_coord_5d

        else: #use precomputed sampling coordinate
            sampling_coord_5d = self._global_to_local_sampling_coord

        return mbvam.Geometry.resample.resampleVolume(vol_3d_glo, sampling_coord_5d, output_shape = output_shape, device=self.beam_config.device)

    
    def mapLocalToGlobal(self, vol_3d_loc, global_grid_vec, use_precomputed_sampling_coord=False, override_axis_angle = None):
        '''
        Map the volume voxel data from the provided local grid (in local coordinate) to the global grid (in global coordinates).
        The sampling grid is an expression of global reference grid in local coordinates.
        The sampling grid is constructed according to the axis_angle and origin_glo parameter stored in holobeamconfig object.

        The original volume data is described in origin grid and is to be mapped to destination grid.
        Origin: local grid -->  Desintation: global reference grid.

        When use_precomputed_sampling_coord is True, the sampling coordinate is computed and stored in self._local_to_global_sampling_coord for later use.
        Subsequent calls to this function with use_precomputed_sampling_coord stay True will use the precomputed sampling coordinate.
        If the parameters of the grid transformation is changed, the precomputed sampling coordinate needs to be manually cleared by calling self.clearSamplingCoord().

        override_axis_angle: None or list of 4 floats, optional. If provided, local to global mapping will use this provided axis_angle instead of that in beam_config.
        This is to enable sub-beam to use a different axis_angle than the parent beam.
        This parameter is ignored if ((use_precomputed_sampling_coord is True) AND (if there exist a sampling coordinate grid stored)).
        '''

        output_shape = [global_grid_vec[0].numel(), global_grid_vec[1].numel(), global_grid_vec[2].numel()]

        if (use_precomputed_sampling_coord==False) or (self._local_to_global_sampling_coord is None): #build sampling coordinate            
            
            #When constructing the sampling coordinate grid, we construct a grid with a physical size of the global grid, offset it by -origin_glo, then (inversely) rotate it.
            #Finally, we normalize the sampling coordinate with the half span of the local grid (because the grid_sample function expects the coordinate to be in [-1,1] range).

            #pre-scale and pre-translate
            global_grid_half_span_x = (torch.max(global_grid_vec[0])-torch.min(global_grid_vec[0]))/2
            global_grid_half_span_y = (torch.max(global_grid_vec[1])-torch.min(global_grid_vec[1]))/2
            global_grid_half_span_z = (torch.max(global_grid_vec[2])-torch.min(global_grid_vec[2]))/2

            pre_grid_scale = torch.as_tensor((global_grid_half_span_x, global_grid_half_span_y, global_grid_half_span_z), device=self.beam_config.device, dtype=self.beam_config.fdtype)
            
            pre_translation = -torch.as_tensor(self.beam_config.origin_glo, device=self.beam_config.device, dtype=self.beam_config.fdtype) #Pre translation is negated
            
            #rotation
            if override_axis_angle is None: #use the axis_angle in beam_config unless overridden
                axis_angle = [self.beam_config.axis_angle[0],
                            self.beam_config.axis_angle[1],
                            self.beam_config.axis_angle[2],
                            -self.beam_config.axis_angle[3]] #The axis is the eigenvector of the rotation matrix so it is unchanged. Angle is negated.
            else:
                axis_angle = [override_axis_angle[0],
                            override_axis_angle[1],
                            override_axis_angle[2],
                            -override_axis_angle[3]] #The axis is the eigenvector of the rotation matrix so it is unchanged. Angle is negated.

            #post-translation
            post_translation = torch.as_tensor((0,0,0), device=self.beam_config.device, dtype=self.beam_config.fdtype) #post_translation is zero

            #post-scale (normalization)
            post_grid_scale = 2/torch.as_tensor(self.beam_config.fov, device=self.beam_config.device, dtype=self.beam_config.fdtype)
            
            sampling_coord_5d = mbvam.Geometry.resample.constructResamplingGrid(
                                                            output_shape=output_shape,
                                                            pre_grid_scale=pre_grid_scale,
                                                            pre_translation=pre_translation,
                                                            axis_angle=axis_angle,
                                                            post_translation=post_translation,
                                                            post_grid_scale=post_grid_scale,
                                                            device=self.beam_config.device
                                                            )
            if use_precomputed_sampling_coord==True:
                self._local_to_global_sampling_coord = sampling_coord_5d

        else: #use precomputed sampling coordinate
            sampling_coord_5d = self._local_to_global_sampling_coord

        return mbvam.Geometry.resample.resampleVolume(vol_3d_loc, sampling_coord_5d, output_shape=output_shape, device=self.beam_config.device)


    def clearSamplingCoord(self):
        self._global_to_local_sampling_coord = None
        self._local_to_global_sampling_coord = None


    # def constructResamplingGrid(self, axis_angle, output_shape, pre_translation=0, pre_grid_scale=1, post_grid_scale=1, post_translation=0):
    #     ''' 
    #     A resampling step is performed to map the original grid to the transformed grid.
    #     T_pre(coordinate) = coordinate + pre_translation
    #     S_pre(coordinate) = coordinate * pre_grid_scale
    #     R(coordinate) = R_rot@coordinate
    #     S_post(coordinate) = coordinate * post_grid_scale
    #     T_post(coordinate) = coordinate + post_translation

    #     Both pre_translation and post_translation are expressed in unit that assumes the boundary of the original grid is [-1,1] in all dimensions.
    
    #     pre_grid_scale and post_grid_scale are the ratio of the physical sizes of the output relative to input, but they are applied before and after rotation.
    #     Scale > 1 resample on a grid that is larger than the original grid.
    #     '''
    #     rot_mat = mbvam.Geometry.resample.axang2rotmat(axis_angle, self.beam_config.device) # Get rotation matrix

    #     if torch.as_tensor(pre_translation).dim() <= 1:
    #         pre_translation = torch.as_tensor(pre_translation, device=self.beam_config.device).expand(3) #expand to a 3-element vector

    #     if torch.as_tensor(pre_grid_scale).dim() <= 1:
    #         pre_grid_scale = torch.as_tensor(pre_grid_scale, device=self.beam_config.device).expand(3) #expand to a 3-element vector

    #     if torch.as_tensor(post_grid_scale).dim() <= 1:
    #         post_grid_scale = torch.as_tensor(post_grid_scale, device=self.beam_config.device).expand(3) #expand to a 3-element vector

    #     if torch.as_tensor(post_translation).dim() <= 1:
    #         post_translation = torch.as_tensor(post_translation, device=self.beam_config.device).expand(3)
        
    #     # Transform the coordinate
    #     x = torch.linspace(-1, 1, output_shape[0], device=self.beam_config.device)+pre_translation[0] * pre_grid_scale[0]
    #     y = torch.linspace(-1, 1, output_shape[1], device=self.beam_config.device)+pre_translation[1] * pre_grid_scale[1] 
    #     z = torch.linspace(-1, 1, output_shape[2], device=self.beam_config.device)+pre_translation[2] * pre_grid_scale[2]
    #     X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
    #     coord = torch.vstack((X.ravel(), Y.ravel(), Z.ravel()))
    #     sampling_coord = torch.mm(rot_mat, coord) #rotate the coordinate, in (3,N) shape
    #     sampling_coord = sampling_coord*post_grid_scale[:,None] + post_translation[:,None] #apply post translation
    #     sampling_coord = sampling_coord.T #transpose to (N,3) shape

    #     return sampling_coord[None,None,None,:,:] #return a 5D tensor in (N,Ch,Z,Y,X) order
    

    '''
        #Alternate implementation of rotation. Not tested.
        Xr = rot_mat[0,0]*X + rot_mat[0,1]*Y + rot_mat[0,2]*Z + post_translation[0]
        Yr = rot_mat[1,0]*X + rot_mat[1,1]*Y + rot_mat[1,2]*Z + post_translation[1]
        Zr = rot_mat[2,0]*X + rot_mat[2,1]*Y + rot_mat[2,2]*Z + post_translation[2]
        Xr = torch.permute(Xr, (2,1,0))
        Yr = torch.permute(Yr, (2,1,0))
        Zr = torch.permute(Zr, (2,1,0))
        rotated_vol = torch.nn.functional.grid_sample(scalar_field_5d, torch.stack((Xr,Yr,Zr), dim=3)[None,...], mode=interp_method, padding_mode='zeros', align_corners=True)
        rotated_vol = torch.squeeze(rotated_vol)
        rotated_vol = torch.permute(rotated_vol, (2,1,0))
        '''

if __name__ == '__main__':
    #Testing for general Beam class
    import mbvam.Beam
    import mbvam.Optim
    import math
    beam_config = mbvam.Beam.holobeamconfig.HoloBeamConfig()
    beam_config.axis_angle = [0,0,1,math.pi/4]

    beam_1 = mbvam.Beam.beam.Beam(beam_config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    #test mapGlobalToLocal
    random_tensor_shape = (125,125,125)
    random_tensor = torch.randn(random_tensor_shape, device=device)
    global_vec = [torch.linspace(-1, 1, random_tensor_shape[0], device=device),
    torch.linspace(-1, 1, random_tensor_shape[1], device=device),
    torch.linspace(-1, 1, random_tensor_shape[2], device=device)]
    
    local_tensor = beam_1.mapGlobalToLocal(random_tensor, global_vec)
    print(f'Global tensor shape: {random_tensor.shape}, dtype:{random_tensor.dtype}')
    print(f'Mapped local tensor shape: {local_tensor.shape}, dtype:{local_tensor.dtype}')

    #test mapLocalToGlobal
    mapped_global_tensor = beam_1.mapLocalToGlobal(local_tensor, global_vec)
    print(f'Local tensor shape: {local_tensor.shape}, dtype:{local_tensor.dtype}')
    print(f'Mapped global tensor shape: {mapped_global_tensor.shape}, dtype:{mapped_global_tensor.dtype}')


    #test mapGlobalToLocal with precomputed sampling coordinate
    local_tensor_1 = beam_1.mapGlobalToLocal(random_tensor, global_vec, use_precomputed_sampling_coord=True)
    beam_1.beam_config.axis_angle = [0,0,1,0]
    local_tensor_2 = beam_1.mapGlobalToLocal(random_tensor, global_vec, use_precomputed_sampling_coord=True)
    #test if the tensors are the same
    eq = torch.equal(local_tensor_1, local_tensor_2)
    print(f'The equality test between tensor 1 and 2: {eq}')
    assert eq, 'Tensor 1 and 2 should be the same but they are different.'
    local_tensor_3 = beam_1.mapGlobalToLocal(random_tensor, global_vec, use_precomputed_sampling_coord=False)
    eq = torch.equal(local_tensor_1, local_tensor_3)
    print(f'The equality test between tensor 1 and 3: {eq}')
    assert not eq, 'Tensor 1 and 3 should be different but they are the same.'

    #test mapLocalToGlobal with precomputed sampling coordinate
    beam_1.beam_config.axis_angle = [0,0,1,math.pi/4] #reset the axis_angle
    mapped_global_tensor_1 = beam_1.mapLocalToGlobal(local_tensor, global_vec, use_precomputed_sampling_coord=True)
    beam_1.beam_config.axis_angle = [0,0,1,0]
    mapped_global_tensor_2 = beam_1.mapLocalToGlobal(local_tensor, global_vec, use_precomputed_sampling_coord=True)
    #test if the tensors are the same
    eq = torch.equal(mapped_global_tensor_1, mapped_global_tensor_2)
    print(f'The equality test between tensor 1 and 2: {eq}')
    assert eq, 'Tensor 1 and 2 should be the same but they are different.'
    mapped_global_tensor_3 = beam_1.mapLocalToGlobal(local_tensor, global_vec, use_precomputed_sampling_coord=False)
    eq = torch.equal(mapped_global_tensor_1, mapped_global_tensor_3)
    print(f'The equality test between tensor 1 and 3: {eq}')
    assert not eq, 'Tensor 1 and 3 should be different but they are the same.'