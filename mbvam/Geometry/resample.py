import torch

def axang2rotmat(axis_angle, device, dtype=torch.float32): 
    '''
    Obtain a rotation matrix from axis-angle representation.
    axis_angle is a 4-element vector [x,y,z,theta]
    Rodrigues rotation formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    '''
    axis_angle = torch.as_tensor(axis_angle, device=device, dtype=dtype) #default data type is float32
    angle = axis_angle[3]
    
    if angle == 0:
        rot_matrix = torch.eye(3, device=device, dtype=dtype)
    else:
        k = axis_angle[:3]
        k /= torch.linalg.vector_norm(k)
        K = torch.tensor([[0, -k[2], k[1]],
                        [k[2], 0, -k[0]],
                        [-k[1], k[0], 0]], device=device, dtype=dtype)
                        
        rot_matrix = torch.eye(3, device=device, dtype=dtype) + torch.sin(angle)*K + (1 - torch.cos(angle))*torch.mm(K,K)

    return rot_matrix
    
def resampleVolume(vol_3d, sampling_coord_5d, output_shape, device):
    '''
    Resample the given vol_3d. This function pre-format the input and post-format the output to the required shape.
    '''

    # Format the input
    vol_3d = vol_3d.to(device)
    vol_3d = torch.permute(vol_3d, (2,1,0)) #Swapping the x and z axis because grid_sample takes input tensor in (N,Ch,Z,Y,X) order
    scalar_field_5d = vol_3d[None, None, :,:,:] #Inser N and Ch axis
    sampling_coord_5d = sampling_coord_5d.to(scalar_field_5d.dtype) #make sure the data type matches. TAG: MIXED_PRECISION
    rotated_vol = torch.nn.functional.grid_sample(scalar_field_5d, sampling_coord_5d, mode='bilinear', padding_mode='zeros', align_corners=True)
    #Note that the query points[0],[1],[2] are still arranged in x,y,z order

    return rotated_vol.reshape(output_shape) #Format the output to the required shape

@torch.no_grad() #tensor resulted from @torch.inference_mode() cannot be directly used in traced process unless it is cloned.
def constructResamplingGrid(output_shape, pre_grid_scale, pre_translation, axis_angle, post_translation, post_grid_scale, device):
    ''' 
    Generate sampling grid in torch.float32 data type.

    One resampling step is performed to map the original grid to the transformed grid.
    S_pre(coordinate) = coordinate * pre_grid_scale
    T_pre(coordinate) = coordinate + pre_translation
    R(coordinate) = R_rot@coordinate
    T_post(coordinate) = coordinate + post_translation
    S_post(coordinate) = coordinate * post_grid_scale

    axis_angle: Define the rotation axis and angle. A 4-element vector [x,y,z,theta]
    output_shape: The shape of the sampling grid. A 3-element vector [X,Y,Z]. The output is a flattened in the num_sample dimension of a 5D tensor in (1,1,1,num_sample,3).
    pre_grid_scale (e.g. 1, tensor([1,1,1]))
    pre_translation (e.g. 0, tensor([0,0,0]))
    post_translation (e.g. 0, tensor([0,0,0]))
    post_grid_scale (e.g. 1, tensor([1,1,1]))
    '''
    pre_grid_scale = torch.as_tensor(pre_grid_scale, device=device)
    if pre_grid_scale.dim() <= 1:
        pre_grid_scale = pre_grid_scale.expand(3) #expand to a 3-element vector

    pre_translation = torch.as_tensor(pre_translation, device=device)
    if pre_translation.dim() <= 1:
        pre_translation = pre_translation.expand(3) #expand to a 3-element vector

    rot_mat = axang2rotmat(axis_angle, device) # Get rotation matrix

    post_translation = torch.as_tensor(post_translation, device=device)
    if post_translation.dim() <= 1:
        post_translation = post_translation.expand(3)
    
    post_grid_scale = torch.as_tensor(post_grid_scale, device=device)
    if post_grid_scale.dim() <= 1:
        post_grid_scale = post_grid_scale.expand(3) #expand to a 3-element vector

    # Transform the coordinate
    x = torch.linspace(-1, 1, output_shape[0], device=device)*pre_grid_scale[0]+pre_translation[0] 
    y = torch.linspace(-1, 1, output_shape[1], device=device)*pre_grid_scale[1]+pre_translation[1]
    z = torch.linspace(-1, 1, output_shape[2], device=device)*pre_grid_scale[2]+pre_translation[2] 
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    coord = torch.vstack((X.ravel(), Y.ravel(), Z.ravel()))
    sampling_coord = torch.mm(rot_mat, coord) #rotate the coordinate, in (3,N) shape
    sampling_coord = sampling_coord.T #transpose to (N,3) shape
    sampling_coord = (sampling_coord+post_translation[None,:])*post_grid_scale[None,:] #apply post translation
    sampling_coord.requires_grad = False #disable gradient
    return sampling_coord[None,None,None,:,:] #return a 5D tensor in (N,Z_out,Y_out,X_out,3) format. https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
