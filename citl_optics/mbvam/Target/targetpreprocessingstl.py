import mbvam.Target
import numpy as np
import torch
import scipy.ndimage
import trimesh

def targetPreprocessingSTL(optim_config):
    '''
    This function parse target parameters, voxelize the STL and build target object.
    '''
    # target_array, Xv, Yv, Zv = voxelizeSTL(optim_config.target_file_path, optim_config.target_num_voxel_z, optim_config.voxelize_by_opengl) # Voxelize STL
    target_tensor, mb_x, mb_y, mb_z = voxelizeSTL(optim_config.target_file_path, optim_config.target_num_voxel_z, optim_config.voxelize_by_opengl) # Voxelize STL and get mesh bounds

    target_tensor = trimTensorZeros(target_tensor) # Trim the target to the bound of the non-zero region

    # Build a coord_vec with number of elements equal to the trimmed array
    coord_vec = (torch.linspace(mb_x[0], mb_x[1], target_tensor.shape[0], device=optim_config.device),
                torch.linspace(mb_y[0], mb_y[1], target_tensor.shape[1], device=optim_config.device),
                torch.linspace(mb_z[0], mb_z[1], target_tensor.shape[2], device=optim_config.device))

    return mbvam.Target.targetobject.TargetObject(
                                                device=optim_config.device,
                                                dtype=optim_config.fdtype,
                                                coord_vec=coord_vec,
                                                array=target_tensor,
                                                eps=0,
                                                w=1,
                                                store_coord_grid=False) #return as target object

@torch.inference_mode()
def voxelizeSTL(filepath, target_num_voxel_z, voxelize_by_opengl=True):
    '''This function voxelize STL mesh and return the voxelized tensor and the mesh bounds that correspond to the bound of the non-zero voxels.'''
    #TODO: Potentially add functions for mesh transformation before voxelization. https://github.com/mikedh/trimesh/blob/main/trimesh/transformations.py

    if voxelize_by_opengl: #Use OpenGL to voxelize the mesh. This is much faster than trimesh.    
        vox = mbvam.Target.voxelize.Voxelizer()
        vox.addMeshes({filepath:'print_body',})
        # layer_thickness = vox.meshes['print_body'].length_z/target_num_voxel_z #If depends on the bound of this particular mesh.
        layer_thickness = vox.global_bounds.length_z/target_num_voxel_z #If depends on the global bound.
        voxelized = vox.voxelize('print_body',layer_thickness=layer_thickness,voxel_value=1,voxel_dtype=np.uint8,square_xy=True,slice_save_path=None)
                    
        voxelized = torch.as_tensor(voxelized, dtype=torch.uint8)
        mesh_bounds_x = vox.global_bounds.xmin, vox.global_bounds.xmax
        mesh_bounds_y = vox.global_bounds.ymin, vox.global_bounds.ymax
        mesh_bounds_z = vox.global_bounds.zmin, vox.global_bounds.zmax
        
        # voxel_grid_vec_x = torch.linspace(vox.global_bounds.xmin, vox.global_bounds.xmax, voxelized.shape[0]) #Caution: Note this is the bound of the MESH.
        # voxel_grid_vec_y = torch.linspace(vox.global_bounds.ymin, vox.global_bounds.ymax, voxelized.shape[1]) #The voxelized.array is FOUND to have a larger XY span than this mesh bound because it is padded with zeros.
        # voxel_grid_vec_z = torch.linspace(vox.global_bounds.zmin, vox.global_bounds.zmax, voxelized.shape[2]) #For this reason, zero trimming is necessary and no longer an option.

    else: #Use trimesh to voxelize the mesh. 
        mesh = trimesh.load(filepath)
        #Determine the extent of the mesh
        body_mesh = mbvam.Target.voxelize.BodyMesh(mesh)
        layer_thickness = body_mesh.bounds.length_z/target_num_voxel_z

        voxels = mesh.voxelized(pitch=layer_thickness) #return trimesh.VoxelGrid object. At this moment the voxels are not filled in and only represent the surface.
        voxels.fill(method='holes') #fill in the interior of the voxels.
        
        voxelized = torch.as_tensor(voxels.matrix, dtype=torch.uint8)

        mesh_bounds_x = mesh.bounds[0][0], mesh.bounds[1][0]
        mesh_bounds_y = mesh.bounds[0][1], mesh.bounds[1][1]
        mesh_bounds_z = mesh.bounds[0][2], mesh.bounds[1][2]

        # voxel_grid_vec_x = torch.linspace(mesh.bounds[0][0], mesh.bounds[1][0], voxelized.shape[0]) #Caution: Note this is the bound of the MESH.
        # voxel_grid_vec_y = torch.linspace(mesh.bounds[0][1], mesh.bounds[1][1], voxelized.shape[1]) #The voxelized.array may have a larger span than this mesh bound because it may be padded with zeros.
        # voxel_grid_vec_z = torch.linspace(mesh.bounds[0][2], mesh.bounds[1][2], voxelized.shape[2]) #For this reason, zero trimming is necessary and no longer an option.

    # return voxelized, voxel_grid_vec_x, voxel_grid_vec_y, voxel_grid_vec_z
    return voxelized, mesh_bounds_x, mesh_bounds_y, mesh_bounds_z

@torch.inference_mode()
def trimTensorZeros(tensor):
    #This function determine the bound of the non-zero region of the target and trim the target to that bound.

    if tensor.dim() != 3:
        raise ValueError('target array should be 3D.')
    
    nonzero_along_x = torch.any(tensor, dim=1, keepdim=True) #reduce 1 dimensions, output stays 3d
    nonzero_along_x = torch.any(nonzero_along_x, dim=2, keepdim=True) #reduce another dimensions, output stays 3d

    nonzero_along_y = torch.any(tensor, dim=0, keepdim=True)
    nonzero_along_y = torch.any(nonzero_along_y, dim=2, keepdim=True)

    nonzero_along_z = torch.any(tensor, dim=0, keepdim=True)
    nonzero_along_z = torch.any(nonzero_along_z, dim=1, keepdim=True)

    nonzero_x_idx = torch.nonzero(nonzero_along_x.squeeze(), as_tuple=False) #returns a 2-D tensor where each row is the index for a nonzero value.
    nonzero_y_idx = torch.nonzero(nonzero_along_y.squeeze(), as_tuple=False)
    nonzero_z_idx = torch.nonzero(nonzero_along_z.squeeze(), as_tuple=False)
    
    # Get min and max indices for each dimension
    min_x, max_x = torch.min(nonzero_x_idx), torch.max(nonzero_x_idx)
    min_y, max_y = torch.min(nonzero_y_idx), torch.max(nonzero_y_idx)
    min_z, max_z = torch.min(nonzero_z_idx), torch.max(nonzero_z_idx)

    # Trim the target
    return tensor[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1].detach().clone() #Add 1 to max_indices because slicing in Python is exclusive at the upper bound

        

if __name__ == '__main__':
    import vedo
    import vedo.applications
    import time
    import mbvam.Optim.optimizationconfig
    optim_config = mbvam.Optim.optimizationconfig.OptimizationConfig()
    # optim_config.target_file_path = r'G:\Shared drives\taylorlab\CAL Projects\3DHL\Codebase\3d-holographic-lithography-py\inputs\3d_cross.stl'
    optim_config.target_file_path = r'G:\Shared drives\taylorlab\CAL Projects\3DHL\Codebase\3d-holographic-lithography-py\inputs\loft_cross.stl'
    
    #Test openGL voxelization
    start_time = time.perf_counter()
    target_array, Xv, Yv, Zv = voxelizeSTL(optim_config.target_file_path, optim_config.target_num_voxel_z, True)
    elapsed_time = time.perf_counter() - start_time

    print('OpenGL voxelization time: ', elapsed_time, 's')
    print(f'target_array.shape: {target_array.shape}')
    print(f'Xv.shape: {Xv.shape}, Yv.shape: {Yv.shape}, Zv.shape: {Zv.shape}')

    target_array_np = target_array.to('cpu').numpy()
    vol = vedo.Volume(target_array_np,mode=0)
    vedo.applications.RayCastPlotter(vol,bg='black').show(viewup="x")
    
    # Test trimesh voxelization
    start_time = time.perf_counter()
    target_array, Xv, Yv, Zv = voxelizeSTL(optim_config.target_file_path, optim_config.target_num_voxel_z, False)
    elapsed_time = time.perf_counter() - start_time

    print('trimesh voxelization time: ', elapsed_time, 's')
    print(f'target_array.shape: {target_array.shape}')
    print(f'Xv.shape: {Xv.shape}, Yv.shape: {Yv.shape}, Zv.shape: {Zv.shape}')
    target_array_np = target_array.to('cpu').numpy()

    vol = vedo.Volume(target_array_np,mode=0)
    vedo.applications.RayCastPlotter(vol,bg='black').show(viewup="x")