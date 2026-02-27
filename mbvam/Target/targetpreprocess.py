from pathlib import Path
import mbvam.Target

def targetPreprocess(optim_config):

    '''
    Multiple dispatch function. Assumes target as 2D image stack if input file path is a folder, and a STL if the file extension is .STL.
    If the extension is .MAT, this function directly import positive_target and zero_target matrices in the .MAT file.
    If the extension is .npy, or .pt, this function directly import positive_target and zero_target matrices in the .MAT file.

    Display convention: WYSIWYG. The image displayed on the computer monitor should be what the hologram looks like,
    when the hologram is intercepted and viewed from -z towards +z. (x,y direction subjected to reversal due to image relay)
    Physically +/-x is horizontal direction (table plane), +/-y is normal to table, +z is light propagation direction.
    Data storage: XY (3D), XYZ (3D). Data is only momentarily transposed for visualization purposes.

    '''
    # (0) Parse file path
    filepath = optim_config.target_file_path
    file_ext = Path(filepath).suffix

    # (1) Get target array and coord
    if not file_ext: # Folder path for image stack
        target = mbvam.Target.targetpreprocessimagestack.targetPreprocessingImageStack(optim_config)
        print('Imported image stack.')

    elif file_ext in ['.stl', '.STL']:
        target = mbvam.Target.targetpreprocessingstl.targetPreprocessingSTL(optim_config)
        print('Voxelized STL target.')

    elif (file_ext == '.npy') or (file_ext == '.pt') or (file_ext == '.mat'):
        # TODO: Load .mat file
        target = mbvam.Target.targetpreprocessarray.targetPreprocessingArray(optim_config)
        print('Imported voxelized target matrices.')

    else:
        raise ValueError('Input file should be a folder, .stl, or .mat')

    # (2) Preprocess target
    target.pad(optim_config.target_padding_mode) #Pad zeros around the array
    target.centerCoord()
    target.scaleCoordSpan(optim_config.target_sizing_mode, optim_config.target_domain_size) #scale the coordinate vectors
    target.rotateTranslateScaleDataInGrid(optim_config.target_rotation, optim_config.target_offset, optim_config.target_scale_in_domain)
    return target

if __name__ == '__main__':
    import mbvam.Optim.optimizationconfig
    from math import pi
    optim_config = mbvam.Optim.optimizationconfig.OptimizationConfig()
    # optim_config.target_file_path = r'G:\Shared drives\taylorlab\CAL Projects\3DHL\Codebase\3d-holographic-lithography-py\inputs\3d_cross.stl'
    optim_config.target_file_path = r'G:\Shared drives\taylorlab\CAL Projects\3DHL\Codebase\3d-holographic-lithography-py\inputs\loft_cross.stl'    
    optim_config.target_rotation = [1, 1, 1, pi/4]
    target = mbvam.Target.targetpreprocessingstl.targetPreprocessingSTL(optim_config)
    print(f'Original target shape: {target.array.shape}')
    print(f'Original target center: {target.coord_center}')
    print(f'Original target half span: {target.coord_half_span}')
    target.show(axes=9)

    target.trimZeros()    
    print(f'Trimmed target shape: {target.array.shape}')
    print(f'Trimmed target center: {target.coord_center}')
    print(f'Trimmed target half span: {target.coord_half_span}')
    target.show(axes=9)

    target.centerCoord()
    print(f'Centered target shape: {target.array.shape}')
    print(f'Centered target center: {target.coord_center}')
    print(f'Centered target half span: {target.coord_half_span}')
    
    target.scaleCoordSpan('fit', optim_config.target_domain_size)
    print(f'Scaled target shape: {target.array.shape}')
    print(f'Scaled target center: {target.coord_center}')
    print(f'Scaled target half span: {target.coord_half_span}')
    target.show(axes=9)


    target.rotateAndTranslateDataInGrid(optim_config.target_rotation, optim_config.target_offset)
    print(f'Rotated and translated target shape: {target.array.shape}')
    print(f'Rotated and translated target center: {target.coord_center}')
    print(f'Rotated and translated target half span: {target.coord_half_span}')
    target.show(axes=9)