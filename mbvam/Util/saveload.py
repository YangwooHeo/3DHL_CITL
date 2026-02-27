import torch
import numpy as np
import os
import vtk
import dill
from copy import deepcopy

def saveArray(volume, file_path):
    '''
    Save pytorch tensor or numpy array as a .npy, .pt, or .vti file.
    '''
    if (not isinstance(volume, torch.Tensor)) and (not isinstance(volume, np.ndarray)):
        raise ValueError('Input volume should be a torch.Tensor or np.ndarray.')

    # Get the full file path
    full_file_path = os.path.abspath(file_path)

    # Get the base name and extension of the file path
    _, extension = os.path.splitext(full_file_path)
    # Append ".vti" extension to the base name

    if extension == '.vti':
        # Convert to numpy array
        if isinstance(volume, torch.Tensor):
            volume = volume.to('cpu').numpy()

        # Convert to VTK array
        vtk_data_array = vtk.util.numpy_support.numpy_to_vtk(num_array=volume.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

        # Create the VTK image data structure
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(volume.shape)
        vtk_image.GetPointData().SetScalars(vtk_data_array)

        # Write to the VTI file
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(full_file_path)
        writer.SetInputData(vtk_image)
        writer.Write()
        print(f'Volume saved to {full_file_path}')

    elif extension == '.npy':
        if isinstance(volume, torch.Tensor):
            volume = volume.to('cpu').numpy()
        np.save(full_file_path, volume)
        print(f'Volume saved to {full_file_path}')  

    elif extension == '.pt':
        torch.save(volume, full_file_path)
        print(f'Volume saved to {full_file_path}')

    else:
        raise ValueError(f'File extension "{extension}" not supported.')
    

def loadArray(file_path, output_format='numpy'):
    '''
    Load .npy, .pt, or .vti file as a numpy array or a PyTorch tensor.
    '''
    # Get the full file path
    full_file_path = os.path.abspath(file_path)

    # Get the base name and extension of the file path
    _, extension = os.path.splitext(full_file_path)

    if extension == '.vti':
        # Read the VTI file
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(full_file_path)
        reader.Update()

        # Get the VTK image data structure
        vtk_image = reader.GetOutput()

        # Convert to numpy array
        volume = vtk.util.numpy_support.vtk_to_numpy(vtk_image.GetPointData().GetScalars())
        volume = volume.reshape(vtk_image.GetDimensions())


    elif extension == '.npy':
        volume = np.load(full_file_path)

    elif extension == '.pt':
        volume = torch.load(full_file_path)

    else:
        raise ValueError('Unsupported file extension.')
    
    #Data conversion
    if output_format == 'numpy':
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
    elif output_format == 'torch':
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume)
    else:
        raise ValueError('Unsupported output format. Choose "numpy" or "torch".')

    print(f'Volume loaded from {full_file_path} as {output_format}')
    return volume

def saveObject(obj, file_path):
    # Save object to file
    with open(file_path, 'wb') as f:
        dill.dump(obj, f)

def loadObject(file_path):
    # Load object from file
    with open(file_path, 'rb') as f:
        obj = dill.load(f)
    return obj

def saveText(text, file_path):
    # Save text to file
    with open(file_path, 'w') as f:
        f.write(text)

def createFolderIfNeeded(directory_path):
    if directory_path is None: #Null means not saving
        return
    elif not os.path.exists(directory_path): #if folder does not exist
        print("Directory does not exist. Creating a new folder at:" + directory_path)
        os.makedirs(directory_path)
        

def saveOptimization(optimizer, directory_path, save_eff_dose:bool=True, save_response:bool=True, save_init:bool=True, restore_debug_info:bool=True):    
    '''
    Save optimization results.
    The following are all available in optimizer or can be generated from optimizer.

    Beam object
        holobeamconfig (obj)
        holobeamconfig (__repr__)
        Beam profile.
        Initial and final phase mask.
        Initial and final beam mean amplitude.

    optimizationconfig (from optimizer) (obj and __repr__)
    Target object
    
    EDM settings (__repr__). Response settings (__repr__).
    Loss function (__repr__) and obj.
    
    Optimizer settings (__repr__).

    Initial EDM volume and response volume.
    Final EDM volume and response volume.

    save_eff_dose toggle saving of effective dose volume.
    save_response toggle saving of response volume.
    save_init toggle saving of initial state of the effective dose and response volume. Ignored if both save_eff_dose and save_response are False.
    restore_debug_info toggle whether we restore the eff_dose and response in memory after saving.
    '''
    createFolderIfNeeded(directory_path)

    #Save the beam objects and config (__repr__)
    for i, beam in enumerate(optimizer.beams):        
        # Write beam config to file
        beam_config_str = beam.beam_config.__repr__()
        beam_config_file_path = os.path.join(directory_path, f'beam_{i}_config.txt')
        saveText(beam_config_str, beam_config_file_path)
        print(f'Saved beam_{i}_config.txt at {beam_config_file_path}')

        # create a copy of the beam object and remove the H attribute
        beam_copy = deepcopy(beam)
        beam_copy.H = None

        # Save the beam object with dill
        beam_file_path = os.path.join(directory_path, f'beam_{i}.dill')
        saveObject(beam_copy, beam_file_path)
        print(f'Saved beam_{i}.dill at {beam_file_path}')
    
    #Save the optimization config (obj and __repr__)
    optim_config = optimizer.optim_config
    optim_config_str = optim_config.__repr__()
    optim_config_txt_file_path = os.path.join(directory_path, 'optim_config.txt')
    saveText(optim_config_str, optim_config_txt_file_path)
    optim_config_dill_file_path = os.path.join(directory_path, 'optim_config.dill')
    saveObject(optim_config, optim_config_dill_file_path)
    print(f'Saved optim_config.txt at {optim_config_txt_file_path}')
    print(f'Saved optim_config.dill at {optim_config_dill_file_path}')
    
    #Save the target object
    target_object_file_path = os.path.join(directory_path, 'target_object.dill')
    saveObject(optimizer.target_object, target_object_file_path)
    print(f'Saved target_object.dill at {target_object_file_path}')

    #Save the EDM settings (__repr__). Response settings (__repr__)
    eff_dose_model_file_path = os.path.join(directory_path, 'eff_dose_model.txt')
    saveText(optimizer.eff_dose_model.__repr__(), eff_dose_model_file_path)
    response_model_file_path = os.path.join(directory_path, 'response_model.txt')
    saveText(optimizer.response_model.__repr__(), response_model_file_path)
    print(f'Saved eff_dose_model.txt at {eff_dose_model_file_path}')
    print(f'Saved response_model.txt at {response_model_file_path}')
    
    #Save the loss function (__repr__) and object
    metric_file_path = os.path.join(directory_path, 'metric.txt')
    saveText(optimizer.metric.__repr__(), metric_file_path)
    metric_dill_file_path = os.path.join(directory_path, 'metric.dill')
    saveObject(optimizer.metric, metric_dill_file_path)
    print(f'Saved metric.txt at {metric_file_path}')
    print(f'Saved metric.dill at {metric_dill_file_path}')

    #Save the optimizer settings (__repr__)
    optimizer_file_path = os.path.join(directory_path, 'optimizer.txt')
    saveText(optimizer.__repr__(), optimizer_file_path) #actual value of the auto-determined learning rates
    print(f'Saved optimizer.txt at {optimizer_file_path}')

    if save_eff_dose or save_response:
        
        # Save the final volumes
        if not optimizer.optim_config.debug_mode: #final effective dose and response volume are available
            optimizer.optim_config.debug_mode = True
            optimizer.forward() #Run forward model to get the final effective dose and response volume
            optimizer.metric.history.pop() #remove the last loss value

        if save_eff_dose:
            eff_dose_iter_file_path = os.path.join(directory_path, 'eff_dose_iter.npy')
            saveArray(optimizer.eff_dose_debug.detach().clone().to('cpu').numpy(), eff_dose_iter_file_path)
            print(f'Saved eff_dose_iter.npy at {eff_dose_iter_file_path}')

        if save_response:
            response_iter_file_path = os.path.join(directory_path, 'response_iter.npy')
            saveArray(optimizer.response_debug.detach().clone().to('cpu').numpy(), response_iter_file_path)
            print(f'Saved response_iter.npy at {response_iter_file_path}')


        if save_init: # in this case, need to regenerate the initial state of the volumes
            #Need to cache the iteration variables to be restored later
            phase_mask_cache = []
            beam_mean_amp_cache = []
            for i, beam in enumerate(optimizer.beams):
                phase_mask_cache.append(beam.phase_mask_iter.detach().clone())
                beam_mean_amp_cache.append(beam.beam_mean_amplitude_iter.detach().clone())
            
                #Set the iteration variables to the initial values
                beam.phase_mask_iter = beam.phase_mask_init.detach().clone()
                beam.beam_mean_amplitude_iter = beam.beam_mean_amplitude_init.detach().clone()
                beam.phase_mask_iter.requires_grad = True
                beam.beam_mean_amplitude_iter.requires_grad = True
            
            #Save the initial effective dose volume and response volume
            optimizer.forward() #Run forward model to get the final effective dose and response volume
            optimizer.metric.history.pop() #remove the last loss value

            if save_eff_dose:
                eff_dose_init_file_path = os.path.join(directory_path, 'eff_dose_init.npy')
                saveArray(optimizer.eff_dose_debug.detach().clone().to('cpu').numpy(), eff_dose_init_file_path)
                print(f'Saved eff_dose_init.npy at {eff_dose_init_file_path}')

            if save_response:
                response_init_file_path = os.path.join(directory_path, 'response_init.npy')
                saveArray(optimizer.response_debug.detach().clone().to('cpu').numpy(), response_init_file_path)
                print(f'Saved response_init.npy at {response_init_file_path}')

            # Restore the iteration variables
            for i, beam in enumerate(optimizer.beams):
                beam.phase_mask_iter = phase_mask_cache[i]
                beam.beam_mean_amplitude_iter = beam_mean_amp_cache[i]
                # Set the optimization variables to require gradient
                beam.phase_mask_iter.requires_grad = True
                beam.beam_mean_amplitude_iter.requires_grad = True

            if restore_debug_info:
                #Save the final effective dose volume and response volume
                optimizer.forward() #Run forward model to get the final effective dose and response volume
                optimizer.metric.history.pop() #remove the last loss value
            else:
                optimizer.dose_tuple_debug = None
                optimizer.eff_dose_debug = None
                optimizer.response_debug = None
        

def loadOptimization(directory_path):
    '''
    Load optimization results.
    '''
    optimization = {}

    # Load the beam objects
    i = 0 #beam index
    while True:
        beam_file_path = os.path.join(directory_path, f'beam_{i}.dill')
        if not os.path.exists(beam_file_path):
            break
        try:
            with open(beam_file_path, 'rb') as f:
                optimization[f'beam_{i}'] = dill.load(f)
        except FileNotFoundError:
            print(f'File {beam_file_path} not found. Skipping.')
        i += 1

    # List of all the other files to load
    files_to_load = ['optim_config.dill', 'target_object.dill', 'metric.dill', 
                     'eff_dose_init.npy', 'response_init.npy', 'eff_dose_iter.npy', 'response_iter.npy']

    for file in files_to_load:
        file_path = os.path.join(directory_path, file)
        try:
            if file.endswith('.dill'):
                with open(file_path, 'rb') as f:
                    optimization[file.replace('.dill', '')] = dill.load(f)
            elif file.endswith('.npy'):
                optimization[file.replace('.npy', '')] = np.load(file_path)
        except FileNotFoundError:
            print(f'File {file_path} not found. Skipping.')

    return optimization

# save_directory = r'G:\Shared drives\taylorlab\CAL Projects\LDCT\test_output\param_sweep\final'
# global_setting_str = f'gratings_p_center={p_center}_eps_center={eps_center:.2f} exit={exit_param_global:.1e}'
# date_time_str = datetime.now().strftime("%Y_%m_%d__%H_%M_")
# print(date_time_str+global_setting_str)
# save_material_response_directory = os.path.join(save_directory,"material_response_sweep", date_time_str+global_setting_str)


# def saveOutput(run,sino,dose,response,logs,options,directory):
#     np.save(os.path.join(directory,f'{run}_sino.npy'),sino)
#     np.save(os.path.join(directory,f'{run}_dose.npy'),dose)
#     np.save(os.path.join(directory,f'{run}_response.npy'),response)
#     with open(os.path.join(directory, f'{run}_options.txt'), 'w') as f:
#         f.write(str(options))
#     with open(os.path.join(directory, f'{run}_logs.pkl'), 'wb') as f:
#         dill.dump(logs, f)
#     plt.savefig(os.path.join(directory,f'{run}_plots.png'),dpi=300)

    
