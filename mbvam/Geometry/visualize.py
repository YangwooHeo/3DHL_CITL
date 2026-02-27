import vedo
vedo.settings.default_backend= 'vtk'
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import torch
import numpy as np
import os

class VedoVisualizer():
    '''
    This class contains functions for visualizing 2D or 3D data.
    The class instances only store settings to visualize the data but not the data itself.

    #TODO: Add export function to export the volume to vti file for visualization in paraview or numpy file for visualization in Tomviz.
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        #Visualization settings: vedo.Volume

        self.cmap = 'viridis' #https://vedo.embl.es/docs/vedo/colors.html#vedo.colors.colorMap.
        #Perceptually Uniform Sequential :['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        # (1) by list of tuples
        # self.cmap = [(0.0,'red')]  # tuple(value:float, color), where color can be RGB, HEX, or color names #https://vedo.embl.es/docs/vedo/colors.html
        # (2) by matplotlib colormap
        # colors = plt.get_cmap('jet')(np.linspace(0, 1, 256)) # Get the colors from the Matplotlib colormap
        # self.cmap = vedo.colors.build_lut(colors) # only supported for mesh.cmap() but not volume.cmap
        # (3) by vedo colormaps, e.g. 'jet'
        self.alpha = [0, 0.1] #[(0, 0), (1, 0.1)] #[(0, 0), (0.5, 0.5), (1, 1)]
        self.mode = 0 
        # 0, composite rendering
        # 1, maximum projection rendering
        # 2, minimum projection rendering
        # 3, average projection rendering
        # 4, additive mode

        #Note that vmin and vmax are not respected with cmap is a list of tuples. In all other cases, vmin and vmax are respected.
        # Reference for version 2023.4.6: d:\miniconda3\envs\3dhl-ldct310\lib\site-packages\vedo\base.py:2021
        # Reference for version 2023.5.0: vedo/vedo/visual.py:376
        self.vmin = None #None is internally converted to the minimum value of the scalar range in vedo package
        self.vmax = None #None is internally converted to the maximum value of the scalar range in vedo package

        # The cmap function also accept alpha as kwarg and the same vmin and vmax.
        # Reference for version 2023.4.6: d:\miniconda3\envs\3dhl-ldct310\lib\site-packages\vedo\base.py:2089
        # Reference for version 2023.5.0: vedo/vedo/visual.py:442

        # Colorbar settings
        self.show_colorbar = False
        self.colorbar_default = {'title':'Colorbar', 'nlabels':4, 'use_alpha':False, 'c':'w'}

        #Visualization settings: vedo.plotter
        self.background_color = 'black'
        self.axes = 9 #https://vedo.embl.es/docs/vedo/plotter.html#Plotter.add_global_axes
        
        # axtype : (int)
        #     0, no axes,
        #     1, draw three gray grid walls
        #     2, show cartesian axes from (0,0,0)
        #     3, show positive range of cartesian axes from (0,0,0)
        #     4, show a triad at bottom left
        #     5, show a cube at bottom left
        #     6, mark the corners of the bounding box
        #     7, draw a 3D ruler at each side of the cartesian axes
        #     8, show the vtkCubeAxesActor object
        #     9, show the bounding box outLine
        #     10, show three circles representing the maximum bounding box
        #     11, show a large grid on the x-y plane
        #     12, show polar axes
        #     13, draw a simple ruler at the bottom of the window
        # Axis type-1 can be fully customized by passing a dictionary axes=dict().

        #Visualization settings: vedo.plotter.camera
        self.camera_position = (0,0,1)
        self.camera_view_up = (1,0,0) #https://vedo.embl.es/docs/vedo/utils.html#camera_from_dict, and https://vtk.org/doc/nightly/html/classvtkCamera.html
        self.camera_focal_point = (0,0,0) # focal point (pointing direction) of the camera in world coordinates. The default focal point is the origin. 
        self.camera_focal_point_distance = None #The distance between the camera position and the focal point.

        self.camera_azimuth = 0.0 #Rotate the camera about the view up vector centered at the focal point.
        self.camera_elevation = 0.0 #Rotate the camera about the cross product of the negative of the direction of projection and the view up vector, using the focal point as the center of rotation.

        self.camera_yaw = 0.0 #Rotate the focal point about the view up vector, using the camera's position as the center of rotation.
        self.camera_pitch = 0.0 #Rotate the focal point about the cross product of the view up vector and the direction of projection, using the camera's position as the center of rotation.
        self.camera_roll = 0.0 #Rotate the camera about the direction of projection.

        #Visualization settings: vedo.plotter.show
        self.interactive=False #Only can take screenshot when interactive is False #TODO

        #Animation parameters
        self.playback_fps=0.01 #TODO
        self.num_frames=128 #TODO
        self.save_video=None #TODO

        #Coordinate system
        self.coord_vec=None #TODO


        

    def vedoRayCastPlot(self, volume, **kwargs): #show the target array
        '''Deprecated. Use vedoShow instead.'''
        if isinstance(volume, torch.Tensor):
            volume = volume.to('cpu').numpy()
        elif isinstance(volume, np.ndarray):
            pass
        
        vol = vedo.Volume(volume,mode=0)

        kwargs.setdefault('axes', 9) #default with bounding box drawn: 9. https://vedo.embl.es/docs/vedo/plotter.html#Plotter
        kwargs.setdefault('bg', 'black') #default black background
        self.plotter = vedo.applications.RayCastPlotter(vol, **kwargs)
        self.plotter.show(viewup="x")    

    def vedoShowSTL(self, list_stl_file_path, **kwargs):
        '''
        Show STL meshes.
        '''
        # Vedo volume list
        vedo_volume_list = []
        for stl_file_path in list_stl_file_path:
            vedo_volume = vedo.load(stl_file_path)
            vedo_volume_list.append(vedo_volume)

        vedo.show(vedo_volume_list, **kwargs)

    def vedoShow(self, volume_seq, cmap_list=None, alpha_list=None, colorbar_kwargs={}, **kwargs): #show the target array
        '''
        Show a tuple of volume on the same plot.
        Note that when plotting multiple volumes, the volume is displayed in the order of the list.
        The first volume is displayed at the back of the screen and the last volume is displayed at the front of the screen.

        kwargs are passed to the RayCastPlotter
        '''
        if not isinstance(volume_seq, (tuple, list)): #if volume_seq is not a tuple or list, make it a tuple
            volume_seq = (volume_seq,)

        for key, value in self.colorbar_default.items():
            colorbar_kwargs.setdefault(key, value)
        
        vedo_volume_list = []

        #for each of the volume in volume_seq, convert it to a vedo.Volume object
        for idx, volume in enumerate(volume_seq):
            if isinstance(volume, torch.Tensor):
                volume = volume.to('cpu').numpy()
            elif isinstance(volume, np.ndarray):
                volume = volume

            vedo_volume = vedo.Volume(volume, mode=self.mode)

            # Apply volume settings
            if cmap_list==None:
                cmap = self.cmap #default use the cmap in the class instance
            else:
                cmap = cmap_list[idx]
            vedo_volume.cmap(cmap, vmin=self.vmin, vmax=self.vmax) # set the colormap and alpha, (can be an int, a str, a list or a list of 2-tuple).

            if alpha_list==None:
                alpha = self.alpha
            else:
                alpha = alpha_list[idx]
            vedo_volume.alpha(alpha, vmin=self.vmin, vmax=self.vmax)

            #vedo_volume.shade(status=0) #Can be set
            if self.show_colorbar:
                vedo_volume.add_scalarbar(**colorbar_kwargs) #cmap and alpha must be set before adding the colorbar.
                # vedo_volume.add_scalarbar3d('colorbar', nlabels=4, draw_box=True, c="w") #https://vedo.embl.es/docs/vedo/visual.html#CommonVisual.add_scalarbar3d

            vedo_volume_list.append(vedo_volume)

        # Set default kwargs for vedo.applications.RayCastPlotter
        kwargs.setdefault('axes', self.axes) #default with bounding box drawn: 9. No axes: 0, https://vedo.embl.es/docs/vedo/plotter.html#Plotter
        # kwargs.setdefault('bg', self.background_color) #background will be set in applyPlotterSettings
        
        # Initialize plotter. This is initialized before volume settings are applied because this initialization will overwrite the volume alpha settings.
        # vedo.show(vedo_volume_list, **kwargs)
        self.plotter = vedo.applications.RayCastPlotter(vedo_volume_list[0], **kwargs)
        for vedo_volume in vedo_volume_list[1:]:
            self.plotter.add(vedo_volume)

        # Reapply volume settings because the plotter initialization will overwrite the volume alpha settings.
        for idx, vedo_volume in enumerate(vedo_volume_list):
            if cmap_list==None:
                cmap = self.cmap #default use the cmap in the class instance
            else:
                cmap = cmap_list[idx]
            vedo_volume.cmap(cmap, vmin=self.vmin, vmax=self.vmax) # set the colormap and alpha, (can be an int, a str, a list or a list of 2-tuple).

            if alpha_list==None:
                alpha = self.alpha
            else:
                alpha = alpha_list[idx]
            vedo_volume.alpha(alpha, vmin=self.vmin, vmax=self.vmax)

        self.applyPlotterSettings()
        self.plotter.show()    

    def applyPlotterSettings(self):
        self.plotter.background(self.background_color)
        self.plotter.camera.SetPosition(self.camera_position)
        self.plotter.camera.SetViewUp(*self.camera_view_up)
        self.plotter.camera.SetFocalPoint(self.camera_focal_point)
        if self.camera_focal_point_distance is not None:
            self.plotter.camera.SetFocalDistance(self.camera_focal_point_distance)

        self.plotter.camera.Elevation(self.camera_elevation)
        self.plotter.camera.Azimuth(self.camera_azimuth)

        self.plotter.camera.Yaw(self.camera_yaw)
        self.plotter.camera.Pitch(self.camera_pitch)
        self.plotter.camera.Roll(self.camera_roll)

    def getPlotterSettings(self):
        cam_pos = self.plotter.camera.GetPosition()
        dir = self.plotter.camera.GetDirectionOfProjection()
        viewangle = self.plotter.camera.GetViewAngle()
        print(f"cam_pos: {cam_pos}")
        print(f"dir: {dir}")
        print(f"viewangle: {viewangle}")
        return cam_pos, dir, viewangle
    
    def screenshot(self, file_path):
        '''
        Take a screenshot of the current plot.
        '''
        if not file_path.endswith('.png'):
            file_path = file_path + '.png'
        self.plotter.screenshot(file_path)

    def close(self):
        '''
        Close the current plot.
        '''
        self.plotter.close()

    def vedoShowScreenshot(self, file_path, *arg, **kwargs):
        '''
        Show the volume and take a screenshot.
        '''
        kwargs.setdefault('interactive', False)
        self.vedoShow(*arg, **kwargs)
        self.screenshot(file_path)
        self.plotter.close()

    def vedoShowRecording(self, directory, *arg, frames_per_rotation=180, elevation=30.0, **kwargs):
        '''
        Record a video rotating the volume.
        '''
        yaw_angles = np.linspace(0, 360.0, frames_per_rotation)
        self.camera_elevation = elevation
        print(f'Start recording {frames_per_rotation} frames...')
        for yaw_angle in yaw_angles:
            print(f'Yaw angle: {yaw_angle}')
            self.camera_yaw = yaw_angle
            kwargs.setdefault('interactive', False)
            self.vedoShow(*arg, **kwargs)
            self.screenshot(os.path.join(directory, f'yaw_{yaw_angle:3.1f}.png'))
            self.plotter.close()
        print('Start recording...')

    # Reference of vedo: d:\miniconda3\envs\3dhl-ldct310\lib\site-packages\vedo\volume.py:1187
    # def cmap(self, c, alpha=None, vmin=None, vmax=None):
    #     """Same as `color()`.

    #     Arguments:
    #         alpha : (list)
    #             use a list to specify transparencies along the scalar range
    #         vmin : (float)
    #             force the min of the scalar range to be this value
    #         vmax : (float)
    #             force the max of the scalar range to be this value
    #     """
    #     return self.color(c, alpha, vmin, vmax)


    def applyStandardSettings(self):
        '''For plotting manuscript figures'''
        self.vmin = 0.0
        self.vmax = 1.0
        self.axes = 0
        self.cmap = 'inferno'
        self.alpha= [0, 0.025]

        self.background_color = 'black'
        self.show_colorbar = False
        self.colorbar_default = {'title':'Colorbar',
                                    'nlabels':0,
                                    'use_alpha':False,
                                    'c':'w',
                                    'pos': (0.775, 0.35), #pos=(0.775, 0.05), default is (0.8, 0.05)
                                    'font_size': 25, #font_size=12,
                                    'size': (100, 400)} #size=(60, 350)

        self.camera_yaw = 35.0
        self.camera_pitch = 0.0
        self.camera_roll = 0.0

        self.camera_azimuth = 0.0
        self.camera_elevation = 30.0 

        self.camera_focal_point_distance = None
        print('Applied standard plotting settings.')

def openCVSliceViewer(volume, axis=2, display_size=(800, 800), normalize_slice_independently=True):
    '''
    axis: 0, 1, 2 select x, y, z planes respectively
    normalize_slice_independently: True to normalize each slice independently, False to normalize the entire volume using the same factor
    '''
    if isinstance(volume, torch.Tensor):
        volume = volume.to('cpu').numpy()

    # Initialize variables
    current_slice = 0
    max_slice = volume.shape[axis] - 1
    s = [slice(None)] * volume.ndim  # create a slice object for each dimension
    s[axis] = current_slice  # replace the slice object for the desired axis with an index
    vol_slice = volume[tuple(s)]

    if normalize_slice_independently:
        # Normalize each slice independently
        slice_normalized = cv2.normalize(vol_slice, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    else:
        # Normalize the entire volume using the same factor
        volume_normalized = cv2.normalize(volume, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        vol_slice = volume_normalized[tuple(s)]
        slice_normalized = cv2.resize(vol_slice, display_size)

    img_resized = cv2.resize(slice_normalized, display_size)

    # Create a window
    window_name = 'Volume slice'
    cv2.namedWindow(window_name)

    def updateSlice(current_slice):
        nonlocal img_resized
        s[axis] = current_slice  # replace the slice object for the desired axis with an index
        vol_slice = volume[tuple(s)]

        if normalize_slice_independently:
            # Normalize each slice independently
            slice_normalized = cv2.normalize(vol_slice, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        else:
            # Normalize the entire volume using the same factor
            vol_slice = volume_normalized[tuple(s)]
            slice_normalized = cv2.resize(vol_slice, display_size)

        img_resized = cv2.resize(slice_normalized, display_size)

    # Create a trackbar for slice selection
    cv2.createTrackbar('Slice', window_name, current_slice, max_slice, updateSlice)

    while True:
        # Display the slice
        cv2.imshow(window_name, img_resized)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # Exit if 'q' is pressed
        if key == ord('q'):
            break

    # Destroy the window
    cv2.destroyAllWindows()

def viewVolumeSlice(volume, axis=2, slice_index=None, title=None, show_axis = True, **kwargs):
    """
    View an arbitrarily oriented slice of a voxel volume using matplotlib.

    Parameters:
        volume (ndarray): The voxel volume.
        axis (int): The axis along which to slice the volume (0 for x-axis, 1 for y-axis, 2 for z-axis).
        slice_index (int): The index of the slice along the specified axis. If None, the middle slice is used.

    Returns:
        None
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.to('cpu').numpy()
    kwargs.setdefault('cmap', 'viridis')

    if slice_index is None:
        slice_index = volume.shape[axis] // 2

    if axis == 0:
        slice_data = volume[slice_index, :, :]
    elif axis == 1:
        slice_data = volume[:, slice_index, :]
    elif axis == 2:
        slice_data = volume[:, :, slice_index]
    else:
        raise ValueError("Invalid axis value. Must be 0, 1, or 2.")

    fig, ax = plt.subplots()
    im = ax.imshow(slice_data, **kwargs)
    plt.colorbar(im)  # Add colorbar
    
    if show_axis:
        ...
    else:
        plt.axis('off')

    direction_dict = {0: 'x', 1: 'y', 2: 'z'}
    if title is None:
        title = "Slice along axis {}".format(direction_dict[axis])
    plt.title(title)  # Add title
    plt.show()
    return fig, ax

def viewVolumeSlices(volume, axis=2, ms_per_frame=100, filename=None, **kwargs):
    """
    View all the slices along the chosen axis of a voxel volume using matplotlib animation.

    Parameters:
        volume (ndarray): The voxel volume.
        axis (int): The axis along which to slice the volume (0 for x-axis, 1 for y-axis, 2 for z-axis).
        ms_per_frame (int): The delay between frames in milliseconds.

    Returns:
        None
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.to('cpu').numpy()
    kwargs.setdefault('cmap', 'viridis')

    fig, ax = plt.subplots()

    if axis == 0:
        slice_data = volume[0, :, :]
    elif axis == 1:
        slice_data = volume[:, 0, :]
    elif axis == 2:
        slice_data = volume[:, :, 0]
    else:
        raise ValueError("Invalid axis value. Must be 0, 1, or 2.")

    im = ax.imshow(slice_data, **kwargs)

    def updatefig(slice_index):
        if axis == 0:
            slice_data = volume[slice_index, :, :]
        elif axis == 1:
            slice_data = volume[:, slice_index, :]
        elif axis == 2:
            slice_data = volume[:, :, slice_index]
        else:
            raise ValueError("Invalid axis value. Must be 0, 1, or 2.")

        im.set_array(slice_data)
        ax.set_title(f'Slice {slice_index}')
        plt.draw()
        return [im]

    ani = animation.FuncAnimation(fig, updatefig, frames=range(volume.shape[axis]), interval=ms_per_frame, blit=True)
    plt.colorbar(im)  # Add colorbar

    if filename is not None:
        ani.save(filename, writer='imagemagick', fps=1000 / ms_per_frame)


    plt.show()  # Display the animation
    return ani 

def volumePlot():
    '''
    Generate volume and slice plots from provided results.
    '''
    pass


if __name__ == '__main__':
    # Test vedoShow
    # import vedo

    # embryo=vedo.Volume(vedo.dataurl + "embryo.slc")
    # head = vedo.Volume(vedo.dataurl+'head.vti')

    # embryo.mode(1).cmap("jet")  # change visual properties

    # # Use the show function
    # # vedo.show([embryo, head], axes=7, bg='black', bg2='blackboard')

    # # Use the RayCastPlotter
    # plt = vedo.applications.RayCastPlotter(embryo, bg='black', bg2='blackboard', axes=7)
    # plt.add(head)
    # plt.show(viewup="x")
    ...