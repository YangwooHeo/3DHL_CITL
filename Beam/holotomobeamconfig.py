from mbvam.Beam.holobeamconfig import HoloBeamConfig
import numpy as np

class HoloTomoBeamConfig(HoloBeamConfig):
    '''
    This class is a subclass of HoloBeamConfig and store additional parameter for holo-tomographic beam configurations.

    TODO: Optional: Composing axis_angle representations to yield one effective axis_angle representation.
    '''
    
    def __init__(self):
        super().__init__()

        self._scan_axis = self.axis_angle[:3] # list of floats, axis of rotation for tomographic scanning
        self._scan_angle_span_rad = (0.0, 2*np.pi) # [rad], tuple of two floats, angular span of tomographic scanning
        self._num_sub_beam = 180 # int, number of sub-beams
        self.updateScanAngles()
    
    @property
    def scan_axis(self):
        return self._scan_axis
    
    @scan_axis.setter
    def scan_axis(self, scan_axis): #This is to safeguard the case where scan_axis_angle is not updated.
        self._scan_axis = scan_axis
        self.updateScanAngles()

    @property
    def scan_angle_span_rad(self):
        return self._scan_angle_span_rad
    
    @scan_angle_span_rad.setter
    def scan_angle_span_rad(self, scan_angle_span_rad): #This is to safeguard the case where scan_axis_angle is not updated.
        self._scan_angle_span_rad = scan_angle_span_rad
        self.updateScanAngles()

    @property
    def num_sub_beam(self):
        return self._num_sub_beam
    
    @num_sub_beam.setter
    def num_sub_beam(self, num_sub_beam): #This is to safeguard the case where scan_axis_angle is not updated.
        self._num_sub_beam = num_sub_beam
        self.updateScanAngles()

    def updateScanAngles(self):
        #The following variables are not made dynamic because they are frequently queried.
        self.scan_angle_step_rad = (self._scan_angle_span_rad[1]-self._scan_angle_span_rad[0])/self._num_sub_beam
        self.scan_angle_rad = np.linspace(self._scan_angle_span_rad[0],
                                        self._scan_angle_span_rad[1]-self.scan_angle_step_rad,
                                        self._num_sub_beam)

        #make axis angle a list of lists, each sublist represents a subbeam
        self.scan_axis_angle = [[*self.scan_axis, scan_angle] for scan_angle in self.scan_angle_rad] 

if __name__ == '__main__':
    holo_tomo_beam_config = HoloTomoBeamConfig()

    # Test scan_axis
    new_scan_axis = [0.5, 0.5, 0.5]
    holo_tomo_beam_config.scan_axis = new_scan_axis
    assert holo_tomo_beam_config.scan_axis == new_scan_axis, "scan_axis not updated correctly"
    
    # Test scan_angle_span_rad
    new_scan_angle_span_rad = (0.0, np.pi)
    holo_tomo_beam_config.scan_angle_span_rad = new_scan_angle_span_rad
    assert holo_tomo_beam_config.scan_angle_span_rad == new_scan_angle_span_rad, "scan_angle_span_rad not updated correctly"

    # Test num_sub_beam
    new_num_sub_beam = 5
    holo_tomo_beam_config.num_sub_beam = new_num_sub_beam
    assert holo_tomo_beam_config.num_sub_beam == new_num_sub_beam, "num_sub_beam not updated correctly"

    # Test updateScanAngles
    holo_tomo_beam_config.updateScanAngles()
    assert len(holo_tomo_beam_config.scan_angle_rad) == new_num_sub_beam, "scan_angle_rad not updated correctly"
    assert len(holo_tomo_beam_config.scan_axis_angle) == new_num_sub_beam, "scan_axis_angle not updated correctly"

    print(f'scan angle step rad: {holo_tomo_beam_config.scan_angle_step_rad}')
    print(f'scan angle rad: {holo_tomo_beam_config.scan_angle_rad}')
    print(f'scan axis angle: {holo_tomo_beam_config.scan_axis_angle}')

    print("All tests passed!")