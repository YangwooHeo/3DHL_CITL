from mbvam.Beam.holobeamconfig import HoloBeamConfig
import numpy as np

class MuxHoloBeamConfig(HoloBeamConfig):
    '''
    This class is a subclass of HoloBeamConfig and store additional parameter for time-multiplexed holographic beam configurations.
    '''
    
    def __init__(self):
        super().__init__()
        self.num_sub_beam = 10 # int, number of sub-beams

if __name__ == '__main__':
    ...