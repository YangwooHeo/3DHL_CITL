import numpy as np

class CoordinateArray(np.ndarray):
    '''
    This subclass of ndarray has additional attributes that stores some attributes of the coordinate vector.
    https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
    https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array

    Parameters
    ----------
    in_free_space: The coordinate attribute 'in_free_space' is used to indicate whether material is inserted into the beam.
    True means no material is inserted. The coordinate represent the corresponding beam position in air.
    False means the stack of material is inserted. The coordinate represent the corresponding beam position in respective media.
    This is checked in HoloBeamConfig.getRefractedZPositions, HoloBeamConfig.getFreeSpaceZPositions, HoloBeam.generateASPropagationTF.
    
    physical_unit: 'm', 'mm', 'um'. This is used to keep track of the physical unit of the array.
    Not checked for now.
    
    frame: 'local' or 'global'. This is used to keep track of the frame of the array.
    Not checked for now.
    '''
    def __new__(cls, input, in_free_space, physical_unit='m', frame=None) -> None:
        '''As you can see, the object can be initialized in the __new__ method or the __init__ method, or both.
        In fact ndarray does not have an __init__ method, because all the initialization is done in the __new__ method.
        Why use __new__ rather than just the usual __init__?
        Because in some cases, as for ndarray, we want to be able to return an object of some other class.'''
        obj = np.asarray(input).astype(float).view(cls)
        obj.in_free_space = in_free_space
        obj.physical_unit = physical_unit
        obj.frame = frame
        return obj
    
    def __array_finalize__(self, obj): 
        '''
        __array_finalize__ is the mechanism that numpy provides to allow subclasses to handle the various ways that new instances get created.
        In cast_a = a.view(subclass), this method is called with "self" being an instance of subclass, and with "obj" = a.

        When called from the explicit constructor __new__, obj is None. For example, when calling, a = CoordinateArray([1,2,3]).
        When called from view casting, obj can be an instance of ANY subclass of ndarray, including our own. For example: cast_a = a.view(C) where a is an instance of any subclass of ndarray.
        When called in new-from-template, obj is another instance of our own subclass, that we might use to update the new self instance. For example: b = a[::2] where a is an instance of our subclass.
        '''
        if obj is None: return
        self.in_free_space = getattr(obj, 'in_free_space', None) #This makes the view's of CoordinateArray have the same in_free_space as the original array.
        self.physical_unit = getattr(obj, 'physical_unit', None) #This makes the view's of CoordinateArray have the same physical_unit as the original array.
        self.frame = getattr(obj, 'frame', None) #This makes the view's of CoordinateArray have the same frame as the original array.



if __name__ == "__main__":

    # Note that any function modifying the object should not modify it in-place, but return a new object.
    # This is because the object is passed by reference, and modifying it in-place will modify the original object.
    def func_1(Xv):
        print(f'In function before mod: in_free_space: {Xv.in_free_space}')
        Xv.in_free_space = not Xv.in_free_space
        print(f'In function after mod: {Xv.in_free_space}')
        return Xv
    

    Xv = CoordinateArray([1,2,3],
                in_free_space=True,
                physical_unit='m',
                frame='local')
    print(f'Xv in_free_space before function call: {Xv.in_free_space}')
    Xv_new = func_1(Xv)
    print(f'Xv in_free_space after function call: {Xv.in_free_space}')
    print(f'Xv_new in_free_space: {Xv_new.in_free_space}')

    #########################################
    from copy import deepcopy
    def func_2(Xv):
        Xv_new = deepcopy(Xv)
        print(f'In function before mod: {Xv_new.in_free_space}')
        Xv_new.in_free_space = not Xv_new.in_free_space
        print(f'In function after mod: {Xv_new.in_free_space}')
        return Xv_new
    

    Xv = CoordinateArray([1,2,3],
                in_free_space=True,
                physical_unit='m',
                frame='local')
    print(f'Xv in_free_space before function call: {Xv.in_free_space}')
    Xv_new = func_2(Xv)
    print(f'Xv in_free_space after function call: {Xv.in_free_space}')
    print(f'Xv_new in_free_space: {Xv_new.in_free_space}')


    # Same goes for normal numpy arrays...
    import numpy as np

    Xv = np.array([1,2,3])
    print(f'Xv before function call: {Xv}')

    def np_fun1(Xv):
        print(f'In function before mod: in_free_space: {Xv}')
        Xv[1] = 100
        print(f'In function after mod: in_free_space: {Xv}')
        return Xv
    
    Xv_new = np_fun1(Xv)
    print(f'Xv after function call: {Xv}')
    print(f'Xv_new : {Xv_new}')