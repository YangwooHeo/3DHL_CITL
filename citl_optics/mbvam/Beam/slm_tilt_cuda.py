"""Optional fused CUDA interpolation using PyTorch's bundled NVRTC runtime.

No nvcc, C++ compiler, pip package, or disk build cache is required. Compile
once per device/process. If NVRTC is unavailable, retain the PyTorch slab path.
Only complex64 is accelerated; complex128 uses the PyTorch implementation.
"""
import ctypes as C
import ctypes.util
from functools import lru_cache
import os
from pathlib import Path
import warnings

import torch


def _check(status, operation):
    if status:
        raise RuntimeError(f'{operation} failed with CUDA/NVRTC status {status}')


def _bind(lib, name, arguments):
    function = getattr(lib, name)
    function.argtypes, function.restype = arguments, C.c_int
    return function


class _Kernel:
    def __init__(self, device):
        lib_dir = Path(torch.__file__).resolve().parent/'lib'
        if os.name == 'nt':
            candidates = sorted(p for p in lib_dir.glob('nvrtc64_*.dll') if '.alt.' not in p.name)
            if not candidates:
                raise OSError('PyTorch bundled NVRTC DLL not found')
            self.nvrtc = C.CDLL(str(candidates[0]))
            self.driver = C.WinDLL('nvcuda.dll')
        else:
            candidates = list(lib_dir.glob('libnvrtc.so*'))
            nvidia = lib_dir.parent.parent/'nvidia/cuda_nvrtc/lib'
            candidates += list(nvidia.glob('libnvrtc.so*'))
            library = str(candidates[0]) if candidates else C.util.find_library('nvrtc')
            if not library:
                raise OSError('NVRTC shared library not found')
            self.nvrtc = C.CDLL(library)
            self.driver = C.CDLL('libcuda.so.1')
        ptr, integer, text = C.c_void_p, C.c_int, C.c_char_p
        create = _bind(self.nvrtc,'nvrtcCreateProgram',
                       [C.POINTER(ptr),text,text,integer,C.POINTER(text),C.POINTER(text)])
        compile_program = _bind(self.nvrtc,'nvrtcCompileProgram',[ptr,integer,C.POINTER(text)])
        log_size = _bind(self.nvrtc,'nvrtcGetProgramLogSize',[ptr,C.POINTER(C.c_size_t)])
        get_log = _bind(self.nvrtc,'nvrtcGetProgramLog',[ptr,ptr])
        ptx_size = _bind(self.nvrtc,'nvrtcGetPTXSize',[ptr,C.POINTER(C.c_size_t)])
        get_ptx = _bind(self.nvrtc,'nvrtcGetPTX',[ptr,ptr])
        destroy = _bind(self.nvrtc,'nvrtcDestroyProgram',[C.POINTER(ptr)])
        program = ptr()
        source = Path(__file__).with_name('slm_tilt_lanczos.cu').read_bytes()
        _check(create(C.byref(program),source,b'slm_tilt_lanczos.cu',0,None,None),'nvrtcCreateProgram')
        try:
            major,minor = torch.cuda.get_device_capability(device)
            options = [f'--gpu-architecture=compute_{major}{minor}'.encode(), b'--std=c++11',
                       b'--fmad=false',b'--ftz=false',b'--prec-div=true',b'--prec-sqrt=true']
            status = compile_program(program,len(options),(text*len(options))(*options))
            if status:
                size = C.c_size_t()
                _check(log_size(program,C.byref(size)),'nvrtcGetProgramLogSize')
                log = C.create_string_buffer(size.value)
                _check(get_log(program,log),'nvrtcGetProgramLog')
                raise RuntimeError(log.value.decode(errors='replace'))
            size = C.c_size_t()
            _check(ptx_size(program,C.byref(size)),'nvrtcGetPTXSize')
            ptx = C.create_string_buffer(size.value)
            _check(get_ptx(program,ptx),'nvrtcGetPTX')
        finally:
            destroy(C.byref(program))
        load = _bind(self.driver,'cuModuleLoadData',[C.POINTER(ptr),ptr])
        get_function = _bind(self.driver,'cuModuleGetFunction',[C.POINTER(ptr),ptr,text])
        self.module,self.function = ptr(),ptr()
        _check(load(C.byref(self.module),ptx),'cuModuleLoadData')
        _check(get_function(C.byref(self.function),self.module,b'lanczos_tilt'),'cuModuleGetFunction')
        self.launch = _bind(self.driver,'cuLaunchKernel',
            [ptr]+[C.c_uint]*7+[ptr,C.POINTER(ptr),C.POINTER(ptr)])

    def __call__(self, spectrum, fx, fy, *, m, k, c, s, n_pitch, adjoint):
        result = torch.zeros_like(spectrum) if adjoint else torch.empty_like(spectrum)
        n, rows = spectrum.shape[1], spectrum.shape[0]
        values = [C.c_void_p(t.data_ptr()) for t in (spectrum,result,fx,fy)]
        values += [C.c_int(n),C.c_int(rows)]
        values += [C.c_float(v) for v in (m*m,k,k*k,k*1e-12,c,s/m,s*m,n_pitch)]
        values += [C.c_int(adjoint)]
        pointers = (C.c_void_p*len(values))(*[C.cast(C.byref(v),C.c_void_p) for v in values])
        stream = C.c_void_p(torch.cuda.current_stream(spectrum.device).cuda_stream)
        _check(self.launch(self.function,(n*rows+255)//256,1,1,256,1,1,0,stream,pointers,None),
               'cuLaunchKernel(lanczos_tilt)')
        return result


@lru_cache(maxsize=None)
def _kernel(device_index):
    try:
        with torch.cuda.device(device_index):
            # CUDA driver module loading requires a current context. Selecting
            # a device alone may not create one on a freshly initialized runtime.
            context_probe = torch.empty(1, device=f'cuda:{device_index}')
            return _Kernel(device_index)
    except (OSError,RuntimeError,AttributeError) as error:
        warnings.warn(f'Fused tilt kernel unavailable; using PyTorch slabs: {error}',RuntimeWarning)
        return None


def rotate_lanczos_cuda(spectrum, fx, fy, *, m, k, c, s, n_pitch, adjoint):
    """Return None when unsupported; preserve original operations on fallback."""
    if spectrum.device.type != 'cuda' or spectrum.dtype != torch.complex64:
        return None
    if spectrum.ndim != 2 or fx.device != spectrum.device or fy.device != spectrum.device:
        raise ValueError('Tilt spectrum and frequency axes must share one CUDA device')
    if spectrum.numel() >= 2**31:
        return None
    if not (spectrum.is_contiguous() and fx.is_contiguous() and fy.is_contiguous()):
        return None
    if fx.dtype != torch.float32 or fy.dtype != torch.float32:
        return None
    if fx.numel() != spectrum.shape[1] or fy.numel() != spectrum.shape[0]:
        raise ValueError('Tilt frequency axes do not match the spectrum slab')
    with torch.cuda.device(spectrum.device):
        kernel = _kernel(spectrum.device.index)
        if kernel is None:
            return None
        return kernel(spectrum,fx,fy,m=m,k=k,c=c,s=s,n_pitch=n_pitch,adjoint=adjoint)
