"""Separable full-grid tilted-SLM reprojection with GPU frequency slabs.

The discrete map is the reference padded FFT / Lanczos-6 gather / inverse FFT
and crop. Padding is only on the incidence axis. No optical tile approximation
or reduced sampling is used: every transform spans its entire physical axis.
The perpendicular frequency is unchanged by the tilt, so each frequency slab
can finish the incidence-axis transform independently. A following real relay
filter can share the perpendicular FFT without moving it across the crop.
"""
import math

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


def _fft_centered(values):
    return torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(values, dim=-1), dim=-1), dim=-1)


def _ifft_centered(values):
    return torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(values, dim=-1), dim=-1), dim=-1)


class _TiltSlabs:
    def __init__(self, field, pitch, wavelength, angle_deg, axis, magnification,
                 padding, processing_device, rows, cols, mask_builder, filter_options, fused):
        if field.ndim != 2 or not field.is_complex():
            raise ValueError('field must be a 2D complex tensor indexed (x, y)')
        if axis not in ('x', 'y'):
            raise ValueError('incidence axis must be x or y')
        if not math.isfinite(angle_deg) or abs(angle_deg) >= 90:
            raise ValueError('SLM incidence must be inside (-90, 90) degrees')
        if any(not math.isfinite(v) or v <= 0 for v in (pitch, wavelength, magnification)):
            raise ValueError('pitch, wavelength and magnification must be positive and finite')
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (padding, rows, cols)):
            raise ValueError('padding and slab sizes must be positive integers')
        self.axis, self.input_device, self.dtype = axis, field.device, field.dtype
        self.device = torch.device(processing_device)
        self.fused = fused
        self.nx, self.ny = field.shape if axis == 'x' else field.T.shape
        self.pad = (self.nx * (padding - 1) + 1) // 2
        self.n = self.nx + 2 * self.pad
        self.rows, self.cols = rows, cols
        self.c, self.s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        self.m, self.k = magnification, 1.0 / wavelength
        self.pitch, self.wavelength = pitch, wavelength
        opts = dict(device=self.device, dtype=field.real.dtype)
        self.fx = torch.fft.fftshift(torch.fft.fftfreq(self.n, d=pitch, **opts))[None, :]
        self.fy = torch.fft.fftshift(torch.fft.fftfreq(self.ny, d=pitch, **opts))[:, None]
        self.origin = torch.exp(-2j * math.pi * pitch * (.5 if self.n % 2 == 0 else 0) * self.fx)
        self.mask_builder = mask_builder
        self.filter_options = dict(filter_options or {})
        if mask_builder is not None:
            self.filter_fx = torch.fft.fftshift(torch.fft.fftfreq(self.nx, d=pitch, **opts))[None, :]

    def _geometry(self, start, stop):
        transverse_sq = self.m**2 * (self.fx**2 + self.fy[start:stop]**2)
        kz = torch.sqrt(torch.clamp(self.k*self.k - transverse_sq, min=0))
        delta_kz = -transverse_sq / (kz + self.k)
        q = self.c * self.fx + (self.s / self.m) * delta_kz
        jacobian = self.c - self.s * self.m * self.fx / kz.clamp_min(self.k * 1e-12)
        sample_index = q * (self.n * self.pitch) + self.n // 2
        valid_ray = (transverse_sq < self.k*self.k) & (jacobian > 0)
        return sample_index, torch.floor(sample_index).long(), jacobian * valid_ray

    def _rotate(self, spectrum, start, stop, adjoint):
        if self.fused:
            from .slm_tilt_cuda import rotate_lanczos_cuda
            result = rotate_lanczos_cuda(spectrum, self.fx, self.fy[start:stop],
                m=self.m, k=self.k, c=self.c, s=self.s, n_pitch=self.n*self.pitch, adjoint=adjoint)
            if result is not None:
                return result
        sample_index, base, ray_weight = self._geometry(start, stop)
        result = torch.zeros_like(spectrum)
        if adjoint:
            spectrum = spectrum * ray_weight
        for offset in range(-5, 7):
            index = base + offset
            distance = sample_index - index
            weight = torch.sinc(distance) * torch.sinc(distance / 6)
            valid = (index >= 0) & (index < self.n) & (distance.abs() < 6)
            if adjoint:
                result.scatter_add_(1, index.clamp(0, self.n - 1), spectrum * weight * valid)
            else:
                result += spectrum.gather(1, index.clamp(0, self.n - 1)) * weight * valid
        if not adjoint:
            result *= ray_weight
        return result

    def _filter(self, spatial_x, start, stop):
        # Keep the unpadded x crop BEFORE the physical Fourier-plane filter.
        fx, fy = self.m * self.filter_fx, self.m * self.fy[start:stop]
        if self.axis == 'y':
            fx, fy = fy, fx
        mask = self.mask_builder(fx, fy, wavelength_m=self.wavelength, **self.filter_options)
        return _ifft_centered(_fft_centered(spatial_x) * mask)

    @torch.no_grad()
    def apply(self, field, adjoint=False):
        source = field if self.axis == 'x' else field.T
        # Transposed CPU staging makes incidence-axis frequency batches contiguous.
        staging = torch.empty((self.ny, self.nx), dtype=self.dtype, device='cpu')
        for start in range(0, self.nx, self.rows):
            stop = min(start + self.rows, self.nx)
            tile = source[start:stop].to(self.device)
            spectral_y = _fft_centered(tile)
            staging[:, start:stop].copy_(spectral_y.T.cpu())
            del tile, spectral_y
        for start in range(0, self.ny, self.cols):
            stop = min(start + self.cols, self.ny)
            tile = staging[start:stop].to(self.device)
            if adjoint and self.mask_builder is not None:
                tile = self._filter(tile, start, stop)
            spectrum = _fft_centered(F.pad(tile, (self.pad, self.pad)))
            spectrum *= self.origin
            rotated = self._rotate(spectrum, start, stop, adjoint)
            del spectrum, tile
            rotated *= self.origin.conj()
            spatial = _ifft_centered(rotated)[:, self.pad:self.pad + self.nx]
            if not adjoint and self.mask_builder is not None:
                spatial = self._filter(spatial, start, stop)
            staging[start:stop].copy_(spatial.cpu())
            del rotated, spatial
        result = torch.empty((self.nx, self.ny), dtype=self.dtype, device=self.input_device)
        for start in range(0, self.nx, self.rows):
            stop = min(start + self.rows, self.nx)
            tile = staging[:, start:stop].T.contiguous().to(self.device)
            result[start:stop].copy_(_ifft_centered(tile).to(self.input_device))
            del tile
        return result if self.axis == 'x' else result.T


class _StreamedTilt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, field, operator):
        ctx.operator = operator
        return operator.apply(field)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return ctx.operator.apply(gradient, adjoint=True), None


def reproject_slm_field_streamed(field, pitch, wavelength, angle_deg=0.0, axis='x',
                                  magnification=1.0, padding=2, processing_device='cuda',
                                  rows=256, cols=64, mask_builder=None, filter_options=None, fused=True):
    """Same angular-spectrum map and optional relay filter, evaluated in slabs.

    Returns on the input device; full fields may stay in host RAM while FFTs,
    Lanczos gather/scatter and filter operations run on the selected device.
    The unchanged perpendicular half-pixel origin cancels through the gather.
    """
    operator = _TiltSlabs(field, pitch, wavelength, angle_deg, axis, magnification,
                          padding, processing_device, rows, cols, mask_builder, filter_options, fused)
    if angle_deg == 0:
        # The zero-angle reference is an exact identity, not sinc interpolation.
        if mask_builder is None:
            return field
        from .slm_tilt import apply_relay_spatial_filter
        return apply_relay_spatial_filter(field, pitch, wavelength, magnification,
                                         mask_builder, filter_options or {})
    return _StreamedTilt.apply(field, operator)
