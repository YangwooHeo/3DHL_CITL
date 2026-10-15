"""Full-grid axicon ASM with CPU staging and an explicit complex adjoint.

Only the execution/storage order changes: every FFT axis retains its full
length, and the same ring mask and sampling conventions are used. No optical
patch approximation is made. The linear map starts AFTER the SLM spatial
filter; trainable source, phase, XY sampling and ROI corrections stay in
ordinary autograd. Nominal z and direct axicon geometry here are constants.
The input grid can have a noninteger ratio to the optical grid; its linear
resampling and matching adjoint are also evaluated in bounded row slabs.
"""
import math

import torch
from torch.autograd.function import once_differentiable

from .field_resample import ComplexFieldResampler, scaled_field_shape


def build_streamed_transfer(beam, z_query, upsample_factor=8, n_medium=1.0,
                            axicon_angle=None, margin_factor=5,
                            axicon_angle_in_medium=False,
                            axicon_transverse_frequency=None,
                            axicon_profile='continuous',
                            axicon_phase_depth=math.pi, axicon_duty_cycle=0.5,
                            axicon_diffraction_orders=None,
                            rows=256, cols=256):
    """Build small geometry metadata. margin_factor=None keeps the full spectrum."""
    cfg = beam.beam_config
    if int(upsample_factor) != upsample_factor or upsample_factor < 1:
        raise ValueError('upsample_factor must be a positive integer')
    if rows < 1 or cols < 1 or int(rows) != rows or int(cols) != cols:
        raise ValueError('streamed FFT row/column sizes must be positive integers')
    medium = float(torch.as_tensor(n_medium, dtype=cfg.fdtype).item())
    if not math.isfinite(medium) or medium <= 0:
        raise ValueError('n_medium must be finite and positive')
    if margin_factor is not None and (not math.isfinite(margin_factor) or margin_factor < 0):
        raise ValueError('margin_factor must be finite and non-negative, or None')
    z = torch.atleast_1d(torch.as_tensor(z_query, device=cfg.device, dtype=cfg.fdtype))
    if z.ndim != 1 or not z.numel() or not bool(torch.isfinite(z).all()):
        raise ValueError('z_query must be a nonempty finite vector')
    if z.requires_grad:
        raise ValueError('Streamed nominal z is fixed; use the proxy ROI delta-z parameter')
    masked = margin_factor is not None and (axicon_angle is not None or axicon_transverse_frequency is not None)
    pitch = float(cfg.psSLM) / int(upsample_factor)
    frequency, orders = None, None
    if masked:
        frequency = float(beam._resolve_axicon_radial_frequency(
            axicon_angle, medium, axicon_angle_in_medium,
            axicon_transverse_frequency).item())
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError('Axicon transverse frequency must be finite and positive')
        orders = beam._resolve_axicon_diffraction_orders(
            axicon_profile, frequency,
            min(math.sqrt(2) / (2 * pitch), medium / cfg.lambda_),
            axicon_phase_depth, axicon_duty_cycle, axicon_diffraction_orders)
    return {
        'type': 'streamed', 'Nx_up': int(cfg.Nx * upsample_factor),
        'Ny_up': int(cfg.Ny * upsample_factor), 'pitch': pitch,
        'wavelength': float(cfg.lambda_), 'n_medium': medium,
        'z_query': z.detach().clone(), 'ring_frequency': frequency,
        'band_half_width': None if margin_factor is None else margin_factor / (cfg.Nx * cfg.psSLM),
        'axicon_diffraction_orders': orders, 'rows': int(rows), 'cols': int(cols),
    }


class _StreamedAxiconMap:
    def __init__(self, field, transfer, repeat, roi, beam, z_index, *,
                 radial_frequency, profile, phase_depth, duty_cycle,
                 radial_offset, shift_x, shift_y, resample_mode='auto',
                 processing_device=None):
        self.shape, self.input_device, self.dtype = tuple(field.shape), field.device, field.dtype
        self.device = field.device if processing_device is None else torch.device(processing_device)
        self.real_dtype = field.real.dtype
        self.roi = int(roi)
        self.nx, self.ny = scaled_field_shape(self.shape, repeat)
        if (self.nx, self.ny) != (transfer['Nx_up'], transfer['Ny_up']):
            raise ValueError('Streamed transfer grid does not match resampled SLM field')
        if roi != self.roi or not 0 < self.roi <= min(self.nx, self.ny):
            raise ValueError('ROI must be a positive integer within the propagation grid')
        self.resampler = ComplexFieldResampler(
            self.shape, (self.nx, self.ny), resample_mode, self.input_device, self.real_dtype)
        # Only the integer-repeat fast path needs complete repeat blocks.
        q = self.resampler.repeat or 1
        self.rows = max(q, int(transfer['rows']) // q * q)
        self.cols = transfer['cols']
        self.x0, self.y0 = self.nx // 2 - self.roi // 2, self.ny // 2 - self.roi // 2
        self.transfer_spec = transfer
        self.z = transfer['z_query'][z_index].to(device=self.device, dtype=self.real_dtype)
        self.axicon_phase = beam._axicon_phase_from_radius
        self.frequency = torch.as_tensor(radial_frequency, device=self.device, dtype=self.real_dtype)
        if self.frequency.requires_grad:
            raise ValueError('The streamed axicon frequency must be fixed')
        self.profile, self.phase_depth, self.duty_cycle = profile, phase_depth, duty_cycle
        self.radial_offset, self.shift_x, self.shift_y = radial_offset, shift_x, shift_y
        opts = dict(device=self.device, dtype=self.real_dtype)
        pitch = transfer['pitch']
        self.x = torch.linspace(-(self.nx - 1) / 2, (self.nx - 1) / 2, self.nx, **opts) * pitch
        self.y = torch.linspace(-(self.ny - 1) / 2, (self.ny - 1) / 2, self.ny, **opts) * pitch
        self.fx = torch.fft.fftfreq(self.nx, d=pitch, **opts)
        self.fy = torch.fft.fftfreq(self.ny, d=pitch, **opts)

    def axicon(self, start, stop):
        radius = torch.sqrt((self.x[start:stop, None] - self.shift_x).square()
                            + (self.y[None] - self.shift_y).square())
        phase = self.axicon_phase(radius, self.frequency, self.profile,
                                 self.phase_depth, self.duty_cycle, self.radial_offset)
        return torch.exp(1j * phase)

    def transfer(self, start, stop):
        # Transposed layout gives contiguous full-length x-axis transforms.
        spec = self.transfer_spec
        kr = torch.sqrt(self.fx[None].square() + self.fy[start:stop, None].square())
        medium = torch.as_tensor(spec['n_medium'], device=self.device, dtype=self.real_dtype)
        gamma_sq = (1 - (spec['wavelength'] * kr / medium).square()
                    if spec['ring_frequency'] is not None else
                    1 - spec['wavelength']**2 * kr.square() / medium.square())
        mask = gamma_sq >= 0
        if spec['ring_frequency'] is not None:
            frequency = torch.as_tensor(spec['ring_frequency'], device=self.device, dtype=self.real_dtype)
            ring_mask = torch.zeros_like(mask)
            for order in spec['axicon_diffraction_orders']:
                ring_mask |= ((kr <= spec['band_half_width']) if order == 0 else
                              (kr - float(order) * frequency).abs() <= spec['band_half_width'])
            mask &= ring_mask
        km = 2 * torch.pi * medium / spec['wavelength']
        return torch.exp(1j * km * self.z * gamma_sq.clamp_min(0).sqrt()) * mask

    @torch.no_grad()
    def forward(self, field):
        staging = torch.empty((self.ny, self.nx), dtype=self.dtype, device='cpu')
        intermediate = torch.empty((self.roi, self.ny), dtype=self.dtype, device=self.device)
        for start in range(0, self.nx, self.rows):
            stop = min(start + self.rows, self.nx)
            tile = self.resampler.tile(field, start, stop).to(self.device)
            tile *= self.axicon(start, stop)
            spectrum = torch.fft.fft(tile, dim=1, norm='ortho')
            host_tile = spectrum.T.contiguous().cpu()
            staging[:, start:stop].copy_(host_tile)
            del tile, spectrum, host_tile
        for start in range(0, self.ny, self.cols):
            stop = min(start + self.cols, self.ny)
            tile = staging[start:stop].to(self.device)
            spectrum = torch.fft.fft(tile, dim=1, norm='ortho')
            spectrum *= self.transfer(start, stop)
            spatial = torch.fft.ifft(spectrum, dim=1, norm='ortho')
            intermediate[:, start:stop].copy_(spatial[:, self.x0:self.x0 + self.roi].T)
            del tile, spectrum, spatial
        result = torch.fft.ifft(intermediate, dim=1, norm='ortho')
        return result[:, self.y0:self.y0 + self.roi].clone()

    @torch.no_grad()
    def adjoint(self, grad):
        staging = torch.empty((self.ny, self.nx), dtype=self.dtype, device='cpu')
        intermediate = torch.zeros((self.roi, self.ny), dtype=self.dtype, device=self.device)
        intermediate[:, self.y0:self.y0 + self.roi] = grad
        intermediate = torch.fft.fft(intermediate, dim=1, norm='ortho')
        for start in range(0, self.ny, self.cols):
            stop = min(start + self.cols, self.ny)
            tile = torch.zeros((stop - start, self.nx), dtype=self.dtype, device=self.device)
            tile[:, self.x0:self.x0 + self.roi] = intermediate[:, start:stop].T
            spectrum = torch.fft.fft(tile, dim=1, norm='ortho')
            spectrum *= self.transfer(start, stop).conj()
            spatial = torch.fft.ifft(spectrum, dim=1, norm='ortho')
            staging[start:stop].copy_(spatial.cpu())
            del tile, spectrum, spatial
        del intermediate
        result = torch.zeros(self.shape, dtype=self.dtype, device=self.input_device)
        for start in range(0, self.nx, self.rows):
            stop = min(start + self.rows, self.nx)
            tile = staging[:, start:stop].T.contiguous().to(self.device)
            spatial = torch.fft.ifft(tile, dim=1, norm='ortho')
            spatial *= self.axicon(start, stop).conj()
            self.resampler.add_adjoint(result, spatial.to(self.input_device), start, stop)
            del tile, spatial
        return result


class _LinearPropagation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, field, operator):
        ctx.operator = operator
        return operator.forward(field)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        return ctx.operator.adjoint(grad), None


def propagate_streamed(field, transfer, repeat, roi, beam, **axicon_geometry):
    """Return HxWxNz fields with first-order gradients to the input field."""
    if field.ndim != 2 or not field.is_complex():
        raise ValueError('Streamed axicon expects a 2D complex field')
    outputs = []
    for z_index in range(transfer['z_query'].numel()):
        operator = _StreamedAxiconMap(field, transfer, repeat, roi, beam,
                                     z_index, **axicon_geometry)
        outputs.append(_LinearPropagation.apply(field, operator))
    return torch.stack(outputs, dim=-1)
