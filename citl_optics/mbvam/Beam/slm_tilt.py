"""Scalar tilted-SLM geometry in a relay aligned to the specular reflected ray.

The carrier has been removed analytically. This is NOT a beam-steering phase
ramp. The separate, normally illuminated axicon is applied AFTER this operator.
All arrays use (x, y); pitch is the untilted pitch AFTER the ideal relay.
"""

import math

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


def tilt_execution_options(config, backend=None, cols=None):
    """Execution only: changing this does not change checkpoint optical geometry."""
    backend = getattr(config, 'slm_tilt_backend', 'auto') if backend is None else backend
    cols = getattr(config, 'slm_tilt_stream_cols', 64) if cols is None else cols
    if backend not in ('auto', 'reference', 'streamed'):
        raise ValueError('slm_tilt_backend must be auto, reference, or streamed')
    if not isinstance(cols, int) or isinstance(cols, bool) or cols < 1:
        raise ValueError('slm_tilt_stream_cols must be a positive integer')
    return backend, cols


def incidence_options(config, angle_deg=None, axis=None, model=None,
                      padding=None, magnification=None):
    """Resolve explicit overrides or beam config; old configs stay at 0 degrees."""
    angle = float(getattr(config, 'slm_incidence_angle_deg', 0.0) if angle_deg is None else angle_deg)
    axis = getattr(config, 'slm_incidence_axis', 'x') if axis is None else axis
    model = getattr(config, 'slm_tilt_model', 'angular_spectrum') if model is None else model
    padding = getattr(config, 'slm_tilt_padding', 2) if padding is None else padding
    magnification = float(getattr(config, 'relay_magnification', 1.0) if magnification is None else magnification)
    if not math.isfinite(angle) or abs(angle) >= 90:
        raise ValueError('SLM incidence must be inside (-90, 90) degrees.')
    if axis not in ('x', 'y') or model not in ('angular_spectrum', 'projection'):
        raise ValueError('SLM incidence axis must be x/y; model angular_spectrum/projection.')
    if not isinstance(padding, int) or isinstance(padding, bool) or padding < 1:
        raise ValueError('SLM tilt padding must be a positive integer.')
    if not math.isfinite(magnification) or magnification <= 0:
        raise ValueError('Relay magnification must be finite and positive.')
    return dict(angle_deg=angle, axis=axis, model=model, padding=padding,
                magnification=magnification)


def apply_relay_spatial_filter(field, pitch, wavelength, magnification,
                               mask_builder, filter_options, chunk_rows=128):
    """Apply the existing center-block/Lee aperture in the reflected-ray frame.

    First Fourier-plane frequencies are M times final relay-plane frequencies.
    Thus a physical aperture stays fixed while the SLM itself is tilted.
    """
    options = (pitch, wavelength, magnification, mask_builder, dict(filter_options), chunk_rows)
    return _RelayFilter.apply(field, options)


class _RelayFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, field, options):
        ctx.options = options
        return _relay_filter(field, *options)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        # The existing center-block/Lee masks are real: F^-1 mask F is Hermitian.
        return _relay_filter(gradient, *ctx.options), None


def _relay_filter(field, pitch, wavelength, magnification,
                  mask_builder, filter_options, chunk_rows):
    fx = torch.fft.fftfreq(field.shape[0], d=pitch, device=field.device, dtype=field.real.dtype)
    fy = torch.fft.fftfreq(field.shape[1], d=pitch, device=field.device, dtype=field.real.dtype)
    spectrum = torch.fft.fft2(field)
    for start in range(0, field.shape[0], chunk_rows):
        mask = mask_builder(magnification * fx[start:start+chunk_rows, None],
                            magnification * fy[None, :], wavelength_m=wavelength,
                            **filter_options)
        spectrum[start:start+chunk_rows] = spectrum[start:start+chunk_rows] * mask
    return torch.fft.ifft2(spectrum)


def relay_magnification(focal_lengths):
    """Two ideal 4f relays, with focal lengths in any common length unit."""
    if len(focal_lengths) != 4 or any(not math.isfinite(f) or f <= 0 for f in focal_lengths):
        raise ValueError("Provide four finite positive relay focal lengths.")
    f1, f2, f3, f4 = focal_lengths
    return (f2 / f1) * (f4 / f3)


def apply_relay_center_block(field, pitch, wavelength, magnification,
                             width_m, focal_length_m=0.250, chunk_rows=128):
    """Square center stop in the FIRST relay Fourier plane, about its chief ray.

    Input has already been reprojected into the perpendicular outgoing frame
    and scaled by the full relay M. Hence first-Fourier-plane coordinate is
    lambda0*f1*M*frequency_out. This keeps the physical stop fixed for tilt.
    The filter is applied to the sampled complex field, including pixel spectra.
    """
    if not math.isfinite(width_m) or width_m < 0:
        raise ValueError('Center-block width must be finite and nonnegative.')
    if width_m == 0:
        return field
    if any(not math.isfinite(v) or v <= 0 for v in (pitch, wavelength, magnification, focal_length_m)):
        raise ValueError('Filter geometry requires finite positive lengths and magnification.')
    fx = torch.fft.fftfreq(field.shape[0], d=pitch, device=field.device, dtype=field.real.dtype)
    fy = torch.fft.fftfreq(field.shape[1], d=pitch, device=field.device, dtype=field.real.dtype)
    limit = width_m / (2 * wavelength * focal_length_m * magnification)
    spectrum = torch.fft.fft2(field)
    y_block = fy.abs()[None, :] <= limit
    for start in range(0, field.shape[0], chunk_rows):
        blocked = (fx[start:start+chunk_rows, None].abs() <= limit) & y_block
        spectrum[start:start+chunk_rows] = spectrum[start:start+chunk_rows] * (~blocked)
    return torch.fft.ifft2(spectrum)


def reproject_slm_field(field, pitch, wavelength, angle_deg=0.0, axis="x",
                        magnification=1.0, model="angular_spectrum",
                        padding=2, chunk_rows=128):
    """Tilt reprojection with a fixed-geometry, first-order memory-bounded adjoint.

    Angular-spectrum backward recomputes Lanczos weights in row slabs and
    scatters their conjugate transpose. No field, full interpolation indices,
    or weights are retained for backward. Full padded CPU FFT arrays are still
    needed. Geometry is fixed; trainable phase/source/crosstalk remain connected.
    The projection-only comparison model uses ordinary autograd.
    """
    options = (pitch, wavelength, angle_deg, axis, magnification, model, padding, chunk_rows)
    if model != 'angular_spectrum' or angle_deg == 0:
        return _reproject_slm_field_reference(field, *options)
    return _TiltReprojection.apply(field, options)


class _TiltReprojection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, field, options):
        ctx.options = options
        return _reproject_slm_field_reference(field, *options)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return _reproject_angular_adjoint(gradient, *ctx.options), None


def _reproject_angular_adjoint(gradient, pitch, wavelength, angle_deg, axis,
                              magnification, model, padding, chunk_rows):
    """Exact discrete adjoint of padding, FFT, Lanczos gather and crop."""
    grad = gradient if axis == 'x' else gradient.T
    nx, ny = grad.shape
    pad = (nx * (padding - 1) + 1) // 2
    padded = F.pad(grad, (0, 0, pad, pad))
    n = padded.shape[0]
    opts = dict(device=grad.device, dtype=grad.real.dtype)
    fx = torch.fft.fftshift(torch.fft.fftfreq(n, d=pitch, **opts))
    fy = torch.fft.fftshift(torch.fft.fftfreq(ny, d=pitch, **opts))
    origin_x = torch.exp(-2j * math.pi * pitch * (0.5 if n % 2 == 0 else 0) * fx[:, None])
    origin_y = torch.exp(-2j * math.pi * pitch * (0.5 if ny % 2 == 0 else 0) * fy[None, :])
    rotated = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(padded)))
    del padded
    rotated *= origin_x
    rotated *= origin_y
    spectrum = torch.zeros_like(rotated)
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    k = 1.0 / wavelength
    for start in range(0, n, chunk_rows):
        f = fx[start:start + chunk_rows, None]
        transverse_sq = magnification**2 * (f**2 + fy[None, :]**2)
        kz = torch.sqrt(torch.clamp(k*k - transverse_sq, min=0))
        delta_kz = -transverse_sq / (kz + k)
        q = c * f + (s / magnification) * delta_kz
        jacobian = c - s * magnification * f / kz.clamp_min(k * 1e-12)
        sample_index = q * (n * pitch) + n // 2
        base = torch.floor(sample_index).to(torch.long)
        valid_ray = (transverse_sq < k*k) & (jacobian > 0)
        tile = rotated[start:start + len(f)] * jacobian * valid_ray
        for offset in range(-5, 7):
            index = base + offset
            distance = sample_index - index
            weight = torch.sinc(distance) * torch.sinc(distance / 6)
            valid = (index >= 0) & (index < n) & (distance.abs() < 6)
            spectrum.scatter_add_(0, index.clamp(0, n - 1), tile * weight * valid)
    del rotated
    spectrum *= origin_x.conj()
    spectrum *= origin_y.conj()
    result = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(spectrum)))[pad:pad + nx]
    return (result if axis == 'x' else result.T).clone(memory_format=torch.contiguous_format)


def _reproject_slm_field_reference(field, pitch, wavelength, angle_deg=0.0, axis="x",
                        magnification=1.0, model="angular_spectrum",
                        padding=2, chunk_rows=128):
    """Map a sampled SLM envelope to the perpendicular relay output plane.

    Assumptions: ideal two-4f imaging in air, centered on the reflected chief
    ray, SLM center at the nominal object plane, no pupil clipping/aberrations.
    Input coordinates are M times SLM-surface coordinates. The signed tilt
    convention is z_SLM = +u*sin(angle); x_SLM = u*cos(angle) for axis='x'.
    'y' exchanges x and y. A negative angle reverses this slope.

    For output frequencies (fx, fy), rotate BEFORE relay magnification:
      qx = cos(theta)*fx + sin(theta)/M * (sqrt(k**2-M**2*f**2)-k)
      qy = fy,  k=1/lambda0 (cycles/m, in air)
      J  = cos(theta) - sin(theta)*M*fx/sqrt(k**2-M**2*f**2).
    Output spectrum = input_envelope_spectrum(qx,qy) * J.
    The subtraction of k removes the known specular carrier. The Jacobian
    is necessary: the low-bandwidth limit is U_out(x,y)=U_in(x/cos(theta),y).

    'projection' retains just this low-bandwidth limit. 'angular_spectrum'
    also accounts for diffraction between the tilted and perpendicular planes.
    Zero padding is along the tilt axis only; Lanczos-6 interpolates the
    centered complex spectrum. Increase padding (2 -> 4) to check convergence.
    Frequencies outside the sampled spectrum or forward propagating branch
    are discarded. Pixel sampling, finite window and scalar optics still limit
    accuracy. No phase-LUT, Fresnel/polarization or throughput calibration here.

    Supports gradients with respect to field; geometry is fixed, not fitted.
    0 degrees returns field unchanged, preserving the existing propagation.
    """
    if field.ndim != 2 or not field.is_complex():
        raise ValueError("field must be a 2D complex tensor, indexed (x, y).")
    if axis not in ("x", "y") or model not in ("projection", "angular_spectrum"):
        raise ValueError("axis must be x/y and model projection/angular_spectrum.")
    if not math.isfinite(angle_deg) or abs(angle_deg) >= 90:
        raise ValueError("SLM incidence angle must be finite and inside (-90, 90) degrees.")
    if any(not math.isfinite(v) or v <= 0 for v in (pitch, wavelength, magnification)):
        raise ValueError("pitch, wavelength and magnification must be positive and finite.")
    if not isinstance(padding, int) or padding < 1 or chunk_rows < 1:
        raise ValueError("padding and chunk_rows must be positive integers.")
    if angle_deg == 0:
        return field

    # A single implementation for both incidence planes; x is array dimension 0.
    source = field if axis == "x" else field.T
    nx, ny = source.shape
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    dtype, device = source.real.dtype, source.device
    if model == "projection":
        values = torch.view_as_real(source).permute(2, 0, 1).unsqueeze(0)
        chunks = []
        y = 2 * (torch.arange(ny, device=device, dtype=dtype) + 0.5) / ny - 1
        for start in range(0, nx, chunk_rows):
            x = torch.arange(start, min(start + chunk_rows, nx), device=device, dtype=dtype)
            x = 2 * ((x - (nx - 1) / 2) / c + nx / 2) / nx - 1
            grid = torch.stack((y.expand(len(x), -1), x[:, None].expand(-1, ny)), -1)
            sampled = F.grid_sample(values, grid[None], mode="bilinear", align_corners=False)
            chunks.append(torch.complex(sampled[0, 0], sampled[0, 1]))
        result = torch.cat(chunks, 0)
    else:
        # Symmetric padding preserves the half-pixel origin for even-sized arrays.
        pad = (nx * (padding - 1) + 1) // 2
        padded = F.pad(source, (0, 0, pad, pad))
        n = padded.shape[0]
        fx = torch.fft.fftshift(torch.fft.fftfreq(n, d=pitch, device=device, dtype=dtype))
        fy = torch.fft.fftshift(torch.fft.fftfreq(ny, d=pitch, device=device, dtype=dtype))
        # fftshift alone uses a sample origin; convert to the physical pixel-center origin.
        half_x, half_y = (0.5 if n % 2 == 0 else 0), (0.5 if ny % 2 == 0 else 0)
        origin_x = torch.exp(-2j * math.pi * pitch * half_x * fx[:, None])
        origin_y = torch.exp(-2j * math.pi * pitch * half_y * fy[None, :])
        spectrum = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(padded)))
        spectrum = spectrum * origin_x
        spectrum = spectrum * origin_y
        del padded
        cols = torch.arange(ny, device=device)[None, :]
        k = 1.0 / wavelength
        rotated = torch.empty_like(spectrum)
        for start in range(0, n, chunk_rows):
            f = fx[start:start + chunk_rows, None]
            transverse_sq = magnification**2 * (f**2 + fy[None, :]**2)
            kz = torch.sqrt(torch.clamp(k*k - transverse_sq, min=0))
            # Stable form of kz-k: direct subtraction loses precision near the axis.
            delta_kz = -transverse_sq / (kz + k)
            q = c * f + (s / magnification) * delta_kz
            jacobian = c - s * magnification * f / kz.clamp_min(k * 1e-12)
            sample_index = q * (n * pitch) + n // 2
            base = torch.floor(sample_index).to(torch.long)
            sampled = torch.zeros_like(q, dtype=source.dtype)
            # Windowed sinc interpolation, with zero extension outside the FFT band.
            for offset in range(-5, 7):
                index = base + offset
                distance = sample_index - index
                weight = torch.sinc(distance) * torch.sinc(distance / 6)
                valid = (index >= 0) & (index < n) & (distance.abs() < 6)
                sampled = sampled + spectrum[index.clamp(0, n - 1), cols] * weight * valid
            valid_ray = (transverse_sq < k*k) & (jacobian > 0)
            rotated[start:start + len(f)] = sampled * jacobian * valid_ray * origin_x[start:start + len(f)].conj() * origin_y.conj()
        del spectrum
        result = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(rotated)))[pad:pad + nx]
    # Do not retain the entire padded backing allocation while streaming the
    # resulting full optical field to the propagation device.
    return (result if axis == "x" else result.T).clone(memory_format=torch.contiguous_format)
