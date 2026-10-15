"""Shared complex optical correction of a pre-propagated simulation field.

Each stage applies a position-dependent complex screen followed by a complex
Fourier transfer.  All samples share the same parameters; the field transform
is complex-linear until the final square-law detector.  This is a learned
correction, not an identified axicon or an explicit free-space propagator.
FFTs are circular on the supplied finite domain. Inputs must retain the native
pixel spacing: this module never resizes an input field.

The state_dict layout is compatible with the successful experimental
``physics_screens_bg`` model (two stages by default). The production module is
self-contained and does not import diagnostics or experimental code.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def _integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _finite(name, value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {bound}, got {value!r}")
    return value


def _boolean(name, value):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return value


def _sizes(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("optical_spatial_sizes must be a nonempty list/tuple of integers >= 2")
    return tuple(_integer("optical_spatial_sizes entry", v, 2) for v in values)


def center_crop(field: torch.Tensor, output_size: int) -> torch.Tensor:
    h, w = field.shape[-2:]
    if min(h, w) < output_size:
        raise ValueError(f"Input {h}x{w} is smaller than output ROI {output_size}")
    y, x = (h - output_size) // 2, (w - output_size) // 2
    return field[..., y:y + output_size, x:x + output_size]


def channels_to_complex(field: torch.Tensor) -> torch.Tensor:
    """Decode raw amplitude/cos(phase)/sin(phase), without image normalization."""
    if field.ndim != 4 or field.shape[1] != 3:
        raise ValueError("Expected Bx3xHxW amplitude/cos/sin field")
    if field.dtype not in (torch.float32, torch.float64):
        raise ValueError("Shared optics requires float32/float64 fields; disable AMP")
    return torch.complex(field[:, 0] * field[:, 1], field[:, 0] * field[:, 2])


class SharedComplexScreen(nn.Module):
    """Multiscale position-dependent amplitude/phase with bounded coefficients."""

    def __init__(self, sizes=(16, 64, 256), reference_size=1024,
                 max_log_amplitude=.7, max_phase=math.pi, channels=1):
        super().__init__()
        sizes = _sizes(sizes)
        self.channels = _integer("optical_channels", channels)
        self.reference_size = _integer("optical_input_size", reference_size, 2)
        self.max_log_amplitude = _finite("optical_max_log_amplitude", max_log_amplitude)
        self.max_phase = _finite("optical_max_phase", max_phase)
        self.maps = nn.ParameterList([
            nn.Parameter(torch.zeros(self.channels, 2, n, n)) for n in sizes])

    def coefficients(self, shape):
        """Return raw two-channel coefficients, before bounded parameterization."""
        h, w = (_integer("screen shape", n) for n in shape)
        if max(h, w) > self.reference_size:
            raise ValueError("Input exceeds the fixed spatial-screen reference FOV")
        value = sum(F.interpolate(p, (self.reference_size, self.reference_size),
                                  mode="bilinear", align_corners=False)
                    for p in self.maps) / math.sqrt(len(self.maps))
        y, x = (self.reference_size - h) // 2, (self.reference_size - w) // 2
        return value[..., y:y + h, x:x + w]

    def forward(self, field):
        value = self.coefficients(field.shape[-2:])
        amplitude = torch.exp(self.max_log_amplitude * torch.tanh(value[:, 0]))
        phase = self.max_phase * torch.tanh(value[:, 1])
        return field * torch.polar(amplitude, phase)


class SharedFourierTransfer(nn.Module):
    """Learned complex transfer over the full Nyquist square, not a low pass."""

    def __init__(self, grid_size=512, max_log_amplitude=.7, max_phase=math.pi, channels=1):
        super().__init__()
        grid_size = _integer("optical_frequency_grid", grid_size, 2)
        self.channels = _integer("optical_channels", channels)
        self.max_log_amplitude = _finite("optical_max_log_amplitude", max_log_amplitude)
        self.max_phase = _finite("optical_max_phase", max_phase)
        self.coefficients = nn.Parameter(torch.zeros(self.channels, 2, grid_size, grid_size))

    def transfer(self, shape):
        """Return complex H in FFT order (use fftshift only for visualization)."""
        h, w = (_integer("transfer shape", n) for n in shape)
        p = self.coefficients
        fy = torch.fft.fftfreq(h, device=p.device, dtype=p.dtype) * 2
        fx = torch.fft.fftfreq(w, device=p.device, dtype=p.dtype) * 2
        gy, gx = torch.meshgrid(fy, fx, indexing="ij")
        grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
        if self.channels > 1:
            grid = grid.expand(self.channels, -1, -1, -1)
        value = F.grid_sample(p, grid, mode="bilinear", padding_mode="border",
                              align_corners=True)
        amplitude = torch.exp(self.max_log_amplitude * torch.tanh(value[:, 0]))
        phase = self.max_phase * torch.tanh(value[:, 1])
        return torch.polar(amplitude, phase)

    def forward(self, field):
        return torch.fft.ifft2(torch.fft.fft2(field, norm="ortho") *
                               self.transfer(field.shape[-2:]), norm="ortho")


class SharedCoherentOperator(nn.Module):
    """Complex-linear optical correction and global intensity gain.

    channels=1 preserves the original S/H computation and checkpoint layout.
    For C>1: E0=p*E, Ej=M_j IFFT(h_j*FFT(s_j*Eprev)), Eout=sum(q*En).
    S/H are channel-wise, M is a spatially shared complex matrix (out,in).
    p and q are complex vectors; readout has no conjugation. Only the final
    single coherent field is squared. There are no field biases/activations.
    """

    def __init__(self, *, output_size=608, reference_size=1024, stages=2,
                 spatial_sizes=(16, 64, 256), frequency_grid=512,
                 intensity_gain=1., learn_gain=True, max_log_amplitude=.7,
                 max_phase=math.pi, channels=1):
        super().__init__()
        self.channels = _integer("optical_channels", channels)
        self.output_size = _integer("model_size", output_size)
        self.reference_size = _integer("optical_input_size", reference_size, 2)
        stages = _integer("depth", stages)
        if self.output_size > self.reference_size:
            raise ValueError("model_size cannot exceed optical_input_size")
        if (self.reference_size - self.output_size) % 2:
            raise ValueError("optical_input_size and model_size must have matching parity "
                             "to preserve the centered native pixel coordinates")
        intensity_gain = _finite("optical_intensity_gain", intensity_gain, positive=True)
        learn_gain = _boolean("optical_learn_gain", learn_gain)
        self.log_intensity_gain = nn.Parameter(torch.tensor(math.log(intensity_gain)),
                                               requires_grad=learn_gain)
        self.screens = nn.ModuleList([
            SharedComplexScreen(spatial_sizes, reference_size, max_log_amplitude, max_phase,
                                channels=self.channels)
            for _ in range(stages)])
        self.transfers = nn.ModuleList([
            SharedFourierTransfer(frequency_grid, max_log_amplitude, max_phase,
                                  channels=self.channels)
            for _ in range(stages)])
        if self.channels > 1:
            # Real/imag coefficient storage keeps .double(), AdamW and legacy
            # real-parameter counting straightforward. These small maps are
            # unconstrained complex weights, unlike the bounded S/H multipliers.
            vector = torch.zeros(2, self.channels)
            vector[0].fill_(1 / math.sqrt(self.channels))
            self.lift_coefficients = nn.Parameter(vector.clone())
            self.readout_coefficients = nn.Parameter(vector.clone())
            identity = torch.stack((torch.eye(self.channels), torch.zeros(self.channels, self.channels)))
            self.mixers = nn.ParameterList([nn.Parameter(identity.clone()) for _ in range(stages)])
            # Break channel symmetry without changing the single-channel path.
            # Centering cancels the first-order screen perturbation at readout.
            # The initial operator is near identity, not exactly identity.
            with torch.no_grad():
                for screen in self.screens:
                    for parameter in screen.maps:
                        noise = torch.randn_like(parameter) * 1e-3
                        parameter.copy_(noise - noise.mean(dim=0, keepdim=True))

    def lift_weights(self):
        if self.channels == 1:
            one = self.log_intensity_gain.new_ones(1)
            return torch.complex(one, torch.zeros_like(one))
        return torch.complex(self.lift_coefficients[0], self.lift_coefficients[1])

    def readout_weights(self):
        if self.channels == 1:
            return self.lift_weights()
        return torch.complex(self.readout_coefficients[0], self.readout_coefficients[1])

    def mixing_matrix(self, stage_index):
        if not 0 <= stage_index < len(self.transfers):
            raise IndexError("Optical stage index out of range")
        if self.channels == 1:
            return self.lift_weights().reshape(1, 1)
        value = self.mixers[stage_index]
        return torch.complex(value[0], value[1])

    def forward_complex(self, field):
        if field.ndim != 3 or field.dtype not in (torch.complex64, torch.complex128):
            raise ValueError("Expected complex64/complex128 BxHxW field")
        expected = (self.reference_size, self.reference_size)
        if tuple(field.shape[-2:]) != expected:
            raise ValueError(f"Expected native optical field {expected}; received "
                             f"{tuple(field.shape[-2:])}. Crop from native data; do not resize.")
        if self.channels == 1:
            for screen, transfer in zip(self.screens, self.transfers):
                field = transfer(screen(field))
            return field
        field = field[:, None] * self.lift_weights()[None, :, None, None]
        for index, (screen, transfer) in enumerate(zip(self.screens, self.transfers)):
            field = transfer(screen(field))
            field = torch.einsum('oc,bchw->bohw',
                                 self.mixing_matrix(index).to(dtype=field.dtype), field)
        return (field * self.readout_weights()[None, :, None, None]).sum(dim=1)

    def forward(self, field):
        corrected = self.forward_complex(channels_to_complex(field))
        corrected = center_crop(corrected, self.output_size)
        return (corrected.abs().square() * self.log_intensity_gain.exp()).unsqueeze(1)


class SharedOpticalModel(nn.Module):
    """Shared field operator plus optional positive camera-intensity background.

    ``predict_camera`` returns (intensity, gain-scaled two-channel complex field).
    The field excludes the additive *intensity* background: in general its
    squared modulus is therefore intensity - background, not total intensity.
    No physically justified background phase is available to add to the field.
    """

    architecture = "shared_optical"

    def __init__(self, operator, *, background=True, background_grid=64,
                 background_initial_intensity=.001, background_reference_intensity=.07):
        super().__init__()
        self.operator = operator
        self.background_enabled = _boolean("optical_background", background)
        self.background_grid = _integer("optical_background_grid", background_grid, 2)
        self.initial_intensity = _finite("optical_background_initial_intensity",
                                         background_initial_intensity, positive=True)
        reference = _finite("optical_background_reference_intensity",
                            background_reference_intensity, positive=True)
        # Retained for state_dict compatibility. Incoherent background does not
        # use this buffer; it was shared with a coherent-reference experiment.
        self.register_buffer("reference_amplitude", torch.tensor(math.sqrt(reference)))
        if self.background_enabled:
            raw = self.initial_intensity + math.log(-math.expm1(-self.initial_intensity))
            self.background_parameters = nn.Parameter(
                torch.full((1, 1, self.background_grid, self.background_grid), raw))
        else:
            self.register_parameter("background_parameters", None)

    @property
    def input_size(self):
        return self.operator.reference_size

    @property
    def depth(self):
        return len(self.operator.transfers)

    @property
    def channels(self):
        return self.operator.channels

    def background(self):
        size = self.operator.output_size
        if not self.background_enabled:
            return self.operator.log_intensity_gain.new_zeros((1, 1, size, size))
        value = F.interpolate(self.background_parameters, (size, size),
                              mode="bilinear", align_corners=False)
        return F.softplus(value)

    def predict_camera(self, field):
        corrected = self.operator.forward_complex(channels_to_complex(field))
        corrected = center_crop(corrected, self.operator.output_size)
        # Keep the experimental intensity operation order for checkpoint parity.
        intensity = (corrected.abs().square() * self.operator.log_intensity_gain.exp()).unsqueeze(1)
        intensity = intensity + self.background()
        scaled = corrected * torch.exp(.5 * self.operator.log_intensity_gain)
        output_field = torch.stack((scaled.real, scaled.imag), dim=1)
        return intensity, output_field

    def forward(self, field):
        return self.predict_camera(field)[0]

    def experiment_metadata(self):
        return {
            "architecture": self.architecture, "depth": self.depth,
            "optical_channels": self.channels,
            "channel_readout": "complex sum before square-law detection",
            "channel_mixing": "shared complex matrix after each channel-wise S/H stage" if self.channels > 1 else "none",
            "operator": type(self.operator).__name__, "input_size": self.input_size,
            "output_size": self.operator.output_size,
            "background_kind": "incoherent" if self.background_enabled else "none",
            "background_grid": self.background_grid if self.background_enabled else None,
            "background_initial_intensity": self.initial_intensity if self.background_enabled else 0.,
            "gain_placement": "gain*|LE|^2+B" if self.background_enabled else "gain*|LE|^2",
            "output_field_excludes_intensity_background": True,
            "global_input_phase_invariance": True,
            "explicit_z_propagation": False,
            "boundary": "circular FFT on the native finite input field",
            "interpretation": "shared complex-linear correction; stages are not identified optical surfaces",
        }


def make_shared_optical_model(cfg: Mapping) -> SharedOpticalModel:
    """Build native field -> central intensity ROI optics from trainer config.

    ``depth`` counts screen/transfer pairs, with independent coefficients per
    stage. ``optical_frequency_grid`` is parameter resolution, not FFT size or
    a frequency cutoff. Gain is an intensity ratio: the trainer may initialize
    it from TRAIN samples only, then persist the resulting value in config.
    """
    if not isinstance(cfg, Mapping):
        raise ValueError("Shared optical configuration must be a mapping")
    if cfg.get("architecture", "shared_optical") != "shared_optical":
        raise ValueError("make_shared_optical_model requires architecture='shared_optical'")
    operator = SharedCoherentOperator(
        output_size=cfg.get("model_size", 608),
        reference_size=cfg.get("optical_input_size", 1024),
        channels=cfg.get("optical_channels", 1),
        stages=cfg.get("depth", 2),
        spatial_sizes=cfg.get("optical_spatial_sizes", (16, 64, 256)),
        frequency_grid=cfg.get("optical_frequency_grid", 512),
        intensity_gain=cfg.get("optical_intensity_gain", 1.),
        learn_gain=cfg.get("optical_learn_gain", True),
        max_log_amplitude=cfg.get("optical_max_log_amplitude", .7),
        max_phase=cfg.get("optical_max_phase", math.pi))
    return SharedOpticalModel(
        operator, background=cfg.get("optical_background", True),
        background_grid=cfg.get("optical_background_grid", 64),
        background_initial_intensity=cfg.get("optical_background_initial_intensity", .001),
        background_reference_intensity=cfg.get("optical_background_reference_intensity", .07))
