"""Differentiable physical propagation and native-grid targets for proxy CGH.

The simulator is imported only when building a system.  Phase input is actual
SLM phase in radians, tensor order (x, y).  The returned complex field is in
the display/camera order used by the simulator's field exports.  No field is
resized: changing the sampling would invalidate a shared optical checkpoint.
"""
from __future__ import annotations

import math
import numbers
import warnings
from pathlib import Path

import numpy as np
import torch


# Reproduces the stored TrainSet_09092026 fields. These values deliberately do
# not follow the simulator's newer AX2510-A experimental defaults. z remains a
# caller-supplied absolute distance (the baseline reference is 6 mm).
LEGACY_TRAINING_SETTINGS = {
    "slm_incidence_angle_deg": 0.0,
    "slm_incidence_axis": "x",
    "slm_tilt_model": "angular_spectrum",
    "slm_tilt_padding": 2,
    "slm_tilt_backend": "auto",
    "slm_tilt_stream_cols": 64,
    "relay_focal_lengths_mm": [250.0, 150.0, 150.0, 200.0],
    "slm_illumination_frame": "beam",
    "slm_shape_xy": [1600, 1200],
    "wavelength_m": 473e-9,
    "focal_slm_m": 0.016,
    "slm_pitch_after_relay_m": 6.4e-6,
    "gaussian_beam_waist_m": 0.005109664,
    "amplitude_profile_type": "gaussian",
    "beam_mean_amplitude": 1.0,
    "upsample_factor": 20,
    "roi_size": 1024,
    "propagation_backend": "streamed",
    "stream_rows": 256,
    "stream_cols": 256,
    "propagation_medium_index": 1.0,
    "axicon_grating_pitch_m": 1.396e-6,
    "axicon_angle_in_medium": False,
    "axicon_profile": "continuous",
    "axicon_phase_depth_rad": math.pi,
    "axicon_duty_cycle": 0.5,
    "axicon_radial_offset_m": 0.0,
    "axicon_lateral_shift_x_m": 0.0,
    "axicon_lateral_shift_y_m": 0.0,
    "axicon_diffraction_orders": None,
    "asm_margin_factor": 8000.0,
    "apply_spatial_filter": True,
    "spatial_filter_mode": "center_block",
    "spatial_filter_focal_length_m": 0.250,
    "center_block_size_m": 550e-6,
    "lee_aperture_diameter_m": 1.8e-3,
    "lee_first_order_offset_m": None,
    "lee_carrier_period_pixels": 4.0,
    "transpose_output_field": True,
}


def _positive(name, value, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or (value < 0 if zero else value <= 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


def _integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


class PhysicalPhaseSystem:
    """SLM phase -> fixed-z complex simulation field, retaining phase gradients."""

    def __init__(self, beam, transfer, settings, forward_options):
        self.beam = beam
        self.beam_config = beam.beam_config
        self.transfer = transfer
        self.settings = settings
        self.forward_options = forward_options

    def forward(self, phase_rad):
        expected = (self.beam_config.Nx, self.beam_config.Ny)
        if tuple(phase_rad.shape) != expected or phase_rad.is_complex():
            raise ValueError(f"Expected real physical phase radians of shape {expected}")
        if not bool(torch.isfinite(phase_rad).all()):
            raise ValueError("Phase must be finite")
        field = self.beam.propagateToVolume_Axicon2(
            phase_mask=phase_rad, H_asm=self.transfer,
            convert_to_intensity=False, **self.forward_options)
        size = self.settings["roi_size"]
        if tuple(field.shape) == (size, size, 1):
            field = field[..., 0]
        if tuple(field.shape) != (size, size) or not field.is_complex():
            raise RuntimeError(f"Simulator returned unexpected field shape/dtype: {field.shape}/{field.dtype}")
        if self.settings["transpose_output_field"]:
            field = field.transpose(0, 1)
        return field

    __call__ = forward


def build_system(z_target_m, device, overrides=None, *, preset="simulator"):
    """Build one physical z plane, measured in metres after the axicon.

    ``overrides`` is a flat JSON-compatible mapping; unknown names fail early.
    Set ``roi_size`` to the checkpoint's ``optical_input_size`` (default 1024).
    ``simulator`` retains the current simulator's experimental defaults.
    ``legacy_training`` pins the 1.396 um axicon / 20x / air / normal-incidence
    setup that reproduces the stored training fields, independently of changes
    to simulator defaults. Explicit overrides are applied after the preset.
    This optimizes physical phase. A measured drive-to-phase LUT is deliberately
    not applied a second time; hardware LUT inversion is a separate operation.
    """
    import axicon_simulator as sim
    from mbvam.Beam.holobeam import HoloBeam

    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("Optical overrides must be a JSON object")
    if preset not in ("simulator", "legacy_training"):
        raise ValueError("System preset must be 'simulator' or 'legacy_training'")
    supplied = dict(overrides or {})
    settings = {
        "slm_incidence_angle_deg": sim.DEFAULT_SLM_INCIDENCE_ANGLE_DEG,
        "slm_incidence_axis": sim.DEFAULT_SLM_INCIDENCE_AXIS,
        "slm_tilt_model": sim.DEFAULT_SLM_TILT_MODEL,
        "slm_tilt_padding": sim.DEFAULT_SLM_TILT_PADDING,
        "slm_tilt_backend": sim.DEFAULT_SLM_TILT_BACKEND,
        "slm_tilt_stream_cols": sim.DEFAULT_SLM_TILT_STREAM_COLS,
        "relay_focal_lengths_mm": list(sim.DEFAULT_RELAY_FOCAL_LENGTHS_MM),
        "slm_illumination_frame": sim.DEFAULT_SLM_ILLUMINATION_FRAME,
        "slm_shape_xy": [1600, 1200],
        "wavelength_m": sim.DEFAULT_WAVELENGTH_M,
        "focal_slm_m": None,
        "slm_pitch_after_relay_m": None,
        "gaussian_beam_waist_m": None,
        "amplitude_profile_type": "gaussian",
        "beam_mean_amplitude": 1.0,
        "upsample_factor": sim.DEFAULT_UPSAMPLE_FACTOR,
        "roi_size": 1024,
        "propagation_backend": sim.DEFAULT_PROPAGATION_BACKEND,
        "stream_rows": sim.DEFAULT_STREAM_ROWS,
        "stream_cols": sim.DEFAULT_STREAM_COLS,
        "propagation_medium_index": sim.DEFAULT_PROPAGATION_MEDIUM_INDEX,
        "axicon_grating_pitch_m": sim.DEFAULT_AXICON_GRATING_PITCH_M,
        "axicon_angle_in_medium": sim.DEFAULT_AXICON_ANGLE_IN_MEDIUM,
        "axicon_profile": sim.DEFAULT_AXICON_PROFILE,
        "axicon_phase_depth_rad": sim.DEFAULT_AXICON_PHASE_DEPTH_RAD,
        "axicon_duty_cycle": sim.DEFAULT_AXICON_DUTY_CYCLE,
        "axicon_radial_offset_m": sim.DEFAULT_AXICON_RADIAL_OFFSET,
        "axicon_lateral_shift_x_m": sim.DEFAULT_AXICON_LATERAL_SHIFT_X_UM * 1e-6,
        "axicon_lateral_shift_y_m": sim.DEFAULT_AXICON_LATERAL_SHIFT_Y_UM * 1e-6,
        "axicon_diffraction_orders": sim.DEFAULT_AXICON_DIFFRACTION_ORDERS,
        "asm_margin_factor": sim.DEFAULT_ASM_MARGIN_FACTOR,
        "apply_spatial_filter": sim.DEFAULT_APPLY_SPATIAL_FILTER,
        "spatial_filter_mode": sim.DEFAULT_SPATIAL_FILTER_MODE,
        "spatial_filter_focal_length_m": sim.DEFAULT_SPATIAL_FILTER_FOCAL_LENGTH_M,
        "center_block_size_m": sim.DEFAULT_CENTER_BLOCK_SIZE_UM * 1e-6,
        "lee_aperture_diameter_m": sim.DEFAULT_LEE_APERTURE_DIAMETER_MM * 1e-3,
        "lee_first_order_offset_m": (None if sim.DEFAULT_LEE_FIRST_ORDER_OFFSET_MM is None
                                      else sim.DEFAULT_LEE_FIRST_ORDER_OFFSET_MM * 1e-3),
        "lee_carrier_period_pixels": sim.DEFAULT_LEE_CARRIER_PERIOD_PIXELS,
        "transpose_output_field": sim.DEFAULT_TRANSPOSE_OUTPUT_FIELD,
    }
    if preset == "legacy_training":
        settings.update(LEGACY_TRAINING_SETTINGS)
    unknown = set(supplied) - set(settings)
    if unknown:
        raise ValueError(f"Unknown optical settings: {sorted(unknown)}. Supported: {sorted(settings)}")
    settings.update(supplied)
    z_target_m = _positive("z_target_m", z_target_m)
    for name in ("upsample_factor", "roi_size", "stream_rows", "stream_cols",
                 "slm_tilt_padding", "slm_tilt_stream_cols"):
        settings[name] = _integer(name, settings[name])
    for name in ("wavelength_m", "beam_mean_amplitude", "propagation_medium_index",
                 "axicon_grating_pitch_m", "spatial_filter_focal_length_m", "lee_aperture_diameter_m"):
        settings[name] = _positive(name, settings[name])
    for name in ("center_block_size_m", "axicon_phase_depth_rad"):
        settings[name] = _positive(name, settings[name], zero=True)
    for name in ("slm_pitch_after_relay_m", "gaussian_beam_waist_m", "lee_carrier_period_pixels", "focal_slm_m"):
        if settings[name] is not None:
            settings[name] = _positive(name, settings[name])
    if settings["asm_margin_factor"] is not None:
        settings["asm_margin_factor"] = _positive("asm_margin_factor", settings["asm_margin_factor"], zero=True)
    for name in ("slm_incidence_angle_deg", "axicon_radial_offset_m", "axicon_lateral_shift_x_m",
                 "axicon_lateral_shift_y_m", "axicon_duty_cycle"):
        settings[name] = _finite(name, settings[name])
    if not 0 < settings["axicon_duty_cycle"] < 1:
        raise ValueError("axicon_duty_cycle must be between 0 and 1")
    if settings["lee_first_order_offset_m"] is not None:
        settings["lee_first_order_offset_m"] = _finite("lee_first_order_offset_m", settings["lee_first_order_offset_m"])
    for name in ("apply_spatial_filter", "transpose_output_field", "axicon_angle_in_medium"):
        if not isinstance(settings[name], bool):
            raise ValueError(f"{name} must be a JSON boolean")
    choices = {"propagation_backend": ("streamed", "dense"),
               "amplitude_profile_type": ("gaussian", "flat_top"),
               "spatial_filter_mode": ("center_block", "lee_left_first_order"),
               "axicon_profile": ("continuous", "binary")}
    for name, allowed in choices.items():
        if settings[name] not in allowed:
            raise ValueError(f"{name} must be one of {allowed}")
    shape = settings["slm_shape_xy"]
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise ValueError("slm_shape_xy must contain [Nx, Ny]")
    settings["slm_shape_xy"] = [_integer("SLM dimension", n, 2) for n in shape]
    if any(n % 2 for n in settings["slm_shape_xy"]):
        raise ValueError("SLM dimensions must be even to preserve centered coordinates")
    if (settings["roi_size"] > min(settings["slm_shape_xy"]) * settings["upsample_factor"]
            or settings["roi_size"] % 2):
        raise ValueError("roi_size must be even and fit inside the upsampled optical grid")
    if preset == "simulator" and "wavelength_m" in supplied and "axicon_grating_pitch_m" not in supplied:
        from mbvam.Beam.axicon import RefractiveAxicon
        settings["axicon_grating_pitch_m"] = RefractiveAxicon(
            wavelength_m=settings["wavelength_m"]).equivalent_pitch_m

    cfg = sim.build_beam_config(**{key: settings[key] for key in (
        "slm_incidence_angle_deg", "slm_incidence_axis", "slm_tilt_model", "slm_tilt_padding",
        "slm_tilt_backend", "slm_tilt_stream_cols", "relay_focal_lengths_mm", "slm_illumination_frame")})
    cfg.device = torch.device(device)
    if cfg.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cfg.Nx_physical, cfg.Ny_physical = settings["slm_shape_xy"]
    cfg.binningFactor = 1
    cfg.croppingFactor = 1
    cfg.lambda_ = settings["wavelength_m"]
    if settings["focal_slm_m"] is not None:
        cfg.focal_SLM = settings["focal_slm_m"]
    if settings["slm_pitch_after_relay_m"] is not None:
        cfg.psSLM_physical = settings["slm_pitch_after_relay_m"]
    if settings["gaussian_beam_waist_m"] is not None:
        cfg.gaussian_beam_waist = settings["gaussian_beam_waist_m"]
    cfg.amplitude_profile_type = settings["amplitude_profile_type"]
    settings["slm_pitch_after_relay_m"] = float(cfg.psSLM)
    settings["gaussian_beam_waist_m"] = float(cfg.gaussian_beam_waist)
    settings["focal_slm_m"] = float(cfg.focal_SLM)
    settings["system_preset"] = preset
    settings["optical_pitch_m"] = float(cfg.psSLM) / settings["upsample_factor"]
    settings["z_target_m"] = z_target_m
    settings["device"] = str(cfg.device)
    settings["phase_input"] = "physical radians, tensor order [Nx, Ny]; no drive LUT"
    frequency = 1.0 / settings["axicon_grating_pitch_m"]
    air_na = cfg.lambda_ * frequency
    if not 0 < air_na < min(1.0, settings["propagation_medium_index"]):
        raise ValueError("Axicon frequency must propagate in air and in the selected medium")
    cone_angle = math.asin(air_na / settings["propagation_medium_index"]
                           if settings["axicon_angle_in_medium"] else air_na)
    beam = HoloBeam(cfg)
    beam.slm_amplitude_profile = beam.buildSLMAmplitudeProfile()
    beam.beam_mean_amplitude_iter = torch.tensor(settings["beam_mean_amplitude"],
                                                device=cfg.device, dtype=cfg.fdtype)
    common = dict(upsample_factor=settings["upsample_factor"],
                  n_medium=settings["propagation_medium_index"], axicon_angle=cone_angle,
                  axicon_angle_in_medium=settings["axicon_angle_in_medium"],
                  axicon_transverse_frequency=frequency, axicon_profile=settings["axicon_profile"],
                  axicon_phase_depth=settings["axicon_phase_depth_rad"],
                  axicon_duty_cycle=settings["axicon_duty_cycle"])
    transfer = sim.build_axicon_transfer_function(
        beam, False, backend=settings["propagation_backend"],
        stream_rows=settings["stream_rows"], stream_cols=settings["stream_cols"],
        z_query=[z_target_m], margin_factor=settings["asm_margin_factor"],
        axicon_diffraction_orders=settings["axicon_diffraction_orders"], **common)
    forward_options = dict(common, roi_size=settings["roi_size"],
        apply_spatial_filter=settings["apply_spatial_filter"],
        spatial_filter_mode=settings["spatial_filter_mode"],
        spatial_filter_focal_length_m=settings["spatial_filter_focal_length_m"],
        center_block_size_m=settings["center_block_size_m"],
        lee_aperture_diameter_m=settings["lee_aperture_diameter_m"],
        lee_first_order_offset_m=settings["lee_first_order_offset_m"],
        lee_carrier_period_pixels=settings["lee_carrier_period_pixels"],
        axicon_radial_offset=settings["axicon_radial_offset_m"],
        axicon_lateral_shift_x=settings["axicon_lateral_shift_x_m"],
        axicon_lateral_shift_y=settings["axicon_lateral_shift_y_m"])
    return PhysicalPhaseSystem(beam, transfer, settings, forward_options)


def _center_canvas(array, size):
    """Use the same centered crop/pad convention as the existing CGH optimizer."""
    target_h, target_w = (size, size) if isinstance(size, numbers.Integral) else size
    output = np.zeros((target_h, target_w), dtype=np.float32)
    h, w = array.shape
    oy, ox = (target_h - h) // 2, (target_w - w) // 2
    dy, dx, sy, sx = max(oy, 0), max(ox, 0), max(-oy, 0), max(-ox, 0)
    ch, cw = min(h - sy, target_h - dy), min(w - sx, target_w - dx)
    if ch > 0 and cw > 0:
        output[dy:dy + ch, dx:dx + cw] = array[sy:sy + ch, sx:sx + cw]
    return output


def prepare_target(path, model_size, beam_config, upsample_factor, mode="physical",
                   target_pixel_size_m=2e-6, *, device=None, dtype=torch.float32,
                   transpose=False, flip_ud=False, flip_lr=False,
                   legacy_optimizer_roi_size=1024):
    """Load nonnegative intensity in display coordinates without peak normalization.

    ``physical`` places a drawing at its specified source pixel pitch on the
    ASM grid, using the existing optimizer's linear interpolation and centered
    crop/pad convention. ``native`` accepts an already sampled camera target;
    larger arrays are center cropped, never resized. NPY intensities retain
    their units (complex arrays become |E|^2). Integer image files use their
    encoded range, e.g. an 8-bit PNG becomes intensity 0..1. Floating-point
    grayscale images retain their units. Orientation changes precede placement.
    ``legacy_optimizer`` reproduces the historical fno_citl drawing layout:
    Abbe-resolution interpolation -> rectangular SLM-size canvas -> bilinear
    square optimizer ROI (normally 1024) -> native camera crop. Its source
    pixel-size argument is a legacy layout parameter, not a direct assignment
    of physical pixel spacing at the camera. Reported effective scales make
    this distinction explicit. No camera or field data are resampled here.
    """
    path = Path(path)
    model_size = _integer("model_size", model_size)
    upsample_factor = _integer("upsample_factor", upsample_factor)
    native_pitch = _positive("beam_config.psSLM", beam_config.psSLM) / upsample_factor
    if mode not in ("physical", "native", "legacy_optimizer"):
        raise ValueError("Target mode must be 'physical', 'native', or 'legacy_optimizer'")
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
        if np.iscomplexobj(array):
            array = np.abs(array) ** 2
        encoding = "NPY intensity units"
    else:
        from PIL import Image
        with Image.open(path) as source:
            if source.mode in ("1", "L", "I", "F") or source.mode.startswith("I;16"):
                array = np.asarray(source)
            else:
                array = np.asarray(source.convert("L"))
        if array.dtype == np.bool_:
            scale = 1.0
        elif np.issubdtype(array.dtype, np.unsignedinteger):
            scale = float(np.iinfo(array.dtype).max)
        elif np.issubdtype(array.dtype, np.signedinteger):
            # PIL can decode 16-bit PNG as mode I / int32.
            scale = 65535.0 if path.suffix.lower() == ".png" else 1.0
        else:
            scale = 1.0
        array = array.astype(np.float32) / scale
        encoding = f"Image intensity divided by encoded range {scale:g}"
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all() or (array < 0).any() or not (array > 0).any():
        raise ValueError("Target must be a finite, nonnegative 2D intensity image with positive signal")
    original_shape = list(array.shape)
    if transpose:
        array = array.T
    if flip_ud:
        array = array[::-1, :]
    if flip_lr:
        array = array[:, ::-1]
    clipping = []
    geometry = {}

    def place_on_canvas(values, size, stage):
        result = _center_canvas(values, size)
        before, after = values.sum(dtype=np.float64), result.sum(dtype=np.float64)
        lost = max(0.0, float(before - after))
        if before > 0 and lost / before > 1e-7:
            clipping.append(dict(stage=stage, input_shape=list(values.shape),
                                 output_shape=list(result.shape),
                                 discarded_intensity_fraction=lost / float(before)))
        return result

    if mode == "physical":
        from scipy.ndimage import zoom
        source_pitch = _positive("target_pixel_size_m", target_pixel_size_m)
        factor = source_pitch / native_pitch
        array = zoom(array, (factor, factor), order=1)
        placed_shape = list(array.shape)
        pixel_scale_xy = [factor, factor]
    elif mode == "legacy_optimizer":
        from scipy.ndimage import zoom
        import torch.nn.functional as F
        source_pitch = _positive("target_pixel_size_m", target_pixel_size_m)
        legacy_size = _integer("legacy_optimizer_roi_size", legacy_optimizer_roi_size)
        if legacy_size < model_size or (legacy_size - model_size) % 2:
            raise ValueError("legacy_optimizer_roi_size must permit an aligned native camera center crop")
        abbe_x = _positive("beam_config.abbe_res_x", beam_config.abbe_res_x)
        abbe_y = _positive("beam_config.abbe_res_y", beam_config.abbe_res_y)
        factor_x, factor_y = source_pitch / abbe_x, source_pitch / abbe_y
        array = zoom(array, (factor_y, factor_x), order=1)
        placed_shape = list(array.shape)
        canvas_shape = (int(beam_config.Ny), int(beam_config.Nx))
        array = place_on_canvas(array, canvas_shape, "legacy_slm_canvas")
        array = F.interpolate(torch.from_numpy(array)[None, None],
                              size=(legacy_size, legacy_size), mode="bilinear",
                              align_corners=False)[0, 0].numpy()
        pixel_scale_xy = [factor_x * legacy_size / canvas_shape[1],
                          factor_y * legacy_size / canvas_shape[0]]
        factor = None  # Legacy layout has two different axis scales.
        geometry = dict(legacy_optimizer_roi_size=legacy_size,
                        legacy_optimizer_canvas_shape=list(canvas_shape),
                        abbe_resolution_xy_m=[abbe_x, abbe_y],
                        abbe_resize_factor_xy=[factor_x, factor_y],
                        layout_description="Abbe resize -> rectangular SLM canvas -> bilinear optimizer ROI -> native camera crop")
    else:
        source_pitch, factor = native_pitch, 1.0
        if min(array.shape) < model_size or any((n - model_size) % 2 for n in array.shape):
            raise ValueError(f"Native target {array.shape} must permit an aligned {model_size}x{model_size} center crop")
        placed_shape = list(array.shape)
        pixel_scale_xy = [1.0, 1.0]
    array = place_on_canvas(array, model_size, "camera_roi")
    if clipping:
        stages = ", ".join(f"{item['stage']}: {100 * item['discarded_intensity_fraction']:.2f}%"
                           for item in clipping)
        warnings.warn(f"Target signal was clipped during placement ({stages}). "
                      "Check the target mode, source pixel size, and target position.",
                      RuntimeWarning, stacklevel=2)
    if not (array > 0).any():
        raise ValueError("Target has no positive signal inside the selected camera ROI")
    if device is None:
        device = beam_config.device
    target = torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=dtype)[None, None]
    report = dict(mode=mode, path=str(path.resolve()), source_shape=original_shape,
                  shape_after_physical_sampling=placed_shape, model_size=model_size,
                  optical_pitch_m=native_pitch, target_pixel_size_m=source_pitch,
                  resize_factor=factor, intensity_encoding=encoding,
                  transpose=bool(transpose), flip_ud=bool(flip_ud), flip_lr=bool(flip_lr),
                  target_peak=float(array.max()), target_mean=float(array.mean()),
                  physical_roi_width_m=model_size * native_pitch,
                  source_to_model_pixel_scale_xy=pixel_scale_xy,
                  effective_source_pixel_size_xy_m=[scale * native_pitch for scale in pixel_scale_xy],
                  signal_clipped=bool(clipping), clipping=clipping, **geometry)
    return target, report
