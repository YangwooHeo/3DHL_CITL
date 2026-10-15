"""Export effective shared optical maps without changing the trained model.

The archive contains exactly the complex multipliers used by the forward pass,
not their raw trainable grids. H maps are saved in FFT order; plots show their
fftshifted versions. S maps retain input-image row/column coordinates. Individual
stages are an operator factorization, not uniquely identified optical surfaces.
Multichannel exports retain the latent-channel axis and coherent lift/mixing/
readout weights. Channels are not claimed to be independently observed modes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stats(array):
    return {"min": float(array.min()), "max": float(array.max()),
            "mean": float(array.mean()), "std": float(array.std())}


def _complex_map(value, shape, name, channels=1):
    if value.shape != (channels, *shape) or not value.is_complex():
        raise ValueError(f"{name} must be a complex {channels}xHxW map, got {value.shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} has nonfinite coefficients")
    # Keep the original two-dimensional archive contract for old checkpoints.
    return (value[0] if channels == 1 else value).detach().cpu().numpy().copy()


def _complex_weights(value, shape, name):
    if tuple(value.shape) != shape or not value.is_complex():
        raise ValueError(f"{name} must have complex shape {shape}, got {value.shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} has nonfinite coefficients")
    return value.detach().cpu().numpy().copy()


def _archive_array(archive, name, array):
    # Stream one map at a time, so exporting many stages does not retain every
    # native-resolution map in memory. np.load reads this as an ordinary NPZ.
    with archive.open(f"{name}.npy", "w", force_zip64=True) as stream:
        np.save(stream, array, allow_pickle=False)


def _plot_stage(hmap, smap, stage, destination, amplitude_limits, roi,
                checkpoint_epoch, channel=None):
    height, width = smap.shape
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    # Pixel centers are the actual FFT sample coordinates, including the -0.5
    # bin for even sizes. Clip the half-bin boundary to the Nyquist square.
    frequency_extent = (fx[0] - .5 / width, fx[-1] + .5 / width,
                        fy[0] - .5 / height, fy[-1] + .5 / height)
    spatial_extent = (-.5, width - .5, height - .5, -.5)
    shifted = np.fft.fftshift(hmap)
    panels = [(np.abs(shifted), f"H{stage}: amplitude", "viridis", amplitude_limits),
              (np.angle(shifted), f"H{stage}: wrapped phase", "twilight", (-math.pi, math.pi)),
              (np.abs(smap), f"S{stage}: amplitude", "viridis", amplitude_limits),
              (np.angle(smap), f"S{stage}: wrapped phase", "twilight", (-math.pi, math.pi))]
    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    try:
        for index, (ax, (data, title, cmap, limits)) in enumerate(zip(axes.flat, panels)):
            spatial = index >= 2
            image = ax.imshow(data, cmap=cmap, vmin=limits[0], vmax=limits[1],
                              origin="upper" if spatial else "lower",
                              extent=spatial_extent if spatial else frequency_extent,
                              interpolation="nearest", rasterized=True)
            ax.set_title(title)
            if spatial:
                y0, y1, x0, x1 = roi
                # White underlay keeps the output ROI visible on either colormap.
                for color, linewidth in (("white", 2.5), ("black", 1.0)):
                    ax.add_patch(Rectangle((x0 - .5, y0 - .5), x1 - x0, y1 - y0,
                                           fill=False, edgecolor=color, linewidth=linewidth,
                                           linestyle="--"))
                ax.set_xlabel("Input x (pixel)")
                ax.set_ylabel("Input y (pixel)")
            else:
                ax.set_xlim(-.5, .5)
                ax.set_ylim(-.5, .5)
                ax.set_xlabel("fx (cycles / input pixel)")
                ax.set_ylabel("fy (cycles / input pixel)")
            bar = figure.colorbar(image, ax=ax, shrink=.82)
            if index % 2:
                bar.set_ticks([-math.pi, 0, math.pi], labels=["-pi", "0", "pi"])
                bar.set_label("rad")
            else:
                bar.set_label("Field amplitude multiplier")
        epoch = "" if checkpoint_epoch is None else f" | checkpoint epoch {checkpoint_epoch}"
        channel_label = "" if channel is None else f" | latent channel {channel}"
        figure.suptitle(f"Shared optical correction: stage {stage}{channel_label}{epoch}\n"
                        "Dashed box: output ROI; shared fixed color limits")
        figure.savefig(destination, dpi=150)
    finally:
        plt.close(figure)


def _plot_channel_weights(lift, readout, mixers, destination, checkpoint_epoch):
    rows = [("Input lift p", lift[:, None], "Input field", "Output channel")]
    rows.extend((f"Stage {stage} mixing M", matrix, "Input channel", "Output channel")
                for stage, matrix in enumerate(mixers, 1))
    rows.append(("Readout q", readout[None, :],
                 "Input channel", "Output field"))
    upper = max(float(np.abs(values).max()) for _, values, _, _ in rows)
    upper = max(upper, np.finfo(np.float32).tiny)
    figure, axes = plt.subplots(len(rows), 2, figsize=(8, 2.5 * len(rows)),
                                squeeze=False, constrained_layout=True)
    try:
        for row, (title, values, xlabel, ylabel) in enumerate(rows):
            for column, (data, suffix, cmap, low, high) in enumerate((
                (np.abs(values), "amplitude", "viridis", 0, upper),
                (np.angle(values), "phase", "twilight", -math.pi, math.pi),
            )):
                axis = axes[row, column]
                rendered = axis.imshow(data, origin="upper", cmap=cmap,
                                       vmin=low, vmax=high, interpolation="nearest", aspect="auto")
                axis.set(title=f"{title}: {suffix}", xlabel=xlabel, ylabel=ylabel)
                axis.set_xticks(range(values.shape[1]), labels=range(1, values.shape[1] + 1))
                axis.set_yticks(range(values.shape[0]), labels=range(1, values.shape[0] + 1))
                bar = figure.colorbar(rendered, ax=axis, shrink=.85)
                if column:
                    bar.set_ticks([-math.pi, 0, math.pi], labels=["-pi", "0", "pi"])
                    bar.set_label("rad")
        epoch = "" if checkpoint_epoch is None else f" | checkpoint epoch {checkpoint_epoch}"
        figure.suptitle("Latent coherent-channel coefficients" + epoch + "\n"
                        "Mixing is after each S/H pair; readout sums fields before squaring")
        figure.savefig(destination, dpi=150)
    finally:
        plt.close(figure)


def visualize_optical_parameters(model, cfg, save_dir, checkpoint_epoch=None):
    """Write native effective H/S maps, stage plots and metadata; return paths.

    ``save_dir`` is a dedicated output directory. The model remains on its
    original device and every module's train/eval flag is restored, even if an
    export fails. H_i is stored in FFT order and S_i in input pixel order in
    ``optical_parameters.npz``. No phase unwrapping or per-map normalization is
    performed. Frequency coordinates are cycles per input pixel, not inferred
    physical units. ``checkpoint_epoch`` must describe the loaded checkpoint.
    Single-channel exports keep legacy two-dimensional maps. For C > 1, maps
    have shape CxHxW; lift_weights, mixing_i (output,input), and readout_weights
    reproduce the field transform, with no conjugation in the coherent readout.
    """
    operator = getattr(model, "operator", None)
    if operator is None or not hasattr(operator, "screens") or not hasattr(operator, "transfers"):
        raise TypeError("Optical parameter visualization requires a shared optical model")
    if len(operator.screens) != len(operator.transfers) or not len(operator.transfers):
        raise ValueError("Expected one spatial screen per Fourier transfer stage")
    input_size = int(getattr(operator, "input_size", operator.reference_size))
    output_size = int(operator.output_size)
    if input_size < output_size or input_size < 2:
        raise ValueError("Input size must cover the output ROI")
    depth = len(operator.transfers)
    channels = int(getattr(operator, "channels", 1))
    if channels < 1:
        raise ValueError("Expected at least one coherent channel")
    shape = (input_size, input_size)
    roi_start = (input_size - output_size) // 2
    roi = (roi_start, roi_start + output_size, roi_start, roi_start + output_size)
    all_maps = [*operator.screens, *operator.transfers]
    bound = max(float(item.max_log_amplitude) for item in all_maps)
    amplitude_limits = (math.exp(-bound), math.exp(bound))
    state_flags = [(module, module.training) for module in model.modules()]
    destination = Path(save_dir)
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "optical_parameters.npz"
    metadata_path = destination / "optical_parameters.json"
    metadata = {
        "format_version": 1,
        "checkpoint_epoch": checkpoint_epoch,
        "depth": depth,
        "map_shape": list(shape),
        "output_shape": [output_size, output_size],
        "output_roi_y0_y1_x0_x1": list(roi),
        "frequency_units": "cycles per input pixel",
        "h_archive_order": "unshifted torch.fft.fft2 order; h_1 through h_n",
        "h_plot_order": "fftshift on both axes; origin lower; fx right, fy up",
        "s_archive_order": "input row/column order; s_1 through s_n",
        "s_plot_order": "origin upper; input x right, input y down",
        "spatial_units": "input pixels; integer coordinates are pixel centers",
        "amplitude_color_limits": list(amplitude_limits),
        "phase_color_limits_radians": [-math.pi, math.pi],
        "phase_representation": "wrapped np.angle; no phase unwrapping",
        "stage_interpretation": "Shared learned factorization; individual stages are not uniquely identified physical surfaces",
        "config": _json_value(cfg),
        "stages": [],
    }
    paths = {"metadata_json": str(metadata_path), "parameters_npz": str(archive_path),
             "stage_pngs": [], "background_png": None}
    if channels > 1:
        metadata.update({
            "format_version": 2,
            "channels": channels,
            "archive_map_shape": [channels, *shape],
            "channel_interpretation": "Latent coherent channels, not independently observed physical modes",
            "field_operator": "e=p*E; for i: e=M_i@IFFT(h_i*FFT(s_i*e)); E_out=sum_c(q_c*e_c)",
            "fft_normalization": "ortho on the final two (spatial) axes",
            "mixing_matrix_axes": ["output_channel", "input_channel"],
            "mixing_placement": "after each spatial-screen/Fourier-transfer pair, including the last",
            "readout": "complex weighted sum with no conjugation; intensity is formed after summation",
            "channel_weight_phase_caveat": "Phase of a zero weight is undefined; np.angle displays it as zero",
        })
        paths["channel_mixing_png"] = None
    try:
        model.eval()
        with torch.no_grad(), ZipFile(archive_path, "w", compression=ZIP_DEFLATED,
                                     compresslevel=4, allowZip64=True) as archive:
            _archive_array(archive, "frequency_x_cycles_per_pixel", np.fft.fftfreq(input_size))
            _archive_array(archive, "frequency_y_cycles_per_pixel", np.fft.fftfreq(input_size))
            _archive_array(archive, "input_x_pixels", np.arange(input_size, dtype=np.int32))
            _archive_array(archive, "input_y_pixels", np.arange(input_size, dtype=np.int32))
            if channels > 1:
                lift = _complex_weights(operator.lift_weights(), (channels,), "Input lift")
                readout = _complex_weights(operator.readout_weights(), (channels,), "Coherent readout")
                _archive_array(archive, "lift_weights", lift)
                _archive_array(archive, "readout_weights", readout)
                mixers = []
                for stage in range(depth):
                    matrix = _complex_weights(operator.mixing_matrix(stage),
                                              (channels, channels), f"Mixing {stage + 1}")
                    _archive_array(archive, f"mixing_{stage + 1}", matrix)
                    mixers.append(matrix)
                mixing_path = destination / "channel_mixing.png"
                _plot_channel_weights(lift, readout, mixers, mixing_path, checkpoint_epoch)
                paths["channel_mixing_png"] = str(mixing_path)
                metadata["channel_mixing_plot"] = mixing_path.name
            for stage, (screen, transfer) in enumerate(zip(operator.screens, operator.transfers), 1):
                coefficients = screen.coefficients(shape)
                if coefficients.shape != (channels, 2, *shape):
                    raise ValueError(f"Spatial screen coefficients must be {channels}x2xHxW")
                amplitude = torch.exp(screen.max_log_amplitude * torch.tanh(coefficients[:, 0]))
                phase = screen.max_phase * torch.tanh(coefficients[:, 1])
                smap = _complex_map(torch.polar(amplitude, phase), shape, f"S{stage}", channels)
                hmap = _complex_map(transfer.transfer(shape), shape, f"H{stage}", channels)
                _archive_array(archive, f"s_{stage}", smap)
                _archive_array(archive, f"h_{stage}", hmap)
                stage_info = {
                    "stage": stage,
                    "h_amplitude": _stats(np.abs(hmap)), "h_phase_radians": _stats(np.angle(hmap)),
                    "s_amplitude": _stats(np.abs(smap)), "s_phase_radians": _stats(np.angle(smap)),
                    "h_parameter_bounds": {"max_log_amplitude": float(transfer.max_log_amplitude),
                                           "max_phase_radians": float(transfer.max_phase)},
                    "s_parameter_bounds": {"max_log_amplitude": float(screen.max_log_amplitude),
                                           "max_phase_radians": float(screen.max_phase)},
                }
                if channels == 1:
                    png_path = destination / f"stage_{stage:02d}.png"
                    _plot_stage(hmap, smap, stage, png_path, amplitude_limits, roi, checkpoint_epoch)
                    paths["stage_pngs"].append(str(png_path))
                    stage_info["plot"] = png_path.name
                else:
                    stage_info["channels"] = []
                    for channel in range(channels):
                        png_path = destination / f"stage_{stage:02d}_channel_{channel + 1:02d}.png"
                        _plot_stage(hmap[channel], smap[channel], stage, png_path,
                                    amplitude_limits, roi, checkpoint_epoch, channel=channel + 1)
                        paths["stage_pngs"].append(str(png_path))
                        stage_info["channels"].append({
                            "channel": channel + 1, "plot": png_path.name,
                            "h_amplitude": _stats(np.abs(hmap[channel])),
                            "h_phase_radians": _stats(np.angle(hmap[channel])),
                            "s_amplitude": _stats(np.abs(smap[channel])),
                            "s_phase_radians": _stats(np.angle(smap[channel])),
                        })
                metadata["stages"].append(stage_info)
            gain = float(operator.log_intensity_gain.exp().detach().cpu())
            if not math.isfinite(gain) or gain <= 0:
                raise ValueError("Optical intensity gain must be finite and positive")
            metadata["intensity_gain"] = gain
            background = model.background().detach().cpu().numpy()
            if background.shape != (1, 1, output_size, output_size) or not np.isfinite(background).all():
                raise ValueError("Background must be a finite 1x1xoutput_size xoutput_size intensity map")
            background = background[0, 0]
            _archive_array(archive, "background_intensity", background)
            metadata["background_enabled"] = bool(getattr(model, "background_enabled", True))
            metadata["background_intensity"] = _stats(background)
            if metadata["background_enabled"]:
                background_path = destination / "background.png"
                figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
                try:
                    max_value = max(float(background.max()), np.finfo(np.float32).tiny)
                    image = axis.imshow(background, vmin=0, vmax=max_value, cmap="magma", origin="upper")
                    axis.set(title="Shared additive intensity background", xlabel="Output x (pixel)",
                             ylabel="Output y (pixel)")
                    figure.colorbar(image, ax=axis, label="Camera target intensity units")
                    figure.savefig(background_path, dpi=150)
                finally:
                    plt.close(figure)
                paths["background_png"] = str(background_path)
                metadata["background_plot_color_limits"] = [0, max_value]
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    finally:
        for module, was_training in state_flags:
            module.training = was_training
    return paths
