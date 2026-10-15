"""Offline phase refinement and new CGH synthesis with a frozen shared_optical proxy.

Run beside the current train_fno_axicon.py / axicon_optical.py / axicon_simulator.py.
The camera is never acquired in this loop. z is an absolute physical propagation
distance, NOT a training-folder offset. See docs/shared_optical_phase_optimizer.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TWO_PI = 2 * math.pi

# For IDE / Run File use. CLI values override these. Paths are deliberately not
# inherited from the older FNO script, whose checkpoint is a different model.
SCRIPT_CONFIG = {
    "checkpoint": Path(r"C:\REVAMP\Yangwoo Heo\SLM_to_Axicon_Optimiaztion\CSSNet_training_10072006_newsim\20261013_114624\best.pt"),  # Path(r"C:\...\shared_optical_run\best.pt")
    #"target": Path(r"H:\Shared drives\taylorlab\3DHL\CITL\Fourier Neural Operator_Training phase masks\Targets_Geometric_Variance\00_baseline.png"),      # drawn PNG/TIFF or intensity .npy
    "phase": Path(r"H:\Shared drives\3DHL + OptiCAL + TFT + SpaceCAL\3DHL\FNO\TrainSet_10072026\0.Phase_Masks\real_00_baseline.npy"),       # None: new CGH; .npy path: refine an existing mask
    "target": Path(r"H:\Shared drives\taylorlab\3DHL\CITL\Fourier Neural Operator_Training phase masks\Targets_Geometric_Variance\00_baseline.png"),      # drawn PNG/TIFF or intensity .npy
    #"phase": None,
    "z_mm": 6.0,         # absolute physical distance from simulation z=0
    "output_dir": Path(r"C:\REVAMP\Yangwoo Heo\SLM_to_Axicon_Optimiaztion\phase_update_10072006_set_newsim"),
    "system_config": None,  # JSON overrides of the field-export optical setup
    # This preset reproduces TrainSet_09092026 fields; simulator uses current optics.
    "system_preset": "legacy_training",
    "z_reference_mm": 6.0,
    "reference_field": None,  # optional explicit saved training field for preflight
    "reference_check": "auto",  # look up this phase/z in the checkpoint's training pool
    "flip_target_lr": True,  # opt in only when the target should be mirrored
    "iters": 30,
    "lr": 0.1,          # Adam, radians; unrelated to old SGD lr=20
    "restarts": 1,
    "seed": 48,
    "target_mode": "legacy_optimizer",  # existing Targets_Geometric_Variance drawings
    "legacy_optimizer_roi_size": 1024,
    "target_pixel_size_um": 2.0,
    "target_peak": None, # None preserves input intensity; otherwise set peak
    "loss": "shape", # training, mse, or shape (relative intensity pattern)
}


def _positive(value, name, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def load_proxy(path, device):
    """Use the trainer's architecture migration and strict state-dict contract."""
    from train_fno_axicon import build_model_from_config, migrate_optical_experiment_config

    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not checkpoint.get("cfg"):
        raise ValueError("Checkpoint must contain cfg and model_state_dict")
    cfg = migrate_optical_experiment_config(checkpoint["cfg"])
    if cfg.get("architecture") != "shared_optical":
        raise ValueError("This optimizer requires a shared_optical checkpoint; use fno_citl.py for FNO")
    if cfg.get("use_slm_phase", False):
        raise ValueError("shared_optical requires a field-only checkpoint")
    mode = cfg.get("field_scale_mode", "global_percentile")
    if mode not in ("raw", "global_percentile"):
        raise ValueError("Offline shared optics requires a fixed raw/global field scale, not sample_norm")
    if mode == "global_percentile":
        _field_scale(cfg)
    model = build_model_from_config(cfg, device, initialize=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model, cfg


def _field_scale(cfg):
    value = cfg.get("field_amp_scale_actual")
    if value is None:
        value = cfg.get("field_amp_scale")
    if value is None:
        raise ValueError("Missing saved field_amp_scale_actual / field_amp_scale")
    return _positive(value, "field amplitude scale")


def camera_from_complex(model, field, cfg):
    """Native complex field -> camera, with fixed training scale and no detach.

    Using real/imaginary components directly avoids undefined phase derivatives
    at field zeros. This is the same coherent operator and detector as training.
    """
    from train_fno_axicon import native_center_crop

    if field.ndim != 2 or not field.is_complex():
        raise ValueError("Expected a single 2D complex field in camera orientation")
    field = native_center_crop(field, int(cfg.get("optical_input_size", 1024)))
    mode = cfg.get("field_scale_mode", "global_percentile")
    if mode == "global_percentile":
        field = field / (_field_scale(cfg) + 1e-8)
    elif mode != "raw":
        raise ValueError(f"Unsupported field_scale_mode={mode!r}; fixed scaling is required")
    corrected = model.operator.forward_complex(field.unsqueeze(0))
    corrected = native_center_crop(corrected, model.operator.output_size)
    return (corrected.abs().square() * model.operator.log_intensity_gain.exp()).unsqueeze(1) + model.background()


def compare_reference_field(field, path, cfg, tolerance=1e-3):
    """Check the physical field against the training export before phase updates.

    Only a constant unit-magnitude phase is aligned: gain, native scale and
    spatial structure must match. This prevents a wrong simulator from being
    hidden by intensity renormalization. Reference arrays are unscaled exports.
    """
    from train_fno_axicon import native_center_crop

    _positive(tolerance, "reference tolerance")
    array = np.load(path, allow_pickle=False)
    if np.iscomplexobj(array) and array.ndim == 2:
        reference = array
    elif array.ndim == 3 and array.shape[0] == 3:
        reference = array[0] * (array[1] + 1j * array[2])
    elif array.ndim == 3 and array.shape[-1] == 3:
        reference = array[..., 0] * (array[..., 1] + 1j * array[..., 2])
    else:
        raise ValueError(f"Unsupported reference field shape {array.shape}; expected complex HW or amp/cos/sin")
    size = int(cfg.get("optical_input_size", 1024))
    ref = native_center_crop(torch.from_numpy(np.ascontiguousarray(reference)), size).to(torch.complex128)
    sim = native_center_crop(field.detach().cpu(), size).to(torch.complex128)
    if not torch.isfinite(ref).all() or not torch.isfinite(sim).all():
        raise ValueError("Nonfinite simulation/reference field")
    ref_norm, sim_norm = torch.linalg.vector_norm(ref), torch.linalg.vector_norm(sim)
    if ref_norm <= 0 or sim_norm <= 0:
        raise ValueError("Simulation/reference field has zero energy")
    dot = (sim.conj() * ref).sum()
    unit_phase = dot / dot.abs() if dot.abs() > 0 else dot.new_ones(())
    error = float(torch.linalg.vector_norm(ref - unit_phase * sim) / ref_norm)
    p, q = sim.abs().square().flatten(), ref.abs().square().flatten()
    p, q = p - p.mean(), q - q.mean()
    denominator = torch.linalg.vector_norm(p) * torch.linalg.vector_norm(q)
    correlation = float((p * q).sum() / denominator) if denominator > 0 else None
    report = {"path": str(Path(path).resolve()), "aligned_relative_error": error,
              "amplitude_ratio": float(sim_norm / ref_norm), "intensity_correlation": correlation}
    if error > tolerance:
        raise ValueError(
            f"Simulation does not match reference training field: relative error={error:.6g}, "
            f"amplitude ratio={report['amplitude_ratio']:.6g}. Check system-preset, pitch, "
            "upsampling, z, phase orientation and filter. No phase update was performed. "
            "Use --reference-check off only for an intentionally different forward setup.")
    print(f"Training field check passed: relative error={error:.3g}, amplitude ratio={report['amplitude_ratio']:.6g}")
    return report


def find_reference_field(args, cfg):
    """Resolve a matching phase stem and z offset without constructing a dataset."""
    if args.reference_check == "off":
        return None
    if args.reference_field is not None:
        if not args.reference_field.is_file():
            raise ValueError(f"Reference field does not exist: {args.reference_field}")
        if args.phase is None:
            raise ValueError("Reference comparison requires --phase for the same mask")
        return args.reference_field
    if args.phase is not None and cfg.get("pool_dir"):
        directory = Path(cfg.get("workflow_field_dir", "1.Forward_Sim"))
        if not directory.is_absolute():
            directory = Path(cfg["pool_dir"]) / directory
        paths = []
        if cfg.get("z_semantics") == "offset_from_reference_plane":
            if args.z_reference_mm is not None and directory.is_dir():
                offset = args.z_mm - args.z_reference_mm
                for folder in directory.iterdir():
                    match = re.fullmatch(r"z_([+-]?\d+(?:\.\d+)?)", folder.name)
                    if folder.is_dir() and match and math.isclose(float(match[1]), offset, abs_tol=1e-6):
                        paths.append(folder / (args.phase.stem + ".npy"))
        else:
            paths.append(directory / (args.phase.stem + ".npy"))
        existing = [path for path in paths if path.is_file()]
        if len(existing) > 1:
            raise ValueError("Ambiguous reference fields; specify --reference-field explicitly")
        if existing:
            return existing[0]
    if args.reference_check == "required":
        raise ValueError("No matching reference field found; specify --reference-field and the corresponding --phase")
    print("No matching training field found for this phase/z; reference check skipped.")
    return None


def load_phase(path, shape, device, units="slm", level_max=1023,
               transpose=True, flip_first_axis=True):
    """Read file orientation into the simulator's internal (x,y) phase array.

    .npy only: this avoids silently reducing a 10/16-bit phase image to 8 bits.
    The phase/radian option changes units, not orientation.
    """
    _positive(level_max, "level_max")
    if Path(path).suffix.lower() != ".npy":
        raise ValueError("Initial phase must be a 2D .npy array")
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2 or np.iscomplexobj(arr) or not np.isfinite(arr).all():
        raise ValueError("Phase must be a finite real 2D array")
    if transpose:
        arr = arr.T
    if flip_first_axis:
        arr = arr[::-1, :]
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"Oriented phase shape {arr.shape} != simulator shape {tuple(shape)}")
    if units == "slm":
        if arr.min() < 0 or arr.max() > level_max:
            raise ValueError(f"SLM commands must be in [0, {level_max}]")
        arr = arr * (TWO_PI / level_max)
    elif units != "rad":
        raise ValueError("Phase units must be slm or rad")
    return torch.tensor(np.ascontiguousarray(arr), dtype=torch.float32, device=device)


def phase_to_slm_array(phase_rad, level_max=1023, transpose=True, flip_first_axis=True):
    """Inverse file orientation with the existing linear [0,level_max] -> [0,2pi] convention."""
    if isinstance(level_max, bool) or int(level_max) != level_max or not 1 <= level_max <= 32767:
        raise ValueError("level_max must be an integer in [1,32767]")
    if phase_rad.ndim != 2 or not torch.isfinite(phase_rad).all():
        raise ValueError("Phase must be a finite 2D tensor")
    levels = torch.round(torch.remainder(phase_rad.detach(), TWO_PI) * (level_max / TWO_PI))
    arr = levels.cpu().numpy()
    if flip_first_axis:
        arr = arr[::-1, :]
    if transpose:
        arr = arr.T
    return np.ascontiguousarray(arr, dtype=np.int16)


def wrapped_tv(phase):
    terms = [(2 - 2 * torch.cos(phase.diff(dim=dim))).mean()
             for dim in (0, 1) if phase.shape[dim] > 1]
    return sum(terms, phase.sum() * 0)


def make_loss(mode, cfg, roi_fraction=1.0):
    from train_fno_axicon import apply_loss_roi, loss_kwargs_from_config, visual_loss

    if not math.isfinite(roi_fraction) or not 0 < roi_fraction <= 1:
        raise ValueError("loss ROI fraction must be in (0,1]")
    if mode not in ("training", "mse", "shape"):
        raise ValueError(f"Unknown loss: {mode}")

    def loss(pred, target):
        pred, target = apply_loss_roi(pred, target, roi_fraction)
        if mode == "training":
            return visual_loss(pred, target, **loss_kwargs_from_config(cfg))[0]
        if mode == "shape":
            # Relative pattern loss has no brightness/efficiency guarantee.
            p = pred / pred.mean().clamp_min(1e-8)
            t = target / target.mean().clamp_min(1e-8)
            return F.smooth_l1_loss(p, t)
        return F.mse_loss(pred, target)
    return loss


def optimize_phase(predict, target, initial_phase, *, iters=50, lr=.03,
                   loss_fn=None, anchor_weight=0., tv_weight=0.,
                   max_phase_change=None, grad_clip=None, print_every=10):
    """Evaluate iteration 0 and every Adam candidate, including the final step.

    ``predict`` is differentiable; freeze model weights before calling. The
    anchor and optional bound are periodic phase differences from initialization.
    Only the phase tensor is passed to Adam. Returns the best evaluated state.
    """
    if isinstance(iters, bool) or int(iters) != iters or iters < 0:
        raise ValueError("iters must be a nonnegative integer")
    _positive(lr, "lr")
    _positive(anchor_weight, "anchor_weight", True)
    _positive(tv_weight, "tv_weight", True)
    if grad_clip is not None:
        _positive(grad_clip, "grad_clip")
    if max_phase_change is not None:
        if not 0 < _positive(max_phase_change, "max_phase_change") <= math.pi:
            raise ValueError("max_phase_change must be in (0,pi]")
    if initial_phase.ndim != 2 or not torch.isfinite(initial_phase).all():
        raise ValueError("Initial phase must be finite and 2D")
    if target.ndim != 4 or target.shape[:2] != (1, 1) or not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError("Target must be finite nonnegative 1x1xHxW intensity")
    if not torch.any(target > 0):
        raise ValueError("Target is empty after preprocessing/cropping")
    if target.device != initial_phase.device:
        raise ValueError("Target and phase must be on the same device")
    loss_fn = loss_fn or F.mse_loss
    initial = initial_phase.detach().clone()
    phase = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([phase], lr=float(lr))
    history, best_loss, best_iteration = [], float("inf"), 0
    best_phase = initial.clone()

    for iteration in range(iters + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(iteration < iters):
            pred = predict(phase)
            if pred.shape != target.shape or not torch.isfinite(pred).all():
                raise ValueError(f"Invalid prediction at iteration {iteration}: shape={tuple(pred.shape)}")
            data_loss = loss_fn(pred, target)
            anchor = (1 - torch.cos(phase - initial)).mean() if anchor_weight else phase.new_zeros(())
            tv = wrapped_tv(phase) if tv_weight else phase.new_zeros(())
            total = data_loss + anchor_weight * anchor + tv_weight * tv
        if not torch.isfinite(total):
            raise FloatingPointError(f"Nonfinite loss at iteration {iteration}; reduce lr/check the forward setup")
        row = {"iteration": iteration, "loss": float(total.detach()),
               "data_loss": float(data_loss.detach()), "anchor": float(anchor.detach()),
               "phase_tv": float(tv.detach()), "raw_mse": float(F.mse_loss(pred.detach(), target))}
        history.append(row)
        if iteration == 0:
            initial_prediction = pred.detach().clone()
        if row["loss"] < best_loss:
            best_loss, best_iteration = row["loss"], iteration
            best_phase = phase.detach().clone()
            best_prediction = pred.detach().clone()
        if print_every and (iteration % print_every == 0 or iteration == iters):
            print(f"  {iteration:4d}/{iters}: loss={row['loss']:.6g}, best={best_loss:.6g}", flush=True)
        if iteration == iters:
            last_prediction = pred.detach().clone()
            break
        total.backward()
        if phase.grad is None or not torch.isfinite(phase.grad).all():
            raise FloatingPointError("Missing/nonfinite phase gradient; the forward path must remain differentiable")
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([phase], grad_clip)
        optimizer.step()
        with torch.no_grad():
            if max_phase_change is not None:
                delta = torch.atan2(torch.sin(phase - initial), torch.cos(phase - initial))
                phase.copy_(initial + delta.clamp(-max_phase_change, max_phase_change))
            else:
                phase.remainder_(TWO_PI)
        del pred, total, data_loss
    return {"best_phase": best_phase, "last_phase": phase.detach().clone(),
            "initial_prediction": initial_prediction, "best_prediction": best_prediction,
            "last_prediction": last_prediction, "history": history,
            "best_iteration": best_iteration, "best_loss": best_loss}


def _json_default(value):
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def phase_comparison_arrays(initial_phase, updated_phase, *, transpose=True, flip_first_axis=True):
    """Continuous phases in SLM-file orientation, and signed circular change [rad]."""
    initial = initial_phase.detach()
    updated = updated_phase.detach()
    change = torch.atan2(torch.sin(updated - initial), torch.cos(updated - initial))

    def orient(value):
        if flip_first_axis:
            value = value.flip(0)
        if transpose:
            value = value.T
        return value.cpu().numpy()

    return (orient(initial.remainder(TWO_PI)), orient(updated.remainder(TWO_PI)), orient(change))


def save_preview(path, target, initial, best, initial_phase, updated_phase, *,
                 transpose_phase=True, flip_phase_first_axis=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = [x.detach().cpu()[0, 0].numpy() for x in (target, initial, best)]
    scale = max(float(np.percentile(np.concatenate([x.ravel() for x in images]), 99.9)), 1e-8)
    phases = phase_comparison_arrays(initial_phase, updated_phase,
                                     transpose=transpose_phase, flip_first_axis=flip_phase_first_axis)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for ax, arr, title in zip(axes[0], images, ("Target intensity", "Initial proxy", "Updated proxy (best)")):
        intensity_image = ax.imshow(arr, cmap="inferno", vmin=0, vmax=scale, interpolation="nearest")
        ax.set_title(title)
    fig.colorbar(intensity_image, ax=list(axes[0]), shrink=.8, label="Proxy intensity (common scale)")
    for ax, arr, title in zip(axes[1, :2], phases[:2], ("Initial phase", "Updated phase (best)")):
        phase_image = ax.imshow(arr, cmap="twilight", vmin=0, vmax=TWO_PI, interpolation="nearest")
        ax.set_title(title + " [SLM file orientation]")
    phase_bar = fig.colorbar(phase_image, ax=list(axes[1, :2]), shrink=.8, ticks=[0, math.pi, TWO_PI])
    phase_bar.ax.set_yticklabels(["0", "pi", "2pi"])
    phase_bar.set_label("Phase [rad]")
    change_image = axes[1, 2].imshow(phases[2], cmap="bwr", vmin=-math.pi, vmax=math.pi, interpolation="nearest")
    axes[1, 2].set_title("Phase change: updated - initial (wrapped)")
    change_bar = fig.colorbar(change_image, ax=axes[1, 2], shrink=.8, ticks=[-math.pi, 0, math.pi])
    change_bar.ax.set_yticklabels(["-pi", "0", "+pi"])
    change_bar.set_label("Phase change [rad]")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"Proxy predictions, not measurements | common intensity p99.9={scale:.4g}")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("checkpoint", "target", "phase", "output_dir", "system_config", "reference_field"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=SCRIPT_CONFIG[name])
    parser.add_argument("--system-preset", choices=("legacy_training", "simulator"), default=SCRIPT_CONFIG["system_preset"])
    parser.add_argument("--reference-check", choices=("auto", "required", "off"), default=SCRIPT_CONFIG["reference_check"])
    parser.add_argument("--z-mm", type=float, default=SCRIPT_CONFIG["z_mm"], help="Absolute physical propagation distance in mm")
    parser.add_argument("--z-reference-mm", type=float, default=SCRIPT_CONFIG["z_reference_mm"], help="Physical reference of training offsets (legacy data: 6 mm)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for name in ("iters", "restarts", "seed"):
        parser.add_argument("--" + name, type=int, default=SCRIPT_CONFIG[name])
    parser.add_argument("--lr", type=float, default=SCRIPT_CONFIG["lr"])
    parser.add_argument("--init", choices=("random", "zero"), default="random", help="Used only without --phase")
    parser.add_argument("--new-cgh", action="store_true", help="Ignore SCRIPT_CONFIG phase and initialize a new mask")
    parser.add_argument("--loss", choices=("training", "mse", "shape"), default=SCRIPT_CONFIG["loss"])
    parser.add_argument("--loss-roi-fraction", type=float, default=None)
    parser.add_argument("--target-mode", choices=("legacy_optimizer", "physical", "native"), default=SCRIPT_CONFIG["target_mode"])
    parser.add_argument("--legacy-optimizer-roi-size", type=int, default=SCRIPT_CONFIG["legacy_optimizer_roi_size"])
    parser.add_argument("--target-pixel-size-um", type=float, default=SCRIPT_CONFIG["target_pixel_size_um"])
    parser.add_argument("--target-peak", type=float, default=SCRIPT_CONFIG["target_peak"], help="Peak in checkpoint camera units; omitted: preserve intensity")
    parser.add_argument("--phase-units", choices=("slm", "rad"), default="slm")
    parser.add_argument("--phase-level-max", type=int, default=1023)
    parser.add_argument("--transpose-phase", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flip-phase-first-axis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transpose-target", action="store_true")
    parser.add_argument("--flip-target-ud", action="store_true")
    parser.add_argument("--flip-target-lr", action=argparse.BooleanOptionalAction,
                        default=SCRIPT_CONFIG["flip_target_lr"])
    parser.add_argument("--anchor-weight", type=float, default=0.)
    parser.add_argument("--tv-weight", type=float, default=0.)
    parser.add_argument("--max-phase-change", type=float, help="Optional periodic change bound from initialization, radians <= pi")
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--print-every", type=int, default=10)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.new_cgh:
        args.phase = None
    from shared_optical_phase_system import build_system, prepare_target

    for name in ("checkpoint", "target"):
        value = getattr(args, name)
        if value is None or not Path(value).is_file():
            raise ValueError(f"Set --{name} (or SCRIPT_CONFIG['{name}']) to an existing file")
    if args.restarts < 1 or (args.phase is not None and args.restarts != 1):
        raise ValueError("restarts must be >=1; existing-phase refinement uses restarts=1")
    if args.init == "zero" and args.phase is None and args.restarts > 1:
        raise ValueError("Multiple restarts require random initialization")
    if args.iters < 0 or args.print_every < 0 or not 1 <= args.phase_level_max <= 32767:
        raise ValueError("iters/print-every must be >=0 and phase-level-max in [1,32767]")
    _positive(args.lr, "lr")
    _positive(args.z_mm, "z_mm")
    model, cfg = load_proxy(args.checkpoint, args.device)
    overrides = {} if args.system_config is None else json.loads(args.system_config.read_text(encoding="utf-8"))
    if not isinstance(overrides, dict):
        raise ValueError("system-config must be a JSON object")
    required_size = int(cfg.get("optical_input_size", 1024))
    if "roi_size" in overrides and overrides["roi_size"] != required_size:
        raise ValueError("system roi_size must equal checkpoint optical_input_size")
    overrides["roi_size"] = required_size
    system = build_system(args.z_mm * 1e-3, args.device, overrides, preset=args.system_preset)
    target, target_report = prepare_target(
        args.target, int(cfg["model_size"]), system.beam_config,
        system.settings["upsample_factor"], mode=args.target_mode,
        target_pixel_size_m=args.target_pixel_size_um * 1e-6,
        transpose=args.transpose_target, flip_ud=args.flip_target_ud, flip_lr=args.flip_target_lr,
        legacy_optimizer_roi_size=args.legacy_optimizer_roi_size)
    if args.target_peak is not None:
        target = target / target.amax().clamp_min(1e-8) * _positive(args.target_peak, "target_peak")
    target_report["effective_peak"] = float(target.max())
    target_report["effective_mean"] = float(target.mean())
    if args.loss == "shape" and not torch.any(target > 0):
        raise ValueError("Shape loss needs a nonzero target")
    roi = cfg.get("loss_roi_fraction", 1.) if args.loss_roi_fraction is None else args.loss_roi_fraction
    loss_fn = make_loss(args.loss, cfg, float(roi))
    z_note = {"physical_z_mm": args.z_mm, "training_z_semantics": cfg.get("z_semantics"),
              "training_z_positions_mm": cfg.get("z_positions_mm"), "z_reference_mm": args.z_reference_mm}
    if cfg.get("z_semantics") == "offset_from_reference_plane":
        if args.z_reference_mm is None:
            warnings.warn("Checkpoint z values are offsets; no reference was saved. Physical --z-mm is used; training-range validity is unverified.")
        else:
            if not math.isfinite(args.z_reference_mm):
                raise ValueError("z-reference-mm must be finite")
            offset = args.z_mm - args.z_reference_mm
            z_note["requested_offset_mm"] = offset
            positions = cfg.get("z_positions_mm", [])
            if positions and not min(positions) <= offset <= max(positions):
                warnings.warn(f"Requested offset {offset:g} mm is outside the trained z range")
    print(f"Frozen shared_optical: native {required_size} -> camera {cfg['model_size']}; z={args.z_mm:g} mm")
    print(f"Optics preset: {args.system_preset}; target layout: {args.target_mode}")
    print(f"Upsampling: {system.settings['upsample_factor']}; training z reference: {args.z_reference_mm} mm")
    print("Target intensities use checkpoint camera units; PNG 0..1 is not a measured exposure.")
    reference_path = find_reference_field(args, cfg)
    reference_report = None
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    shape = (system.beam_config.Nx, system.beam_config.Ny)

    def predict(phase):
        nonlocal reference_report
        field = system.forward(phase)
        if reference_path is not None and reference_report is None:
            reference_report = compare_reference_field(field, reference_path, cfg)
        return camera_from_complex(model, field, cfg)

    chosen, chosen_initial, chosen_restart = None, None, None
    restart_metrics, history = [], []
    for restart in range(args.restarts):
        if args.phase is not None:
            initial = load_phase(args.phase, shape, args.device, args.phase_units,
                                 args.phase_level_max, args.transpose_phase, args.flip_phase_first_axis)
        elif args.init == "zero":
            initial = torch.zeros(shape, device=args.device)
        else:
            initial = torch.rand(shape, generator=generator, device=args.device) * TWO_PI
        print(f"Restart {restart + 1}/{args.restarts}", flush=True)
        result = optimize_phase(predict, target, initial, iters=args.iters, lr=args.lr,
                                loss_fn=loss_fn, anchor_weight=args.anchor_weight, tv_weight=args.tv_weight,
                                max_phase_change=args.max_phase_change, grad_clip=args.grad_clip,
                                print_every=args.print_every)
        history.extend({"restart": restart, **row} for row in result["history"])
        restart_metrics.append({"restart": restart, "best_iteration": result["best_iteration"], "best_loss": result["best_loss"]})
        if chosen is None or result["best_loss"] < chosen["best_loss"]:
            chosen, chosen_initial, chosen_restart = result, initial.detach().clone(), restart
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    mask_path = run_dir / "phase_best_slm.npy"
    np.save(mask_path, phase_to_slm_array(chosen["best_phase"], args.phase_level_max,
                                        args.transpose_phase, args.flip_phase_first_axis))
    print(f"SLM mask: {mask_path.resolve()}")
    config = {"arguments": vars(args), "checkpoint_cfg": cfg, "system": system.settings,
              "target": target_report, "z": z_note, "reference_field_check": reference_report,
              "phase_response": "linear commands 0..level_max map to 0..2pi"}
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=_json_default), encoding="utf-8")
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with torch.no_grad():
        metrics = {"selected_restart": chosen_restart, "best_iteration": chosen["best_iteration"],
                   "best_total_loss": chosen["best_loss"], "restarts": restart_metrics}
        for label, pred in (("initial", chosen["initial_prediction"]), ("best", chosen["best_prediction"]),
                            ("last", chosen["last_prediction"])):
            metrics[label] = {"data_loss": float(loss_fn(pred, target)), "raw_mse": float(F.mse_loss(pred, target)),
                              "mean_intensity": float(pred.mean()), "max_intensity": float(pred.max())}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_preview(run_dir / "comparison.png", target, chosen["initial_prediction"],
                 chosen["best_prediction"], chosen_initial, chosen["best_phase"],
                 transpose_phase=args.transpose_phase, flip_phase_first_axis=args.flip_phase_first_axis)
    print(f"Saved: {run_dir.resolve()}")
    return run_dir


if __name__ == "__main__":
    main()
