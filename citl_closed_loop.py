"""Single image / single plane camera-in-the-loop phase refinement.

    python citl_closed_loop.py --config citl_config.example.json --preflight
    python citl_closed_loop.py --config citl_config.example.json
    python citl_closed_loop.py --dry-run --iters 5 --output-dir runs_citl

Only this entry point is run. Hardware, training preprocessing and the frozen
CSSNet adapter remain separate so they can be verified without a connected SLM.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


@dataclass
class LoopConfig:
    iters: int = 30
    lr: float = 0.1
    grad_clip: float | None = None
    max_phase_change: float | None = None
    min_valid_fraction: float = 0.9
    plot_every: int = 1

    def validate(self):
        for name in ("iters", "plot_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (name == "plot_every"):
                raise ValueError(f"{name} must be an integer >= {int(name == 'plot_every')}")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(self.min_valid_fraction) or not 0 < self.min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be in (0,1]")
        for name in ("grad_clip", "max_phase_change"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.max_phase_change is not None and self.max_phase_change > math.pi:
            raise ValueError("max_phase_change must be <= pi")


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def as_image(tensor):
    return tensor.detach().cpu().numpy().squeeze().astype(np.float32)


def save_plots(run_dir, iteration_dir, history, target, measured, predicted, initial, best):
    # Fixed intensity display range across iterations; no per-frame rescaling.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = max(float(np.max(target)), float(np.max(initial)), 1e-8)
    support = np.load(run_dir / "loss_support.npy", allow_pickle=False)
    initial, measured, best = (np.where(support, a, np.nan) for a in (initial, measured, best))
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#a0a0a0")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, array, title in zip(axes.flat,
            (target, initial, measured, predicted, best, np.abs(measured - target)),
            ("Target", "Initial measured", "Current measured", "CSSNet prediction", "Best measured", "Absolute error")):
        im = axis.imshow(array, cmap=cmap, vmin=0, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.75, label="Checkpoint camera intensity")
    fig.suptitle(f"Iteration {history[-1]['iteration']} | measured loss {history[-1]['loss']:.6g}")
    fig.savefig(iteration_dir / "comparison.png", dpi=140)
    fig.savefig(run_dir / "comparison_latest.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    x = [row["iteration"] for row in history]
    ax.plot(x, [row["loss"] for row in history], "o-", label="Measured camera loss")
    ax.plot(x, [row["proxy_loss"] for row in history], "--", label="Proxy loss (diagnostic)")
    ax.plot(x, np.minimum.accumulate([row["loss"] for row in history]), ":", label="Best measured")
    ax.set(xlabel="Completed phase updates (0 = initial)", ylabel="Loss")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(run_dir / "loss_curve.png", dpi=160)
    plt.close(fig)


def save_bracket(directory, arrays, exposures):
    """Persist untouched headerless little-endian Bayer mosaics, before any LUT."""
    directory.mkdir(parents=True)
    if set(arrays) != set(exposures):
        raise ValueError("Camera bracket exposures differ from calibration")
    paths = {}
    for exposure in exposures:
        array = np.asarray(arrays[exposure])
        if array.ndim != 2 or array.dtype != np.uint16:
            raise ValueError("Camera must return unpacked uint16 2D Bayer mosaics")
        path = directory / f"Exp{exposure}.raw"
        np.ascontiguousarray(array, dtype="<u2").tofile(path)
        paths[exposure] = path
    return paths


class LiveMeasurement:
    def __init__(self, hardware, preprocessor, reference_commands):
        self.hardware = hardware
        self.preprocessor = preprocessor
        self.reference_commands = reference_commands

    def __call__(self, commands, iteration_dir):
        exposures = self.preprocessor.exposures_us
        self.hardware.upload_phase(self.reference_commands)
        reference_paths = save_bracket(iteration_dir / "raw_reference",
                                      self.hardware.capture_bracket(exposures), exposures)
        self.hardware.upload_phase(commands)
        target_paths = save_bracket(iteration_dir / "raw_target",
                                   self.hardware.capture_bracket(exposures), exposures)
        write_json(iteration_dir / "hardware.json", self.hardware.metadata)
        return self.preprocessor.process(target_paths, reference_paths, stem=iteration_dir.name)


def run_loop(proxy, measure, run_dir, config):
    """Perform N updates and N+1 measured evaluations, including the final phase.

    measure(commands, iteration_dir) returns intensity + HDR quality masks.
    Camera loss is differentiated at the measurement, then used as the upstream
    gradient for the proxy. Only the phase parameter belongs to Adam.
    """
    config.validate()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "history.csv").exists():
        raise FileExistsError("Use a new run directory; existing measurements are never overwritten")
    initial_phase = proxy.initial_phase.detach().clone()
    phase = torch.nn.Parameter(initial_phase.clone())
    optimizer = torch.optim.Adam([phase], lr=config.lr)
    target = proxy.target.detach()
    np.save(run_dir / "target.npy", as_image(target))
    history, best_loss, best_iteration = [], float("inf"), None
    fixed_mask, first_image, best_image = None, None, None
    start = time.monotonic()
    try:
        for iteration in range(config.iters + 1):
            iteration_dir = run_dir / "iterations" / f"{iteration:04d}"
            iteration_dir.mkdir(parents=True, exist_ok=False)
            commands = proxy.phase_to_slm(phase)
            np.save(iteration_dir / "phase_slm.npy", commands)
            np.save(iteration_dir / "phase_rad_internal.npy", as_image(phase))
            frame = measure(commands, iteration_dir)
            # This mask was measured even if a later quality/gradient check fails.
            np.save(run_dir / "phase_last_measured_slm.npy", commands)
            # Keep the calibrated HDR result, including NaNs and quality, exactly.
            np.save(iteration_dir / "aligned_hdr.npy", frame.intensity)
            np.savez_compressed(iteration_dir / "quality.npz", valid_mask=frame.valid_mask,
                                confidence=frame.confidence, fallback_mask=frame.fallback_mask)
            write_json(iteration_dir / "preprocessing.json", frame.diagnostics)
            valid_array = np.asarray(frame.valid_mask, dtype=bool) & np.isfinite(frame.intensity)
            if valid_array.shape != np.shape(frame.intensity):
                raise ValueError("HDR validity shape must match intensity")
            # Invalid pixels are placeholders only; the mask excludes their loss.
            safe_intensity = np.where(valid_array, frame.intensity, 0).astype(np.float32)
            measured = proxy.camera_tensor(safe_intensity).detach().requires_grad_(True)
            valid = proxy.mask_tensor(valid_array).bool()
            if measured.shape != target.shape or valid.shape != target.shape:
                raise ValueError("Measured camera / quality / target grids do not match")
            valid_fraction = float(valid.float().mean())
            if valid_fraction < config.min_valid_fraction:
                raise ValueError(f"Only {valid_fraction:.1%} camera pixels valid; check exposure / affine")
            if fixed_mask is None:
                fixed_mask = valid.detach().clone()
                np.save(run_dir / "loss_support.npy", as_image(fixed_mask).astype(bool))
            elif bool((fixed_mask & ~valid).any()):
                raise ValueError("A pixel in the initial loss support became invalid. Stopping before an update; "
                                 "check saturation and registration (loss support stays fixed for comparisons).")
            optimizer.zero_grad(set_to_none=True)
            predicted = proxy.predict(phase)
            if predicted.shape != target.shape or not bool(torch.isfinite(predicted).all()):
                raise FloatingPointError("Invalid proxy camera prediction")
            data_loss = proxy.loss(measured, mask=fixed_mask)
            proxy_loss = proxy.loss(predicted.detach(), mask=fixed_mask)
            if not bool(torch.isfinite(data_loss)) or not bool(torch.isfinite(proxy_loss)):
                raise FloatingPointError("Nonfinite measured / proxy loss")
            raw_mse = ((measured.detach() - target).square()[fixed_mask]).mean()
            row = dict(iteration=iteration, loss=float(data_loss.detach()),
                       proxy_loss=float(proxy_loss), raw_mse=float(raw_mse),
                       valid_fraction=valid_fraction, gradient_norm=0.,
                       elapsed_s=time.monotonic() - start)
            measured_array, predicted_array = as_image(measured), as_image(predicted)
            np.save(iteration_dir / "camera_scaled.npy", measured_array)
            np.save(iteration_dir / "proxy.npy", predicted_array)
            if first_image is None:
                first_image = measured_array.copy()
            if row["loss"] < best_loss:
                best_loss, best_iteration = row["loss"], iteration
                best_image = measured_array.copy()
                np.save(run_dir / "phase_best_slm.npy", commands)
                np.save(run_dir / "phase_best_rad_internal.npy", as_image(phase))
                np.save(run_dir / "camera_best.npy", best_image)
            if iteration < config.iters:
                camera_gradient, = torch.autograd.grad(data_loss, measured)
                predicted.backward(camera_gradient.detach())
                if phase.grad is None or not bool(torch.isfinite(phase.grad).all()):
                    raise FloatingPointError("Missing/nonfinite dL/df * d(proxy)/d(phase)")
                row["gradient_norm"] = float(torch.linalg.vector_norm(phase.grad))
                if config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_([phase], config.grad_clip)
            history.append(row)
            with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            write_json(run_dir / "status.json", dict(status="running", **row,
                       best_iteration=best_iteration, best_loss=best_loss))
            if iteration % config.plot_every == 0 or iteration == config.iters:
                save_plots(run_dir, iteration_dir, history, as_image(target), measured_array,
                           predicted_array, first_image, best_image)
            print(f"[{iteration:04d}/{config.iters:04d}] camera loss={row['loss']:.6g} "
                  f"best={best_loss:.6g} valid={valid_fraction:.2%}", flush=True)
            if iteration == config.iters:
                np.save(run_dir / "phase_final_slm.npy", commands)
                np.save(run_dir / "phase_final_rad_internal.npy", as_image(phase))
                np.save(run_dir / "camera_final.npy", measured_array)
                break
            optimizer.step()
            with torch.no_grad():
                if config.max_phase_change is not None:
                    delta = torch.atan2(torch.sin(phase - initial_phase), torch.cos(phase - initial_phase))
                    phase.copy_(initial_phase + delta.clamp(-config.max_phase_change, config.max_phase_change))
                else:
                    phase.remainder_(2 * math.pi)
            del predicted, measured, data_loss
        result = dict(status="complete", updates=config.iters, measurements=len(history),
                      initial_loss=history[0]["loss"], final_loss=history[-1]["loss"],
                      best_loss=best_loss, best_iteration=best_iteration)
        write_json(run_dir / "status.json", result)
        return result
    except BaseException as error:
        write_json(run_dir / "status.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   error=f"{type(error).__name__}: {error}", completed_measurements=len(history),
                   best_iteration=best_iteration, best_loss=best_loss if math.isfinite(best_loss) else None))
        raise


class SyntheticProxy:
    """Small deterministic differentiable toy model for exercising the controller."""
    def __init__(self):
        from citl_proxy import make_camera_loss
        self.initial_phase = torch.full((32, 40), .65)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 40), indexing="ij")
        self.target = (.35 + .5 * torch.exp(-5 * (xx.square() + yy.square())))[None, None]
        self._loss = make_camera_loss("mse")
        self.metadata = {"synthetic": True, "note": "Toy response; does not validate optical calibration or real hardware"}

    def phase_to_slm(self, phase):
        return torch.round(phase.detach().remainder(2 * math.pi) * (1023 / (2 * math.pi))).numpy().astype(np.uint16)

    def predict(self, phase):
        from citl_proxy import quantize_phase_ste
        quantized = quantize_phase_ste(phase, 1023)
        return (torch.sin(quantized).square() + .03)[None, None]

    def camera_tensor(self, array):
        return torch.as_tensor(array, dtype=torch.float32)[None, None]

    def mask_tensor(self, array):
        return torch.as_tensor(array, dtype=torch.bool)[None, None]

    def loss(self, value, mask=None):
        return self._loss(value, self.target, mask)


def synthetic_measurement(commands, iteration_dir):
    phase = commands.astype(np.float32) * (2 * np.pi / 1023)
    intensity = (.92 * np.square(np.sin(phase)) + .04).astype(np.float32)
    return SimpleNamespace(intensity=intensity, valid_mask=np.ones_like(intensity, dtype=bool),
                           confidence=np.ones_like(intensity), fallback_mask=np.zeros_like(intensity, dtype=bool),
                           diagnostics={"synthetic": True})


def prepare_live(settings, worker_log_path):
    """Validate paths/calibration and a complete proxy derivative before opening devices."""
    from citl_proxy import ProxyAdapter
    from citl_camera_preprocess import CameraPreprocessConfig
    from citl_camera_worker import CameraPreprocessorWorker
    from citl_hardware import HardwareConfig, prepare_slm_commands

    proxy_settings = dict(settings["proxy"])
    for key in ("proxy_source_dir", "checkpoint", "phase", "target"):
        if not proxy_settings.get(key):
            raise ValueError(f"Set proxy.{key} to the experimental PC's local path")
    reference_path = settings.get("reference_phase")
    if not reference_path or not Path(reference_path).is_file():
        raise ValueError("Set reference_phase to the SAME fixed reference mask used during training captures")
    reference = np.load(reference_path, allow_pickle=False)
    if reference.ndim != 2 or not reference.size or np.iscomplexobj(reference) or not np.isfinite(reference).all() or not np.equal(reference, np.rint(reference)).all() or reference.min() < 0 or reference.max() > 1023:
        raise ValueError("reference_phase must contain integer 0..1023 SLM commands in file orientation")
    reference = np.ascontiguousarray(reference, dtype=np.uint16)
    pp_settings = dict(settings["preprocessing"])
    expected_offset = float(proxy_settings["z_mm"]) - float(proxy_settings.get("z_reference_mm", 6.))
    if not math.isclose(float(pp_settings.get("z_mm", 0.)), expected_offset, abs_tol=1e-6):
        raise ValueError("preprocessing.z_mm is a training folder offset: use proxy.z_mm - proxy.z_reference_mm")
    hardware_config = HardwareConfig(**settings["hardware"])
    if not Path(hardware_config.config_path).is_file():
        raise FileNotFoundError("hardware.config_path must point to the acquisition PC's hololith setting.yaml")
    if hardware_config.hololith_root and not (Path(hardware_config.hololith_root) / "hololith").is_dir():
        raise FileNotFoundError("hardware.hololith_root must contain the hololith package")
    if int(proxy_settings.get("phase_level_max", 1023)) != 1023:
        raise ValueError("The current SLM hardware adapter requires phase_level_max=1023")
    prepare_slm_commands(reference, hardware_config.slm_shape)
    preprocessor = CameraPreprocessorWorker(CameraPreprocessConfig(**pp_settings), worker_log_path)
    try:
        if tuple(hardware_config.raw_shape) != tuple(preprocessor.raw_shape):
            raise ValueError("Hardware raw_shape differs from training calibration raw_shape")
        proxy = ProxyAdapter.from_args(SimpleNamespace(**proxy_settings))
        prepare_slm_commands(proxy.phase_to_slm(proxy.initial_phase), hardware_config.slm_shape)
        proxy.camera_tensor(np.zeros((preprocessor.output_size, preprocessor.output_size), np.float32))
        # A bad or detached proxy must fail before any laser/device access.
        phase = proxy.initial_phase.detach().clone().requires_grad_(True)
        predicted = proxy.predict(phase)
        loss = proxy.loss(predicted)
        gradient, = torch.autograd.grad(loss, phase)
        if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Proxy preflight prediction / gradient is nonfinite")
        return proxy, preprocessor, hardware_config, reference
    except BaseException:
        preprocessor.close()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="JSON with proxy, preprocessing, hardware and loop settings")
    parser.add_argument("--dry-run", action="store_true", help="Run a small simulated loop; never import device SDKs")
    parser.add_argument("--preflight", action="store_true", help="Validate real calibration and proxy gradient without opening hardware")
    parser.add_argument("--iters", type=int, help="Override number of updates; N+1 measurements are saved")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.dry_run and args.preflight:
        parser.error("Choose --dry-run or --preflight")
    if not args.dry_run and args.config is None:
        parser.error("--config is required for real runs / preflight")
    settings = {} if args.config is None else json.loads(args.config.read_text(encoding="utf-8-sig"))
    loop_settings = dict(settings.get("loop", {}))
    if args.iters is not None:
        loop_settings["iters"] = args.iters
    config = LoopConfig(**loop_settings)
    config.validate()
    output = args.output_dir or Path(settings.get("output_dir", "runs_citl"))
    run_dir = output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "run_config.json", {"settings": settings, "loop": vars(config), "dry_run": args.dry_run})
    if args.dry_run:
        proxy = SyntheticProxy()
        write_json(run_dir / "provenance.json", proxy.metadata)
        run_loop(proxy, synthetic_measurement, run_dir, config)
    else:
        try:
            proxy, preprocessor, hardware_config, reference = prepare_live(
                settings, run_dir / "preprocessing_worker.log")
        except BaseException as error:
            write_json(run_dir / "status.json", {"status": "preflight_failed", "hardware_opened": False,
                       "error": f"{type(error).__name__}: {error}"})
            raise
        try:
            write_json(run_dir / "provenance.json", {"proxy": proxy.metadata, "preprocessing": preprocessor.provenance})
            if args.preflight:
                write_json(run_dir / "status.json", {"status": "preflight_passed", "hardware_opened": False})
            else:
                from citl_hardware import HololithCameraHardware
                with HololithCameraHardware(hardware_config) as hardware:
                    measure = LiveMeasurement(hardware, preprocessor, reference)
                    write_json(run_dir / "hardware.json", hardware.metadata)
                    try:
                        run_loop(proxy, measure, run_dir, config)
                    finally:
                        write_json(run_dir / "hardware.json", hardware.metadata)
        except BaseException as error:
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get("status") not in ("failed", "interrupted"):
                write_json(status_path, {**status, "status": "failed",
                           "error": f"{type(error).__name__}: {error}"})
            raise
        finally:
            try:
                preprocessor.close()
            except BaseException as error:
                status_path = run_dir / "status.json"
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                write_json(status_path, {**status, "status": "failed",
                           "cleanup_error": f"{type(error).__name__}: {error}"})
                raise
    print(f"Saved: {run_dir.resolve()}")
    return run_dir


if __name__ == "__main__":
    main()
