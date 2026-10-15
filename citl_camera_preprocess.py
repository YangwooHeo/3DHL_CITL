"""Reuse training camera calibration for a single-plane camera-in-the-loop run.

The supplied preprocessing script remains the numerical implementation.  Geometry,
LUT and dark/flat masters are prepared once; each iteration estimates only the
current reference translation and power gain and reconstructs its HDR target.
Targets and references must contain native sensor DN, never demosaiced grayscale
or independently normalized intensities.  A ``.raw`` file is headerless little
endian uint16 Bayer RG (12-bit DN stored in 16-bit words).

The training report does not record every original command-line setting.  Missing
settings use the referenced script's defaults and are listed in provenance.  Put
any nondefault settings from the training invocation into ``config.options``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CameraPreprocessConfig:
    training_root: Path | str
    preprocessing_script: Path | str
    lut_path: Path | str | None = None
    z_mm: float = 0.0
    report_path: Path | str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    allow_refit: bool = False


@dataclass(frozen=True)
class ProcessedCameraFrame:
    intensity: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray
    fallback_mask: np.ndarray
    diagnostics: dict[str, Any]


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def _load_backend(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"preprocessing script does not exist: {path}")
    name = "_citl_camera_backend_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import preprocessing script: {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses with postponed annotations need the defining module registered.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    required = ("parse_args", "validate_args", "discover_exposure_folders",
                "build_exposure_index", "load_response_lut", "load_radiometric_calibration",
                "build_reference_registration_model", "process_one_stem", "ExposureIndex")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ValueError(f"incompatible preprocessing script; missing API: {missing}")
    for dependency in ("CV2_IMPORT_ERROR", "SCIPY_IMPORT_ERROR"):
        if getattr(module, dependency, None) is not None:
            raise ImportError("camera preprocessing requires opencv-python and scipy") from getattr(module, dependency)
    return module


def _default_arguments(backend: ModuleType) -> argparse.Namespace:
    """Read backend defaults without changing the application's sys.argv."""
    class DefaultsParser(argparse.ArgumentParser):
        def parse_args(self, args: Any = None, namespace: Any = None) -> argparse.Namespace:
            return super().parse_args([], namespace)

    original = backend.argparse
    backend.argparse = SimpleNamespace(
        ArgumentParser=DefaultsParser,
        RawDescriptionHelpFormatter=argparse.RawDescriptionHelpFormatter,
        Namespace=argparse.Namespace,
    )
    try:
        return backend.parse_args()
    finally:
        backend.argparse = original


def _report_options(report: Mapping[str, Any]) -> dict[str, Any]:
    direct = ("output_size", "output_channels", "output_dtype", "png_dn_scale",
              "saturation_dn", "nd_transmission", "reference_gain_mode",
              "invalid_output", "hdr_weighting")
    values = {key: report[key] for key in direct if key in report}
    renamed = {
        "exposures_us": "exposures", "anchor_exposure_us": "anchor_exposure",
        "geometry_exposure_us": "geometry_exposure", "transform_model": "transform",
        "homography_pattern_names": "homography_patterns",
        "subtract_lut_zero_without_dark": "subtract_lut_zero",
    }
    values.update({dest: report[src] for src, dest in renamed.items() if src in report})
    if "raw_shape" in report:
        values["raw_height"], values["raw_width"] = report["raw_shape"]
    registration = report.get("reference_registration", {})
    for source, dest in {
        "channel": "reference_registration_channel", "method": "registration_method",
        "mask_dilate_radius": "registration_mask_dilate_radius",
        "blur_sigma": "registration_blur_sigma", "highpass_sigma": "registration_highpass_sigma",
        "max_expected_shift_px": "registration_max_shift", "downsample": "registration_downsample",
    }.items():
        if source in registration:
            values[dest] = registration[source]
    if "enabled" in registration:
        values["disable_reference_registration"] = not registration["enabled"]
    # A future report may include a complete invocation.  Existing v3 reports do not.
    values.update(report.get("effective_options", {}))
    return values


def _validate_matrix(matrix: Any) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("training shared_matrix must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("training shared_matrix is singular")
    return matrix


class CameraPreprocessor:
    """Prepared camera reconstruction with one fixed training reference anchor.

    ``reference_anchor_stem`` identifies a historical reference capture, not the
    SLM reference mask filename.  The latter must be supplied separately to the
    hardware loop and must generate the same reference pattern used for training.
    Instances process one bracket at a time and are not thread-safe.
    """

    def __init__(self, config: CameraPreprocessConfig):
        self.config = config
        self.training_root = Path(config.training_root).expanduser().resolve()
        if not self.training_root.is_dir():
            raise FileNotFoundError(f"training dataset root does not exist: {self.training_root}")
        if not math.isfinite(config.z_mm):
            raise ValueError("z_mm must be finite")
        self.backend = _load_backend(Path(config.preprocessing_script).expanduser().resolve())
        backend = self.backend
        self.args = args = _default_arguments(backend)
        self.z_label = backend.z_label(float(config.z_mm))
        report_path = self._resolve_report(config)
        report: dict[str, Any] = {}
        z_report: dict[str, Any] = {}
        if report_path is not None:
            with report_path.open("r", encoding="utf-8-sig") as handle:
                report = json.load(handle)
            if report.get("algorithm_version") != "raw-reference-hdr-v3":
                raise ValueError("training report must use algorithm_version raw-reference-hdr-v3")
            try:
                z_report = report["positions"][self.z_label]
            except KeyError as error:
                raise ValueError(f"training report has no position {self.z_label}") from error
        self.training_report_path = report_path
        reported_options = _report_options(report)
        options = dict(config.options)
        unknown = (set(reported_options) | set(options)) - set(vars(args))
        if unknown:
            raise ValueError(f"unknown preprocessing options: {sorted(unknown)}")
        for key, value in options.items():
            if key in reported_options and value != reported_options[key]:
                raise ValueError(f"option {key!r}={value!r} conflicts with training report "
                                 f"value {reported_options[key]!r}")
        for key, value in {**reported_options, **options}.items():
            setattr(args, key, value)
        args.input_root = self.training_root
        args.z_positions = [self.z_label]
        for key in ("dark_root", "flat_root", "sim_root"):
            value = getattr(args, key)
            if value is not None:
                setattr(args, key, Path(value).expanduser().resolve())

        # Gain scales inferred from old log rows cannot describe a new live frame.
        if args.reference_gain_mode != "raw":
            raise ValueError("live CITL requires training with reference_gain_mode='raw'; "
                             "legacy logged/linearized reference gains cannot be reused")
        if args.disable_reference_registration:
            raise ValueError("live CITL requires reference registration enabled")
        if args.missing_reference != "error" or args.missing_reference_registration != "error":
            raise ValueError("live CITL requires missing-reference policies set to 'error'")
        if args.output_channels not in ("blue", "red"):
            raise ValueError("live CITL supports one native blue or red Bayer channel")
        if args.reference_registration_channel != args.output_channels:
            raise ValueError("reference registration and intensity output channels must match")
        # A quality mask still accompanies every output.  Keep scientific data at
        # full precision; the optimizer explicitly excludes invalid measurements.
        args.output_dtype = "float32"
        args.invalid_output = "nan"
        try:
            backend.validate_args(args)
        except SystemExit as error:
            raise ValueError(f"invalid preprocessing configuration: {error}") from error
        self.exposures_us = tuple(int(e) for e in args.exposures)
        self.output_size = int(args.output_size)
        self.raw_shape = (int(args.raw_height), int(args.raw_width))

        report_anchors = {
            value["anchor_stem"]
            for value in z_report.get("reference_registration_summary", {}).values()
            if value.get("anchor_stem")
        }
        if len(report_anchors) > 1:
            raise ValueError("training report does not use one common reference anchor")
        if report_anchors:
            report_anchor = next(iter(report_anchors))
            if args.reference_anchor_pattern and backend.normalize_stem(args.reference_anchor_pattern) != report_anchor:
                raise ValueError("reference_anchor_pattern conflicts with training report")
            args.reference_anchor_pattern = report_anchor
        self.reference_anchor_stem = backend.normalize_stem(
            args.reference_anchor_pattern or args.homography_patterns[0])
        self.reference_stem = self.reference_anchor_stem

        lut_value = config.lut_path or report.get("lut", {}).get("path")
        if not lut_value:
            raise ValueError("lut_path is required when no camera response LUT path is in the report")
        lut_path = Path(lut_value).expanduser().resolve()
        self.lut = backend.load_response_lut(lut_path)
        args.lut = lut_path
        lut_report = report.get("lut", {})
        for key, actual in (("rows", len(self.lut.dn)), ("min_dn", self.lut.min_dn),
                            ("max_dn", self.lut.max_dn)):
            if key in lut_report and not np.isclose(float(lut_report[key]), float(actual), rtol=0, atol=1e-9):
                raise ValueError(f"provided camera LUT {key} does not match training report")
        historical_lut = Path(lut_report["path"]) if lut_report.get("path") else None
        lut_identity = _file_identity(lut_path)
        if historical_lut and historical_lut.is_file() and historical_lut.resolve() != lut_path:
            if _file_identity(historical_lut)["sha256"] != lut_identity["sha256"]:
                raise ValueError("provided camera LUT differs from the still-accessible training LUT")

        discovered = backend.discover_exposure_folders(self.training_root, args.exposures,
                                                      [self.z_label])
        self.position, folders = discovered[self.z_label]
        self._training_indices = {
            exposure: backend.build_exposure_index(folder, exposure, self.position)
            for exposure, folder in folders.items()
        }
        self._radiometric = {}
        calibration_files: dict[str, Any] = {}
        for exposure in self.exposures_us:
            historical = z_report.get("radiometric_calibration", {}).get(str(exposure), {})
            for kind in ("dark", "flat"):
                root = getattr(args, f"{kind}_root")
                was_applied = bool(historical.get(f"{kind}_applied", False))
                if was_applied and root is None:
                    raise ValueError(f"training applied {kind} correction for Exp{exposure}; "
                                     f"provide options.{kind}_root from that calibration")
                if historical and root is not None and not was_applied:
                    raise ValueError(f"training did not apply {kind} correction for Exp{exposure}")
                paths = backend.calibration_files(root, exposure)
                expected_count = historical.get(f"{kind}_frames")
                if expected_count is not None and len(paths) != int(expected_count):
                    raise ValueError(f"Exp{exposure} {kind} calibration file count changed "
                                     f"({len(paths)} != {expected_count})")
                calibration_files[f"Exp{exposure}_{kind}"] = [_file_identity(p) for p in paths]
            self._radiometric[exposure] = backend.load_radiometric_calibration(
                exposure, args.dark_root, args.flat_root, self.lut, args.png_dn_scale,
                args.raw_width, args.raw_height, args.output_channels, args.subtract_lut_zero)
        common_anchor = self._training_indices[args.geometry_exposure]
        self._registrations = {
            exposure: backend.build_reference_registration_model(index, args, common_anchor)
            for exposure, index in self._training_indices.items()
        }
        anchor_path = self._registrations[args.geometry_exposure].fixed_path
        if anchor_path is None:
            raise ValueError("preprocessing backend did not identify the training reference anchor")
        self.reference_anchor_path = Path(anchor_path)
        self._reference_models = {
            exposure: backend.ReferenceModel(exposure, "raw", {}, 1.0, 0, 0, 1.0, 1.0, 1.0)
            for exposure in self.exposures_us
        }

        if report:
            try:
                matrix = _validate_matrix(z_report["transforms"]["shared_matrix"])
            except KeyError as error:
                raise ValueError("training report lacks the shared geometry matrix") from error
            transform_report = z_report["transforms"]
            geometry_source = "training_report"
        else:
            # Explicit opt-in, done only once before the hardware is opened.
            sim_root = args.sim_root or (self.training_root / "Forward_sim")
            sim_dir = backend.resolve_simulation_dir(sim_root, self.position)
            for pattern in args.homography_patterns:
                backend.reference_gain_from_raw(
                    self._registrations[args.geometry_exposure], common_anchor,
                    backend.normalize_stem(pattern), self.lut, args,
                    self._radiometric[args.geometry_exposure], self._radiometric[args.geometry_exposure])
            matrix, transform_report = backend.calibrate_exposure_transform(
                common_anchor, sim_dir, args.homography_patterns,
                self._registrations[args.geometry_exposure], args)
            matrix = _validate_matrix(matrix)
            geometry_source = "explicit_one_time_refit"
        self._transforms = {exposure: matrix.copy() for exposure in self.exposures_us}
        self._static_scales = {exposure: 1.0 for exposure in self.exposures_us}
        self._reserved_stems = {self.reference_anchor_stem}
        self._reserved_stems.update(backend.normalize_stem(s) for s in args.homography_patterns)
        self._baseline_caches = {
            exposure: (dict(model.shifts), copy.deepcopy(model.reports))
            for exposure, model in self._registrations.items()
        }

        processing_keys = (
            "keep_negative", "low_dn", "low_full_weight_dn", "high_full_weight_dn",
            "minimum_output_weight", "reference_gain_low_dn", "reference_gain_min_pixels",
            "reference_gain_max_nrmse", "max_reference_gain", "registration_saturation_dn",
            "registration_dark_dn", "registration_roi_size", "registration_min_correlation",
        )
        unrecorded = [key for key in processing_keys if key not in reported_options and key not in options]
        self.provenance: dict[str, Any] = {
            "adapter": "single-plane-live-hdr-v1",
            "preprocessing_script": _file_identity(Path(backend.__file__)),
            "training_root": str(self.training_root),
            "training_report": _file_identity(report_path) if report_path else None,
            "camera_response_lut": lut_identity,
            "z_label": self.z_label, "z_mm": float(config.z_mm),
            "exposures_us": list(self.exposures_us),
            "raw_shape": list(self.raw_shape),
            "reference_anchor_stem": self.reference_anchor_stem,
            "reference_anchor": _file_identity(self.reference_anchor_path),
            "reference_pattern_requirement": "live fixed reference phase must match training reference_captures",
            "geometry_source": geometry_source,
            "shared_matrix": matrix.tolist(), "geometry_diagnostics": transform_report,
            "radiometric_calibration_files": calibration_files,
            "effective_options": backend.json_ready(vars(args)),
            "unrecorded_options_using_source_defaults": unrecorded,
            "units": "LUT units / second / ND transmission, at training reference anchor power",
            "settings_note": (
                "The legacy report omits some settings and calibration content hashes. "
                "Use options for the original nondefault settings; current files are SHA256 recorded."
            ),
        }

    def _resolve_report(self, config: CameraPreprocessConfig) -> Path | None:
        if config.report_path is not None:
            path = Path(config.report_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"training preprocessing report does not exist: {path}")
            return path
        candidates = [self.training_root / "Aligned_camera_v3" / "preprocessing_report.json",
                      self.training_root / "preprocessing_report.json"]
        found = [path for path in candidates if path.is_file()]
        if len(found) > 1:
            raise ValueError("multiple preprocessing reports found; set report_path explicitly")
        if found:
            return found[0]
        if config.allow_refit:
            return None
        raise FileNotFoundError(
            "no training preprocessing_report.json found; set report_path to the report used "
            "for proxy training (or explicitly allow_refit for a one-time new calibration)")

    def process(
        self,
        target_paths: Mapping[int, Path | str],
        reference_paths: Mapping[int, Path | str],
        stem: str,
    ) -> ProcessedCameraFrame:
        """Reconstruct one fresh HDR bracket; never reuse a previous live gain.

        Both mappings must contain exactly ``exposures_us``.  A `.npy` input is
        already a native channel DN plane, while `.raw` is the full Bayer mosaic.
        Scientific intensity is unnormalized and retains NaN outside valid_mask.
        """
        if not stem or Path(stem).name != stem or stem in self._reserved_stems:
            raise ValueError("stem must be a nonempty iteration identifier distinct from calibration stems")
        expected = set(self.exposures_us)
        for label, paths in (("target", target_paths), ("reference", reference_paths)):
            if set(paths) != expected:
                raise ValueError(f"{label} exposure keys {sorted(paths)} must equal {sorted(expected)}")
            for exposure, filename in paths.items():
                path = Path(filename)
                if not path.is_file():
                    raise FileNotFoundError(f"missing {label} Exp{exposure} capture: {path}")
                if path.suffix.lower() not in self.backend.SUPPORTED_CAMERA_EXTENSIONS:
                    raise ValueError(f"unsupported {label} capture format: {path.suffix}")
                if path.suffix.lower() == ".raw":
                    expected_bytes = self.raw_shape[0] * self.raw_shape[1] * 2
                    if path.stat().st_size != expected_bytes:
                        raise ValueError(f"wrong RAW size for {path}: expected {expected_bytes} bytes")
        indices = {
            exposure: self.backend.ExposureIndex(
                exposure, self.position, Path(target_paths[exposure]).parent,
                {stem: Path(target_paths[exposure])}, {stem: Path(reference_paths[exposure])},
                {stem: {"status": "OK", "target_status": "OK"}}, [stem])
            for exposure in self.exposures_us
        }
        # Bound memory and guarantee repeated identifiers still fit fresh data.
        for exposure, model in self._registrations.items():
            shifts, reports = self._baseline_caches[exposure]
            model.shifts = dict(shifts)
            model.reports = copy.deepcopy(reports)
            model.recent_reference = None
        quality: dict[str, np.ndarray] = {}
        intensity, report = self.backend.process_one_stem(
            stem, indices, self._reference_models, self._registrations, self._transforms,
            self._radiometric, self._static_scales, self.lut, self.args, quality=quality)
        valid_mask = np.asarray(quality["valid_mask"], dtype=bool)
        if intensity.shape != (self.output_size, self.output_size) or valid_mask.shape != intensity.shape:
            raise ValueError("preprocessing backend returned an unexpected output shape")
        if not np.any(valid_mask) or not np.all(np.isfinite(intensity[valid_mask])):
            raise ValueError("preprocessing produced no valid finite measured intensity")
        if set(report["used_exposures"]) != expected or report.get("dropped_exposures"):
            raise ValueError(f"preprocessing dropped an HDR exposure: {report}")
        report["reference_fit_diagnostics"] = {
            str(exposure): copy.deepcopy(self._registrations[exposure].reports[stem])
            for exposure in self.exposures_us
        }
        report["target_paths"] = {str(e): str(Path(p).resolve()) for e, p in target_paths.items()}
        report["reference_paths"] = {str(e): str(Path(p).resolve()) for e, p in reference_paths.items()}
        report["units"] = self.provenance["units"]
        return ProcessedCameraFrame(
            np.asarray(intensity, dtype=np.float32), valid_mask,
            np.asarray(quality["confidence"], dtype=np.float32),
            np.asarray(quality["fallback_mask"], dtype=bool), report)
