#!/usr/bin/env python3
"""Reconstruct reference-registered, linear HDR camera intensity for each z.

Inputs: Exp300, Exp1000, Exp5000 (and _z+/-Nmm variants), with paired
reference_captures/<stem>.raw, and Forward_sim/z_N calibration intensity NPY.
Headerless RAW is 2168x4096 little-endian uint16 Bayer RG. Native blue is
mosaic[1::2,1::2], a 1084x2048 grid. All reported shifts use this channel grid.

Default calibration uses p008/p016 a045 checkerboards: their first-order dots
are separated and constrain scale/shear over a wider region. p032 a000's
brightest peaks may be different diffraction orders in camera and simulation.
One affine is fitted per z after reference registration; all exposures share
the SAME geometry-exposure reference anchor. Calibration residuals and
conditioning are checked before processing targets.

For each reference, cached-ROI phase correlation initializes native-resolution
translation ECC. Pixelwise inverse-response LUT, optional dark/flat correction,
and exposure normalization precede joint robust gain/translation refinement.
The default RAW reference gain does not use logged CSV intensity means/gains.
It places all exposures on the reference anchor's relative power scale.

Targets: native DN validity -> pixelwise LUT -> dark/flat -> exposure/power
correction -> one composed affine @ translation interpolation -> HDR fusion.
Saturation and invalid interpolation footprints are excluded. Output is float32
1024x1024 in LUT-units/second/ND-transmission, relative to the per-z anchor.
There is no claim of absolute W/m2 calibration. DN=0 is a low-signal measurement,
not missing data. If all HDR weights vanish, use the supported exposure with
the longest effective integration time. Truly missing measurements remain NaN;
--invalid-output zero supplies a finite placeholder, never a measured zero.
Every NPY has quality/<stem>.npz: valid_mask, confidence, fallback_mask, in the
same array order. Training must use valid_mask, even with finite placeholders.
Confidence is a DN-taper heuristic, not a calibrated inverse noise variance.
LUT(0) is preserved unless --subtract-lut-zero is explicitly requested: a
nonzero fitted irradiance at DN=0 is not by itself a measured dark calibration.

Output defaults to Aligned_camera_v3, preserving previous preprocessing results.
--dry-run calibrates and writes diagnostics without target reconstruction.
--validate-input-only checks layout. --registration-trends-only needs no LUT
and reports ECC shifts (full HDR reports the jointly refined shifts).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import numpy as np

CV2_IMPORT_ERROR: Exception | None = None
SCIPY_IMPORT_ERROR: Exception | None = None
try:
    import cv2
except ModuleNotFoundError as error:  # permits --help/--validate-input-only
    CV2_IMPORT_ERROR = error

    class _MissingCV2:
        IMREAD_UNCHANGED = -1
        INTER_NEAREST = 0
        INTER_LINEAR = 1

        def __getattr__(self, name: str) -> Any:
            raise ModuleNotFoundError(
                "OpenCV is required for calibration/registration/HDR processing"
            ) from CV2_IMPORT_ERROR

    cv2 = _MissingCV2()  # type: ignore[assignment]

try:
    from scipy.ndimage import gaussian_filter, maximum_filter
except ModuleNotFoundError as error:  # permits --help/--validate-input-only
    SCIPY_IMPORT_ERROR = error

    def gaussian_filter(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError("SciPy is required for image preprocessing") from SCIPY_IMPORT_ERROR

    def maximum_filter(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError("SciPy is required for core localization") from SCIPY_IMPORT_ERROR


DEFAULT_INPUT_ROOT = Path(
    r"D:\PhDResearch\Measurement\09_08_2026_FNOtraining"
)
DEFAULT_LUT_PATH = Path(
    r"G:\공유 드라이브\3DHL + OptiCAL + TFT + SpaceCAL\3DHL"
    r"\Camera_Response_LUT_0_4091_(4088-4091).xlsx"
)
DEFAULT_EXPOSURES = (300, 1000, 5000)
DEFAULT_HOMOGRAPHY_PATTERNS = (
    # a000 has brighter higher diffraction orders in this simulation. Selecting
    # its four brightest peaks pairs different physical orders with the camera.
    # Wider first-order constellations constrain scale/shear much better than
    # p032's clustered peaks and are less contaminated by overlapping rings.
    "sys_checker_p008_a045_off0_d0256",
    "sys_checker_p008_a045_off1_d0256",
    "sys_checker_p016_a045_off0_d0256",
    "sys_checker_p016_a045_off1_d0256",
)
SUPPORTED_CAMERA_EXTENSIONS = (".raw", ".png", ".tif", ".tiff", ".npy")
SUPPORTED_SIM_EXTENSIONS = (".npy", ".png", ".tif", ".tiff")
CORE_ORDER = ("P1", "P2", "P3", "P4")
RAW_WIDTH = 4096
RAW_HEIGHT = 2168
EXPOSURE_FOLDER_RE = re.compile(
    r"^Exp(?P<exposure>\d+)(?:_z(?P<z>[+-]?\d+(?:\.\d+)?)mm)?$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Unicode-safe image I/O and camera-response LUT
# ---------------------------------------------------------------------------


def imread_unicode(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """Read an image through numpy so Windows Unicode paths work reliably."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def load_reference_array(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """Load a reference from an image file or numpy array file."""
    if path.suffix.lower() == ".npy":
        try:
            return np.load(str(path), allow_pickle=False)
        except (OSError, ValueError):
            return None
    return imread_unicode(path, flags)


def normalize_reference_array(arr: np.ndarray) -> np.ndarray:
    """Normalize HxW, HxWxC, 1xHxW, HxWx1, and CxHxW arrays."""
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] > 4:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[-1] not in (2, 3, 4):
        raise ValueError(f"unsupported 3D reference shape: {arr.shape}")
    if arr.ndim not in (2, 3):
        raise ValueError(f"reference must be 2D/3D after squeeze; got {arr.shape}")
    return arr


def finite_float32(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if np.all(np.isfinite(out)):
        return out
    finite = out[np.isfinite(out)]
    high = float(np.max(finite)) if finite.size else 0.0
    return np.nan_to_num(out, nan=0.0, posinf=high, neginf=0.0)


def intensity_plane(img: np.ndarray, channel: str) -> np.ndarray:
    """Return a float image used for camera/simulation core localization."""
    img = normalize_reference_array(img)
    if img.ndim == 2:
        return finite_float32(img)
    bgr = finite_float32(img[..., :3])
    if channel == "blue":
        return bgr[..., 0]
    if channel == "green":
        return bgr[..., 1]
    if channel == "red":
        return bgr[..., 2]
    if channel == "gray":
        return 0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]
    if channel == "max":
        return np.max(bgr, axis=2)
    raise ValueError(f"unknown channel {channel!r}")


def _column_number(cell_ref: str) -> int:
    match = re.match(r"([A-Za-z]+)", cell_ref)
    if match is None:
        return -1
    result = 0
    for char in match.group(1).upper():
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def _read_first_xlsx_sheet_numeric_rows(path: Path) -> list[list[float]]:
    """Read numeric cells from the first XLSX sheet without openpyxl."""
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(path, "r") as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = workbook.find(f"{{{main_ns}}}sheets/{{{main_ns}}}sheet")
        if sheet is None:
            raise ValueError(f"XLSX contains no worksheet: {path}")
        relationship_id = sheet.attrib[f"{{{rel_ns}}}id"]
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels.findall(f"{{{pkg_rel_ns}}}Relationship"):
            if rel.attrib.get("Id") == relationship_id:
                target = rel.attrib.get("Target")
                break
        if not target:
            raise ValueError(f"could not resolve first worksheet in {path}")
        normalized_target = target.replace("\\", "/").lstrip("/")
        if not normalized_target.startswith("xl/"):
            normalized_target = "xl/" + normalized_target
        worksheet = ET.fromstring(archive.read(normalized_target))
        rows: list[list[float]] = []
        for row_node in worksheet.findall(f".//{{{main_ns}}}row"):
            by_column: dict[int, float] = {}
            for cell in row_node.findall(f"{{{main_ns}}}c"):
                if cell.attrib.get("t") in {"s", "str", "inlineStr", "b", "e"}:
                    continue
                value_node = cell.find(f"{{{main_ns}}}v")
                if value_node is None or value_node.text is None:
                    continue
                try:
                    value = float(value_node.text)
                except ValueError:
                    continue
                column = _column_number(cell.attrib.get("r", ""))
                if column >= 0:
                    by_column[column] = value
            if by_column:
                width = max(by_column) + 1
                rows.append([by_column.get(index, float("nan")) for index in range(width)])
        return rows


@dataclass(frozen=True)
class ResponseLUT:
    dn: np.ndarray
    linear: np.ndarray
    source: str
    integer_table: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dn = np.asarray(self.dn, dtype=np.float64)
        linear = np.asarray(self.linear, dtype=np.float64)
        if (dn.ndim != 1 or dn.shape != linear.shape or len(dn) < 2
                or not np.all(np.isfinite(dn)) or not np.all(np.isfinite(linear))
                or np.any(np.diff(dn) <= 0) or np.any(np.diff(linear) < 0)):
            raise ValueError("LUT requires finite, increasing DN and monotone response")
        table = np.interp(np.arange(65536), dn, linear).astype(np.float32)
        object.__setattr__(self, "integer_table", table)

    @property
    def min_dn(self) -> float:
        return float(self.dn[0])

    @property
    def max_dn(self) -> float:
        return float(self.dn[-1])

    def apply(self, values: np.ndarray) -> np.ndarray:
        if np.asarray(values).dtype == np.uint16:
            return self.integer_table[values]
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        mapped = np.interp(flat, self.dn, self.linear).astype(np.float32)
        return mapped.reshape(np.shape(values))

    def scalar(self, value: float) -> float:
        return float(np.interp(float(value), self.dn, self.linear))


def load_response_lut(path: Path) -> ResponseLUT:
    if not path.is_file():
        raise FileNotFoundError(f"camera-response LUT does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        rows = _read_first_xlsx_sheet_numeric_rows(path)
        pairs = [(row[0], row[1]) for row in rows if len(row) >= 2 and np.isfinite(row[0]) and np.isfinite(row[1])]
        data = np.asarray(pairs, dtype=np.float64)
    elif suffix == ".csv":
        data = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    elif suffix == ".npy":
        data = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    else:
        raise ValueError("LUT must be .xlsx, .csv, or .npy")
    if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 2:
        raise ValueError(f"LUT must contain at least two numeric columns/rows: {path}")
    dn = data[:, 0]
    linear = data[:, 1]
    order = np.argsort(dn)
    dn = dn[order]
    linear = linear[order]
    if np.any(np.diff(dn) <= 0):
        raise ValueError("LUT DN column must be strictly increasing and unique")
    if np.any(np.diff(linear) < 0):
        raise ValueError("inverse-response LUT must be monotonic non-decreasing")
    return ResponseLUT(dn.astype(np.float64), linear.astype(np.float64), str(path))


def map_raw_mosaic(
    path: Path,
    width: int = RAW_WIDTH,
    height: int = RAW_HEIGHT,
) -> np.memmap:
    """Memory-map a validated headerless little-endian uint16 Bayer frame."""
    expected_bytes = int(width) * int(height) * np.dtype("<u2").itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"RAW size mismatch for {path}: {actual_bytes} bytes, expected "
            f"{expected_bytes} ({height}x{width} uint16)"
        )
    return np.memmap(
        str(path),
        dtype="<u2",
        mode="r",
        shape=(int(height), int(width)),
    )


def read_raw_bayer_plane(
    path: Path,
    channel: str = "blue",
    width: int = RAW_WIDTH,
    height: int = RAW_HEIGHT,
) -> np.ndarray:
    """Read only the requested Bayer-derived plane, avoiding unused RGB copies.

    ``read_Rawfile.m`` labels raw[odd, odd] as R and raw[even, even] as B.
    Python's zero-based equivalents are used here.  The blue-only default reads
    just ``mosaic[1::2, 1::2]`` into the 1084x2048 working plane.
    """
    mosaic = map_raw_mosaic(path, width, height)
    normalized_channel = str(channel).lower()
    if normalized_channel == "blue":
        return np.array(mosaic[1::2, 1::2], dtype=np.uint16)
    if normalized_channel == "red":
        return np.array(mosaic[0::2, 0::2], dtype=np.uint16)
    green = 0.5 * (
        mosaic[0::2, 1::2].astype(np.float32)
        + mosaic[1::2, 0::2].astype(np.float32)
    )
    if normalized_channel == "green":
        return green
    blue = mosaic[1::2, 1::2].astype(np.float32)
    red = mosaic[0::2, 0::2].astype(np.float32)
    if normalized_channel == "gray":
        return (0.114 * blue + 0.587 * green + 0.299 * red).astype(np.float32)
    if normalized_channel == "max":
        return np.maximum(np.maximum(blue, green), red).astype(np.float32)
    raise ValueError(f"unknown Bayer output channel {channel!r}")


def read_raw_bayer_bgr(
    path: Path,
    width: int = RAW_WIDTH,
    height: int = RAW_HEIGHT,
) -> np.ndarray:
    """Read all Bayer-derived channels in internal BGR order when requested."""
    mosaic = map_raw_mosaic(path, width, height)
    red = mosaic[0::2, 0::2].astype(np.float32)
    green = 0.5 * (
        mosaic[0::2, 1::2].astype(np.float32)
        + mosaic[1::2, 0::2].astype(np.float32)
    )
    blue = mosaic[1::2, 1::2].astype(np.float32)
    return np.stack([blue, green, red], axis=2)


def load_camera_dn_channels(
    path: Path,
    png_dn_scale: float,
    raw_width: int = RAW_WIDTH,
    raw_height: int = RAW_HEIGHT,
) -> np.ndarray:
    """Load camera data as HxWx3 float32 BGR sensor-DN values."""
    if path.suffix.lower() == ".raw":
        return read_raw_bayer_bgr(path, raw_width, raw_height)
    arr = load_reference_array(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"could not read camera frame: {path}")
    # Keep nonfinite sensor data invalid until the HDR validity check.
    arr = np.asarray(normalize_reference_array(arr), dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] < 3:
        raise ValueError(f"camera frame must provide three channels: {path} has {arr.shape}")
    scale = 1.0 if path.suffix.lower() == ".npy" else float(png_dn_scale)
    if scale <= 0:
        raise ValueError("PNG DN scale must be positive")
    return arr[..., :3].astype(np.float32, copy=False) / scale


def load_camera_dn(
    path: Path,
    png_dn_scale: float,
    camera_channel: str,
    raw_width: int = RAW_WIDTH,
    raw_height: int = RAW_HEIGHT,
) -> np.ndarray:
    if path.suffix.lower() == ".raw":
        return read_raw_bayer_plane(path, camera_channel, raw_width, raw_height)
    channels = load_camera_dn_channels(path, png_dn_scale, raw_width, raw_height)
    if camera_channel in ("blue", "green", "red"):
        return channels[..., {"blue":0,"green":1,"red":2}[camera_channel]]
    if camera_channel == "gray":
        return .114*channels[...,0]+.587*channels[...,1]+.299*channels[...,2]
    if camera_channel == "max":
        return np.max(channels,axis=2)
    raise ValueError(f"unknown camera channel {camera_channel}")


# ---------------------------------------------------------------------------
# 4-core detection adapted from homography_calibrate_4dots.py
# ---------------------------------------------------------------------------


def parabolic_subpixel(smoothed: np.ndarray, py: int, px: int) -> tuple[float, float]:
    """3-point parabolic peak refinement around integer peak (py, px)."""
    dy = 0.0
    dx = 0.0
    if 1 <= py < smoothed.shape[0] - 1:
        before, peak, after = smoothed[py - 1, px], smoothed[py, px], smoothed[py + 1, px]
        denom = before - 2.0 * peak + after
        if abs(denom) > 1e-12:
            dy = 0.5 * (before - after) / denom
    if 1 <= px < smoothed.shape[1] - 1:
        before, peak, after = smoothed[py, px - 1], smoothed[py, px], smoothed[py, px + 1]
        denom = before - 2.0 * peak + after
        if abs(denom) > 1e-12:
            dx = 0.5 * (before - after) / denom
    return float(dy), float(dx)


def blurred_peak_subpixel(
    plane: np.ndarray,
    cy0: float,
    cx0: float,
    *,
    win: int,
    sigma: float,
) -> tuple[float, float]:
    h_img, w_img = plane.shape
    y0 = max(0, int(round(cy0 - win)))
    y1 = min(h_img, int(round(cy0 + win)))
    x0 = max(0, int(round(cx0 - win)))
    x1 = min(w_img, int(round(cx0 + win)))
    patch = plane[y0:y1, x0:x1].astype(np.float32)
    smoothed = gaussian_filter(patch, sigma)
    py, px = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    dy, dx = parabolic_subpixel(smoothed, int(py), int(px))
    return float(y0 + py + dy), float(x0 + px + dx)


def locate_four_cores_quadrants(
    plane: np.ndarray,
    *,
    seed_sigma: float,
    win: int,
    sigma: float,
) -> dict[str, tuple[float, float]]:
    h_img, w_img = plane.shape
    rough = gaussian_filter(plane.astype(np.float32), seed_sigma)
    mid_y, mid_x = h_img // 2, w_img // 2
    quadrants = {
        "TL": (0, mid_y, 0, mid_x),
        "TR": (0, mid_y, mid_x, w_img),
        "BL": (mid_y, h_img, 0, mid_x),
        "BR": (mid_y, h_img, mid_x, w_img),
    }
    cores: dict[str, tuple[float, float]] = {}
    for name, (y0, y1, x0, x1) in quadrants.items():
        yy, xx = np.unravel_index(np.argmax(rough[y0:y1, x0:x1]), (y1 - y0, x1 - x0))
        cy0, cx0 = y0 + yy, x0 + xx
        cores[name] = blurred_peak_subpixel(plane, cy0, cx0, win=win, sigma=sigma)
    return cores


def central_roi_bounds(
    image_shape: tuple[int, int], fraction: float
) -> tuple[int, int, int, int]:
    """Return a centered rectangular ROI as ``y0, y1, x0, x1``."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("core ROI fraction must be in (0, 1]")
    h_img, w_img = image_shape
    roi_h = max(4, min(h_img, int(round(h_img * fraction))))
    roi_w = max(4, min(w_img, int(round(w_img * fraction))))
    y0 = (h_img - roi_h) // 2
    x0 = (w_img - roi_w) // 2
    return y0, y0 + roi_h, x0, x0 + roi_w


def calibration_pattern_layout(pattern_name: str) -> str:
    """Map the checkerboard orientation token to its expected four-core topology."""
    stem = normalize_stem(pattern_name).lower()
    match = re.search(r"(?:^|_)a(?P<angle>\d{3})(?:_|$)", stem)
    if match is None:
        raise ValueError(
            f"pattern-aware localization needs an _aNNN_ token: {pattern_name!r}"
        )
    angle = int(match.group("angle")) % 90
    distance_to_axis = min(angle, 90 - angle)
    if distance_to_axis <= 15:
        return "quadrants"
    if abs(angle - 45) <= 15:
        return "cardinal-sectors"
    raise ValueError(
        f"unsupported checkerboard orientation a{int(match.group('angle')):03d} in "
        f"{pattern_name!r}; expected an axis-aligned or 45-degree pattern"
    )


def core_sector_masks(
    dy: np.ndarray, dx: np.ndarray, layout: str, guard: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return clockwise non-overlapping sector masks around a pattern center."""
    if layout == "quadrants":
        return (
            (dy < 0) & (dx < 0),
            (dy < 0) & (dx >= 0),
            (dy >= 0) & (dx >= 0),
            (dy >= 0) & (dx < 0),
        )
    if layout == "cardinal-sectors":
        return (
            (dy < 0) & ((-dy) >= np.abs(dx) + guard),
            (dx > 0) & (dx >= np.abs(dy) + guard),
            (dy > 0) & (dy >= np.abs(dx) + guard),
            (dx < 0) & ((-dx) >= np.abs(dy) + guard),
        )
    raise ValueError(f"unknown core layout {layout!r}")


def refine_peak_in_core_sector(
    plane: np.ndarray,
    seed_y: float,
    seed_x: float,
    *,
    center_y: float,
    center_x: float,
    layout: str,
    sector_index: int,
    sector_guard: float,
    center_exclusion_radius: float,
    roi_bounds: tuple[int, int, int, int],
    win: int,
    sigma: float,
) -> tuple[float, float]:
    """Subpixel-refine a seed without allowing it to jump to another sector."""
    h_img, w_img = plane.shape
    roi_y0, roi_y1, roi_x0, roi_x1 = roi_bounds
    y0 = max(roi_y0, 0, int(round(seed_y - win)))
    y1 = min(roi_y1, h_img, int(round(seed_y + win)))
    x0 = max(roi_x0, 0, int(round(seed_x - win)))
    x1 = min(roi_x1, w_img, int(round(seed_x + win)))
    patch = plane[y0:y1, x0:x1].astype(np.float32)
    if patch.size == 0:
        raise RuntimeError("empty sector refinement patch")
    smoothed = gaussian_filter(patch, sigma)
    yy, xx = np.indices(smoothed.shape, dtype=np.float32)
    dy = yy + float(y0) - center_y
    dx = xx + float(x0) - center_x
    allowed = core_sector_masks(dy, dx, layout, sector_guard)[sector_index]
    allowed &= np.hypot(dy, dx) >= center_exclusion_radius
    if not np.any(allowed):
        raise RuntimeError("empty allowed region in sector refinement patch")
    masked = np.where(allowed, smoothed, -np.inf)
    peak_y, peak_x = np.unravel_index(np.argmax(masked), masked.shape)
    sub_y, sub_x = parabolic_subpixel(smoothed, int(peak_y), int(peak_x))
    return float(y0 + peak_y + sub_y), float(x0 + peak_x + sub_x)


def locate_four_cores_pattern_aware(
    plane: np.ndarray,
    pattern_name: str,
    *,
    center_sigma: float,
    seed_sigma: float,
    win: int,
    sigma: float,
    roi_fraction: float,
    candidate_min_distance: int,
    sector_guard: float,
    center_exclusion_radius: float,
    minimum_separation: float,
) -> tuple[dict[str, tuple[float, float]], str]:
    """Locate checkerboard cores using a central ROI and orientation-aware sectors.

    Separated local maxima from the compact ``seed_sigma`` map estimate the
    center of the four-core constellation.  A broad Gaussian is only a fallback
    if fewer than four compact maxima exist.  Final subpixel positions use
    ``sigma`` in a local patch.
    Axis-aligned a000 masks are split into quadrants around the estimated
    pattern center; a045 masks are split into top/right/bottom/left sectors.
    """
    work = finite_float32(plane)
    y0, y1, x0, x1 = central_roi_bounds(work.shape, roi_fraction)
    compact = gaussian_filter(work, seed_sigma)
    roi = compact[y0:y1, x0:x1]
    local_filter_size = max(3, 2 * int(candidate_min_distance) + 1)
    local_max = roi == maximum_filter(roi, size=local_filter_size, mode="nearest")
    candidate_yx = np.argwhere(local_max)
    if len(candidate_yx) >= 4:
        candidate_values = roi[candidate_yx[:, 0], candidate_yx[:, 1]]
        strongest = np.argsort(candidate_values)[::-1][:4]
        center_y = float(y0 + np.mean(candidate_yx[strongest, 0]))
        center_x = float(x0 + np.mean(candidate_yx[strongest, 1]))
        center_method = "four-strongest-compact-peaks"
    else:
        center_map = gaussian_filter(work, center_sigma)
        center_patch = center_map[y0:y1, x0:x1]
        center_local_y, center_local_x = np.unravel_index(
            np.argmax(center_patch), center_patch.shape
        )
        center_y = float(y0 + center_local_y)
        center_x = float(x0 + center_local_x)
        center_method = "broad-envelope-fallback"

    yy, xx = np.indices(roi.shape, dtype=np.float32)
    dy = yy + float(y0) - center_y
    dx = xx + float(x0) - center_x
    layout = calibration_pattern_layout(pattern_name)
    masks = tuple(
        mask & (np.hypot(dy, dx) >= center_exclusion_radius)
        for mask in core_sector_masks(dy, dx, layout, sector_guard)
    )

    seeds: list[tuple[float, float]] = []
    for sector_index, mask in enumerate(masks, start=1):
        if not np.any(mask):
            raise RuntimeError(
                f"empty core sector {sector_index} for {pattern_name!r}"
            )
        sector_values = np.where(mask, roi, -np.inf)
        local_y, local_x = np.unravel_index(
            np.argmax(sector_values), sector_values.shape
        )
        seeds.append((float(y0 + local_y), float(x0 + local_x)))

    refined = [
        refine_peak_in_core_sector(
            work,
            seed_y,
            seed_x,
            center_y=center_y,
            center_x=center_x,
            layout=layout,
            sector_index=sector_index,
            sector_guard=sector_guard,
            center_exclusion_radius=center_exclusion_radius,
            roi_bounds=(y0, y1, x0, x1),
            win=win,
            sigma=sigma,
        )
        for sector_index, (seed_y, seed_x) in enumerate(seeds)
    ]
    cores = order_points_clockwise(refined)
    min_distance = minimum_pair_distance(cores)
    if min_distance < minimum_separation:
        raise RuntimeError(
            f"pattern-aware cores collapsed for {pattern_name!r}: minimum "
            f"separation {min_distance:.3f}px < {minimum_separation:.3f}px"
        )
    for core_y, core_x in cores.values():
        if not (y0 <= core_y < y1 and x0 <= core_x < x1):
            raise RuntimeError(
                f"refined core left the central ROI for {pattern_name!r}: "
                f"(y={core_y:.3f}, x={core_x:.3f})"
            )
    return cores, f"pattern-aware/{layout}/{center_method}"


def order_points_clockwise(points_yx: list[tuple[float, float]]) -> dict[str, tuple[float, float]]:
    pts = np.asarray(points_yx, dtype=np.float64)
    center_yx = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 0] - center_yx[0], pts[:, 1] - center_yx[1])
    order = list(np.argsort(angles))
    ordered = pts[order]
    y_span = float(np.ptp(ordered[:, 0]))
    top_y = float(np.min(ordered[:, 0]))
    top_candidates = np.where(ordered[:, 0] <= top_y + 0.20 * max(y_span, 1.0))[0]
    if len(top_candidates) == 1:
        start = int(top_candidates[0])
    else:
        start = int(top_candidates[np.argmin(ordered[top_candidates, 1])])
    ordered = np.roll(ordered, -start, axis=0)
    return {name: (float(y), float(x)) for name, (y, x) in zip(CORE_ORDER, ordered)}


def cyclic_points(points_yx: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_yx, dtype=np.float64)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 0] - center[0], pts[:, 1] - center[1])
    return pts[np.argsort(angles)]


def convex_quad(points_yx: np.ndarray) -> bool:
    pts_xy = np.asarray(points_yx, dtype=np.float64)[:, ::-1]
    edges = np.roll(pts_xy, -1, axis=0) - pts_xy
    next_edges = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    return bool(np.all(crosses > 1e-6) or np.all(crosses < -1e-6))


def select_geometric_core_quadruple(
    candidate_yx: np.ndarray,
    strengths: np.ndarray,
    *,
    image_shape: tuple[int, int],
    min_distance: float,
    max_pattern_fraction: float,
    center_tolerance: float,
) -> list[tuple[float, float]]:
    h_img, w_img = image_shape
    frame_center = np.array([(h_img - 1) / 2.0, (w_img - 1) / 2.0])
    half_frame = np.array([max(h_img / 2.0, 1.0), max(w_img / 2.0, 1.0)])
    min_dimension = float(min(h_img, w_img))
    strengths = np.asarray(strengths, dtype=np.float64)
    strength_low = float(np.min(strengths))
    strength_span = max(float(np.max(strengths)) - strength_low, 1e-12)
    normalized_strength = (strengths - strength_low) / strength_span
    best_score = float("inf")
    best_points: np.ndarray | None = None
    rejected_scale = 0
    rejected_geometry = 0
    for indices in itertools.combinations(range(len(candidate_yx)), 4):
        pts = cyclic_points(candidate_yx[list(indices)])
        if not convex_quad(pts):
            rejected_geometry += 1
            continue
        edges = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        diag_a_vec = pts[2] - pts[0]
        diag_b_vec = pts[3] - pts[1]
        diag_a = float(np.linalg.norm(diag_a_vec))
        diag_b = float(np.linalg.norm(diag_b_vec))
        mean_diag = 0.5 * (diag_a + diag_b)
        min_edge = float(np.min(edges))
        if (
            min_edge < 0.8 * min_distance
            or mean_diag < 1.15 * min_distance
            or mean_diag > max_pattern_fraction * min_dimension
        ):
            rejected_scale += 1
            continue
        center = np.mean(pts, axis=0)
        center_distance = float(np.linalg.norm((center - frame_center) / half_frame))
        if center_distance > center_tolerance:
            rejected_scale += 1
            continue
        midpoint_a = 0.5 * (pts[0] + pts[2])
        midpoint_b = 0.5 * (pts[1] + pts[3])
        midpoint_error = float(np.linalg.norm(midpoint_a - midpoint_b) / mean_diag)
        edge_cv = float(np.std(edges) / max(np.mean(edges), 1e-12))
        diagonal_balance = abs(math.log(max(diag_a, 1e-12) / max(diag_b, 1e-12)))
        diagonal_orthogonality = abs(float(np.dot(diag_a_vec, diag_b_vec))) / max(
            diag_a * diag_b, 1e-12
        )
        opposite_balance = 0.5 * (
            abs(math.log(max(edges[0], 1e-12) / max(edges[2], 1e-12)))
            + abs(math.log(max(edges[1], 1e-12) / max(edges[3], 1e-12)))
        )
        combo_strength = float(np.mean(normalized_strength[list(indices)]))
        score = (
            4.0 * midpoint_error
            + 1.4 * edge_cv
            + 0.9 * diagonal_balance
            + 0.8 * diagonal_orthogonality
            + 1.0 * opposite_balance
            + 0.8 * center_distance
            - 0.35 * combo_strength
        )
        if score < best_score:
            best_score = score
            best_points = pts
    if best_points is None:
        raise RuntimeError(
            "could not find a plausible four-core quadrilateral among "
            f"{len(candidate_yx)} candidates ({rejected_scale} scale/center rejects, "
            f"{rejected_geometry} non-convex rejects)"
        )
    return [(float(y), float(x)) for y, x in best_points]


def locate_four_cores_peaks(
    plane: np.ndarray,
    *,
    seed_sigma: float,
    win: int,
    sigma: float,
    min_distance: int | None = None,
    margin_fraction: float = 0.06,
    candidate_count: int = 32,
    background_sigma: float | None = None,
    max_pattern_fraction: float = 0.65,
    center_tolerance: float = 0.85,
) -> dict[str, tuple[float, float]]:
    h_img, w_img = plane.shape
    work = finite_float32(plane)
    clip_high = float(np.percentile(work, 99.95))
    if np.isfinite(clip_high) and clip_high > 0:
        work = np.minimum(work, clip_high)
    if background_sigma is None:
        background_sigma = max(3.5 * seed_sigma, seed_sigma + 8.0)
    if background_sigma <= seed_sigma:
        raise ValueError("peak background sigma must be larger than seed sigma")
    compact = gaussian_filter(work, seed_sigma)
    background = gaussian_filter(work, background_sigma)
    rough = compact - background
    margin_y = int(round(h_img * margin_fraction))
    margin_x = int(round(w_img * margin_fraction))
    if margin_y > 0:
        rough[:margin_y, :] = np.min(rough)
        rough[h_img - margin_y :, :] = np.min(rough)
    if margin_x > 0:
        rough[:, :margin_x] = np.min(rough)
        rough[:, w_img - margin_x :] = np.min(rough)
    if min_distance is None:
        min_distance = max(20, int(round(min(h_img, w_img) * 0.06)))
    filt_size = max(3, 2 * int(min_distance) + 1)
    local_max = rough == maximum_filter(rough, size=filt_size, mode="nearest")
    local_max &= rough > 0
    candidate_yx = np.argwhere(local_max)
    if candidate_yx.size == 0:
        raise RuntimeError("could not find any local maxima for core detection")
    values = rough[candidate_yx[:, 0], candidate_yx[:, 1]]
    if len(candidate_yx) < 4:
        raise RuntimeError(f"found only {len(candidate_yx)} compact-core candidates")
    order = np.argsort(values)[::-1][: max(4, int(candidate_count))]
    candidate_yx = candidate_yx[order].astype(np.float64)
    values = values[order].astype(np.float64)
    selected = select_geometric_core_quadruple(
        candidate_yx,
        values,
        image_shape=plane.shape,
        min_distance=float(min_distance),
        max_pattern_fraction=max_pattern_fraction,
        center_tolerance=center_tolerance,
    )
    refined = [
        blurred_peak_subpixel(plane, cy0, cx0, win=win, sigma=sigma)
        for cy0, cx0 in selected
    ]
    return order_points_clockwise(refined)


def classify_core_layout(cores: dict[str, tuple[float, float]]) -> str:
    pts = np.asarray([cores[name] for name in CORE_ORDER], dtype=np.float64)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 0] - center[0], pts[:, 1] - center[1])
    axis_error = float(np.mean(np.abs(np.sin(2.0 * angles))))
    diagonal_error = float(np.mean(np.abs(np.cos(2.0 * angles))))
    return "diamond" if axis_error < diagonal_error else "square"


def minimum_pair_distance(cores: dict[str, tuple[float, float]]) -> float:
    pts = np.asarray([cores[name] for name in cores], dtype=np.float64)
    distances = [
        float(np.linalg.norm(pts[i] - pts[j]))
        for i in range(len(pts))
        for j in range(i + 1, len(pts))
    ]
    return min(distances) if distances else 0.0


def geometry_summary(cores: dict[str, tuple[float, float]]) -> tuple[float, float, float]:
    pts = np.asarray(list(cores.values()), dtype=np.float64)
    dy = float(np.ptp(pts[:, 0]))
    dx = float(np.ptp(pts[:, 1]))
    ratio = np.inf if abs(dx) < 1e-12 else dy / dx
    return dy, dx, float(ratio)


def locate_four_cores(
    plane: np.ndarray,
    *,
    seed_sigma: float,
    win: int,
    sigma: float,
    locator: str,
    peak_min_distance: int | None,
    peak_margin_fraction: float,
    peak_candidates: int,
    peak_background_sigma: float | None,
    max_pattern_fraction: float,
    pattern_center_tolerance: float,
) -> tuple[dict[str, tuple[float, float]], str]:
    if locator in ("auto", "peaks"):
        try:
            geometric_cores = locate_four_cores_peaks(
                plane,
                seed_sigma=seed_sigma,
                win=win,
                sigma=sigma,
                min_distance=peak_min_distance,
                margin_fraction=peak_margin_fraction,
                candidate_count=peak_candidates,
                background_sigma=peak_background_sigma,
                max_pattern_fraction=max_pattern_fraction,
                center_tolerance=pattern_center_tolerance,
            )
            return geometric_cores, f"geometric-peaks/{classify_core_layout(geometric_cores)}"
        except RuntimeError:
            if locator == "peaks":
                raise
    quadrant_cores = locate_four_cores_quadrants(
        plane, seed_sigma=seed_sigma, win=win, sigma=sigma
    )
    if locator == "quadrants":
        ordered = order_points_clockwise(list(quadrant_cores.values()))
        return ordered, f"quadrants/{classify_core_layout(ordered)}"
    dy, dx, ratio = geometry_summary(quadrant_cores)
    min_dist = minimum_pair_distance(quadrant_cores)
    too_close = min_dist < max(5.0, min(plane.shape) * 0.03)
    bad_geometry = (
        not np.isfinite(ratio)
        or abs(dx) < 1e-6
        or abs(dy) < 1e-6
        or ratio < 0.5
        or ratio > 2.0
    )
    if not too_close and not bad_geometry:
        ordered = order_points_clockwise(list(quadrant_cores.values()))
        return ordered, f"quadrants-fallback/{classify_core_layout(ordered)}"
    raise RuntimeError("geometric peak selection and quadrant fallback both failed")


def cores_to_xy(cores: dict[str, tuple[float, float]]) -> np.ndarray:
    return np.array([[cores[name][1], cores[name][0]] for name in CORE_ORDER], dtype=np.float32)


def project_points(matrix: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points_xy, dtype=np.float32).reshape(1, -1, 2),
        np.asarray(matrix, dtype=np.float64),
    )[0].astype(np.float64)


def normalize_homography(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if abs(float(matrix[2, 2])) < 1e-12:
        norm = float(np.linalg.norm(matrix))
        if norm < 1e-12:
            raise ValueError("degenerate homography")
        return matrix / norm
    return matrix / matrix[2, 2]


def fit_affine_lstsq(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    inlier_mask: np.ndarray | None = None,
) -> np.ndarray:
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    if inlier_mask is not None:
        keep = np.asarray(inlier_mask).reshape(-1).astype(bool)
        src = src[keep]
        dst = dst[keep]
    if len(src) < 3:
        raise RuntimeError("at least three inlier points are required for affine fitting")
    design = np.column_stack([src, np.ones(len(src), dtype=np.float64)])
    coefficients, _, rank, _ = np.linalg.lstsq(design, dst, rcond=None)
    if rank < 3:
        raise RuntimeError("affine calibration points are rank deficient")
    affine = np.eye(3, dtype=np.float64)
    affine[:2, :] = coefficients.T
    return affine


def fit_pooled_affine(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    ransac_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    affine_2x3, mask = cv2.estimateAffine2D(
        np.asarray(src_xy, dtype=np.float64),
        np.asarray(dst_xy, dtype=np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold),
        maxIters=10000,
        confidence=0.999,
        refineIters=20,
    )
    if affine_2x3 is None:
        raise RuntimeError("global affine RANSAC failed")
    inliers = np.ones(len(src_xy), dtype=bool) if mask is None else mask.reshape(-1).astype(bool)
    if np.count_nonzero(inliers) < 3:
        raise RuntimeError("global affine has fewer than three RANSAC inliers")
    return fit_affine_lstsq(src_xy, dst_xy, inliers), inliers


def fit_pooled_homography(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    ransac_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    initial, mask = cv2.findHomography(
        np.asarray(src_xy, dtype=np.float64),
        np.asarray(dst_xy, dtype=np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold),
        maxIters=10000,
        confidence=0.999,
    )
    if initial is None:
        raise RuntimeError("global homography RANSAC failed")
    inliers = np.ones(len(src_xy), dtype=bool) if mask is None else mask.reshape(-1).astype(bool)
    if np.count_nonzero(inliers) < 4:
        raise RuntimeError("global homography has fewer than four RANSAC inliers")
    refit, _ = cv2.findHomography(
        np.asarray(src_xy, dtype=np.float64)[inliers],
        np.asarray(dst_xy, dtype=np.float64)[inliers],
        method=0,
    )
    return normalize_homography(initial if refit is None else refit), inliers


def residual_statistics(errors: np.ndarray) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    return {
        "rmse_px": float(np.sqrt(np.mean(values * values))),
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "max_px": float(np.max(values)),
    }


# ---------------------------------------------------------------------------
# Dataset, reference gain, dark/flat, and exposure-specific calibration
# ---------------------------------------------------------------------------


def normalize_stem(name: str) -> str:
    return Path(str(name).strip()).stem


def find_stem_file(folder: Path, stem: str, extensions: Iterable[str]) -> Path:
    for extension in extensions:
        candidate = folder / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    by_casefold = {
        path.stem.casefold(): path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in set(extensions)
    }
    result = by_casefold.get(stem.casefold())
    if result is None:
        raise FileNotFoundError(f"could not find stem {stem!r} in {folder}")
    return result


@dataclass(frozen=True)
class ZPosition:
    value_mm: float
    label: str


def z_label(value_mm: float) -> str:
    value = float(value_mm)
    if abs(value) < 1e-12:
        return "z_0"
    magnitude = f"{abs(value):.1f}" if float(value).is_integer() else f"{abs(value):g}"
    return f"z_{'+' if value > 0 else '-'}{magnitude}"


def parse_z_selector(value: str) -> str:
    text = str(value).strip().lower().replace("mm", "")
    if text.startswith("z_"):
        text = text[2:]
    elif text.startswith("z"):
        text = text[1:]
    if text in ("", "+0", "-0"):
        text = "0"
    try:
        return z_label(float(text))
    except ValueError as error:
        raise ValueError(f"invalid z position {value!r}") from error


def discover_exposure_folders(
    input_root: Path,
    exposures: Iterable[int],
    selected_z: list[str] | None = None,
) -> dict[str, tuple[ZPosition, dict[int, Path]]]:
    """Discover ``Exp<us>[_z<signed-mm>mm]`` folders grouped by z."""
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {input_root}")
    wanted_exposures = {int(value) for value in exposures}
    wanted_z = None if not selected_z else {parse_z_selector(value) for value in selected_z}
    discovered: dict[str, tuple[ZPosition, dict[int, Path]]] = {}
    for folder in sorted(path for path in input_root.iterdir() if path.is_dir()):
        match = EXPOSURE_FOLDER_RE.fullmatch(folder.name)
        if match is None:
            continue
        exposure_us = int(match.group("exposure"))
        if exposure_us not in wanted_exposures:
            continue
        value_mm = 0.0 if match.group("z") is None else float(match.group("z"))
        label = z_label(value_mm)
        if wanted_z is not None and label not in wanted_z:
            continue
        if label not in discovered:
            discovered[label] = (ZPosition(value_mm, label), {})
        folders = discovered[label][1]
        if exposure_us in folders:
            raise RuntimeError(
                f"duplicate exposure {exposure_us} folder for {label}: "
                f"{folders[exposure_us]} and {folder}"
            )
        folders[exposure_us] = folder
    if not discovered:
        raise RuntimeError(f"no matching exposure/z folders found in {input_root}")
    for label, (_, folders) in discovered.items():
        missing = sorted(wanted_exposures - set(folders))
        if missing:
            raise RuntimeError(f"{label} is missing exposure folders: {missing}")
    return dict(sorted(discovered.items(), key=lambda item: item[1][0].value_mm))


@dataclass
class ExposureIndex:
    exposure_us: int
    z_position: ZPosition
    folder: Path
    paths: dict[str, Path]
    reference_paths: dict[str, Path]
    capture_rows: dict[str, dict[str, str]]
    order: list[str]


def build_exposure_index(
    folder: Path,
    exposure_us: int,
    position: ZPosition | None = None,
) -> ExposureIndex:
    if not folder.is_dir():
        raise FileNotFoundError(f"exposure folder does not exist: {folder}")
    if position is None:
        position = ZPosition(0.0, "z_0")
    capture_rows: dict[str, dict[str, str]] = {}
    order: list[str] = []
    capture_log = folder / "capture_log.csv"
    if capture_log.is_file():
        with capture_log.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                output_name = row.get("output_name") or row.get("mask_name") or ""
                stem = normalize_stem(output_name)
                if not stem:
                    continue
                capture_rows[stem] = dict(row)
                order.append(stem)
    reference_log = folder / "reference_gain_log.csv"
    if reference_log.is_file():
        with reference_log.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                stem = normalize_stem(row.get("mask_name", ""))
                if not stem:
                    continue
                if stem in capture_rows:
                    for key, value in row.items():
                        if key not in capture_rows[stem] or capture_rows[stem][key] in (None, ""):
                            capture_rows[stem][key] = value
                else:
                    capture_rows[stem] = dict(row)
                if stem not in order:
                    order.append(stem)
    paths: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_CAMERA_EXTENSIONS:
            paths[path.stem] = path
    if not paths:
        raise RuntimeError(f"no camera frames found in {folder}")
    reference_folder = folder / "reference_captures"
    reference_paths: dict[str, Path] = {}
    if reference_folder.is_dir():
        for path in sorted(reference_folder.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_CAMERA_EXTENSIONS:
                reference_paths[path.stem] = path
    if not order:
        order = sorted(paths)
    else:
        logged = set(order)
        order.extend(stem for stem in sorted(paths) if stem not in logged)
    return ExposureIndex(
        exposure_us,
        position,
        folder,
        paths,
        reference_paths,
        capture_rows,
        order,
    )


@dataclass
class ReferenceModel:
    exposure_us: int
    mode: str
    gains: dict[str, float]
    target_linear: float
    valid_rows: int
    rejected_rows: int
    gain_min: float
    gain_median: float
    gain_max: float


def _safe_float(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def load_reference_model(
    csv_path: Path,
    exposure_us: int,
    lut: ResponseLUT,
    mode: str,
    max_gain: float,
) -> ReferenceModel:
    if not csv_path.is_file():
        raise FileNotFoundError(f"reference gain CSV does not exist: {csv_path}")
    rows: list[tuple[str, dict[str, str], float]] = []
    rejected = 0
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            stem = normalize_stem(row.get("mask_name", ""))
            status = str(row.get("status", "")).strip().upper()
            target_status = str(row.get("target_status", "")).strip().upper()
            signal_mean = _safe_float(row, "signal_mean")
            background = _safe_float(row, "background_level")
            if (
                not stem
                or not status.startswith("OK")
                or (target_status and not target_status.startswith("OK"))
                or signal_mean is None
                or background is None
                or signal_mean > lut.max_dn
                or background < lut.min_dn
            ):
                rejected += 1
                continue
            reference_linear = lut.scalar(signal_mean) - lut.scalar(background)
            if not np.isfinite(reference_linear) or reference_linear <= 0:
                rejected += 1
                continue
            rows.append((stem, dict(row), reference_linear))
    if not rows:
        raise RuntimeError(f"no valid reference rows in {csv_path}")
    target_linear = float(np.median([item[2] for item in rows]))
    gains: dict[str, float] = {}
    for stem, row, reference_linear in rows:
        if mode == "linearized":
            gain = target_linear / reference_linear
        elif mode == "logged":
            parsed = _safe_float(row, "correction_gain")
            gain = float("nan") if parsed is None else parsed
        else:
            raise ValueError(f"unknown reference gain mode: {mode}")
        if np.isfinite(gain) and 0 < gain <= max_gain:
            gains[stem] = float(gain)
        else:
            rejected += 1
    if not gains:
        raise RuntimeError(f"no usable reference gains in {csv_path}")
    gain_values = np.asarray(list(gains.values()), dtype=np.float64)
    return ReferenceModel(
        exposure_us=exposure_us,
        mode=mode,
        gains=gains,
        target_linear=target_linear,
        valid_rows=len(gains),
        rejected_rows=rejected,
        gain_min=float(np.min(gain_values)),
        gain_median=float(np.median(gain_values)),
        gain_max=float(np.max(gain_values)),
    )


# ---------------------------------------------------------------------------
# Subpixel temporal drift from paired reference RAW captures
# ---------------------------------------------------------------------------


def calculate_pearson(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    av = np.asarray(a[valid], dtype=np.float64)
    bv = np.asarray(b[valid], dtype=np.float64)
    av -= float(np.mean(av))
    bv -= float(np.mean(bv))
    denominator = float(np.sqrt(np.sum(av * av) * np.sum(bv * bv)))
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(np.sum(av * bv) / denominator)


def translation_matrix(shift_xy: tuple[float, float] | list[float] | np.ndarray) -> np.ndarray:
    dx, dy = np.asarray(shift_xy, dtype=np.float64).reshape(2)
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)


def warp_translation(
    image: np.ndarray,
    shift_xy: tuple[float, float],
    interpolation: int = cv2.INTER_LINEAR,
    border_value: float = 0.0,
) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image,
        translation_matrix(shift_xy)[:2],
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _gradient_correlation_image(image: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


@dataclass
class TranslationRegistrationResult:
    shift_xy: tuple[float, float]
    peak_correlation: float
    correlation_before: float
    correlation_after: float
    saturation_threshold_fixed: float
    dark_threshold_moving: float
    valid_fraction: float
    exceeds_expected_shift: bool


@dataclass
class ReferenceRegistrationModel:
    exposure_us: int
    z_position: ZPosition
    anchor_stem: str
    fixed_plane: np.ndarray
    shifts: dict[str, tuple[float, float]]
    reports: dict[str, dict[str, Any]]
    fixed_exposure_us: int = 0
    fixed_path: Path | None = None
    prepared: PreparedRegistration | None = None
    gain_template: tuple[np.ndarray, np.ndarray] | None = None
    recent_reference: tuple[str, np.ndarray] | None = None


@dataclass
class PreparedRegistration:
    bounds: tuple[int, int, int, int]
    feature: np.ndarray
    coarse: np.ndarray
    window: np.ndarray
    valid: np.ndarray
    saturation_dn: float


def registration_feature(image: np.ndarray, blur_sigma: float, highpass_sigma: float) -> np.ndarray:
    work = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if blur_sigma > 0:
        work = cv2.GaussianBlur(work, (0, 0), blur_sigma)
    if highpass_sigma > 0:
        work = work - cv2.GaussianBlur(work, (0, 0), highpass_sigma)
    work -= float(np.percentile(work, 10))
    scale = float(np.percentile(np.abs(work), 99.5))
    if scale <= 1e-8:
        raise RuntimeError("reference has insufficient texture for registration")
    return np.ascontiguousarray(work / scale, dtype=np.float32)


def prepare_registration(fixed: np.ndarray, *, saturation_dn: float = 4088.0,
                         blur_sigma: float = 1.0, highpass_sigma: float = 0.0,
                         downsample: int = 2, roi_size: int = 768) -> PreparedRegistration:
    """Prepare the common anchor once; crop before filters/FFT, never inpaint background."""
    fixed = np.asarray(fixed)
    h, w = fixed.shape
    small = cv2.resize(fixed.astype(np.float32), (max(1, w//8), max(1, h//8)), interpolation=cv2.INTER_AREA)
    signal = np.maximum(small - np.percentile(small, 98), 0)
    yy, xx = np.indices(small.shape)
    if signal.sum() <= 0:
        raise RuntimeError("reference anchor has no measurable signal")
    cx = (float(np.sum(xx * signal) / signal.sum()) + .5) * w / small.shape[1] - .5
    cy = (float(np.sum(yy * signal) / signal.sum()) + .5) * h / small.shape[0] - .5
    rh, rw = min(h, roi_size), min(w, roi_size)
    y0 = int(np.clip(round(cy-rh/2), 0, h-rh))
    x0 = int(np.clip(round(cx-rw/2), 0, w-rw))
    crop = fixed[y0:y0+rh, x0:x0+rw]
    feature = registration_feature(crop, blur_sigma, highpass_sigma)
    size = (max(8, rw//downsample), max(8, rh//downsample))
    coarse = cv2.resize(feature, size, interpolation=cv2.INTER_AREA)
    window = cv2.createHanningWindow(size, cv2.CV_32F)
    return PreparedRegistration((y0,y0+rh,x0,x0+rw), feature, coarse, window,
                                np.isfinite(crop) & (crop < saturation_dn), saturation_dn)


def register_intensity_translation(
    fixed: np.ndarray, moving: np.ndarray, *, saturation_threshold_fixed: float | None = None,
    dark_threshold_moving: float | None = None, mask_dilate_radius: int = 0,
    blur_sigma: float = 1.0, highpass_sigma: float = 0.0, method: str = "ecc",
    max_expected_shift: float = 50.0, downsample: int = 2,
    prepared: PreparedRegistration | None = None, roi_size: int = 768,
) -> TranslationRegistrationResult:
    """Return moving->anchor translation on the native Bayer-channel pixel grid.

    Phase correlation gives a coarse initialization; translation-only ECC
    refines it on the native ROI. The ECC matrix is anchor->moving and must be
    negated. Dark background is retained; independently masked saturation is
    excluded from fitting and from identical before/after diagnostic support.
    """
    if fixed.ndim != 2 or fixed.shape != moving.shape:
        raise ValueError("registration requires matching 2D channel planes")
    if downsample < 1 or max_expected_shift <= 0:
        raise ValueError("registration scale and shift limit must be positive")
    sat = (prepared.saturation_dn if prepared is not None else 4088.0) if saturation_threshold_fixed is None else float(saturation_threshold_fixed)
    prep = prepared or prepare_registration(fixed, saturation_dn=sat, blur_sigma=blur_sigma,
        highpass_sigma=highpass_sigma, downsample=downsample, roi_size=roi_size)
    y0,y1,x0,x1 = prep.bounds
    raw = moving[y0:y1,x0:x1]
    feature = registration_feature(raw, blur_sigma, highpass_sigma)
    size = (prep.coarse.shape[1], prep.coarse.shape[0])
    coarse = cv2.resize(feature, size, interpolation=cv2.INTER_AREA)
    a, b = prep.coarse.copy(), coarse
    if method == "gradcorr":
        a = _gradient_correlation_image(a)
        b = _gradient_correlation_image(b)
    shift_small, response = cv2.phaseCorrelate(b, a, prep.window)
    # Actual resize ratios matter when the ROI dimensions are not divisible.
    ratio = np.array([feature.shape[1]/size[0], feature.shape[0]/size[1]])
    shift = np.array(shift_small) * ratio
    if not np.all(np.isfinite(shift)) or np.any(np.abs(shift) > max_expected_shift):
        raise RuntimeError(f"unreliable phase registration: shift={shift}, response={response:.3g}")
    moving_valid = np.isfinite(raw) & (raw < sat)
    if dark_threshold_moving is not None:
        moving_valid &= raw > dark_threshold_moving
    radius = max(2, int(mask_dilate_radius))
    invalid = cv2.dilate((~moving_valid).astype(np.uint8), np.ones((2*radius+1,2*radius+1),np.uint8)) > 0
    if method == "ecc":
        warp = np.eye(2,3,dtype=np.float32)
        warp[:,2] = -shift
        # OpenCV's mask is in moving coordinates. Map fixed invalidity there.
        fixed_valid_moving = warp_translation(prep.valid.astype(np.float32), tuple(-shift)) > .999
        mask = ((~invalid) & fixed_valid_moving).astype(np.uint8)
        try:
            _, warp = cv2.findTransformECC(prep.feature, feature, warp, cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5), mask, 5)
        except cv2.error as error:
            raise RuntimeError("reference ECC did not converge") from error
        shift = -warp[:,2].astype(np.float64)
    elif method not in ("phasecorr", "gradcorr"):
        raise ValueError(f"unknown registration method: {method}")
    registered = warp_translation(feature, tuple(shift))
    overlap = prep.valid & (warp_translation(moving_valid.astype(np.float32), tuple(shift)) > .999)
    border = int(math.ceil(max_expected_shift)) + 3
    overlap[:border] = False
    overlap[-border:] = False
    overlap[:,:border] = False
    overlap[:,-border:] = False
    # Avoid background-dominated correlations; include anchor's signal texture.
    overlap &= prep.feature > np.percentile(prep.feature, 70)
    before = calculate_pearson(prep.feature, feature, overlap)
    after = calculate_pearson(prep.feature, registered, overlap)
    return TranslationRegistrationResult(tuple(float(s) for s in shift), float(response), before, after,
        sat, float(dark_threshold_moving or 0), float(np.mean(overlap)),
        bool(np.any(np.abs(shift) > max_expected_shift)))


def build_reference_registration_model(
    exposure_index: ExposureIndex,
    args: argparse.Namespace,
    common_anchor: ExposureIndex | None = None,
) -> ReferenceRegistrationModel:
    if not exposure_index.reference_paths:
        raise RuntimeError(f"no reference_captures RAW files in {exposure_index.folder}")
    anchor = normalize_stem(args.reference_anchor_pattern or args.homography_patterns[0])
    common_anchor = common_anchor or exposure_index
    anchor_path = common_anchor.reference_paths.get(anchor)
    if anchor_path is None:
        raise FileNotFoundError(
            f"reference registration anchor {anchor!r} missing in "
            f"{exposure_index.folder / 'reference_captures'}"
        )
    fixed = load_camera_dn(
        anchor_path,
        args.png_dn_scale,
        args.reference_registration_channel,
        args.raw_width,
        args.raw_height,
    )
    model = ReferenceRegistrationModel(
        exposure_index.exposure_us,
        exposure_index.z_position,
        anchor,
        fixed,
        {anchor: (0.0, 0.0)},
        {
            anchor: {
                "shift_xy": [0.0, 0.0],
                "peak_correlation": 1.0,
                "correlation_before": 1.0,
                "correlation_after": 1.0,
                "valid_fraction": 1.0,
                "exceeds_expected_shift": False,
            }
        },
    )
    model.fixed_exposure_us = common_anchor.exposure_us
    model.fixed_path = anchor_path
    model.prepared = prepare_registration(fixed,
        saturation_dn=args.registration_saturation_dn or args.saturation_dn,
        blur_sigma=args.registration_blur_sigma, highpass_sigma=args.registration_highpass_sigma,
        downsample=args.registration_downsample, roi_size=args.registration_roi_size)
    if common_anchor.folder != exposure_index.folder:
        model.shifts.clear()
        model.reports.clear()
    return model


def reference_shift_for_stem(
    model: ReferenceRegistrationModel,
    exposure_index: ExposureIndex,
    stem: str,
    args: argparse.Namespace,
) -> tuple[float, float]:
    if stem in model.shifts:
        return model.shifts[stem]
    if args.disable_reference_registration:
        model.shifts[stem] = (0.0, 0.0)
        model.reports[stem] = {"shift_xy": [0.0, 0.0], "disabled": True}
        return model.shifts[stem]
    moving_path = exposure_index.reference_paths.get(stem)
    if moving_path is None:
        policy = args.missing_reference_registration
        if policy == "error":
            raise FileNotFoundError(f"missing reference RAW for {stem} in {exposure_index.folder}")
        model.shifts[stem] = (0.0, 0.0)
        model.reports[stem] = {"shift_xy": [0.0, 0.0], "missing": True}
        return model.shifts[stem]
    moving = load_camera_dn(
        moving_path,
        args.png_dn_scale,
        args.reference_registration_channel,
        args.raw_width,
        args.raw_height,
    )
    model.recent_reference = (stem, moving)
    result = register_intensity_translation(
        model.fixed_plane,
        moving,
        saturation_threshold_fixed=args.registration_saturation_dn,
        dark_threshold_moving=args.registration_dark_dn,
        mask_dilate_radius=args.registration_mask_dilate_radius,
        blur_sigma=args.registration_blur_sigma,
        highpass_sigma=args.registration_highpass_sigma,
        method=args.registration_method,
        max_expected_shift=args.registration_max_shift,
        downsample=args.registration_downsample,
        prepared=model.prepared,
        roi_size=args.registration_roi_size,
    )
    if (result.exceeds_expected_shift or not np.isfinite(result.correlation_after)
            or result.correlation_after < args.registration_min_correlation
            or result.correlation_after + 0.01 < result.correlation_before):
        raise RuntimeError(f"reference registration rejected for Exp{model.exposure_us}/{stem}: {result}")
    model.shifts[stem] = result.shift_xy
    model.reports[stem] = {
        "shift_xy": list(result.shift_xy),
        "peak_correlation": result.peak_correlation,
        "correlation_before": result.correlation_before,
        "correlation_after": result.correlation_after,
        "saturation_threshold_fixed": result.saturation_threshold_fixed,
        "dark_threshold_moving": result.dark_threshold_moving,
        "valid_fraction": result.valid_fraction,
        "exceeds_expected_shift": result.exceeds_expected_shift,
        "correlation_stage": "ECC before joint radiometric refinement",
    }
    return result.shift_xy


def reference_gain_from_raw(model: ReferenceRegistrationModel, index: ExposureIndex,
                            stem: str, lut: ResponseLUT, args: argparse.Namespace,
                            calibration: RadiometricCalibration,
                            fixed_calibration: RadiometricCalibration) -> float:
    """Estimate power ratio from linearized, registered, unsaturated RAW pixels.

    Ratios use exposure-normalized linear values. Thus this gain already puts
    every exposure on the common anchor's power scale: no CSV/static scale may
    be multiplied into it. Gain/offset and translation are jointly refined on
    a bounded sample; target images are still interpolated only once.
    """
    from scipy.ndimage import map_coordinates
    from scipy.optimize import least_squares

    report = model.reports.setdefault(stem, {})
    if "raw_reference_gain" in report:
        return float(report["raw_reference_gain"])
    shift = np.asarray(reference_shift_for_stem(model, index, stem, args), dtype=np.float64)
    # Registration creates/replaces this record on the first call. Reacquire
    # it so joint diagnostics and cached gain belong to the live model record.
    report = model.reports[stem]
    if model.recent_reference is not None and model.recent_reference[0] == stem:
        moving = model.recent_reference[1]
    else:
        path = index.reference_paths.get(stem)
        if path is None:
            raise FileNotFoundError(f"missing reference RAW for gain: {stem} in {index.folder}")
        moving = load_camera_dn(path,args.png_dn_scale,args.reference_registration_channel,
                                args.raw_width,args.raw_height)
    y0,y1,x0,x1 = model.prepared.bounds

    def linear_crop(dn: np.ndarray, cal: RadiometricCalibration, exposure: int) -> np.ndarray:
        values = lut.apply(dn[y0:y1,x0:x1])
        if cal.dark is not None:
            values -= cal.dark[y0:y1,x0:x1]
        elif args.subtract_lut_zero:
            values -= lut.scalar(lut.min_dn)
        if cal.flat is not None:
            values /= cal.flat[y0:y1,x0:x1]
        return values / (exposure * 1e-6)

    if model.gain_template is None:
        fixed_dn = model.fixed_plane[y0:y1,x0:x1]
        fixed_linear = linear_crop(model.fixed_plane, fixed_calibration, model.fixed_exposure_us)
        model.gain_template = (fixed_dn, fixed_linear)
    fixed_dn, fixed_linear = model.gain_template
    moving_dn = moving[y0:y1,x0:x1]
    moving_linear = linear_crop(moving, calibration, index.exposure_us)
    # Erode invalid support before smoothing, including LUT range and flats.
    def good_dn(dn: np.ndarray, linear: np.ndarray) -> np.ndarray:
        valid = (np.isfinite(linear) & (dn >= lut.min_dn)
                 & (dn < min(lut.max_dn, args.saturation_dn)))
        # Four-pixel Gaussian halo plus one pixel of joint translation search.
        footprint = cv2.erode(valid.astype(np.uint8), np.ones((11,11),np.uint8)) > 0
        signal = cv2.GaussianBlur(dn.astype(np.float32),(0,0),1.0) >= args.reference_gain_low_dn
        return footprint & signal
    fixed_valid = good_dn(fixed_dn, fixed_linear)
    moving_valid = good_dn(moving_dn, moving_linear)
    margin = int(math.ceil(args.registration_max_shift)) + 4
    yy,xx = np.mgrid[margin:fixed_dn.shape[0]-margin:2,margin:fixed_dn.shape[1]-margin:2]
    yy,xx = yy.ravel(),xx.ravel()
    keep = fixed_valid[yy,xx] & (map_coordinates(moving_valid.astype(np.float32),
        [yy-shift[1],xx-shift[0]],order=1,mode='constant') > .999)
    yy,xx = yy[keep],xx[keep]
    if len(yy) < args.reference_gain_min_pixels:
        raise RuntimeError(f"insufficient unsaturated reference overlap: {len(yy)} samples for {stem}")
    step = max(1, int(math.ceil(len(yy)/4096)))
    yy,xx = yy[::step],xx[::step]
    # LUT precedes smoothing/interpolation, preserving nonlinear response order.
    fixed_smooth = cv2.GaussianBlur(np.nan_to_num(fixed_linear,nan=0,posinf=0,neginf=0), (0,0), 1.0)
    moving_smooth = cv2.GaussianBlur(np.nan_to_num(moving_linear,nan=0,posinf=0,neginf=0), (0,0), 1.0)
    fv = fixed_smooth[yy,xx].astype(np.float64)
    def sample(s: np.ndarray) -> np.ndarray:
        return map_coordinates(moving_smooth,[yy-s[1],xx-s[0]],order=1,mode='constant').astype(np.float64)
    mv = sample(shift)
    gain = float(np.median(fv/np.maximum(mv,1e-12)))
    norm = max(float(np.percentile(fv,90)),1e-12)
    # Only a small refinement about a verified registration is allowed. The
    # offset is a diagnostic nuisance parameter, never subtracted from targets.
    def residual(v: np.ndarray) -> np.ndarray:
        return (v[2]*sample(v[:2]) - fv)/norm + v[3]
    def jacobian(v: np.ndarray) -> np.ndarray:
        eps = .05
        dx = (sample(v[:2]+[eps,0])-sample(v[:2]-[eps,0]))/(2*eps)
        dy = (sample(v[:2]+[0,eps])-sample(v[:2]-[0,eps]))/(2*eps)
        return np.column_stack([v[2]*dx/norm,v[2]*dy/norm,sample(v[:2])/norm,np.ones(len(fv))])
    if args.disable_reference_registration:
        def gain_residual(v: np.ndarray) -> np.ndarray:
            return (v[0]*mv-fv)/norm+v[1]
        fit = least_squares(gain_residual,[gain,0.0],loss='soft_l1',f_scale=.03,
                            bounds=([1/args.max_reference_gain,-1],[args.max_reference_gain,1]))
        fitted = np.array([*shift,fit.x[0],fit.x[1]])
    else:
        fit = least_squares(residual,[*shift,gain,0.0],jac=jacobian,loss='soft_l1',f_scale=.03,
            bounds=([*(shift-1.0),1/args.max_reference_gain,-1],
                    [*(shift+1.0),args.max_reference_gain,1]),max_nfev=50,
                    ftol=1e-5,xtol=1e-4,gtol=1e-5)
        fitted = fit.x
        if np.any(np.abs(fitted[:2]) > args.registration_max_shift):
            raise RuntimeError(f"joint reference shift exceeds limit for {stem}")
        if np.any(np.abs(fitted[:2]-shift) > .95):
            raise RuntimeError(f"joint reference refinement reached its search boundary for {stem}")
    gain = float(fitted[2])
    if not fit.success or not np.isfinite(gain) or not 1/args.max_reference_gain < gain < args.max_reference_gain:
        raise RuntimeError(f"reference gain fit failed for {stem}: {fit.message}")
    residuals = residual(fitted)
    nrmse = float(np.sqrt(np.mean(residuals**2)))
    if nrmse > args.reference_gain_max_nrmse:
        raise RuntimeError(f"reference is inconsistent with a scalar gain for {stem}: NRMSE={nrmse:.3f}")
    model.shifts[stem] = tuple(float(v) for v in fitted[:2])
    report.update(raw_reference_gain=gain,raw_reference_gain_samples=len(fv),
        raw_reference_gain_nrmse=nrmse,raw_reference_offset=float(fitted[3]*norm),
        raw_reference_scale_only_nrmse=float(np.sqrt(np.mean((residuals-fitted[3])**2))),
        raw_reference_offset_fraction_of_p90=float(fitted[3]),
        requires_offset_calibration=bool(abs(fitted[3]) > .05),
        registration_shift_before_joint_xy=shift.tolist(), shift_xy=fitted[:2].tolist(),
        reference_gain_basis="pixelwise LUT, common-anchor radiance, robust gain+offset+translation")
    model.recent_reference = None
    return gain


def summarize_reference_registration(model: ReferenceRegistrationModel) -> dict[str, Any]:
    shifts = np.asarray(list(model.shifts.values()), dtype=np.float64)
    magnitudes = np.linalg.norm(shifts, axis=1) if len(shifts) else np.empty(0)
    automatic = [report for report in model.reports.values() if "peak_correlation" in report]
    responses = np.asarray(
        [report["peak_correlation"] for report in automatic], dtype=np.float64
    )
    correlations_after = np.asarray(
        [report.get("correlation_after", np.nan) for report in automatic], dtype=np.float64
    )
    flagged = [
        stem
        for stem, report in model.reports.items()
        if report.get("exceeds_expected_shift", False)
    ]
    return {
        "anchor_stem": model.anchor_stem,
        "registered_stem_count": len(model.shifts),
        "shift_dx_min_px": float(np.min(shifts[:, 0])) if len(shifts) else 0.0,
        "shift_dx_max_px": float(np.max(shifts[:, 0])) if len(shifts) else 0.0,
        "shift_dy_min_px": float(np.min(shifts[:, 1])) if len(shifts) else 0.0,
        "shift_dy_max_px": float(np.max(shifts[:, 1])) if len(shifts) else 0.0,
        "shift_magnitude_median_px": float(np.median(magnitudes)) if len(magnitudes) else 0.0,
        "shift_magnitude_max_px": float(np.max(magnitudes)) if len(magnitudes) else 0.0,
        "peak_correlation_median": float(np.nanmedian(responses)) if len(responses) else None,
        "correlation_after_median": (
            float(np.nanmedian(correlations_after)) if len(correlations_after) else None
        ),
        "exceeds_expected_shift_count": len(flagged),
        "exceeds_expected_shift_examples": flagged[:20],
    }


def _integer_or_default(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def collect_registration_trend_rows(
    position: ZPosition,
    stems: list[str],
    exposure_indices: dict[int, ExposureIndex],
    reference_registrations: dict[int, ReferenceRegistrationModel],
    args: argparse.Namespace,
    *, cached_only: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate/copy cached reference shifts in phase-mask acquisition order."""
    rows: list[dict[str, Any]] = []
    for exposure_us in args.exposures:
        index = exposure_indices[exposure_us]
        registration = reference_registrations[exposure_us]
        for order_index, stem in enumerate(stems, start=1):
            if cached_only and stem not in registration.shifts:
                continue
            shift = reference_shift_for_stem(registration, index, stem, args)
            capture_row = index.capture_rows.get(stem, {})
            diagnostic = registration.reports.get(stem, {})
            rows.append(
                {
                    "z_label": position.label,
                    "z_position_mm": position.value_mm,
                    "exposure_us": exposure_us,
                    "phase_mask_index": _integer_or_default(
                        capture_row.get("index"), order_index
                    ),
                    "order_index": order_index,
                    "chunk": _integer_or_default(capture_row.get("chunk"), 0),
                    "elapsed_s": _safe_float(capture_row, "elapsed_s"),
                    "stem": stem,
                    "dx_px": float(shift[0]),
                    "dy_px": float(shift[1]),
                    "dx_raw_sensor_px": float(2.0 * shift[0]),
                    "dy_raw_sensor_px": float(2.0 * shift[1]),
                    "peak_correlation": diagnostic.get("peak_correlation"),
                    "correlation_before": diagnostic.get("correlation_before"),
                    "correlation_after": diagnostic.get("correlation_after"),
                    "valid_fraction": diagnostic.get("valid_fraction"),
                    "raw_reference_gain": diagnostic.get("raw_reference_gain"),
                    "raw_reference_gain_nrmse": diagnostic.get("raw_reference_gain_nrmse"),
                    "exceeds_expected_shift": bool(
                        diagnostic.get("exceeds_expected_shift", False)
                    ),
                }
            )
    return rows


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    width = max(1, int(window))
    if width <= 1 or len(data) <= 2:
        return data.copy()
    if width % 2 == 0:
        width += 1
    radius = width // 2
    result = np.empty_like(data)
    for index in range(len(data)):
        start = max(0, index - radius)
        stop = min(len(data), index + radius + 1)
        result[index] = np.nanmedian(data[start:stop])
    return result


def save_registration_trend_outputs(
    output_dir: Path,
    position: ZPosition,
    rows: list[dict[str, Any]],
    smoothing_window: int,
) -> dict[str, str]:
    """Save a long-form CSV and a two-panel dx/dy trend PNG for one z."""
    if not rows:
        raise ValueError(f"no reference registration rows for {position.label}")
    csv_path = output_dir / f"{position.label}_reference_shift_vs_index.csv"
    png_path = output_dir / f"{position.label}_reference_shift_vs_index.png"
    atomic_write_csv(csv_path, rows)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "matplotlib is required for registration trend PNGs; "
            "the numeric trend CSV was written successfully"
        ) from error

    exposures = sorted({int(row["exposure_us"]) for row in rows})
    figure, axes = plt.subplots(2, 1, figsize=(12.0, 7.5), sharex=True, constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for series_index, exposure_us in enumerate(exposures):
        series = sorted(
            (row for row in rows if int(row["exposure_us"]) == exposure_us),
            key=lambda row: (int(row["phase_mask_index"]), int(row["order_index"])),
        )
        x = np.asarray([row["phase_mask_index"] for row in series], dtype=np.float64)
        color = colors(series_index % 10)
        for axis, key in zip(axes, ("dx_px", "dy_px")):
            values = np.asarray([row[key] for row in series], dtype=np.float64)
            axis.plot(x, values, color=color, linewidth=0.65, alpha=0.28)
            axis.plot(
                x,
                rolling_median(values, smoothing_window),
                color=color,
                linewidth=1.6,
                label=f"Exp{exposure_us}",
            )
            flagged = np.asarray(
                [bool(row["exceeds_expected_shift"]) for row in series], dtype=bool
            )
            if np.any(flagged):
                axis.scatter(
                    x[flagged],
                    values[flagged],
                    color=color,
                    edgecolors="black",
                    linewidths=0.45,
                    s=24,
                    zorder=4,
                )
    axes[0].set_ylabel("x shift, dx (Bayer-split px; +right)")
    axes[1].set_ylabel("y shift, dy (Bayer-split px; +down)")
    axes[1].set_xlabel("phase-mask index (capture order within each exposure)")
    for axis in axes:
        axis.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
        axis.grid(True, color="0.85", linewidth=0.55)
        axis.legend(loc="best", ncols=min(3, len(exposures)), frameon=False)
    figure.suptitle(
        f"Reference registration drift — {position.label} ({position.value_mm:+g} mm)\n"
        f"thin: individual shifts; thick: rolling median (window={max(1, smoothing_window)})"
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_png = png_path.with_name(png_path.stem + ".tmp.png")
    figure.savefig(temporary_png, dpi=180)
    plt.close(figure)
    os.replace(temporary_png, png_path)
    return {"csv": str(csv_path), "png": str(png_path)}


def calibration_files(root: Path | None, exposure_us: int) -> list[Path]:
    if root is None:
        return []
    candidates = (root / f"Exp{exposure_us}", root / f"exposure{exposure_us}")
    folder = next((candidate for candidate in candidates if candidate.is_dir()), root)
    if not folder.is_dir():
        raise FileNotFoundError(f"calibration folder does not exist: {folder}")
    return [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_CAMERA_EXTENSIONS
    ]


def build_master_calibration(
    files: list[Path],
    lut: ResponseLUT,
    png_dn_scale: float,
    raw_width: int,
    raw_height: int,
    output_channels: str,
    dark_linear: np.ndarray | None = None,
) -> np.ndarray | None:
    if not files:
        return None
    linear_frames: list[np.ndarray] = []
    shape: tuple[int, ...] | None = None
    for path in files:
        if output_channels == "rgb":
            dn = load_camera_dn_channels(path, png_dn_scale, raw_width, raw_height)
        else:
            dn = load_camera_dn(
                path,
                png_dn_scale,
                output_channels,
                raw_width,
                raw_height,
            )
        if shape is None:
            shape = dn.shape
        elif dn.shape != shape:
            raise ValueError(f"calibration frame shape mismatch: {path} has {dn.shape}, expected {shape}")
        valid = (dn >= lut.min_dn) & (dn <= lut.max_dn)
        linear = lut.apply(np.clip(dn, lut.min_dn, lut.max_dn))
        linear[~valid] = np.nan
        if dark_linear is not None:
            if dark_linear.shape != linear.shape:
                raise ValueError("dark and flat calibration shapes differ")
            linear = linear - dark_linear
        linear_frames.append(linear)
    return np.nanmedian(np.stack(linear_frames, axis=0), axis=0).astype(np.float32)


@dataclass
class RadiometricCalibration:
    dark: np.ndarray | None
    flat: np.ndarray | None
    dark_frames: int
    flat_frames: int


def load_radiometric_calibration(
    exposure_us: int,
    dark_root: Path | None,
    flat_root: Path | None,
    lut: ResponseLUT,
    png_dn_scale: float,
    raw_width: int,
    raw_height: int,
    output_channels: str,
    subtract_lut_zero: bool,
) -> RadiometricCalibration:
    dark_files = calibration_files(dark_root, exposure_us)
    flat_files = calibration_files(flat_root, exposure_us)
    dark = build_master_calibration(
        dark_files,
        lut,
        png_dn_scale,
        raw_width,
        raw_height,
        output_channels,
    )
    flat_signal = build_master_calibration(
        flat_files,
        lut,
        png_dn_scale,
        raw_width,
        raw_height,
        output_channels,
        dark_linear=dark,
    )
    if flat_signal is not None and dark is None and subtract_lut_zero:
        flat_signal = flat_signal - lut.scalar(lut.min_dn)
    flat = None
    if flat_signal is not None:
        positive = flat_signal[np.isfinite(flat_signal) & (flat_signal > 0)]
        if not positive.size:
            raise RuntimeError(f"flat calibration is non-positive for exposure {exposure_us}")
        normalization = float(np.median(positive))
        flat = flat_signal / normalization
        flat[(~np.isfinite(flat)) | (flat <= 0.05)] = np.nan
        flat = flat.astype(np.float32)
    return RadiometricCalibration(dark, flat, len(dark_files), len(flat_files))


def calibrate_exposure_transform(
    exposure_index: ExposureIndex,
    sim_dir: Path,
    pattern_names: list[str],
    reference_registration: ReferenceRegistrationModel,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for requested_name in pattern_names:
        stem = normalize_stem(requested_name)
        camera_path = find_stem_file(exposure_index.folder, stem, SUPPORTED_CAMERA_EXTENSIONS)
        sim_path = find_stem_file(sim_dir, stem, SUPPORTED_SIM_EXTENSIONS)
        camera_plane = load_camera_dn(
            camera_path,
            args.png_dn_scale,
            args.camera_channel,
            args.raw_width,
            args.raw_height,
        )
        sim_img = load_reference_array(sim_path, cv2.IMREAD_UNCHANGED)
        if sim_img is None:
            raise RuntimeError(f"could not read calibration pair {stem}")
        sim_plane = intensity_plane(sim_img, args.sim_channel)
        if args.core_locator == "pattern":
            camera_cores, camera_locator = locate_four_cores_pattern_aware(
                camera_plane,
                stem,
                center_sigma=args.core_center_sigma,
                seed_sigma=args.camera_seed_sigma,
                win=args.camera_win,
                sigma=args.camera_sigma,
                roi_fraction=args.core_roi_fraction,
                candidate_min_distance=args.core_candidate_min_distance,
                sector_guard=args.core_sector_guard,
                center_exclusion_radius=args.core_center_exclusion_radius,
                minimum_separation=args.core_min_separation,
            )
            sim_cores, sim_locator = locate_four_cores_pattern_aware(
                sim_plane,
                stem,
                center_sigma=args.core_center_sigma,
                seed_sigma=args.sim_seed_sigma,
                win=args.sim_win,
                sigma=args.sim_sigma,
                roi_fraction=args.core_roi_fraction,
                candidate_min_distance=args.core_candidate_min_distance,
                sector_guard=args.core_sector_guard,
                center_exclusion_radius=args.core_center_exclusion_radius,
                minimum_separation=args.core_min_separation,
            )
        else:
            camera_cores, camera_locator = locate_four_cores(
                camera_plane,
                seed_sigma=args.camera_seed_sigma,
                win=args.camera_win,
                sigma=args.camera_sigma,
                locator=args.core_locator,
                peak_min_distance=args.peak_min_distance,
                peak_margin_fraction=args.peak_margin_fraction,
                peak_candidates=args.peak_candidates,
                peak_background_sigma=args.peak_background_sigma,
                max_pattern_fraction=args.max_pattern_fraction,
                pattern_center_tolerance=args.pattern_center_tolerance,
            )
            sim_cores, sim_locator = locate_four_cores(
                sim_plane,
                seed_sigma=args.sim_seed_sigma,
                win=args.sim_win,
                sigma=args.sim_sigma,
                locator=args.core_locator,
                peak_min_distance=args.peak_min_distance,
                peak_margin_fraction=args.peak_margin_fraction,
                peak_candidates=args.peak_candidates,
                peak_background_sigma=args.peak_background_sigma,
                max_pattern_fraction=args.max_pattern_fraction,
                pattern_center_tolerance=args.pattern_center_tolerance,
            )
        src_raw = cores_to_xy(camera_cores).astype(np.float64)
        reference_shift = reference_shift_for_stem(
            reference_registration,
            exposure_index,
            stem,
            args,
        )
        src = src_raw.copy()
        src[:, 0] += reference_shift[0]
        src[:, 1] += reference_shift[1]
        dst = cores_to_xy(sim_cores).astype(np.float64)
        sim_h, sim_w = sim_plane.shape
        # OpenCV resize maps pixel centres, not pixel corners.
        dst[:, 0] = (dst[:, 0] + .5) * args.output_size / sim_w - .5
        dst[:, 1] = (dst[:, 1] + .5) * args.output_size / sim_h - .5
        pairs.append(
            {
                "stem": stem,
                "camera_path": str(camera_path),
                "simulation_path": str(sim_path),
                "camera_locator": camera_locator,
                "simulation_locator": sim_locator,
                "reference_shift_xy": list(reference_shift),
                "camera_points_raw_xy": src_raw,
                "camera_points_xy": src,
                "simulation_points_xy": dst,
                "camera_plane": camera_plane,
            }
        )
    src_all = np.vstack([pair["camera_points_xy"] for pair in pairs])
    dst_all = np.vstack([pair["simulation_points_xy"] for pair in pairs])
    if args.transform == "affine":
        matrix, inliers = fit_pooled_affine(src_all, dst_all, args.ransac_threshold)
    else:
        matrix, inliers = fit_pooled_homography(src_all, dst_all, args.ransac_threshold)
    # A bad initial sector centre can select a side lobe. A supported pooled
    # fit supplies a local search region for re-detection, not invented points.
    # Every recovered coordinate still comes from the measured image peak.
    if np.mean(inliers) >= .625 and not np.all(inliers):
        predicted = project_points(np.linalg.inv(matrix), dst_all)
        offset = 0
        for pair in pairs:
            raw_predictions = predicted[offset:offset+4] - np.asarray(pair["reference_shift_xy"])
            recovered = []
            for x,y in raw_predictions:
                ry,rx = blurred_peak_subpixel(pair["camera_plane"],y,x,win=args.core_guided_window,sigma=args.camera_sigma)
                recovered.append((rx,ry))
            pair["camera_points_raw_xy"] = np.asarray(recovered)
            pair["camera_points_xy"] = np.asarray(recovered)+np.asarray(pair["reference_shift_xy"])
            pair["camera_locator"] += "/pooled-guided-redetection"
            offset += 4
        src_all = np.vstack([pair["camera_points_xy"] for pair in pairs])
        if args.transform == "affine":
            matrix,inliers = fit_pooled_affine(src_all,dst_all,args.ransac_threshold)
        else:
            matrix,inliers = fit_pooled_homography(src_all,dst_all,args.ransac_threshold)
    errors = np.linalg.norm(project_points(matrix, src_all) - dst_all, axis=1)
    offset = 0
    pair_reports: list[dict[str, Any]] = []
    for pair in pairs:
        count = len(pair["camera_points_xy"])
        pair_reports.append(
            {
                "stem": pair["stem"],
                "camera_path": pair["camera_path"],
                "simulation_path": pair["simulation_path"],
                "camera_locator": pair["camera_locator"],
                "simulation_locator": pair["simulation_locator"],
                "reference_shift_xy": pair["reference_shift_xy"],
                "camera_points_raw_xy": pair["camera_points_raw_xy"].astype(float).tolist(),
                "camera_points_xy": pair["camera_points_xy"].astype(float).tolist(),
                "simulation_points_xy": pair["simulation_points_xy"].astype(float).tolist(),
                "residual": residual_statistics(errors[offset : offset + count]),
                "inlier_count": int(np.count_nonzero(inliers[offset : offset + count])),
            }
        )
        offset += count
    report = {
        "exposure_us": exposure_index.exposure_us,
        "z_position_mm": exposure_index.z_position.value_mm,
        "z_label": exposure_index.z_position.label,
        "model": args.transform,
        "core_locator": args.core_locator,
        "core_roi_fraction": float(args.core_roi_fraction),
        "core_center_sigma_px": float(args.core_center_sigma),
        "matrix": matrix.astype(float).tolist(),
        "ransac_threshold_px": float(args.ransac_threshold),
        "inlier_count": int(np.count_nonzero(inliers)),
        "point_count": int(len(inliers)),
        "residual": residual_statistics(errors),
        "pairs": pair_reports,
    }
    singular_values = np.linalg.svd(matrix[:2,:2], compute_uv=False)
    report["affine_axis_ratio"] = float(singular_values[0]/max(singular_values[-1],1e-12))
    report["residual_inliers"] = residual_statistics(errors[inliers])
    report["inlier_mask"] = inliers.tolist()
    if not all(np.count_nonzero(inliers[i:i+4]) >= 3 for i in range(0,len(inliers),4)):
        raise RuntimeError("calibration lacks three supported corners in each pattern")
    # Independent period holdout: fit the wide p008 points and evaluate p016
    # (and vice versa). These errors are not the pooled training residual.
    if args.transform == "affine":
        periods = [re.search(r"_p(\d+)_", pair["stem"]) for pair in pairs]
        period_ids = np.repeat([m.group(1) if m else "unknown" for m in periods], 4)
        holdouts = {}
        for period in sorted(set(period_ids)):
            test = period_ids == period
            if np.count_nonzero(~test) >= 4 and np.any(test):
                held_matrix = fit_affine_lstsq(src_all[~test],dst_all[~test])
                holdouts[str(period)] = residual_statistics(np.linalg.norm(
                    project_points(held_matrix,src_all[test])-dst_all[test],axis=1))
        report["held_out_period_residuals"] = holdouts
    if (np.max(errors) > args.max_calibration_error
            or np.mean(inliers) < args.min_calibration_inlier_fraction
            or singular_values[-1] < 1e-6
            or report["affine_axis_ratio"] > args.max_affine_axis_ratio):
        raise RuntimeError("calibration rejected: " + json.dumps(report))
    return matrix.astype(np.float64), report


# ---------------------------------------------------------------------------
# HDR processing and final-only output
# ---------------------------------------------------------------------------


def hdr_dn_weight(
    dn: np.ndarray,
    valid: np.ndarray,
    low_dn: float,
    low_full_weight_dn: float,
    high_full_weight_dn: float,
    saturation_dn: float,
) -> np.ndarray:
    dn = np.asarray(dn, dtype=np.float32)
    rise_denominator = max(low_full_weight_dn - low_dn, 1e-6)
    fall_denominator = max(saturation_dn - high_full_weight_dn, 1e-6)
    rise = np.clip((dn - low_dn) / rise_denominator, 0.0, 1.0)
    fall = np.clip((saturation_dn - dn) / fall_denominator, 0.0, 1.0)
    weight = rise * fall
    weight[~valid] = 0.0
    return weight.astype(np.float32)


def warp_hdr_exposure(
    radiance: np.ndarray,
    weight: np.ndarray,
    valid: np.ndarray,
    matrix: np.ndarray,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate intensity independently of its HDR weight, using full support.

    warp(L*w)/warp(w) biases spatial interpolation toward brighter neighbours
    when w depends on DN. A zero-DN neighbour must retain its geometric weight.
    Saturation, bad calibration and out-of-frame neighbours remain excluded.
    """
    is_affine = np.allclose(matrix[2], [0,0,1], atol=1e-12)
    warp = cv2.warpAffine if is_affine else cv2.warpPerspective
    warp_matrix = matrix[:2] if is_affine else matrix
    warped_radiance = warp(
        np.where(valid, radiance, 0).astype(np.float32),
        warp_matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_weight = warp(
        weight.astype(np.float32),
        warp_matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    support = warp(valid.astype(np.float32), warp_matrix,
                   (output_size, output_size), flags=cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    supported = support >= .999
    warped_radiance[~supported] = 0
    warped_weight[~supported] = 0
    return warped_radiance, warped_weight, supported


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def atomic_save_quality(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Keep quality arrays separate from top-level camera NPY discovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def save_preview_png(path: Path, image: np.ndarray) -> None:
    display = np.asarray(image, dtype=np.float32)
    if display.ndim == 3 and display.shape[0] == 3 and display.shape[-1] != 3:
        # Saved scientific output is CHW RGB; OpenCV encoding expects HWC BGR.
        display = np.moveaxis(display, 0, 2)[..., ::-1]
    finite = display[np.isfinite(display) & (display > 0)]
    if finite.size:
        low, high = np.percentile(finite, [0.1, 99.9])
        if high <= low:
            high = low + 1.0
        preview = np.clip((display - low) / (high - low), 0.0, 1.0)
    else:
        preview = np.zeros_like(display, dtype=np.float32)
    encoded_input = np.rint(np.nan_to_num(preview) * 65535.0).astype(np.uint16)
    ok, encoded = cv2.imencode(".png", encoded_input, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"could not encode preview: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def process_one_stem(
    stem: str,
    exposure_indices: dict[int, ExposureIndex],
    reference_models: dict[int, ReferenceModel],
    reference_registrations: dict[int, ReferenceRegistrationModel],
    transforms: dict[int, np.ndarray],
    radiometric: dict[int, RadiometricCalibration],
    static_scales: dict[int, float],
    lut: ResponseLUT,
    args: argparse.Namespace,
    *,
    quality: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    output_size = int(args.output_size)
    output_shape = (
        (output_size, output_size, 3)
        if args.output_channels == "rgb"
        else (output_size, output_size)
    )
    hdr_numerator = np.zeros(output_shape, dtype=np.float32)
    hdr_weight = np.zeros(output_shape, dtype=np.float32)
    fallback_radiance = np.zeros(output_shape, dtype=np.float32)
    fallback_exposure = np.zeros(output_shape, dtype=np.float32)
    confidence = np.zeros(output_shape, dtype=np.float32)
    used_exposures: list[int] = []
    dropped: dict[str, str] = {}
    native_valid_fraction: dict[str, float] = {}
    native_low_signal_fraction: dict[str, float] = {}
    logged_reference_sat_pct: dict[str, float | None] = {}
    reference_shifts: dict[str, list[float]] = {}
    reference_gains: dict[str, float] = {}
    for exposure_us in args.exposures:
        index = exposure_indices[exposure_us]
        path = index.paths.get(stem)
        if path is None:
            dropped[str(exposure_us)] = "missing camera frame"
            continue
        capture_row = index.capture_rows.get(stem, {})
        capture_status = str(capture_row.get("status", "OK")).strip().upper()
        target_status = str(capture_row.get("target_status", "OK")).strip().upper()
        if (
            (capture_status and not capture_status.startswith("OK"))
            or (target_status and not target_status.startswith("OK"))
        ):
            dropped[str(exposure_us)] = f"capture status {capture_status}"
            continue
        sat_value = _safe_float(capture_row, "sat_pct")
        if sat_value is None:
            sat_value = _safe_float(capture_row, "reference_sat_pct")
        logged_reference_sat_pct[str(exposure_us)] = sat_value
        if args.reference_gain_mode == "raw":
            gain = reference_gain_from_raw(reference_registrations[exposure_us],index,stem,lut,args,
                radiometric[exposure_us],radiometric[args.geometry_exposure])
        else:
            gain = reference_models[exposure_us].gains.get(stem)
        if gain is None:
            if args.missing_reference == "error":
                raise RuntimeError(f"missing valid reference gain for {stem} at exposure {exposure_us}")
            if args.missing_reference == "identity":
                gain = 1.0
            else:
                dropped[str(exposure_us)] = "missing valid reference gain"
                continue
        if args.output_channels == "rgb":
            dn = load_camera_dn_channels(
                path,
                args.png_dn_scale,
                args.raw_width,
                args.raw_height,
            )
        else:
            dn = load_camera_dn(
                path,
                args.png_dn_scale,
                args.output_channels,
                args.raw_width,
                args.raw_height,
            )
        valid = (
            np.isfinite(dn)
            & (dn < args.saturation_dn)
            & (dn >= lut.min_dn)
            & (dn <= lut.max_dn)
        )
        linear = lut.apply(dn)
        calibration = radiometric[exposure_us]
        if calibration.dark is not None:
            if calibration.dark.shape != linear.shape:
                raise ValueError(
                    f"dark shape {calibration.dark.shape} != frame shape {linear.shape} for exposure {exposure_us}"
                )
            linear = linear - calibration.dark
        elif args.subtract_lut_zero:
            linear = linear - lut.scalar(lut.min_dn)
        if calibration.flat is not None:
            if calibration.flat.shape != linear.shape:
                raise ValueError(
                    f"flat shape {calibration.flat.shape} != frame shape {linear.shape} for exposure {exposure_us}"
                )
            valid &= np.isfinite(calibration.flat)
            linear = linear / calibration.flat
        valid &= np.isfinite(linear)
        exposure_seconds = float(exposure_us) * 1e-6
        radiance = (
            linear
            * float(gain)
            * float(static_scales[exposure_us])
            / (exposure_seconds * float(args.nd_transmission))
        )
        if not args.keep_negative:
            radiance = np.maximum(radiance, 0.0)
        valid &= np.isfinite(radiance)
        weight = hdr_dn_weight(
            dn,
            valid,
            args.low_dn,
            args.low_full_weight_dn,
            args.high_full_weight_dn,
            args.saturation_dn,
        )
        # In radiance units shot-noise variance scales approximately as g/t.
        # Taper still suppresses saturation and read-noise-dominated shadows.
        # Full inverse variance weighting requires a measured camera noise model.
        effective_exposure = float(exposure_us) / (float(gain) * static_scales[exposure_us])
        shift = reference_shift_for_stem(
            reference_registrations[exposure_us],
            index,
            stem,
            args,
        )
        reference_shifts[str(exposure_us)] = list(shift)
        reference_gains[str(exposure_us)] = float(gain)
        composed_transform = transforms[exposure_us] @ translation_matrix(shift)
        warped_radiance, warped_weight, supported = warp_hdr_exposure(
            radiance, weight, valid, composed_transform, output_size
        )
        np.maximum(confidence, warped_weight, out=confidence)
        # DN taper is a fusion preference, not a test of measurement existence.
        # Shadows with zero total taper retain the longest supported exposure,
        # including the supplied LUT intercept (no invented dark subtraction).
        take_fallback = supported & (effective_exposure > fallback_exposure)
        fallback_radiance[take_fallback] = warped_radiance[take_fallback]
        fallback_exposure[take_fallback] = effective_exposure
        if args.hdr_weighting == "exposure":
            warped_weight *= effective_exposure / max(args.exposures)
        hdr_numerator += warped_radiance * warped_weight
        hdr_weight += warped_weight
        native_valid_fraction[str(exposure_us)] = float(np.mean(valid))
        native_low_signal_fraction[str(exposure_us)] = float(np.mean(valid & (dn <= args.low_dn)))
        used_exposures.append(int(exposure_us))
    final_working = np.full(output_shape, np.nan if args.invalid_output == "nan" else 0, dtype=np.float32)
    fused = hdr_weight > float(args.minimum_output_weight)
    good = fallback_exposure > 0
    fallback = good & ~fused
    final_working[fused] = (hdr_numerator[fused] / hdr_weight[fused]).astype(np.float32)
    final_working[fallback] = fallback_radiance[fallback]
    confidence[~good] = 0
    if not used_exposures:
        raise RuntimeError(f"no usable exposure remains for {stem}")
    if not np.any(good):
        raise RuntimeError(f"no valid HDR output pixels for {stem}")
    if args.output_channels == "rgb":
        final = np.moveaxis(final_working[..., ::-1], 2, 0)
        output_channel_order = "CHW_RGB"
    else:
        final = final_working
        output_channel_order = f"HW_{args.output_channels.upper()}"
    if quality is not None:
        def output_order(array: np.ndarray) -> np.ndarray:
            return np.moveaxis(array[..., ::-1], 2, 0) if args.output_channels == "rgb" else array
        quality.update(valid_mask=output_order(good),
                       confidence=output_order(confidence),
                       fallback_mask=output_order(fallback))
    if args.output_dtype == "float16":
        if np.nanmax(np.abs(final)) > np.finfo(np.float16).max:
            raise OverflowError("float16 would overflow; use --output-dtype float32")
        final = final.astype(np.float16)
    report = {
        "stem": stem,
        "used_exposures": used_exposures,
        "dropped_exposures": dropped,
        "native_valid_fraction": native_valid_fraction,
        "native_low_signal_fraction": native_low_signal_fraction,
        "logged_reference_sat_pct": logged_reference_sat_pct,
        "reference_shift_xy": reference_shifts,
        "reference_gain": reference_gains,
        "output_valid_fraction": float(np.mean(good)),
        "output_hdr_weighted_fraction": float(np.mean(fused)),
        "output_fallback_fraction": float(np.mean(fallback)),
        "output_confidence_mean": float(np.mean(confidence)),
        "output_valid_fraction_by_channel": (
            {
                "blue": float(np.mean(good[..., 0])),
                "green": float(np.mean(good[..., 1])),
                "red": float(np.mean(good[..., 2])),
            }
            if good.ndim == 3
            else {args.output_channels: float(np.mean(good))}
        ),
        "output_channel_order": output_channel_order,
        "output_min": float(np.nanmin(final)),
        "output_max": float(np.nanmax(final)),
        "output_mean": float(np.nanmean(final)),
    }
    return final, report


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def resolve_simulation_dir(sim_root: Path, position: ZPosition) -> Path:
    """Resolve the documented z folder while accepting source-style aliases."""
    value = position.value_mm
    signed = f"{value:+g}" if abs(value) >= 1e-12 else "0"
    candidates = [
        sim_root / position.label,
        sim_root / f"z{signed}mm",
        sim_root / f"z_{signed}mm",
    ]
    if abs(value) < 1e-12:
        candidates.extend([sim_root / "z0", sim_root])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def validate_input_layout(
    discovered: dict[str, tuple[ZPosition, dict[int, Path]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Read-only validation that deliberately does not require simulations/LUT."""
    expected_raw_bytes = int(args.raw_width) * int(args.raw_height) * 2
    result: dict[str, Any] = {"positions": {}, "ok": True}
    for label, (position, folders) in discovered.items():
        indices = {
            exposure_us: build_exposure_index(folder, exposure_us, position)
            for exposure_us, folder in folders.items()
        }
        stem_sets = [set(index.paths) for index in indices.values()]
        common = set.intersection(*stem_sets)
        union = set.union(*stem_sets)
        exposure_stems_equal = all(stems == stem_sets[0] for stems in stem_sets[1:])
        result["ok"] = bool(result["ok"] and exposure_stems_equal)
        position_report: dict[str, Any] = {
            "z_position_mm": position.value_mm,
            "common_stem_count": len(common),
            "union_stem_count": len(union),
            "exposure_stems_equal": exposure_stems_equal,
            "exposures": {},
        }
        for exposure_us, index in indices.items():
            main_stems = set(index.paths)
            reference_stems = set(index.reference_paths)
            bad_raw_sizes = [
                path.name
                for path in itertools.chain(index.paths.values(), index.reference_paths.values())
                if path.suffix.lower() == ".raw" and path.stat().st_size != expected_raw_bytes
            ]
            csv_path = index.folder / "reference_gain_log.csv"
            csv_stems: set[str] = set()
            if csv_path.is_file():
                with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
                    csv_stems = {
                        normalize_stem(row.get("mask_name", ""))
                        for row in csv.DictReader(handle)
                        if normalize_stem(row.get("mask_name", ""))
                    }
            geometry_source = exposure_us == args.geometry_exposure
            missing_patterns = (
                [
                    normalize_stem(name)
                    for name in args.homography_patterns
                    if normalize_stem(name) not in main_stems
                    or normalize_stem(name) not in reference_stems
                ]
                if geometry_source
                else []
            )
            exposure_ok = not (
                bad_raw_sizes
                or (main_stems - reference_stems)
                or (reference_stems - main_stems)
                or (args.reference_gain_mode != "raw" and (main_stems - csv_stems))
                or missing_patterns
            )
            result["ok"] = bool(result["ok"] and exposure_ok)
            position_report["exposures"][str(exposure_us)] = {
                "folder": str(index.folder),
                "main_count": len(main_stems),
                "reference_count": len(reference_stems),
                "csv_stem_count": len(csv_stems),
                "missing_reference_count": len(main_stems - reference_stems),
                "orphan_reference_count": len(reference_stems - main_stems),
                "missing_csv_count": len(main_stems - csv_stems),
                "bad_raw_size_count": len(bad_raw_sizes),
                "bad_raw_size_examples": bad_raw_sizes[:10],
                "missing_homography_patterns": missing_patterns,
                "geometry_source": geometry_source,
                "ok": exposure_ok,
            }
        result["positions"][label] = position_report
    return result


def select_stems(index: ExposureIndex, position_label: str, args: argparse.Namespace) -> list[str]:
    stems = [stem for stem in index.order if stem in index.paths]
    if args.only_stems:
        requested = {normalize_stem(value) for value in args.only_stems}
        stems = [stem for stem in stems if stem in requested]
        missing = sorted(requested - set(stems))
        if missing:
            raise SystemExit(f"requested stems missing from {position_label}: {missing}")
    if args.limit is not None:
        stems = stems[: args.limit]
    if not stems:
        raise SystemExit(f"no target stems selected for {position_label}")
    return stems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--sim-root", "--sim-dir", dest="sim_root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--lut", type=Path, default=DEFAULT_LUT_PATH)
    parser.add_argument("--exposures", nargs="+", type=int, default=list(DEFAULT_EXPOSURES))
    parser.add_argument("--anchor-exposure", type=int, default=1000)
    parser.add_argument(
        "--geometry-exposure",
        "--affine-exposure",
        dest="geometry_exposure",
        type=int,
        default=None,
        help=(
            "single exposure used to fit each z-specific transform; the matrix is "
            "shared by all exposures (default: median --exposures, normally 1000 us)"
        ),
    )
    parser.add_argument(
        "--z-positions",
        nargs="*",
        default=None,
        help="optional z subset, e.g. 0 -0.5 +0.5 (default: discover all)",
    )
    parser.add_argument(
        "--homography-patterns",
        nargs="+",
        default=list(DEFAULT_HOMOGRAPHY_PATTERNS),
        help="same-stem 4-dot pattern names; file extensions are optional",
    )
    parser.add_argument("--transform", choices=("affine", "homography"), default="affine")
    parser.add_argument("--output-size", type=int, default=1024)
    parser.add_argument("--ransac-threshold", type=float, default=1.5)
    parser.add_argument("--max-calibration-error", type=float, default=2.0,
                        help="reject ANY calibration point above this output-pixel error")
    parser.add_argument("--min-calibration-inlier-fraction", type=float, default=.75)
    parser.add_argument("--max-affine-axis-ratio", type=float, default=1.5)
    parser.add_argument("--core-guided-window", type=int, default=6)

    parser.add_argument("--raw-width", type=int, default=RAW_WIDTH)
    parser.add_argument("--raw-height", type=int, default=RAW_HEIGHT)
    parser.add_argument("--png-dn-scale", type=float, default=16.0)
    parser.add_argument("--saturation-dn", type=float, default=4088.0)
    parser.add_argument("--low-dn", type=float, default=0.0,
        help="lower DN taper threshold; low-signal pixels remain measured, with fallback if necessary")
    parser.add_argument("--low-full-weight-dn", type=float, default=128.0)
    parser.add_argument("--high-full-weight-dn", type=float, default=3600.0)
    parser.add_argument("--minimum-output-weight", type=float, default=1e-6)
    parser.add_argument("--nd-transmission", type=float, default=1.0)
    parser.add_argument("--keep-negative", action="store_true")

    parser.add_argument("--reference-gain-mode", choices=("raw", "linearized", "logged"), default="raw",
        help="raw jointly fits registration/gain from pixelwise-linearized reference RAW (default); other modes are legacy")
    parser.add_argument("--reference-gain-low-dn", type=float, default=64.0)
    parser.add_argument("--reference-gain-min-pixels", type=int, default=100)
    parser.add_argument("--reference-gain-max-nrmse", type=float, default=.30)
    parser.add_argument("--hdr-weighting", choices=("exposure", "dn"), default="exposure")
    parser.add_argument("--invalid-output", choices=("nan", "zero"), default="nan",
        help="truly unmeasured pixels: NaN (default) or zero placeholder; ALWAYS use the saved valid_mask for training")
    parser.add_argument("--opencv-threads", type=int, default=4)
    parser.add_argument("--missing-reference", choices=("drop", "identity", "error"), default="error")
    parser.add_argument("--max-reference-gain", type=float, default=20.0)
    parser.add_argument("--disable-inter-exposure-scale", action="store_true")
    parser.add_argument(
        "--reference-anchor-pattern",
        default=None,
        help="fixed reference RAW stem (default: first homography pattern)",
    )
    parser.add_argument("--disable-reference-registration", action="store_true")
    parser.add_argument(
        "--missing-reference-registration",
        choices=("error", "identity"),
        default="error",
    )
    parser.add_argument(
        "--reference-registration-channel",
        choices=("blue", "green", "red", "gray", "max"),
        default="blue",
    )
    parser.add_argument(
        "--registration-method", choices=("ecc", "gradcorr", "phasecorr"), default="ecc"
    )
    parser.add_argument("--registration-saturation-dn", type=float, default=None)
    parser.add_argument("--registration-dark-dn", type=float, default=None)
    parser.add_argument("--registration-mask-dilate-radius", type=int, default=0)
    parser.add_argument("--registration-blur-sigma", type=float, default=1.0)
    parser.add_argument("--registration-highpass-sigma", type=float, default=0.0)
    parser.add_argument("--registration-max-shift", type=float, default=50.0)
    parser.add_argument("--registration-downsample", type=int, default=2)
    parser.add_argument("--registration-roi-size", type=int, default=768)
    parser.add_argument("--registration-min-correlation", type=float, default=.65)
    parser.add_argument(
        "--registration-trend-window",
        type=int,
        default=21,
        help="rolling-median window for dx/dy trend lines (default: 21 masks)",
    )
    parser.add_argument(
        "--no-registration-trends",
        dest="write_registration_trends",
        action="store_false",
        help="do not write per-z reference-shift CSV/PNG diagnostics",
    )
    parser.add_argument(
        "--registration-trends-only",
        action="store_true",
        help="compute per-z shift CSV/PNG without LUT, simulations, homography, or HDR",
    )
    parser.set_defaults(write_registration_trends=True)
    parser.add_argument("--dark-root", type=Path, default=None)
    parser.add_argument("--flat-root", type=Path, default=None)
    parser.add_argument(
        "--subtract-lut-zero", dest="subtract_lut_zero", action="store_true",
        help="explicitly treat LUT(min DN) as a subtractable dark offset when no dark frames exist",
    )
    parser.add_argument(
        "--no-lut-zero-subtraction",
        dest="subtract_lut_zero",
        action="store_false",
        help="preserve calibrated LUT values when no dark frames exist (default)",
    )
    parser.set_defaults(subtract_lut_zero=False)

    parser.add_argument("--output-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument(
        "--output-channels",
        choices=("rgb", "blue", "green", "red", "gray", "max"),
        default="blue",
        help="blue (default) writes 1024x1024; rgb explicitly writes CHW RGB",
    )
    parser.add_argument("--write-preview-png", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-stems", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validate-input-only",
        action="store_true",
        help="read-only structure validation; does not require LUT/simulations or write files",
    )

    parser.add_argument(
        "--camera-channel",
        choices=("blue", "green", "red", "gray", "max"),
        default="blue",
        help="camera plane used for geometry; RAW defaults to the Bayer blue plane only",
    )
    parser.add_argument("--camera-seed-sigma", type=float, default=3.0)
    parser.add_argument("--camera-sigma", type=float, default=1.5)
    parser.add_argument("--camera-win", type=int, default=12)
    parser.add_argument(
        "--sim-channel",
        choices=("blue", "green", "red", "gray", "max"),
        default="blue",
        help=(
            "simulation plane used for geometry; blue selects plane 0 of the "
            "supplied CHW NPY, which is the positive intensity plane"
        ),
    )
    parser.add_argument("--sim-seed-sigma", type=float, default=1.5)
    parser.add_argument("--sim-sigma", type=float, default=1.5)
    parser.add_argument("--sim-win", type=int, default=12)
    parser.add_argument(
        "--core-locator",
        choices=("pattern", "auto", "quadrants", "peaks"),
        default="pattern",
        help=(
            "pattern uses a000 quadrants and a045 cardinal sectors inside a central ROI"
        ),
    )
    parser.add_argument(
        "--core-roi-fraction",
        type=float,
        default=0.60,
        help="centered fraction of each axis searched for checkerboard cores (default: 0.60)",
    )
    parser.add_argument(
        "--core-center-sigma",
        type=float,
        default=80.0,
        help=(
            "fallback Gaussian sigma for the four-core envelope center when four "
            "compact center candidates cannot be found; never used for coordinates"
        ),
    )
    parser.add_argument(
        "--core-min-separation",
        type=float,
        default=8.0,
        help="reject a calibration pattern if two refined cores are closer than this",
    )
    parser.add_argument(
        "--core-center-exclusion-radius",
        type=float,
        default=10.0,
        help="ignore peaks this many pixels from the estimated pattern center",
    )
    parser.add_argument(
        "--core-sector-guard",
        type=float,
        default=4.0,
        help="leave this gap between adjacent topology sectors to prevent peak leakage",
    )
    parser.add_argument(
        "--core-candidate-min-distance",
        type=int,
        default=12,
        help="minimum separation used while estimating the four-core pattern center",
    )
    parser.add_argument("--peak-min-distance", type=int, default=None)
    parser.add_argument("--peak-margin-fraction", type=float, default=0.06)
    parser.add_argument("--peak-candidates", type=int, default=32)
    parser.add_argument("--peak-background-sigma", type=float, default=None)
    parser.add_argument("--max-pattern-fraction", type=float, default=0.65)
    parser.add_argument("--pattern-center-tolerance", type=float, default=0.85)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.exposures = [int(value) for value in args.exposures]
    if not args.exposures or any(e <= 0 for e in args.exposures):
        raise SystemExit("exposures must be positive")
    if (args.reference_gain_mode == "raw" and
            (args.output_channels not in ("blue", "red")
             or args.reference_registration_channel != args.output_channels)):
        raise SystemExit("raw reference gain requires matching blue/red native output and registration channels")
    if (args.reference_gain_min_pixels < 20 or args.reference_gain_max_nrmse <= 0
            or args.reference_gain_low_dn < 0 or args.reference_gain_low_dn >= args.saturation_dn):
        raise SystemExit("invalid reference gain quality thresholds")
    if args.registration_roi_size <= 2*args.registration_max_shift+16:
        raise SystemExit("registration ROI must exceed twice the maximum shift plus 16 pixels")
    if not 0 <= args.registration_min_correlation <= 1 or args.opencv_threads < 1 or args.core_guided_window < 2:
        raise SystemExit("invalid registration/core settings")
    if (not 0 < args.min_calibration_inlier_fraction <= 1 or args.max_calibration_error <= 0
            or args.max_affine_axis_ratio < 1 or args.max_reference_gain <= 1):
        raise SystemExit("invalid calibration/gain thresholds")
    if len(set(args.exposures)) != len(args.exposures):
        raise SystemExit("--exposures contains duplicates")
    if args.geometry_exposure is None:
        args.geometry_exposure = sorted(args.exposures)[len(args.exposures) // 2]
    args.geometry_exposure = int(args.geometry_exposure)
    if args.anchor_exposure not in args.exposures:
        raise SystemExit("--anchor-exposure must be one of --exposures")
    if args.geometry_exposure not in args.exposures:
        raise SystemExit("--geometry-exposure must be one of --exposures")
    if args.output_size <= 0 or args.ransac_threshold <= 0:
        raise SystemExit("output size and RANSAC threshold must be positive")
    if (
        args.raw_width <= 0
        or args.raw_height <= 0
        or args.raw_width % 2
        or args.raw_height % 2
    ):
        raise SystemExit("RAW width and height must be positive even integers")
    if args.png_dn_scale <= 0 or args.nd_transmission <= 0:
        raise SystemExit("PNG DN scale and ND transmission must be positive")
    if not (args.low_dn < args.low_full_weight_dn < args.high_full_weight_dn < args.saturation_dn):
        raise SystemExit(
            "require low_dn < low_full_weight_dn < high_full_weight_dn < saturation_dn"
        )
    if not np.isfinite(args.minimum_output_weight) or args.minimum_output_weight < 0:
        raise SystemExit("minimum output weight must be finite and non-negative")
    if args.peak_candidates < 4:
        raise SystemExit("--peak-candidates must be at least 4")
    if not 0.0 < args.core_roi_fraction <= 1.0:
        raise SystemExit("--core-roi-fraction must be in (0, 1]")
    if (
        args.core_center_sigma <= 0
        or args.core_min_separation <= 0
        or args.core_center_exclusion_radius < 0
        or args.core_sector_guard < 0
        or args.core_candidate_min_distance < 1
    ):
        raise SystemExit(
            "core center sigma/minimum separation must be positive and center "
            "exclusion radius non-negative"
        )
    if (
        args.camera_seed_sigma <= 0
        or args.camera_sigma <= 0
        or args.sim_seed_sigma <= 0
        or args.sim_sigma <= 0
        or args.camera_win < 2
        or args.sim_win < 2
    ):
        raise SystemExit("core sigmas must be positive and refinement windows at least 2 px")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if not args.homography_patterns:
        raise SystemExit("at least one --homography-patterns value is required")
    if args.registration_mask_dilate_radius < 0:
        raise SystemExit("registration mask dilation radius must be non-negative")
    if args.registration_blur_sigma < 0 or args.registration_highpass_sigma < 0:
        raise SystemExit("registration Gaussian sigmas must be non-negative")
    if args.registration_max_shift <= 0 or args.registration_downsample < 1:
        raise SystemExit("registration max shift/downsample must be positive")
    if args.registration_trend_window < 1:
        raise SystemExit("--registration-trend-window must be positive")
    if args.registration_trends_only and args.disable_reference_registration:
        raise SystemExit(
            "--registration-trends-only cannot be combined with "
            "--disable-reference-registration"
        )
    if args.z_positions:
        try:
            args.z_positions = [parse_z_selector(value) for value in args.z_positions]
        except ValueError as error:
            raise SystemExit(str(error)) from error


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_root = args.input_root.resolve()
    sim_root = (args.sim_root or (input_root / "Forward_sim")).resolve()
    output_root = (args.output_root or (input_root / "Aligned_camera_v3")).resolve()
    if output_root == input_root:
        raise SystemExit("output root must not be the raw input root")
    started = time.perf_counter()
    print(f"Input root:  {input_root}")
    print(f"Simulation:  {sim_root}")
    print(f"Output root: {output_root}")
    print(f"Exposures:   {args.exposures} us")
    print(f"Geometry:    Exp{args.geometry_exposure} shared within each z position")
    discovered = discover_exposure_folders(input_root, args.exposures, args.z_positions)
    print(
        "Z positions: "
        + ", ".join(
            f"{position.label} ({position.value_mm:+g} mm)"
            for position, _ in discovered.values()
        )
    )
    if args.validate_input_only:
        validation = validate_input_layout(discovered, args)
        for label, position_report in validation["positions"].items():
            print(f"{label}: common stems={position_report['common_stem_count']}")
            for exposure_us, exposure_report in position_report["exposures"].items():
                state = "OK" if exposure_report["ok"] else "FAILED"
                print(
                    f"  Exp{exposure_us}: {state}; main={exposure_report['main_count']}, "
                    f"reference={exposure_report['reference_count']}, "
                    f"CSV={exposure_report['csv_stem_count']}, "
                    f"bad-size={exposure_report['bad_raw_size_count']}"
                )
        print(f"Input validation: {'PASS' if validation['ok'] else 'FAIL'}")
        if not validation["ok"]:
            print(json.dumps(validation, indent=2, ensure_ascii=False))
            raise SystemExit(2)
        return

    missing_dependencies = []
    if CV2_IMPORT_ERROR is not None:
        missing_dependencies.append("opencv-python")
    if SCIPY_IMPORT_ERROR is not None:
        missing_dependencies.append("scipy")
    trend_png_requested = args.registration_trends_only or (
        args.write_registration_trends and not args.dry_run
    )
    if trend_png_requested and importlib.util.find_spec("matplotlib") is None:
        missing_dependencies.append("matplotlib")
    if missing_dependencies:
        raise SystemExit(
            "missing Python dependencies for processing: " + ", ".join(missing_dependencies)
        )
    cv2.setNumThreads(args.opencv_threads)

    if args.registration_trends_only:
        trend_root = output_root / "registration_trends"
        trend_report: dict[str, Any] = {
            "input_root": str(input_root),
            "output_root": str(trend_root),
            "registration_method": args.registration_method,
            "registration_channel": args.reference_registration_channel,
            "smoothing_window_masks": args.registration_trend_window,
            "positions": {},
        }
        for label, (position, folders) in discovered.items():
            exposure_indices = {
                exposure_us: build_exposure_index(folder, exposure_us, position)
                for exposure_us, folder in folders.items()
            }
            reference_registrations = {
                exposure_us: build_reference_registration_model(index, args, exposure_indices[args.geometry_exposure])
                for exposure_us, index in exposure_indices.items()
            }
            stems = select_stems(
                exposure_indices[args.anchor_exposure], label, args
            )
            print(
                f"Computing reference registration trends for {label}: "
                f"{len(stems)} masks x {len(args.exposures)} exposures..."
            )
            rows = collect_registration_trend_rows(
                position,
                stems,
                exposure_indices,
                reference_registrations,
                args,
            )
            outputs = save_registration_trend_outputs(
                trend_root,
                position,
                rows,
                args.registration_trend_window,
            )
            trend_report["positions"][label] = {
                "z_position_mm": position.value_mm,
                "selected_stem_count": len(stems),
                "outputs": outputs,
                "reference_registration_summary": {
                    str(exposure_us): summarize_reference_registration(model)
                    for exposure_us, model in reference_registrations.items()
                },
            }
            print(f"  CSV: {outputs['csv']}")
            print(f"  PNG: {outputs['png']}")
        report_path = trend_root / "registration_trend_report.json"
        atomic_write_json(report_path, trend_report)
        print(f"Registration trend generation complete: {report_path}")
        return

    lut = load_response_lut(args.lut)
    print(f"LUT: {lut.source}  DN=[{lut.min_dn:g}, {lut.max_dn:g}] rows={len(lut.dn)}")
    setup_report: dict[str, Any] = {
        "algorithm_version": "raw-reference-hdr-v3",
        "output_units": "LUT units / second / ND transmission, per-z reference anchor power",
        "invalid_output": args.invalid_output,
        "hdr_weighting": args.hdr_weighting,
        "low_signal_policy": "retain DN=0; longest effective exposure fallback if total HDR weight is too small",
        "spatial_interpolation": "bilinear radiance before HDR weighting; require full hard-valid footprint",
        "quality_sidecar": {
            "path": "<z>/quality/<stem>.npz",
            "array_order": "same shape and channel order as camera NPY",
            "valid_mask": "bool: a complete unsaturated in-range interpolation footprint exists",
            "confidence": "float32 [0,1]: maximum supported interpolated DN taper; NOT calibrated SNR",
            "fallback_mask": "bool: used one supported exposure because total HDR weight was too small",
        },
        "reference_anchor_scope": "one shared geometry-exposure anchor per z for ALL exposures",
        "script": str(Path(__file__).resolve()),
        "input_root": str(input_root),
        "simulation_root": str(sim_root),
        "output_root": str(output_root),
        "lut": {
            "path": lut.source,
            "rows": len(lut.dn),
            "min_dn": lut.min_dn,
            "max_dn": lut.max_dn,
        },
        "exposures_us": list(args.exposures),
        "anchor_exposure_us": args.anchor_exposure,
        "geometry_exposure_us": args.geometry_exposure,
        "reference_gain_mode": args.reference_gain_mode,
        "reference_registration": {
            "enabled": not args.disable_reference_registration,
            "channel": args.reference_registration_channel,
            "method": args.registration_method,
            "mask_dilate_radius": args.registration_mask_dilate_radius,
            "blur_sigma": args.registration_blur_sigma,
            "highpass_sigma": args.registration_highpass_sigma,
            "max_expected_shift_px": args.registration_max_shift,
            "downsample": args.registration_downsample,
            "trend_outputs_enabled": args.write_registration_trends,
            "trend_rolling_median_window_masks": args.registration_trend_window,
        },
        "transform_model": args.transform,
        "transform_sharing": "one z-specific matrix shared by all exposures",
        "core_localization": {
            "locator": args.core_locator,
            "camera_channel": args.camera_channel,
            "simulation_channel": args.sim_channel,
            "roi_fraction": args.core_roi_fraction,
            "center_sigma_px": args.core_center_sigma,
            "camera_seed_sigma_px": args.camera_seed_sigma,
            "camera_refine_sigma_px": args.camera_sigma,
            "camera_refine_half_window_px": args.camera_win,
            "simulation_seed_sigma_px": args.sim_seed_sigma,
            "simulation_refine_sigma_px": args.sim_sigma,
            "simulation_refine_half_window_px": args.sim_win,
            "minimum_core_separation_px": args.core_min_separation,
            "center_exclusion_radius_px": args.core_center_exclusion_radius,
            "sector_guard_px": args.core_sector_guard,
            "candidate_min_distance_px": args.core_candidate_min_distance,
        },
        "homography_pattern_names": [normalize_stem(name) for name in args.homography_patterns],
        "saturation_dn": args.saturation_dn,
        "raw_shape": [args.raw_height, args.raw_width],
        "bayer_working_shape": (
            [args.raw_height // 2, args.raw_width // 2, 3]
            if args.output_channels == "rgb"
            else [args.raw_height // 2, args.raw_width // 2]
        ),
        "png_dn_scale": args.png_dn_scale,
        "nd_transmission": args.nd_transmission,
        "subtract_lut_zero_without_dark": args.subtract_lut_zero,
        "output_size": args.output_size,
        "output_dtype": args.output_dtype,
        "output_channels": args.output_channels,
        "output_array_order": "CHW_RGB" if args.output_channels == "rgb" else "HW",
        "missing_reference_policy": args.missing_reference,
        "positions": {},
    }

    total_written = 0
    skipped_existing = 0
    failures: list[dict[str, str]] = []
    for label, (position, folders) in discovered.items():
        print(f"\n=== {label} ({position.value_mm:+g} mm) ===")
        sim_dir = resolve_simulation_dir(sim_root, position)
        if not sim_dir.is_dir():
            raise FileNotFoundError(
                f"simulation folder for {label} does not exist: {sim_dir}. "
                "Use --validate-input-only until z-specific simulations are ready."
            )
        z_output = output_root / label
        exposure_indices = {
            exposure_us: build_exposure_index(folder, exposure_us, position)
            for exposure_us, folder in folders.items()
        }
        reference_models: dict[int, ReferenceModel] = {}
        reference_registrations: dict[int, ReferenceRegistrationModel] = {}
        radiometric: dict[int, RadiometricCalibration] = {}
        transforms: dict[int, np.ndarray] = {}
        transform_reports: dict[str, Any] = {}
        for exposure_us in args.exposures:
            index = exposure_indices[exposure_us]
            if args.reference_gain_mode == "raw":
                reference_models[exposure_us] = ReferenceModel(exposure_us,"raw",{},1.0,0,0,1.0,1.0,1.0)
            else:
                reference_models[exposure_us] = load_reference_model(
                    index.folder / "reference_gain_log.csv", exposure_us, lut,
                    args.reference_gain_mode, args.max_reference_gain)
            reference_registrations[exposure_us] = build_reference_registration_model(
                index, args, exposure_indices[args.geometry_exposure]
            )
            radiometric[exposure_us] = load_radiometric_calibration(
                exposure_us,
                args.dark_root,
                args.flat_root,
                lut,
                args.png_dn_scale,
                args.raw_width,
                args.raw_height,
                args.output_channels,
                args.subtract_lut_zero,
            )
        geometry_exposure = args.geometry_exposure
        geometry_index = exposure_indices[geometry_exposure]
        if args.reference_gain_mode == "raw":
            for pattern in args.homography_patterns:
                reference_gain_from_raw(reference_registrations[geometry_exposure], geometry_index,
                    normalize_stem(pattern),lut,args,radiometric[geometry_exposure],radiometric[geometry_exposure])
        print(
            f"Calibrating one shared {args.transform} at {label} from "
            f"Exp{geometry_exposure} and {len(args.homography_patterns)} "
            "checkerboard pattern(s)..."
        )
        matrix, calibration_report = calibrate_exposure_transform(
            geometry_index,
            sim_dir,
            args.homography_patterns,
            reference_registrations[geometry_exposure],
            args,
        )
        transforms = {
            exposure_us: matrix.copy() for exposure_us in args.exposures
        }
        transform_reports = {
            "geometry_exposure_us": geometry_exposure,
            "applied_to_exposures_us": list(args.exposures),
            "shared_matrix": matrix.astype(float).tolist(),
            "calibration": calibration_report,
        }
        residual = calibration_report["residual"]
        print(
            f"  shared {args.transform} RMSE={residual['rmse_px']:.4f}px, "
            f"max={residual['max_px']:.4f}px, "
            f"inliers={calibration_report['inlier_count']}/{calibration_report['point_count']}; "
            f"applying to exposures {args.exposures}"
        )
        reference_radiance = {
            exposure_us: reference_models[exposure_us].target_linear / (exposure_us * 1e-6)
            for exposure_us in args.exposures
        }
        if args.disable_inter_exposure_scale or args.reference_gain_mode == "raw":
            static_scales = {exposure_us: 1.0 for exposure_us in args.exposures}
        else:
            anchor_level = reference_radiance[args.anchor_exposure]
            static_scales = {
                exposure_us: anchor_level / reference_radiance[exposure_us]
                for exposure_us in args.exposures
            }
        if args.reference_gain_mode == "raw":
            print("Reference power: joint RAW fits to the shared anchor; no CSV/static rescaling")
        else:
            for exposure_us in args.exposures:
                print(f"Legacy Exp{exposure_us}: static scale={static_scales[exposure_us]:.6g}")
        anchor_index = exposure_indices[args.anchor_exposure]
        stems = select_stems(anchor_index, label, args)
        frame_reports: list[dict[str, Any]] = []
        trend_outputs: dict[str, str] | None = None
        if not args.dry_run:
            z_output.mkdir(parents=True, exist_ok=True)
            for number, stem in enumerate(stems, start=1):
                output_path = z_output / f"{stem}.npy"
                quality_path = z_output / "quality" / f"{stem}.npz"
                if output_path.exists() and not args.overwrite:
                    if not quality_path.is_file():
                        message = "existing camera output has no quality sidecar; rerun with --overwrite or a new --output-root"
                        failures.append({"z": label, "stem": stem, "error": message})
                        print(f"[{label} {number}/{len(stems)}] {message}", file=sys.stderr)
                        continue
                    skipped_existing += 1
                    print(f"[{label} {number}/{len(stems)}] skip existing {output_path.name}")
                    continue
                try:
                    frame_started = time.perf_counter()
                    quality_arrays: dict[str, np.ndarray] = {}
                    final, frame_report = process_one_stem(
                        stem,
                        exposure_indices,
                        reference_models,
                        reference_registrations,
                        transforms,
                        radiometric,
                        static_scales,
                        lut,
                        args,
                        quality=quality_arrays,
                    )
                    frame_report["processing_seconds"] = time.perf_counter() - frame_started
                    atomic_save_npy(output_path, final)
                    atomic_save_quality(quality_path, quality_arrays)
                    frame_report["quality_path"] = str(quality_path)
                    frame_report["processing_and_save_seconds"] = time.perf_counter() - frame_started
                    if args.write_preview_png:
                        save_preview_png(z_output / f"{stem}.png", final.astype(np.float32))
                    frame_reports.append(frame_report)
                    total_written += 1
                    print(
                        f"[{label} {number}/{len(stems)}] {output_path.name}  "
                        f"exp={frame_report['used_exposures']}  "
                        f"valid={100.0 * frame_report['output_valid_fraction']:.3f}%"
                    )
                except Exception as error:  # preserve batch failures in the report
                    failures.append({"z": label, "stem": stem, "error": repr(error)})
                    print(
                        f"[{label} {number}/{len(stems)}] FAILED {stem}: {error!r}",
                        file=sys.stderr,
                    )
            if args.write_registration_trends:
                trend_rows = collect_registration_trend_rows(position,stems,exposure_indices,
                    reference_registrations,args,cached_only=True)
                if trend_rows:
                    trend_outputs = save_registration_trend_outputs(output_root / "registration_trends",
                        position,trend_rows,args.registration_trend_window)
        setup_report["positions"][label] = {
            "z_position_mm": position.value_mm,
            "simulation_dir": str(sim_dir),
            "output_dir": str(z_output),
            "selected_stem_count": len(stems),
            "written_count": len(frame_reports),
            "registration_trend_outputs": trend_outputs,
            "reference_models": ({
                str(e): {"mode": "raw", "anchor_exposure_us": model.fixed_exposure_us,
                         "anchor_path": str(model.fixed_path),
                         "gain_source": "reference_frame_diagnostics", "static_scale": 1.0}
                for e, model in reference_registrations.items()
            } if args.reference_gain_mode == "raw" else {
                str(exposure_us): {
                    "valid_rows": model.valid_rows,
                    "rejected_rows": model.rejected_rows,
                    "target_linear": model.target_linear,
                    "gain_min": model.gain_min,
                    "gain_median": model.gain_median,
                    "gain_max": model.gain_max,
                    "reference_radiance": reference_radiance[exposure_us],
                    "static_scale": static_scales[exposure_us],
                }
                for exposure_us, model in reference_models.items()
            }),
            "reference_registration_summary": {
                str(exposure_us): summarize_reference_registration(model)
                for exposure_us, model in reference_registrations.items()
            },
            "reference_frame_diagnostics": {
                str(exposure_us): model.reports for exposure_us, model in reference_registrations.items()
            },
            "radiometric_calibration": {
                str(exposure_us): {
                    "dark_frames": calibration.dark_frames,
                    "flat_frames": calibration.flat_frames,
                    "dark_applied": calibration.dark is not None,
                    "flat_applied": calibration.flat is not None,
                }
                for exposure_us, calibration in radiometric.items()
            },
            "transforms": transform_reports,
            "frames": frame_reports,
        }
    setup_report.update(
        {
            "dry_run": bool(args.dry_run),
            "written_count": total_written,
            "skipped_existing_count": skipped_existing,
            "failure_count": len(failures),
            "failures": failures,
            "frames": frame_reports,
            "elapsed_s": time.perf_counter() - started,
        }
    )
    report_path = output_root / "preprocessing_report.json"
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_path, setup_report)
    print(
        f"Complete: wrote {total_written}, skipped {skipped_existing}, "
        f"failed {len(failures)}. Report: {report_path}"
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
