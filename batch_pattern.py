"""
Batch patterning + capture (473 nm only).

For every phase_mask_*.npy in MASK_FOLDER:
    1. Randomize the exposure order to reduce systematic laser-intensity drift.
    2. Load up to CHUNK_SIZE .npy files into one temporary MaskStack.
    3. Upload that stack to SLM memory slots 1..N.
    4. Start one long exposure/playback run for the chunk.
    5. Capture one Basler frame per SLM frame and save it as a PNG with the SAME
       basename as the phase mask (phase_mask_dot.npy -> phase_mask_dot.npy
       in npy_raw16 mode, or phase_mask_dot.png in png mode).
    6. Upload the next chunk over the same SLM memory slots and continue.

The on-disk training pool stays as individual .npy files. Only the in-memory
stack is chunked. The Basler, SLM, and Arduino are opened once for the run.

Optional reference-gain tracking inserts a fixed reference phase immediately
before every target phase. Reference images are reduced to a robust scalar in
memory and are not saved. The resulting temporal correction factors are written
to reference_gain_log.csv.
"""

import os
import glob
import time
import csv
import ctypes
import json
from datetime import datetime

import numpy as np
import cv2  # only used for PNG writing
from pypylon import pylon

import hololith
import hololith.SLM.slmcontrol as slmcontrol
import hololith.SLM._slm_win as SLMLib


# ============================================================
# 1. USER CONFIGURATION  (the only cell you normally edit)
# ============================================================

CONFIG_PATH = r'G:\Shared drives\taylorlab\3DHL\Codebase\3DHL-Python-Control\hololith\util\setting.yaml'

# Folder that holds the phase_mask_*.npy files to expose in sequence.
#MASK_FOLDER = r'G:\Shared drives\taylorlab\3DHL\Phase masks\optimized_with_mbvam\sweep_p8'
#MASK_FOLDER = r'G:\Shared drives\taylorlab\3DHL\Phase masks\optimized_with_mbvam\linearity_masks'
#MASK_FOLDER = r'G:\Shared drives\taylorlab\3DHL\Phase masks\optimized_with_mbvam\LEE NEW EXPSplayback_sequence'
#MASK_FOLDER = r'C:\CITL\LEE NEW EXPSplayback_sequence'
#MASK_FOLDER = r'C:\CITL\Refonly_multiples'
#MASK_FOLDER = r'C:\CITL\TM_NEW_Alignment_smallerset'
#MASK_FOLDER = r'G:\Shared drives\taylorlab\3DHL\CITL\Fourier Neural Operator_Training phase masks\test_sine_8pixel'
#MASK_FOLDER = r'G:\Shared drives\taylorlab\3DHL\Phase masks\optimized_with_mbvam\Pinhole test for TM'
#MASK_FOLDER = r'C:\CITL\LEE NEW EXPSplayback_sequence'
MASK_FOLDER = r'G:\Shared drives\3DHL + OptiCAL + TFT + SpaceCAL\3DHL\FNO\0.Phase_Masks'

# Where to write the captured frames. Defaults to a "captures" subfolder of MASK_FOLDER.
#OUTPUT_FOLDER = os.path.join(MASK_FOLDER, 'captures')
OUTPUT_FOLDER = r'C:\CITL\FNO_Training\08_14_2026_whole_z7.1mm\exposure30'
#OUTPUT_FOLDER = r'G:\Shared drives\taylorlab\3DHL\CITL\TM_16STEPS_31JULY_PWM255_30US'
#OUTPUT_FOLDER = r'G:\Shared drives\taylorlab\3DHL\CITL\TM_PinholeTest_5AUGUST_PWM255_300US_withWC'
#OUTPUT_FOLDER = r'C:\CITL\LEE_NEW_ALIGNMENT\TM_081026_PWM255_30US_allmodes'



# Glob pattern for the masks inside MASK_FOLDER.
MASK_GLOB = '*.npy'

# Optional wavefront-correction CSV. None to skip.
WFC_FILE_PATH = r'G:\Shared drives\taylorlab\3DHL\Phase masks\wavefront correction\520nm_wf_correction.csv'

# ---- Exposure parameters (single wavelength, beam 1 = 473 nm) ----
EXPOSURE_PARAMS = dict(
    pwm_1=255,          # 473 nm laser PWM (0-255)
    pwm_2=0,            # ignored in single-wavelength mode
    fps=4,              # legacy single-mask fps; chunk playback derives fps from duration_ms
    duration_ms=2500,   # per-mask hold time; chunk exposure uses duration_ms * N
    suppress_ms=50,
)

# ---- Camera ----
# Auto-detect the first Basler if SERIAL is None; otherwise pin to a serial number.
CAMERA_SERIAL = None
# Camera exposure time for the grab, in microseconds. Tune so the target is
# bright but not saturated. The 473 nm sample frame was badly underexposed at
# 20 ms, so start at 60 ms and tune: aim for brightest pixels ~3000-3800 / 4095
# (the per-frame min/max/sat% printed below tells you exactly where you are).
CAMERA_EXPOSURE_US = 30
PIXEL_FORMAT = 'BayerRG12'   # color raw mosaic, debayered to 16-bit RGB on save

# OpenCV debayer constant. NOTE: Basler's Bayer naming is offset by one vs OpenCV.
# Basler "BayerRG12" usually debayers correctly with COLOR_BayerRG2BGR here, but if
# your colors look wrong (red/blue swapped or greenish), try the alternatives below:
#   cv2.COLOR_BayerRG2BGR   <- default for Basler BayerRG
#   cv2.COLOR_BayerBG2BGR
#   cv2.COLOR_BayerGR2BGR
#   cv2.COLOR_BayerGB2BGR
DEBAYER_CODE = cv2.COLOR_BayerRG2BGR

# ---- Capture output ----
# 'npy_raw16' is fastest and preserves the camera's raw uint16 Bayer frame
# exactly. Use 'png' only when you need directly viewable images.
# Options: 'npy_raw16', 'raw16', 'png'
CAPTURE_SAVE_FORMAT = 'raw16'
RAW16_WRITE_SIDECAR_JSON = False #True

# PNG-specific options. PNG compression is lossless, but higher compression is
# slower. The gray modes debayer first so the Bayer mosaic is not saved as
# pore-like grayscale texture.
# Options: 'gray8_debayered', 'gray16_debayered', 'bgr16'
PNG_SAVE_MODE = 'gray8_debayered'
PNG_COMPRESSION = 0

# Fraction of the exposure to wait before grabbing (0.5 = middle of exposure).
CAPTURE_AT_FRACTION = 0.5

# ---- Chunked SLM playback ----
# SLM memory supports locations 1..128. Start with 64 to keep RAM/upload behavior
# conservative; raise toward 128 after the timing looks stable.
CHUNK_SIZE = 60
SLM_MEMORY_CAPACITY = 128

# Randomizing exposure order helps decorrelate captures from slow laser drift.
# Filenames and output paths are unchanged. Set RANDOM_SEED to an int to replay
# the exact same random order.
RANDOMIZE_ORDER = False
RANDOM_SEED = None

# The hololith defaults poll SLM readiness at 0.2 Hz, which can add 5 s after a
# busy response. These patches only affect this script's process.
FAST_SLM_READY_CHECK_HZ = 20.0
FAST_SLM_UPLOAD_POLL_S = 0.05

# Correctness-first mode: after a chunk is uploaded, each SLM memory slot is
# displayed manually. The next phase is not displayed until the current PNG
# write has succeeded.
MANUAL_ADVANCE_AFTER_SAVE = True
MANUAL_SLM_SETTLE_S = 0.30
MANUAL_LASER_SETTLE_S = 0.10
MANUAL_CAPTURE_DELAY_S = None  # None -> duration_ms * CAPTURE_AT_FRACTION
CAMERA_STALE_FRAMES_TO_FLUSH = 2

# The Arduino command parser is most reliable through ArduinoController.writeCommand,
# even though it blocks on the serial read timeout. Use this correctness-first path
# for manual laser on/off.
MANUAL_LASER_TIMING_MODE = 'run_static'  # 'per_frame', 'chunk_static', or 'run_static'
MANUAL_LASER_USE_BLOCKING_COMMAND = None  # legacy override: True -> blocking, False -> nonblocking
MANUAL_LASER_COMMAND_MODE = 'blocking'  # 'blocking', 'short_timeout', or 'nonblocking'
MANUAL_LASER_SHORT_TIMEOUT_S = 0.10

# Print per-image timing so we can see whether the bottleneck is Arduino, camera,
# saving, or SLM display.
PROFILE_TIMING = True

# ---- Loop pacing ----
INTER_CHUNK_PAUSE_S    = 2.0   # now applied between uploaded chunks
POST_PATTERN_BUFFER_S = 0.5
OVERWRITE_EXISTING    = True   # if False, skip masks whose PNG already exists
VERBOSE               = True

# Restrict / reorder files. None = every match, sorted. Or list of basenames.
# To test a SINGLE mask, set this to a one-element list, e.g.:
#   MASK_FILES_OVERRIDE = ['phase_mask_a7f3e9.npy']
MASK_FILES_OVERRIDE = None

# Burn the mask name + frame index into the top-left corner of each saved PNG,
# so a capture is identifiable even if its filename is ever lost. The raw pixel
# data outside the small text box is untouched.
STAMP_NAME_ON_IMAGE = True

# Write a capture_log.csv (index, mask name, output name, min, max, saturated %, OK/FAIL).
WRITE_CSV_LOG = True

# ---- Optional temporal intensity reference ----
# When enabled, display and capture this fixed phase mask immediately before
# every target. The reference frame itself is not saved; only robust intensity
# metrics and correction gains are appended to REFERENCE_GAIN_CSV_NAME.
REFERENCE_GAIN_ENABLED = False
REFERENCE_MASK_PATH = None  # e.g. r'C:\CITL\reference_phase.npy'
REFERENCE_GAIN_CSV_NAME = 'reference_gain_log.csv'

# The first reference frame defines fixed signal/background pixel regions.
# Every later frame is measured over those same pixels, avoiding the instability
# of a single maximum pixel while remaining insensitive to dark image borders.
REFERENCE_SIGNAL_PERCENTILE = 90.0
REFERENCE_BACKGROUND_PERCENTILE = 20.0
REFERENCE_MIN_ROI_PIXELS = 1024
REFERENCE_MAX_SAT_PCT = 0.5

# Per-file exposure overrides. Key = mask basename, value = pc.pattern kwargs.
PER_FILE_OVERRIDE = {
    # 'phase_mask_dot.npy': dict(pwm_1=130, duration_ms=3000),
}


# ============================================================
# 2. HELPERS
# ============================================================

def _log(msg):
    if VERBOSE:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}')


def patch_hololith_timing():
    """Speed up SLM readiness polling and upload polling for this run only."""
    if FAST_SLM_READY_CHECK_HZ and FAST_SLM_READY_CHECK_HZ > 0:
        def fast_wait_til_ready(self, check_freq=FAST_SLM_READY_CHECK_HZ, timeout=10):
            interval_s = 1.0 / float(check_freq)
            start = time.perf_counter()
            while time.perf_counter() - start < timeout:
                ret = SLMLib.SLM_Ctrl_ReadSU(self.SLM_NUMBER)
                if ret == SLMLib.SLM_OK:
                    return
                time.sleep(interval_s)
            print('Timeout %d seconds' % timeout)

        slmcontrol.SLMControl.waitTilReady = fast_wait_til_ready

    if FAST_SLM_UPLOAD_POLL_S and FAST_SLM_UPLOAD_POLL_S > 0:
        def fast_upload_phase_mask(self, mask, memory_location, timeout_s=15):
            assert mask.dtype == np.uint16, 'Mask should be of type np.uint16'
            assert 1 <= memory_location <= SLM_MEMORY_CAPACITY, (
                'memory_location must be within 1 and 128')

            mask_view = np.ascontiguousarray(mask, dtype=np.uint16)
            data_ptr = mask_view.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

            self.waitTilReady()
            print(f'Starting the upload of image to memory location {memory_location}')
            start = time.perf_counter()
            ret = SLMLib.SLM_Ctrl_WriteMI(
                self.SLM_NUMBER, memory_location, self.WIDTH, self.HEIGHT,
                ctypes.c_uint(0), data_ptr)
            if ret != SLMLib.SLM_OK:
                raise RuntimeError(f'SLM_Ctrl_WriteMI failed with status {ret}')

            while SLMLib.SLM_Ctrl_ReadSU(self.SLM_NUMBER) != SLMLib.SLM_OK:
                time.sleep(float(FAST_SLM_UPLOAD_POLL_S))
                if time.perf_counter() - start > timeout_s:
                    raise TimeoutError(
                        f'Timed out after {timeout_s}s uploading memory {memory_location}')

            print(f'Finished the upload of image to memory location {memory_location} '
                  f'in time {time.perf_counter() - start :.2f}')
            return True

        slmcontrol.SLMControl.uploadPhaseMask = fast_upload_phase_mask


def build_mask_file_list():
    if MASK_FILES_OVERRIDE is not None:
        files = [os.path.join(MASK_FOLDER, name) for name in MASK_FILES_OVERRIDE]
    else:
        files = sorted(glob.glob(os.path.join(MASK_FOLDER, MASK_GLOB)))
    files = [f for f in files if os.path.isfile(f)]
    if REFERENCE_GAIN_ENABLED and REFERENCE_MASK_PATH:
        reference_path = os.path.normcase(os.path.abspath(REFERENCE_MASK_PATH))
        files = [f for f in files
                 if os.path.normcase(os.path.abspath(f)) != reference_path]
    return files


def validate_reference_config():
    if not REFERENCE_GAIN_ENABLED:
        return
    if not REFERENCE_MASK_PATH:
        raise ValueError('REFERENCE_MASK_PATH is required when REFERENCE_GAIN_ENABLED=True')
    if not os.path.isfile(REFERENCE_MASK_PATH):
        raise FileNotFoundError(f'reference phase mask not found: {REFERENCE_MASK_PATH}')
    if not (0.0 <= REFERENCE_BACKGROUND_PERCENTILE < REFERENCE_SIGNAL_PERCENTILE <= 100.0):
        raise ValueError(
            'reference percentiles must satisfy 0 <= background < signal <= 100')
    if REFERENCE_MIN_ROI_PIXELS < 1:
        raise ValueError('REFERENCE_MIN_ROI_PIXELS must be positive')


def slm_frames_per_target():
    if REFERENCE_GAIN_ENABLED and not MANUAL_ADVANCE_AFTER_SAVE:
        return 2
    return 1


def max_target_chunk_size():
    if not REFERENCE_GAIN_ENABLED:
        return SLM_MEMORY_CAPACITY
    if MANUAL_ADVANCE_AFTER_SAVE:
        return SLM_MEMORY_CAPACITY - 1
    return SLM_MEMORY_CAPACITY // 2


def reference_memory_location(target_offset):
    if not REFERENCE_GAIN_ENABLED:
        return None
    if MANUAL_ADVANCE_AFTER_SAVE:
        return 1
    return target_offset * 2 + 1


def target_memory_location(target_offset):
    if REFERENCE_GAIN_ENABLED:
        if MANUAL_ADVANCE_AFTER_SAVE:
            return target_offset + 2
        return target_offset * 2 + 2
    return target_offset + 1


def open_camera():
    """Open the Basler once and configure it for single on-demand grabs."""
    tlf = pylon.TlFactory.GetInstance()

    if CAMERA_SERIAL is None:
        cam = pylon.InstantCamera(tlf.CreateFirstDevice())
    else:
        info = pylon.DeviceInfo()
        info.SetSerialNumber(str(CAMERA_SERIAL))
        cam = pylon.InstantCamera(tlf.CreateDevice(info))

    cam.Open()

    # Free-run-free configuration: software-triggered single frames.
    try:
        cam.PixelFormat.SetValue(PIXEL_FORMAT)
    except Exception as e:
        _log(f'  PixelFormat {PIXEL_FORMAT} not set ({e!r}); using camera default')

    # Manual exposure so brightness is reproducible across all 600 masks.
    try:
        cam.ExposureAuto.SetValue('Off')
    except Exception:
        pass
    try:
        cam.ExposureTime.SetValue(float(CAMERA_EXPOSURE_US))      # newer SFNC
    except Exception:
        try:
            cam.ExposureTimeAbs.SetValue(float(CAMERA_EXPOSURE_US))  # older SFNC
        except Exception as e:
            _log(f'  could not set exposure time ({e!r})')

    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    model = cam.GetDeviceInfo().GetModelName()
    serial = cam.GetDeviceInfo().GetSerialNumber()
    _log(f'Camera open: {model} (SN {serial}), exposure={CAMERA_EXPOSURE_US} us, fmt={PIXEL_FORMAT}')
    return cam


def grab_frame(cam):
    """Grab a single frame as a numpy array. Flushes stale frames first so we
    get a fresh capture of the current target rather than a buffered older one."""
    # Throw away frames that may have been captured before the mask/laser settled.
    for _ in range(CAMERA_STALE_FRAMES_TO_FLUSH):
        stale = cam.RetrieveResult(2000, pylon.TimeoutHandling_Return)
        if stale:
            stale.Release()

    res = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if not res.GrabSucceeded():
        code, desc = res.GetErrorCode(), res.GetErrorDescription()
        res.Release()
        raise RuntimeError(f'grab failed: {code} {desc}')
    img = res.Array.copy()
    res.Release()
    return img


def raw_image_stats(img):
    if img.dtype != np.uint16:
        img = img.astype(np.uint16)

    raw_min = int(img.min())
    raw_max = int(img.max())
    sat_pct = 100.0 * np.count_nonzero(img >= 4091) / img.size
    stats = dict(raw_min=raw_min, raw_max=raw_max, sat_pct=sat_pct)
    return img, stats


class ReferenceGainTracker:
    """Reduce reference frames to a stable scalar and append one CSV row per target."""

    CSV_FIELDS = [
        'index', 'chunk', 'mask_name', 'reference_name', 'captured_at',
        'elapsed_s', 'status', 'reference_metric', 'baseline_metric',
        'relative_intensity', 'correction_gain', 'background_level',
        'signal_mean', 'roi_pixels', 'reference_raw_min',
        'reference_raw_max', 'reference_sat_pct', 'target_status'
    ]

    def __init__(self, csv_path, reference_path):
        self.csv_path = csv_path
        self.reference_path = reference_path
        self.started_at = time.perf_counter()
        self.signal_mask = None
        self.background_mask = None
        self.baseline_metric = None

        with open(self.csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=self.CSV_FIELDS).writeheader()

    def _initialize_regions(self, raw):
        flat = raw.reshape(-1)
        signal_fraction = (100.0 - REFERENCE_SIGNAL_PERCENTILE) / 100.0
        n_signal = max(REFERENCE_MIN_ROI_PIXELS,
                       int(np.ceil(flat.size * signal_fraction)))
        n_signal = min(n_signal, flat.size)
        n_background = max(
            1, int(np.ceil(flat.size * REFERENCE_BACKGROUND_PERCENTILE / 100.0)))
        n_background = min(n_background, flat.size)

        signal_indices = np.argpartition(flat, flat.size - n_signal)[-n_signal:]
        background_indices = np.argpartition(flat, n_background - 1)[:n_background]
        signal_mask = np.zeros(flat.size, dtype=bool)
        background_mask = np.zeros(flat.size, dtype=bool)
        signal_mask[signal_indices] = True
        background_mask[background_indices] = True
        signal_mask = signal_mask.reshape(raw.shape)
        background_mask = background_mask.reshape(raw.shape)

        self.signal_mask = signal_mask
        self.background_mask = background_mask
        _log(
            f'  reference ROI initialized: signal={np.count_nonzero(signal_mask)} px, '
            f'background={np.count_nonzero(background_mask)} px')

    def measure(self, img):
        raw, stats = raw_image_stats(img)
        if self.signal_mask is None:
            self._initialize_regions(raw)
        elif raw.shape != self.signal_mask.shape:
            raise RuntimeError(
                f'reference camera shape changed from {self.signal_mask.shape} to {raw.shape}')

        raw_float = raw.astype(np.float32, copy=False)
        background_level = float(np.median(raw_float[self.background_mask]))
        signal_mean = float(np.mean(raw_float[self.signal_mask]))
        metric = signal_mean - background_level
        if not np.isfinite(metric) or metric <= 0:
            raise RuntimeError(f'invalid reference metric {metric!r}')

        if self.baseline_metric is None:
            self.baseline_metric = metric
            _log(f'  reference baseline initialized: {metric:.6g}')

        relative_intensity = metric / self.baseline_metric
        correction_gain = self.baseline_metric / metric
        status = 'SATURATED' if stats['sat_pct'] > REFERENCE_MAX_SAT_PCT else 'OK'
        return {
            'captured_at': datetime.now().isoformat(timespec='milliseconds'),
            'elapsed_s': time.perf_counter() - self.started_at,
            'status': status,
            'reference_metric': metric,
            'baseline_metric': self.baseline_metric,
            'relative_intensity': relative_intensity,
            'correction_gain': correction_gain,
            'background_level': background_level,
            'signal_mean': signal_mean,
            'roi_pixels': int(np.count_nonzero(self.signal_mask)),
            'reference_raw_min': stats['raw_min'],
            'reference_raw_max': stats['raw_max'],
            'reference_sat_pct': stats['sat_pct'],
        }

    def failed_measurement(self, error):
        return {
            'captured_at': datetime.now().isoformat(timespec='milliseconds'),
            'elapsed_s': time.perf_counter() - self.started_at,
            'status': f'FAIL: {error}',
        }

    def record(self, idx, chunk_no, mask_path, measurement, target_status):
        row = {field: '' for field in self.CSV_FIELDS}
        row.update(measurement)
        row.update({
            'index': idx,
            'chunk': chunk_no,
            'mask_name': os.path.basename(mask_path),
            'reference_name': os.path.basename(self.reference_path),
            'target_status': target_status,
        })
        for key in (
                'elapsed_s', 'reference_metric', 'baseline_metric',
                'relative_intensity', 'correction_gain', 'background_level',
                'signal_mean', 'reference_sat_pct'):
            if row.get(key) != '':
                row[key] = f'{row[key]:.9g}'

        with open(self.csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.CSV_FIELDS).writerow(row)


def capture_reference_measurement(cam, tracker):
    """Capture and measure a reference without retaining or saving its image."""
    try:
        measurement = tracker.measure(grab_frame(cam))
        _log(
            '  reference -> '
            f'metric={measurement["reference_metric"]:.6g}  '
            f'rel={measurement["relative_intensity"]:.6f}  '
            f'gain={measurement["correction_gain"]:.6f}  '
            f'sat={measurement["reference_sat_pct"]:.7f}%')
        if measurement['status'] == 'SATURATED':
            _log('  !! WARNING reference is saturated; correction gain may be biased')
        return measurement
    except Exception as e:
        _log(f'  reference measurement failed (target capture will continue): {e!r}')
        return tracker.failed_measurement(repr(e))


def save_png(img, out_path, label=None):
    """Debayer a raw BayerRG12 frame to a 16-bit color PNG and return its stats.

    The 12-bit sensor data sits in the low bits of a uint16 array (0-4095). We
    measure exposure on the raw 12-bit values, then shift up (<< 4) to fill the
    full 16-bit range and debayer to BGR (OpenCV stores/writes BGR order).

    Returns dict: raw_min, raw_max (0-4095 scale), sat_pct (% pixels == 4095)."""
    img, stats = raw_image_stats(img)

    img16 = img << 4                            # 12-bit -> full 16-bit range
    bgr16 = cv2.cvtColor(img16, DEBAYER_CODE)   # raw mosaic -> 16-bit BGR

    if PNG_SAVE_MODE == 'bgr16':
        out_img = bgr16
        text_color = (65535, 65535, 65535)
    elif PNG_SAVE_MODE == 'gray16_debayered':
        out_img = cv2.cvtColor(bgr16, cv2.COLOR_BGR2GRAY)
        text_color = 65535
    elif PNG_SAVE_MODE == 'gray8_debayered':
        gray16 = cv2.cvtColor(bgr16, cv2.COLOR_BGR2GRAY)
        out_img = (gray16 >> 8).astype(np.uint8)
        text_color = 255
    else:
        raise ValueError(
            "PNG_SAVE_MODE must be 'gray8_debayered', 'gray16_debayered', or 'bgr16'")

    if STAMP_NAME_ON_IMAGE and label:
        cv2.putText(out_img, label, (40, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, text_color, 4, cv2.LINE_AA)

    ok = cv2.imwrite(out_path, out_img, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise RuntimeError(f'cv2.imwrite failed for {out_path}')
    return stats


def output_extension():
    if CAPTURE_SAVE_FORMAT == 'png':
        return '.png'
    if CAPTURE_SAVE_FORMAT == 'npy_raw16':
        return '.npy'
    if CAPTURE_SAVE_FORMAT == 'raw16':
        return '.raw'
    raise ValueError("CAPTURE_SAVE_FORMAT must be 'npy_raw16', 'raw16', or 'png'")


def save_capture_file(img, out_path, label=None):
    """Save a captured Basler frame and return raw 12-bit exposure stats."""
    if CAPTURE_SAVE_FORMAT == 'png':
        return save_png(img, out_path, label=label)

    img, stats = raw_image_stats(img)
    raw = np.ascontiguousarray(img, dtype=np.uint16)

    if CAPTURE_SAVE_FORMAT == 'npy_raw16':
        np.save(out_path, raw, allow_pickle=False)
        return stats

    if CAPTURE_SAVE_FORMAT == 'raw16':
        raw.tofile(out_path)
        if RAW16_WRITE_SIDECAR_JSON:
            meta_path = os.path.splitext(out_path)[0] + '.json'
            with open(meta_path, 'w') as f:
                json.dump({
                    'shape': list(raw.shape),
                    'dtype': str(raw.dtype),
                    'pixel_format': PIXEL_FORMAT,
                    'camera_exposure_us': CAMERA_EXPOSURE_US,
                    'note': 'Raw uint16 Bayer frame; 12-bit sensor values are in the low bits.'
                }, f, indent=2)
        return stats

    raise ValueError("CAPTURE_SAVE_FORMAT must be 'npy_raw16', 'raw16', or 'png'")


def output_path_for_mask(mask_path):
    name = os.path.basename(mask_path)
    return os.path.join(OUTPUT_FOLDER, os.path.splitext(name)[0] + output_extension())


def maybe_randomize_mask_order(mask_files):
    ordered = list(mask_files)
    if not RANDOMIZE_ORDER:
        return ordered

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(ordered)
    seed_msg = 'entropy' if RANDOM_SEED is None else str(RANDOM_SEED)
    _log(f'Randomized exposure order for {len(ordered)} mask(s), seed={seed_msg}')
    return ordered


def exposure_params_for_mask(mask_path):
    name = os.path.basename(mask_path)
    return {**EXPOSURE_PARAMS, **PER_FILE_OVERRIDE.get(name, {})}


def iter_param_chunks(mask_files, chunk_size):
    """Yield chunks that fit SLM memory and share one exposure parameter set."""
    max_targets = max_target_chunk_size()
    if chunk_size < 1 or chunk_size > max_targets:
        raise ValueError(
            f'CHUNK_SIZE must be between 1 and {max_targets} with '
            f'REFERENCE_GAIN_ENABLED={REFERENCE_GAIN_ENABLED}')

    chunk = []
    chunk_params = None
    for mask_path in mask_files:
        params = exposure_params_for_mask(mask_path)
        if chunk and (len(chunk) >= chunk_size or params != chunk_params):
            yield chunk, chunk_params
            chunk = []
            chunk_params = None

        if not chunk:
            chunk_params = params
        chunk.append(mask_path)

    if chunk:
        yield chunk, chunk_params


def build_chunk_mask_stack(chunk_files):
    """Build a stack, optionally ordered as reference,target pairs."""
    masks = []

    def load_mask(mask_path):
        ms = hololith.Mask.maskstack.MaskStack(
            mask_path, wavefront_correction=WFC_FILE_PATH, pad_mode='constant')
        if WFC_FILE_PATH is None:
            return ms.master_mask
        return np.atleast_3d(ms.getMaskWithWavefrontCorrection())

    reference_mask = load_mask(REFERENCE_MASK_PATH) if REFERENCE_GAIN_ENABLED else None

    if reference_mask is not None and MANUAL_ADVANCE_AFTER_SAVE:
        masks.append(reference_mask)

    for mask_path in chunk_files:
        if reference_mask is not None and not MANUAL_ADVANCE_AFTER_SAVE:
            masks.append(reference_mask)
        masks.append(load_mask(mask_path))

    chunk_array = np.concatenate(masks, axis=2).astype(np.uint16, copy=False)
    return hololith.Mask.maskstack.MaskStack(
        chunk_array, wavefront_correction=None, pad_mode='constant')


def format_arduino_number(value):
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f'{value:.6g}'


def chunk_pattern_params(frame_params, n_frames):
    """Convert per-mask exposure params to one chunk playback command."""
    per_frame_ms = int(frame_params['duration_ms'])
    if per_frame_ms <= 0:
        raise ValueError('duration_ms must be positive')

    fps = 1000.0 / float(per_frame_ms)
    fps_for_command = int(round(fps)) if abs(fps - round(fps)) < 1e-9 else fps

    params = dict(frame_params)
    params['duration_ms'] = int(round(per_frame_ms * n_frames))
    params['fps'] = fps_for_command
    return params, per_frame_ms / 1000.0


def start_chunk_pattern_nonblocking(pc, pattern_params, wl_first=1):
    """Start Arduino exposure without waiting for serial read timeout."""
    pc._resetStartFrame()
    pwm = pattern_params['pwm_1'] if wl_first == 1 else pattern_params['pwm_2']
    cmd = (
        f's_exp -wl_select {wl_first} '
        f'-d {int(pattern_params["duration_ms"])} '
        f'-pwm {int(pwm)} '
        f'-fps {format_arduino_number(pattern_params["fps"])} '
        f'-suppress_ms {int(pattern_params["suppress_ms"])} -y'
    )

    try:
        pc.ard_ctrl.serial_port.reset_input_buffer()
    except Exception:
        pass
    pc.ard_ctrl.serial_port.write(cmd.encode('utf-8'))
    return cmd


def send_arduino_nonblocking(pc, cmd):
    try:
        pc.ard_ctrl.serial_port.reset_input_buffer()
    except Exception:
        pass
    pc.ard_ctrl.serial_port.write(cmd.encode('utf-8'))
    return cmd


def set_laser(pc, pwm_1=0, pwm_2=0):
    cmd = f'laser {int(pwm_1)} {int(pwm_2)}'
    if MANUAL_LASER_USE_BLOCKING_COMMAND is True:
        mode = 'blocking'
    elif MANUAL_LASER_USE_BLOCKING_COMMAND is False:
        mode = 'nonblocking'
    else:
        mode = MANUAL_LASER_COMMAND_MODE

    if mode == 'blocking':
        pc.ard_ctrl.writeCommand(cmd)
        return cmd

    if mode == 'short_timeout':
        ser = pc.ard_ctrl.serial_port
        old_timeout = ser.timeout
        try:
            ser.timeout = MANUAL_LASER_SHORT_TIMEOUT_S
            ser.write(cmd.encode('utf-8'))
            ser.flush()
            lines = ser.readlines()
            for line in lines:
                print(line.decode('utf-8', errors='replace').rstrip())
        finally:
            ser.timeout = old_timeout
        return cmd

    if mode == 'nonblocking':
        return send_arduino_nonblocking(pc, cmd)

    raise ValueError("MANUAL_LASER_COMMAND_MODE must be 'blocking', 'short_timeout', or 'nonblocking'")


def drain_arduino_lines(pc):
    """Print any already-buffered Arduino replies without blocking the capture loop."""
    try:
        ser = pc.ard_ctrl.serial_port
        waiting = ser.in_waiting
        if waiting:
            text = ser.read(waiting).decode('utf-8', errors='replace')
            if text.strip():
                print(text.rstrip())
    except Exception as e:
        _log(f'  Arduino drain failed (ignored): {e!r}')


def save_captured_mask(img, mask_path, idx):
    name = os.path.basename(mask_path)
    out_path = output_path_for_mask(mask_path)
    label = f'[{idx}] {name}' if idx is not None else name
    stats = save_capture_file(img, out_path, label=label)
    _log(f'  captured -> {os.path.basename(out_path)} ({CAPTURE_SAVE_FORMAT})  '
         f'max={stats["raw_max"]}/4091  min={stats["raw_min"]}  '
         f'sat={stats["sat_pct"]:.7f}%')

    if stats['sat_pct'] > 0.5:
        _log(f'  !! WARNING clipping ({stats["sat_pct"]:.2f}% at max) - '
             'lower CAMERA_EXPOSURE_US or pwm_1')
    elif stats['raw_max'] < 1500:
        _log(f'  !! WARNING dim (max={stats["raw_max"]}/4091) - '
             'raise CAMERA_EXPOSURE_US or pwm_1')
    return stats


def capture_and_save_mask(cam, mask_path, idx):
    img = grab_frame(cam)
    return save_captured_mask(img, mask_path, idx)


def run_manual_chunk_capture(
        pc, cam, chunk_files, frame_params, start_idx,
        chunk_no, reference_tracker=None):
    """Capture targets manually, optionally measuring a reference before each one."""
    if MANUAL_LASER_TIMING_MODE not in ('per_frame', 'chunk_static', 'run_static'):
        raise ValueError(
            "MANUAL_LASER_TIMING_MODE must be 'per_frame', 'chunk_static', or 'run_static'")

    results = []
    per_frame_s = frame_params['duration_ms'] / 1000.0
    if MANUAL_CAPTURE_DELAY_S is None:
        capture_delay_s = max(MANUAL_LASER_SETTLE_S, per_frame_s * CAPTURE_AT_FRACTION)
    else:
        capture_delay_s = float(MANUAL_CAPTURE_DELAY_S)

    chunk_laser_on = False
    try:
        if MANUAL_LASER_TIMING_MODE == 'chunk_static':
            _log('  laser timing mode: chunk_static (laser stays on across this chunk)')
            t0 = time.perf_counter()
            set_laser(pc, pwm_1=frame_params['pwm_1'], pwm_2=0)
            chunk_laser_on = True
            t_chunk_laser_on = time.perf_counter() - t0
            if MANUAL_LASER_SETTLE_S > 0:
                time.sleep(MANUAL_LASER_SETTLE_S)
            if PROFILE_TIMING:
                _log(f'  chunk laser_on_s={t_chunk_laser_on:.3f}')

        for j, mask_path in enumerate(chunk_files):
            idx = start_idx + j
            name = os.path.basename(mask_path)
            t_frame = time.perf_counter()
            laser_on = False
            t_laser_on = 0.0
            t_laser_off = 0.0
            t_reference = 0.0
            measurement = None
            try:
                if REFERENCE_GAIN_ENABLED:
                    ref_location = reference_memory_location(j)
                    _log(
                        f'  display memory {ref_location} -> reference for {name}')
                    t0 = time.perf_counter()
                    pc.slm_ctrl.displayPatternAtMemory(ref_location)
                    if MANUAL_SLM_SETTLE_S > 0:
                        time.sleep(MANUAL_SLM_SETTLE_S)

                    if MANUAL_LASER_TIMING_MODE == 'per_frame':
                        t_laser_start = time.perf_counter()
                        set_laser(pc, pwm_1=frame_params['pwm_1'], pwm_2=0)
                        t_laser_on = time.perf_counter() - t_laser_start
                        laser_on = True

                    time.sleep(capture_delay_s)
                    measurement = capture_reference_measurement(cam, reference_tracker)
                    t_reference = time.perf_counter() - t0

                memory_location = target_memory_location(j)
                _log(f'  display memory {memory_location} -> {name}')
                t0 = time.perf_counter()
                pc.slm_ctrl.displayPatternAtMemory(memory_location)
                t_display = time.perf_counter() - t0
                t0 = time.perf_counter()
                if MANUAL_SLM_SETTLE_S > 0:
                    time.sleep(MANUAL_SLM_SETTLE_S)
                t_slm_settle = time.perf_counter() - t0

                if MANUAL_LASER_TIMING_MODE == 'per_frame':
                    if not laser_on:
                        t0 = time.perf_counter()
                        set_laser(pc, pwm_1=frame_params['pwm_1'], pwm_2=0)
                        t_laser_on = time.perf_counter() - t0
                        laser_on = True

                t0 = time.perf_counter()
                time.sleep(capture_delay_s)
                t_capture_delay = time.perf_counter() - t0
                t0 = time.perf_counter()
                img = grab_frame(cam)
                t_grab = time.perf_counter() - t0

                if MANUAL_LASER_TIMING_MODE == 'per_frame':
                    t0 = time.perf_counter()
                    set_laser(pc, pwm_1=0, pwm_2=0)
                    t_laser_off = time.perf_counter() - t0
                    laser_on = False

                t0 = time.perf_counter()
                stats = save_captured_mask(img, mask_path, idx=idx)
                t_save = time.perf_counter() - t0
                results.append((idx, name, True, stats))
                if reference_tracker is not None:
                    reference_tracker.record(
                        idx, chunk_no, mask_path, measurement, target_status='OK')
                if PROFILE_TIMING:
                    t_total = time.perf_counter() - t_frame
                    _log(
                        '  timing_s '
                        f'mode={MANUAL_LASER_TIMING_MODE} total={t_total:.3f} '
                        f'display={t_display:.3f} slm_settle={t_slm_settle:.3f} '
                        f'laser_on={t_laser_on:.3f} capture_wait={t_capture_delay:.3f} '
                        f'grab={t_grab:.3f} laser_off={t_laser_off:.3f} '
                        f'reference={t_reference:.3f} save={t_save:.3f}'
                    )
            except Exception as e:
                if reference_tracker is not None:
                    if measurement is None:
                        measurement = reference_tracker.failed_measurement(repr(e))
                    reference_tracker.record(
                        idx, chunk_no, mask_path, measurement, target_status='FAIL')
                # Keep the current phase from advancing to the next target if saving failed.
                if laser_on:
                    set_laser(pc, pwm_1=0, pwm_2=0)
                    laser_on = False
                raise
            finally:
                if laser_on:
                    set_laser(pc, pwm_1=0, pwm_2=0)
                drain_arduino_lines(pc)
    finally:
        if chunk_laser_on:
            t0 = time.perf_counter()
            set_laser(pc, pwm_1=0, pwm_2=0)
            t_chunk_laser_off = time.perf_counter() - t0
            if PROFILE_TIMING:
                _log(f'  chunk laser_off_s={t_chunk_laser_off:.3f}')

    return results


# ============================================================
# 3. SHARED HOLOLITH OBJECTS
# ============================================================

config = hololith.Util.config.readConfig(CONFIG_PATH)
tm_1 = hololith.Mask.tonemapper.ToneMapper(
    method='power_curve', power_curve_max_value=1023, power_curve_gamma=1.0)


# ============================================================
# 4. PER-CHUNK ROUTINE
# ============================================================

def run_one_chunk(
        pc, cam, chunk_files, frame_params, chunk_no, total_chunks, start_idx,
        reference_tracker=None):
    n_frames = len(chunk_files)
    _log(f'===== chunk [{chunk_no}/{total_chunks}] {n_frames} mask(s) =====')

    results = []
    try:
        ms_1 = build_chunk_mask_stack(chunk_files)
        pc.mask_stack_1 = ms_1
        pc.mask_stack_2 = None
        pc.updateDerivedVariables()

        pc.upload()

        if MANUAL_ADVANCE_AFTER_SAVE:
            _log('  manual advance mode: next phase waits for capture save success')
            return run_manual_chunk_capture(
                pc, cam, chunk_files, frame_params, start_idx,
                chunk_no, reference_tracker=reference_tracker)

        n_slm_frames = n_frames * slm_frames_per_target()
        pattern_params, frame_s = chunk_pattern_params(frame_params, n_slm_frames)
        _log(f'  chunk exposure -> pwm_1={pattern_params["pwm_1"]}, '
             f'fps={format_arduino_number(pattern_params["fps"])}, '
             f'total_duration_ms={pattern_params["duration_ms"]}, '
             f'per_slm_frame_ms={frame_params["duration_ms"]}')

        cmd = start_chunk_pattern_nonblocking(pc, pattern_params, wl_first=1)
        t0 = time.perf_counter()
        _log(f'  Arduino command sent: {cmd}')

        for j, mask_path in enumerate(chunk_files):
            idx = start_idx + j
            name = os.path.basename(mask_path)
            measurement = None

            if REFERENCE_GAIN_ENABLED:
                reference_t = t0 + (2 * j + CAPTURE_AT_FRACTION) * frame_s
                wait_s = reference_t - time.perf_counter()
                if wait_s > 0:
                    time.sleep(wait_s)
                measurement = capture_reference_measurement(cam, reference_tracker)

            target_frame_offset = j * slm_frames_per_target()
            if REFERENCE_GAIN_ENABLED:
                target_frame_offset += 1
            target_t = t0 + (target_frame_offset + CAPTURE_AT_FRACTION) * frame_s
            wait_s = target_t - time.perf_counter()
            if wait_s > 0:
                time.sleep(wait_s)

            try:
                stats = capture_and_save_mask(cam, mask_path, idx=idx)
                results.append((idx, name, True, stats))
                if reference_tracker is not None:
                    reference_tracker.record(
                        idx, chunk_no, mask_path, measurement, target_status='OK')
            except Exception as e:
                _log(f'  ERROR on {name}: {e!r}')
                results.append((idx, name, False, None))
                if reference_tracker is not None:
                    if measurement is None:
                        measurement = reference_tracker.failed_measurement(repr(e))
                    reference_tracker.record(
                        idx, chunk_no, mask_path, measurement, target_status='FAIL')

        end_t = t0 + n_slm_frames * frame_s + POST_PATTERN_BUFFER_S
        wait_s = end_t - time.perf_counter()
        if wait_s > 0:
            time.sleep(wait_s)
        drain_arduino_lines(pc)

    except Exception as e:
        _log(f'  ERROR on chunk {chunk_no}: {e!r}')
        for j, mask_path in enumerate(chunk_files):
            results.append((start_idx + j, os.path.basename(mask_path), False, None))
    finally:
        try:
            pc.slm_ctrl.displayConstantValue(phase_integer=0)
        except Exception as e:
            _log(f'  blank SLM after chunk failed (ignored): {e!r}')

    return results


# ============================================================
# 5. MAIN LOOP
# ============================================================

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    patch_hololith_timing()
    validate_reference_config()

    mask_files = build_mask_file_list()
    _log(f'{len(mask_files)} mask file(s) queued from {MASK_FOLDER}')
    _log(f'Captured frames will be written to {OUTPUT_FOLDER} as {CAPTURE_SAVE_FORMAT}')
    if not mask_files:
        _log('Nothing to do.')
        return

    results = []   # (idx, name, ok, stats)
    pending_files = []
    for i, mask_path in enumerate(mask_files):
        name = os.path.basename(mask_path)
        out_path = output_path_for_mask(mask_path)
        if (not OVERWRITE_EXISTING) and os.path.isfile(out_path):
            _log(f'  SKIP - output already exists: {os.path.basename(out_path)}')
            results.append((i + 1, name, True, None))
        else:
            pending_files.append(mask_path)

    pending_files = maybe_randomize_mask_order(pending_files)
    chunks = list(iter_param_chunks(pending_files, CHUNK_SIZE))
    _log(f'{len(pending_files)} mask(s) pending in {len(chunks)} chunk(s), '
         f'CHUNK_SIZE={CHUNK_SIZE}')
    if REFERENCE_GAIN_ENABLED:
        slot_layout = (
            'one reusable reference slot plus one slot per target'
            if MANUAL_ADVANCE_AFTER_SAVE else
            'interleaved reference,target slots')
        _log(
            f'Reference gain enabled: {REFERENCE_MASK_PATH} '
            f'({slot_layout}; reference images are not saved)')
    if not pending_files:
        _log('Nothing pending after skip check.')

    cam = None
    pc = None
    run_laser_on = False
    reference_tracker = None
    t_start = time.time()

    try:
        if pending_files:
            if REFERENCE_GAIN_ENABLED:
                reference_csv_path = os.path.join(
                    OUTPUT_FOLDER, REFERENCE_GAIN_CSV_NAME)
                reference_tracker = ReferenceGainTracker(
                    reference_csv_path, REFERENCE_MASK_PATH)
                _log(f'Reference gains will be appended to {reference_csv_path}')

            cam = open_camera()
            pc = hololith.main.PatterningControl(
                config,
                mask_stack_1=None,
                mask_stack_2=None,
                tonemapper_1=tm_1,
                tonemapper_2=None,
                dual_wl_patterning=False,
                wl_first=1,
                enable_camera_tuple=(),   # hololith's own camera stays off; we drive the Basler ourselves
            )
            pc.open()
            if MANUAL_ADVANCE_AFTER_SAVE:
                try:
                    _log('Manual mode: disarming Arduino trigger interrupts for direct laser control')
                    pc._disarmArduinoTriggers()
                except Exception as e:
                    _log(f'  Arduino trigger disarm failed (ignored): {e!r}')

                if MANUAL_LASER_TIMING_MODE == 'run_static':
                    _log('Manual mode: run_static laser on for the entire pending capture run')
                    static_pwm_1 = chunks[0][1]['pwm_1']
                    static_pwm_values = {params['pwm_1'] for _, params in chunks}
                    if len(static_pwm_values) > 1:
                        _log(f'  !! WARNING run_static ignores varying pwm_1 values: '
                             f'{sorted(static_pwm_values)}; using {static_pwm_1}')
                    t0 = time.perf_counter()
                    set_laser(pc, pwm_1=static_pwm_1, pwm_2=0)
                    run_laser_on = True
                    if MANUAL_LASER_SETTLE_S > 0:
                        time.sleep(MANUAL_LASER_SETTLE_S)
                    if PROFILE_TIMING:
                        _log(f'  run laser_on_s={time.perf_counter() - t0:.3f}')

            next_idx = len(results) + 1
            for chunk_no, (chunk_files, params) in enumerate(chunks, start=1):
                chunk_results = run_one_chunk(
                    pc, cam, chunk_files, frame_params=params,
                    chunk_no=chunk_no, total_chunks=len(chunks), start_idx=next_idx,
                    reference_tracker=reference_tracker)
                results.extend(chunk_results)
                next_idx += len(chunk_files)

                if chunk_no < len(chunks) and INTER_CHUNK_PAUSE_S > 0:
                    time.sleep(INTER_CHUNK_PAUSE_S)
    finally:
        if run_laser_on and pc is not None:
            try:
                t0 = time.perf_counter()
                set_laser(pc, pwm_1=0, pwm_2=0)
                run_laser_on = False
                if PROFILE_TIMING:
                    _log(f'  run laser_off_s={time.perf_counter() - t0:.3f}')
            except Exception as e:
                _log(f'run laser off failed before close (ignored): {e!r}')
        if pc is not None:
            try:
                pc.close()
                _log('SLM/Arduino closed.')
            except Exception as e:
                _log(f'pc close failed (ignored): {e!r}')
        try:
            if cam is not None:
                cam.StopGrabbing()
                cam.Close()
                _log('Camera closed.')
        except Exception as e:
            _log(f'camera close failed (ignored): {e!r}')

    # Write the run log CSV.
    if WRITE_CSV_LOG:
        csv_path = os.path.join(OUTPUT_FOLDER, 'capture_log.csv')
        try:
            with open(csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    'index', 'mask_name', 'output_name', 'save_format',
                    'status', 'raw_min', 'raw_max', 'sat_pct'
                ])
                for idx, name, ok, st in results:
                    output_name = os.path.basename(
                        os.path.splitext(name)[0] + output_extension())
                    if st:
                        w.writerow([idx, name, output_name, CAPTURE_SAVE_FORMAT,
                                    'OK' if ok else 'FAIL',
                                    st['raw_min'], st['raw_max'], f'{st["sat_pct"]:.3f}'])
                    else:
                        w.writerow([idx, name, output_name, CAPTURE_SAVE_FORMAT,
                                    'OK' if ok else 'FAIL', '', '', ''])
            _log(f'Wrote log: {csv_path}')
        except Exception as e:
            _log(f'CSV log failed (ignored): {e!r}')

    t_total = time.time() - t_start
    n_ok = sum(1 for _, _, ok, _ in results if ok)
    _log(f'=== Done in {t_total:.1f} s | {n_ok}/{len(results)} OK ===')
    for idx, name, ok, _ in results:
        print(f'  {"OK  " if ok else "FAIL"}  [{idx}] {name}')


if __name__ == '__main__':
    main()
