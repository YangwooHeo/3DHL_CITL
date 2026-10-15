"""Version 2 camera-intensity objectives for FNO training.

All display terms share *one target-derived scale*. There is no prediction
standard-deviation or prediction-mean normalization. The positive scale floor
is fitted on the training split, saved in the run configuration, and reused for
validation/resume. Its units are those of the camera tensors entering this loss.

``log1p`` is the default: it compresses bright lobes without the additional
bright-pixel saturation of ``rational``. Its SSIM constants use a fixed unit in
log-display space (not a per-image data range). ``rational`` maps to [0, 1).
Neither transformation changes the raw MSE metric or pointwise intensity term.
"""

import math

import torch
import torch.nn.functional as F


def _mean_per_sample(value):
    return value.flatten(1).mean(1)


def _positive_scalar(value, name, *, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        relation = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}, got {value}")
    return value


def _validate_pair(pred, target):
    if pred.ndim != 4 or pred.shape != target.shape:
        raise ValueError("pred and target must have the same nonempty BCHW shape")
    if any(size < 1 for size in pred.shape):
        raise ValueError("pred and target must have the same nonempty BCHW shape")
    if pred.device != target.device:
        raise ValueError("pred and target must be on the same device")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError("pred and target must be real floating-point tensors")
    # Accumulate local moments/FFT in at least FP32, also under mixed precision.
    dtype = torch.promote_types(pred.dtype, target.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    return pred.to(dtype=dtype), target.to(dtype=dtype)


def camera_target_scale(target, scale_floor):
    """Detached per-image scale sqrt(nonnegative_target_mean**2 + floor**2).

    The floor limits the pointwise SmoothL1 intensity derivative to 1/floor
    before pixel/sample averaging. ``hypot`` avoids squaring the floor or mean
    in low precision. Prediction values never enter this denominator.
    """
    floor = _positive_scalar(scale_floor, "scale_floor")
    target_mean = target.detach().clamp_min(0).mean((1, 2, 3), keepdim=True)
    return torch.hypot(target_mean, target_mean.new_tensor(floor))


def shared_positive_display(pred, target, scale, transform="log1p"):
    """Apply the same nonnegative display map to prediction and target.

    Negative predictions are clipped only for display terms; the pointwise
    SmoothL1 still penalizes their signed raw error. Camera targets are expected
    to be nonnegative, as ensured by the training data preprocessing.
    """
    if transform == "log1p":
        return torch.log1p(pred.clamp_min(0) / scale), torch.log1p(target.clamp_min(0) / scale)
    if transform == "rational":
        pred_pos, target_pos = pred.clamp_min(0), target.clamp_min(0)
        return pred_pos / (pred_pos + scale), target_pos / (target_pos + scale)
    raise ValueError(f"Unsupported display_transform: {transform!r}")


def _reflect_pool(value, window):
    """Odd local window that also supports singleton and very small images."""
    size = min(int(window), value.shape[-2], value.shape[-1])
    size = max(1, size if size % 2 else size - 1)
    if size == 1:
        return value
    pad = size // 2
    return F.avg_pool2d(F.pad(value, (pad, pad, pad, pad), mode="reflect"), size, stride=1)


def positive_ssim_loss_per_sample(pred, target, window=11, c1=0.01**2, c2=0.03**2):
    """SSIM on a common positive display with stable exact-identity behavior.

    The contrast/structure factor is written as
    ``1 - Var(pred-target)/(Var(pred)+Var(target)+c2)``. This is algebraically
    equivalent to the usual covariance formula. Unlike subtracting nearly
    equal second moments for the covariance, it gives zero loss and zero
    gradient for identical flat/zero/tiny images. Population moments use a
    reflective window; small negative variances from roundoff are clipped.
    No epsilon is added to the product of the already-positive denominators.
    """
    if int(window) < 1:
        raise ValueError("window must be positive")
    c1 = _positive_scalar(c1, "c1")
    c2 = _positive_scalar(c2, "c2")
    mean_p, mean_t = _reflect_pool(pred, window), _reflect_pool(target, window)
    var_p = (_reflect_pool(pred.square(), window) - mean_p.square()).clamp_min(0)
    var_t = (_reflect_pool(target.square(), window) - mean_t.square()).clamp_min(0)
    difference = pred - target
    mean_difference = _reflect_pool(difference, window)
    var_difference = (_reflect_pool(difference.square(), window) - mean_difference.square()).clamp_min(0)
    luminance = 1 - (mean_p - mean_t).square() / (mean_p.square() + mean_t.square() + c1)
    structure = 1 - var_difference / (var_p + var_t + c2)
    ssim = (luminance * structure).clamp(-1, 1)
    return _mean_per_sample(1 - ssim)


def display_gradient_loss_per_sample(pred, target):
    """Spatial finite-difference error on the shared positive display."""
    result = pred.new_zeros(pred.shape[0])
    if pred.shape[-1] > 1:
        result = result + _mean_per_sample((pred.diff(dim=-1) - target.diff(dim=-1)).abs())
    if pred.shape[-2] > 1:
        result = result + _mean_per_sample((pred.diff(dim=-2) - target.diff(dim=-2)).abs())
    return result


def _fft_texture_loss_per_sample(pred, target):
    pred_hp, target_hp = pred - _reflect_pool(pred, 31), target - _reflect_pool(target, 31)
    pred_mag = torch.log1p(torch.fft.rfft2(pred_hp, norm="ortho").abs())
    target_mag = torch.log1p(torch.fft.rfft2(target_hp, norm="ortho").abs())
    # Magnitude is only an auxiliary texture statistic; it cannot locate speckles.
    # No independent spectrum standardization or division by its small std.
    return _mean_per_sample(F.smooth_l1_loss(pred_mag, target_mag, reduction="none"))


def _tail_loss_per_sample(error, margin, top_fraction):
    margin = _positive_scalar(margin, "tail margin", allow_zero=True)
    excess = (error - margin).clamp_min(0)
    # SmoothL1 bounds the derivative; the old squared penalty did not.
    penalty = F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction="none").flatten(1)
    if top_fraction is not None:
        fraction = _positive_scalar(top_fraction, "top_fraction", allow_zero=True)
        if fraction > 1:
            raise ValueError("top_fraction must be in [0, 1], or None")
        if fraction > 0:
            count = max(1, int(penalty.shape[1] * fraction))
            penalty = penalty.topk(count, dim=1).values
    return penalty.mean(1)


def camera_loss_v2_per_sample(
    pred,
    target,
    *,
    scale_floor,
    w_ssim=1.0,
    w_grad=0.1,
    w_mean_norm_l1=1.0,
    w_log_display_smooth_l1=0.0,
    w_fft=0.0,
    w_peak=0.0,
    w_dark=0.0,
    peak_margin=0.10,
    peak_top_fraction=0.002,
    dark_margin=0.10,
    dark_top_fraction=0.002,
    display_transform="log1p",
):
    """Return ``(loss[B], components[str -> tensor[B]])``.

    Component names intentionally match loss version 1 for existing history
    and plotting code. ``mean_norm_l1`` now uses the smooth positive floor;
    ``log_display_smooth_l1`` uses the selected shared display transform. All
    other components except raw_mse are zero when their weight is zero.

    Finite input checking belongs at the data/training boundary, avoiding a
    CPU/GPU synchronization per loss call. Invalid scalar settings fail fast;
    nonfinite image values propagate to the trainer's nonfinite-loss guard.
    """
    pred, target = _validate_pair(pred, target)
    weights = {
        "ssim": w_ssim,
        "grad": w_grad,
        "mean_norm_l1": w_mean_norm_l1,
        "log_display_smooth_l1": w_log_display_smooth_l1,
        "fft": w_fft,
        "peak": w_peak,
        "dark": w_dark,
    }
    weights = {key: _positive_scalar(weight, f"w_{key}", allow_zero=True) for key, weight in weights.items()}
    if display_transform not in ("log1p", "rational"):
        raise ValueError(f"Unsupported display_transform: {display_transform!r}")
    scale = camera_target_scale(target, scale_floor)
    zeros = pred.new_zeros(pred.shape[0])
    components = {key: zeros for key in weights}
    if any(weights[key] > 0 for key in ("ssim", "grad", "log_display_smooth_l1", "fft")):
        pred_d, target_d = shared_positive_display(pred, target, scale, display_transform)
        if weights["ssim"] > 0:
            components["ssim"] = positive_ssim_loss_per_sample(pred_d, target_d)
        if weights["grad"] > 0:
            components["grad"] = display_gradient_loss_per_sample(pred_d, target_d)
        if weights["log_display_smooth_l1"] > 0:
            components["log_display_smooth_l1"] = _mean_per_sample(F.smooth_l1_loss(pred_d, target_d, reduction="none"))
        if weights["fft"] > 0:
            components["fft"] = _fft_texture_loss_per_sample(pred_d, target_d)
    normalized_error = (pred - target) / scale
    if weights["mean_norm_l1"] > 0:
        components["mean_norm_l1"] = _mean_per_sample(F.smooth_l1_loss(normalized_error, torch.zeros_like(normalized_error), reduction="none"))
    if weights["peak"] > 0:
        components["peak"] = _tail_loss_per_sample(normalized_error, peak_margin, peak_top_fraction)
    if weights["dark"] > 0:
        components["dark"] = _tail_loss_per_sample(-normalized_error, dark_margin, dark_top_fraction)
    # Maintain an autograd edge even when all terms are deliberately disabled.
    total = pred.flatten(1).sum(1) * 0
    for key, weight in weights.items():
        total = total + weight * components[key]
    components["raw_mse"] = _mean_per_sample((pred - target).square())
    return total, components
