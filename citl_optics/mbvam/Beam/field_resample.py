"""Pixel-centred complex-field resizing and its slab-wise linear adjoint."""
import math

import torch


def scaled_field_shape(shape, scale):
    """Keep the requested physical grid exact, without truncating a ratio."""
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('upsample_factor must be finite and positive')
    target = tuple(round(n * scale) for n in shape)
    if any(m < 1 or not math.isclose(n * scale, m, rel_tol=0, abs_tol=1e-7)
           for n, m in zip(shape, target)):
        raise ValueError('upsample_factor must give integer output grid dimensions')
    return target


def resolve_resample_mode(mode, shape, target):
    if mode not in ('auto', 'nearest', 'bilinear'):
        raise ValueError('slm_resample_mode must be auto, nearest, or bilinear')
    if mode == 'auto':
        return 'nearest' if all(m % n == 0 for n, m in zip(shape, target)) else 'bilinear'
    return mode


def resize_complex_field(field, target, mode='auto'):
    """Dense reference: same pixel centres, edge clamping, no antialias filter."""
    mode = resolve_resample_mode(mode, field.shape, target)
    if tuple(field.shape) == tuple(target):
        return field.clone()  # Caller multiplies the axicon phase in place.
    # Share double-precision coordinate construction with the streamed path.
    # F.interpolate's float32 coordinates otherwise drift on large grids.
    resampler = ComplexFieldResampler(field.shape, target, mode, field.device, field.real.dtype)
    return resampler.tile(field, 0, target[0])


class ComplexFieldResampler:
    """Resize only requested output rows; accumulate R^H into the input grid.

    Coordinates follow align_corners=False: (j + 0.5) * input/output - 0.5.
    Bilinear weights are real, so the complex adjoint uses those same weights.
    No full target-sized GPU field or interpolation graph is retained.
    """
    def __init__(self, shape, target, mode, device, dtype):
        self.shape, self.target = tuple(shape), tuple(target)
        self.mode = resolve_resample_mode(mode, shape, target)
        self.repeat = (target[0] // shape[0]
                       if self.mode == 'nearest' and
                       all(m % n == 0 for n, m in zip(shape, target)) and
                       target[0] // shape[0] == target[1] // shape[1] else None)
        if self.repeat is None:
            self.x = self._axis(shape[0], target[0], device, dtype)
            self.y = self._axis(shape[1], target[1], device, dtype)

    def _axis(self, source, target, device, dtype):
        # Compute coordinates in double precision to avoid position quantization
        # on the 32000-pixel output grid; only O(target) metadata is stored.
        centre = (torch.arange(target, device=device, dtype=torch.float64) + .5) * (source / target)
        if self.mode == 'nearest':
            return centre.floor().long().clamp(0, source - 1), None, None
        position = (centre - .5).clamp(0, source - 1)
        lower = position.floor().long()
        upper = (lower + 1).clamp_max(source - 1)
        return lower, upper, (position - lower).to(dtype)

    def tile(self, field, start, stop):
        if self.repeat is not None:
            q = self.repeat
            return field[start // q:stop // q].repeat_interleave(q, 0).repeat_interleave(q, 1)
        x0, x1, wx = self.x
        y0, y1, wy = self.y
        rows = field.index_select(0, x0[start:stop])
        if self.mode == 'nearest':
            return rows.index_select(1, y0)
        rows = rows * (1 - wx[start:stop, None]) + field.index_select(0, x1[start:stop]) * wx[start:stop, None]
        return rows.index_select(1, y0) * (1 - wy[None]) + rows.index_select(1, y1) * wy[None]

    def add_adjoint(self, result, tile, start, stop):
        if self.repeat is not None:
            q = self.repeat
            result[start // q:stop // q] = tile.reshape(
                (stop - start) // q, q, self.shape[1], q).sum((1, 3))
            return
        x0, x1, wx = self.x
        y0, y1, wy = self.y
        rows = tile.new_zeros((stop - start, self.shape[1]))
        if self.mode == 'nearest':
            rows.index_add_(1, y0, tile)
            result.index_add_(0, x0[start:stop], rows)
        else:
            rows.index_add_(1, y0, tile * (1 - wy[None]))
            rows.index_add_(1, y1, tile * wy[None])
            result.index_add_(0, x0[start:stop], rows * (1 - wx[start:stop, None]))
            result.index_add_(0, x1[start:stop], rows * wx[start:stop, None])
