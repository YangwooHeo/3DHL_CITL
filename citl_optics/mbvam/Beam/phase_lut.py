"""Measured SLM command-to-phase response, shared by simulation/calibration.

File columns: MaskInput, PhaseOutput_Rad, optional IsReachable. Commands use
raw SLM levels; forward() takes normalized drive (command / phase_level_max).
Reachable measurements are linearly interpolated without wrapping or inversion.
Unreachable/blank rows are excluded. Bounds either saturate or raise an error.
"""
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.autograd.function import once_differentiable


class _InterpolatePhase(torch.autograd.Function):
    """Chunk interpolation so P=5 does not retain many full-size index arrays."""
    @staticmethod
    def forward(ctx, drive, x, y, bounds):
        ctx.save_for_backward(drive, x, y)
        result = torch.empty(drive.shape, device=drive.device, dtype=drive.dtype)
        flat, output = drive.reshape(-1), result.view(-1)
        for start in range(0, flat.numel(), 1_048_576):
            values = flat[start:start + 1_048_576]
            if bounds == 'error' and bool(((values < x[0]) | (values > x[-1])).any()):
                raise ValueError('SLM drive is outside the measured phase LUT range; use clamp bounds to saturate')
            query = values.clamp(x[0], x[-1])
            index = torch.searchsorted(x, query.contiguous(), right=True).clamp(1, x.numel() - 1)
            x0, x1, y0, y1 = x[index - 1], x[index], y[index - 1], y[index]
            output[start:start + values.numel()] = y0 + (query - x0) * ((y1 - y0) / (x1 - x0))
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        drive, x, y = ctx.saved_tensors
        result = torch.empty(drive.shape, device=drive.device, dtype=drive.dtype)
        flat, upstream, output = drive.reshape(-1), grad.reshape(-1), result.view(-1)
        for start in range(0, flat.numel(), 1_048_576):
            values = flat[start:start + 1_048_576]
            index = torch.searchsorted(x, values.contiguous(), right=True).clamp(1, x.numel() - 1)
            slope = (y[index] - y[index - 1]) / (x[index] - x[index - 1])
            slope = torch.where((values >= x[0]) & (values <= x[-1]), slope, 0.0)
            output[start:start + values.numel()] = upstream[start:start + values.numel()] * slope
        return result, None, None, None


class PhaseResponseLUT(nn.Module):
    def __init__(self, commands, phases_rad, *, phase_level_max=1023.0,
                 bounds='clamp', source=None, sha256=None, omitted_rows=0):
        super().__init__()
        commands = np.asarray(commands, dtype=np.float64)
        phases = np.asarray(phases_rad, dtype=np.float64)
        if commands.ndim != 1 or phases.shape != commands.shape or commands.size < 2:
            raise ValueError('Phase LUT requires at least two command/phase pairs')
        if not np.isfinite(commands).all() or not np.isfinite(phases).all():
            raise ValueError('Phase LUT values must be finite')
        if not np.all(np.diff(commands) > 0):
            raise ValueError('Phase LUT commands must be strictly increasing, with no duplicates')
        if bounds not in ('clamp', 'error'):
            raise ValueError('Phase LUT bounds must be clamp or error')
        if not math.isfinite(phase_level_max) or phase_level_max <= 0:
            raise ValueError('phase_level_max must be finite and positive')
        self.spec = dict(commands=commands.tolist(), phases_rad=phases.tolist(),
                         phase_level_max=float(phase_level_max), bounds=bounds,
                         source=source, sha256=sha256, omitted_rows=int(omitted_rows))
        # Embedded in checkpoint metadata, not learned parameter state. This
        # preserves old state_dict compatibility and allows portable resume.
        self.register_buffer('drive_knots', torch.tensor(commands / phase_level_max, dtype=torch.float64), persistent=False)
        self.register_buffer('phase_knots', torch.tensor(phases, dtype=torch.float64), persistent=False)
        self.bounds = bounds

    def forward(self, drive):
        return _InterpolatePhase.apply(
            drive, self.drive_knots.to(device=drive.device, dtype=drive.dtype),
            self.phase_knots.to(device=drive.device, dtype=drive.dtype), self.bounds)

    def to_config(self):
        return dict(self.spec)

    @classmethod
    def from_config(cls, config):
        return None if config is None else cls(**config)

    @classmethod
    def from_file(cls, path, *, phase_level_max=1023.0, bounds='clamp'):
        path = Path(path)
        if path.suffix.lower() == '.xlsx':
            try:
                import openpyxl
            except ImportError as error:
                raise ImportError('Reading .xlsx phase LUTs requires openpyxl; CSV is also supported') from error
            workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
            try:
                rows = list(workbook.worksheets[0].values)
            finally:
                workbook.close()
        elif path.suffix.lower() in ('.csv', '.tsv'):
            with path.open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.reader(handle, delimiter='\t' if path.suffix.lower() == '.tsv' else ','))
        else:
            raise ValueError('Phase LUT must be .xlsx, .csv, or .tsv with MaskInput and PhaseOutput_Rad columns')
        if not rows:
            raise ValueError('Phase LUT file is empty')
        headers = [str(value).strip().lower() for value in rows[0]]
        try:
            xi, yi = headers.index('maskinput'), headers.index('phaseoutput_rad')
        except ValueError as error:
            raise ValueError('Phase LUT requires MaskInput and PhaseOutput_Rad headers') from error
        ri = headers.index('isreachable') if 'isreachable' in headers else None
        commands, phases, omitted = [], [], 0
        for row_number, row in enumerate(rows[1:], start=2):
            if not any(value is not None and str(value).strip() for value in row):
                continue
            if len(row) <= max(xi, yi, ri or 0):
                raise ValueError(f'Malformed phase LUT row {row_number}')
            reachable = True
            if ri is not None:
                flag = str(row[ri]).strip().lower()
                if flag not in ('true', 'false', '1', '0', '1.0', '0.0'):
                    raise ValueError(f'Invalid IsReachable value on row {row_number}')
                reachable = flag in ('true', '1', '1.0')
            if not reachable or row[yi] is None or str(row[yi]).strip() == '':
                omitted += 1
                continue
            command, phase = float(row[xi]), float(row[yi])
            if not np.isfinite(phase):
                omitted += 1
                continue
            commands.append(command)
            phases.append(phase)
        return cls(commands, phases, phase_level_max=phase_level_max, bounds=bounds,
                   source=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   omitted_rows=omitted)

    def description(self):
        return (f"{len(self.spec['commands'])} measured rows, command range "
                f"[{self.spec['commands'][0]:g}, {self.spec['commands'][-1]:g}], "
                f"bounds={self.bounds}, last phase={self.spec['phases_rad'][-1]:.9g} rad, "
                f"omitted rows={self.spec['omitted_rows']}")
