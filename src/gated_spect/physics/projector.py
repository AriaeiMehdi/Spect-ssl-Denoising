"""A small, deterministic ideal parallel-hole SPECT projector.

The reference operator is deliberately explicit rather than highly optimized:
each axial detector row maps to the corresponding volume slice and a sparse
2-D Siddon-style ray matrix is applied independently to every slice.  The same
matrix transpose is used for the adjoint, making the operator suitable for
float64 validation before an optimized implementation is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class Geometry:
    """Acquisition and reconstruction-grid geometry in millimetres/radians."""

    angles_rad: tuple[float, ...]
    detector_shape: tuple[int, int]
    volume_shape: tuple[int, int, int]
    voxel_spacing_mm: tuple[float, float, float] = (6.4, 6.4, 6.4)
    detector_spacing_mm: tuple[float, float] = (6.4, 6.4)

    def __post_init__(self) -> None:
        if not self.angles_rad:
            raise ValueError("at least one view angle is required")
        if len(self.detector_shape) != 2 or min(self.detector_shape) < 1:
            raise ValueError("detector_shape must be (H,W) with positive dimensions")
        if len(self.volume_shape) != 3 or min(self.volume_shape) < 1:
            raise ValueError("volume_shape must be (Z,Y,X) with positive dimensions")
        if self.detector_shape[0] != self.volume_shape[0]:
            raise ValueError("detector rows must equal volume Z for the reference axial map")
        if len(self.voxel_spacing_mm) != 3 or any(float(x) <= 0 for x in self.voxel_spacing_mm):
            raise ValueError("voxel_spacing_mm must contain three positive values")
        if len(self.detector_spacing_mm) != 2 or any(
            float(x) <= 0 for x in self.detector_spacing_mm
        ):
            raise ValueError("detector_spacing_mm must contain two positive values")
        if any(not math.isfinite(float(x)) for x in self.angles_rad):
            raise ValueError("angles_rad must be finite")

    @property
    def num_views(self) -> int:
        return len(self.angles_rad)

    @property
    def detector_height(self) -> int:
        return self.detector_shape[0]

    @property
    def detector_width(self) -> int:
        return self.detector_shape[1]

    def cache_key(self, algorithm_version: str = "ideal-siddon-v1") -> str:
        payload = repr(
            (
                algorithm_version,
                tuple(float(x) for x in self.angles_rad),
                self.detector_shape,
                self.volume_shape,
                tuple(float(x) for x in self.voxel_spacing_mm),
                tuple(float(x) for x in self.detector_spacing_mm),
            )
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _ray_segments(
    angle: float,
    detector_offset: float,
    *,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
) -> list[tuple[int, float]]:
    """Return flattened pixel indices and line-integral lengths for one ray."""

    # The detector offset is the signed distance from the volume centre along
    # the normal n.  The ray direction d is tangent to that normal.
    normal = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
    direction = np.asarray((-math.sin(angle), math.cos(angle)), dtype=np.float64)
    point = detector_offset * normal
    x_half = 0.5 * nx * dx
    y_half = 0.5 * ny * dy

    bounds = ((-x_half, x_half, float(point[0]), float(direction[0])),
              (-y_half, y_half, float(point[1]), float(direction[1])))
    t_enter = -math.inf
    t_exit = math.inf
    for low, high, p, d in bounds:
        if abs(d) < 1.0e-14:
            if p < low or p > high:
                return []
            continue
        t0 = (low - p) / d
        t1 = (high - p) / d
        if t0 > t1:
            t0, t1 = t1, t0
        t_enter = max(t_enter, t0)
        t_exit = min(t_exit, t1)
    if not math.isfinite(t_enter) or not math.isfinite(t_exit) or t_exit <= t_enter:
        return []

    ts = [t_enter, t_exit]
    if abs(direction[0]) >= 1.0e-14:
        ts.extend(((-x_half + i * dx - point[0]) / direction[0] for i in range(1, nx)))
    if abs(direction[1]) >= 1.0e-14:
        ts.extend(((-y_half + j * dy - point[1]) / direction[1] for j in range(1, ny)))
    ts = sorted(t for t in ts if t_enter <= t <= t_exit)

    entries: list[tuple[int, float]] = []
    for left, right in zip(ts[:-1], ts[1:]):
        if right - left <= 1.0e-12:
            continue
        midpoint = point + ((left + right) * 0.5) * direction
        ix = int(math.floor((float(midpoint[0]) + x_half) / dx))
        iy = int(math.floor((float(midpoint[1]) + y_half) / dy))
        # A boundary midpoint can round to nx/ny at the positive edge.
        if not (0 <= ix < nx and 0 <= iy < ny):
            continue
        entries.append((iy * nx + ix, float(right - left)))
    return entries


@lru_cache(maxsize=16)
def _cached_sparse_arrays(
    angles: tuple[float, ...],
    detector_width: int,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    detector_spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    detector_half = 0.5 * detector_width * dx
    for view, angle in enumerate(angles):
        for column in range(detector_width):
            offset = (column + 0.5 - detector_width / 2.0) * detector_spacing
            # The detector has the same nominal lateral sampling as the volume.
            # Rays outside the square FOV naturally have no nonzero entries.
            for pixel, length in _ray_segments(
                float(angle),
                offset,
                nx=nx,
                ny=ny,
                dx=dx,
                dy=dy,
            ):
                rows.append(view * detector_width + column)
                cols.append(pixel)
                values.append(length)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
        (len(angles) * detector_width, ny * nx),
    )


class IdealParallelProjector:
    """Sparse ideal parallel-hole forward projector with exact transpose."""

    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry
        self._matrix_cpu: Tensor | None = None

    def _matrix(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("the reference projector supports float32 and float64 only")
        if self._matrix_cpu is None:
            z, ny, nx = self.geometry.volume_shape
            rows, cols, values, shape = _cached_sparse_arrays(
                tuple(float(x) for x in self.geometry.angles_rad),
                self.geometry.detector_width,
                nx,
                ny,
                float(self.geometry.voxel_spacing_mm[2]),
                float(self.geometry.voxel_spacing_mm[1]),
                float(self.geometry.detector_spacing_mm[1]),
            )
            indices = torch.from_numpy(np.stack((rows, cols), axis=0))
            self._matrix_cpu = torch.sparse_coo_tensor(
                indices,
                torch.from_numpy(values),
                size=shape,
                dtype=torch.float64,
            ).coalesce()
        matrix = self._matrix_cpu.to(device=device, dtype=dtype)
        return matrix

    def forward(self, activity_rate: Tensor) -> Tensor:
        """Project ``[B,G,Z,Y,X]`` activity rate to ``[B,G,V,H,W]``."""

        if activity_rate.ndim != 5:
            raise ValueError("activity_rate must have shape [B,G,Z,Y,X]")
        b, g, z, y, x = activity_rate.shape
        if (z, y, x) != self.geometry.volume_shape:
            raise ValueError("activity_rate shape does not match projector volume_shape")
        matrix = self._matrix(device=activity_rate.device, dtype=activity_rate.dtype)
        slices = activity_rate.reshape(b * g * z, y * x)
        projected = torch.sparse.mm(matrix, slices.transpose(0, 1)).transpose(0, 1)
        projected = projected.reshape(b, g, z, self.geometry.num_views, self.geometry.detector_width)
        return projected.permute(0, 1, 3, 2, 4).contiguous()

    def adjoint(self, projections: Tensor) -> Tensor:
        """Apply the exact transpose to ``[B,G,V,H,W]`` projections."""

        if projections.ndim != 5:
            raise ValueError("projections must have shape [B,G,V,H,W]")
        b, g, v, h, w = projections.shape
        if (v, h, w) != (
            self.geometry.num_views,
            self.geometry.detector_height,
            self.geometry.detector_width,
        ):
            raise ValueError("projection shape does not match projector detector geometry")
        z, y, x = self.geometry.volume_shape
        matrix = self._matrix(device=projections.device, dtype=projections.dtype)
        ordered = projections.permute(0, 1, 3, 2, 4).reshape(b * g * z, v * w)
        volume = torch.sparse.mm(matrix.transpose(0, 1), ordered.transpose(0, 1)).transpose(0, 1)
        return volume.reshape(b, g, z, y, x).contiguous()
