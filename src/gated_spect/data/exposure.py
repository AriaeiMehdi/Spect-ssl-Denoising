"""Exposure estimation with an explicit vendor-first fallback policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import median_filter

from ..types import ExposureResult


class ExposureError(ValueError):
    """Raised when an exposure matrix cannot be estimated safely."""


@dataclass(frozen=True)
class ExposureConfig:
    q_clip: tuple[float, float] = (0.02, 1.25)
    low_information_q: float = 0.20
    stable_gate_z: float = 2.5
    stable_gate_ratio_floor: float = 1.25
    view_variation_threshold: float = 0.10
    smooth_width: int = 5


def _check_inputs(counts: np.ndarray, view_dwell_s: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(counts)
    if y.ndim != 4:
        raise ExposureError(f"counts must have shape [G,V,H,W], got {y.shape}")
    if np.any(~np.isfinite(y)) or np.any(y < 0):
        raise ExposureError("counts must be finite and nonnegative")
    dwell = np.asarray(view_dwell_s, dtype=np.float64)
    if dwell.shape != (y.shape[1],) or np.any(~np.isfinite(dwell)) or np.any(dwell <= 0):
        raise ExposureError("view dwell must be finite, positive, and have one value per view")
    return y.astype(np.float64, copy=False), dwell


def _relative_q_from_exposure(exposure: np.ndarray, valid: np.ndarray, q_clip: tuple[float, float]) -> np.ndarray:
    q = np.zeros_like(exposure, dtype=np.float64)
    for view in range(exposure.shape[1]):
        eligible = valid[:, view] & np.isfinite(exposure[:, view]) & (exposure[:, view] > 0)
        if not np.any(eligible):
            continue
        baseline = float(np.median(exposure[eligible, view]))
        q[eligible, view] = np.clip(exposure[eligible, view] / max(baseline, 1e-12), *q_clip)
    return q


def _robust_cv(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    med = float(np.median(values))
    if med <= 1e-12:
        return float("inf")
    return float(1.4826 * np.median(np.abs(values - med)) / med)


def estimate_exposure(
    counts: np.ndarray,
    view_dwell_s: Sequence[float],
    *,
    vendor_exposure_s: np.ndarray | None = None,
    valid_frames: np.ndarray | None = None,
    config: ExposureConfig = ExposureConfig(),
) -> ExposureResult:
    """Estimate [G,V] exposure and preserve the relative occupancy used by QC.

    When vendor exposure is supplied, it is used as-is after validation. The
    count-derived path is deliberately rank-based and normalizes occupancies
    per view so the exposure sum equals view dwell time.
    """

    y, dwell = _check_inputs(counts, view_dwell_s)
    g, v = y.shape[:2]
    valid = np.ones((g, v), dtype=bool) if valid_frames is None else np.asarray(valid_frames, dtype=bool)
    if valid.shape != (g, v):
        raise ExposureError(f"valid_frames must have shape {(g, v)}, got {valid.shape}")
    if not (0 < config.q_clip[0] <= config.q_clip[1]):
        raise ExposureError("q_clip must be positive and ordered")

    if vendor_exposure_s is not None:
        exposure = np.asarray(vendor_exposure_s, dtype=np.float64)
        if exposure.shape != (g, v):
            raise ExposureError(f"vendor exposure must have shape {(g, v)}, got {exposure.shape}")
        if np.any(~np.isfinite(exposure[valid])) or np.any(exposure[valid] <= 0):
            raise ExposureError("vendor exposure for valid frames must be finite and positive")
        q = _relative_q_from_exposure(exposure, valid, config.q_clip)
        exposure = np.where(valid, exposure, 0.0)
        return ExposureResult(
            frame_exposure_s=exposure.astype(np.float32),
            relative_exposure_q=q.astype(np.float32),
            source="vendor",
            stable_gates=tuple(range(g)),
        )

    frame_counts = y.sum(axis=(2, 3))
    frame_counts = np.where(valid, frame_counts, 0.0)
    gate_counts = frame_counts.sum(axis=1)
    log_gate = np.log(gate_counts + 0.5)
    center = float(np.median(log_gate))
    mad = float(np.median(np.abs(log_gate - center)))
    robust_scale = 1.4826 * mad
    threshold = max(config.stable_gate_z * robust_scale, np.log(config.stable_gate_ratio_floor))
    stable = np.abs(log_gate - center) <= threshold
    stable &= gate_counts > 0
    if int(stable.sum()) < int(np.ceil(g / 2)):
        raise ExposureError(
            f"fewer than half of gates are stable for fallback exposure: {np.flatnonzero(stable).tolist()}"
        )

    baseline_by_view = np.median(frame_counts[stable], axis=0)
    raw_q = frame_counts / np.maximum(baseline_by_view[None, :], 1.0)
    raw_q = np.clip(raw_q, config.q_clip[0], config.q_clip[1])
    raw_q[~valid] = 0.0

    q = np.zeros_like(raw_q)
    warnings: list[str] = []
    width = int(config.smooth_width)
    if width < 1 or width % 2 == 0:
        raise ExposureError("smooth_width must be a positive odd integer")
    for gate in range(g):
        row = raw_q[gate]
        if _robust_cv(row[valid[gate]]) <= config.view_variation_threshold:
            q[gate, valid[gate]] = float(np.median(row[valid[gate]]))
        else:
            filtered = median_filter(row, size=width, mode="nearest")
            q[gate, valid[gate]] = filtered[valid[gate]]
            warnings.append(f"gate {gate} has view-varying fallback occupancy")

    exposure = np.zeros((g, v), dtype=np.float64)
    for view in range(v):
        denom = float(q[:, view].sum())
        if denom <= 0:
            raise ExposureError(f"no valid exposure estimate for view {view}")
        exposure[:, view] = dwell[view] * q[:, view] / denom
    low_q = np.flatnonzero(q < config.low_information_q)
    if low_q.size:
        warnings.append(f"{low_q.size} gate/view frames have q below {config.low_information_q}")
    return ExposureResult(
        frame_exposure_s=exposure.astype(np.float32),
        relative_exposure_q=q.astype(np.float32),
        source="derived",
        stable_gates=tuple(int(x) for x in np.flatnonzero(stable)),
        warnings=tuple(warnings),
    )
