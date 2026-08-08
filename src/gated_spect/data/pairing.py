"""Low/high exposure pairing and independent-pair motion checks."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..types import GatedSPECTSample, PairProvenance, PairResult


def _phase_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float, float]:
    """Return signed (row, col) shift and normalized peak using FFT phase correlation."""

    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    if ref.shape != mov.shape or ref.ndim != 2:
        raise ValueError("phase-correlation inputs must be equal-shaped 2D arrays")
    h, w = ref.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    ref = (ref - ref.mean()) * window
    mov = (mov - mov.mean()) * window
    cross = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    cross /= np.maximum(np.abs(cross), 1e-12)
    correlation = np.abs(np.fft.ifft2(cross))
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    shift = np.asarray(peak, dtype=np.float64)
    for axis, size in enumerate((h, w)):
        if shift[axis] > size / 2:
            shift[axis] -= size
    peak_value = float(correlation[peak] / max(correlation.sum(), 1e-12))
    return float(shift[0]), float(shift[1]), peak_value


def _geometry_mismatch_reason(
    low: GatedSPECTSample,
    high: GatedSPECTSample,
    *,
    radius_tolerance_mm: float,
) -> tuple[str | None, float]:
    if low.shape != high.shape:
        return "shape mismatch", float("inf")
    if not np.allclose(low.angles_rad, high.angles_rad, atol=1e-6, rtol=0):
        return "angle mismatch", float("inf")
    if low.detector_spacing_mm != high.detector_spacing_mm:
        return "detector spacing mismatch", float("inf")
    if low.collimator_id != high.collimator_id:
        return "collimator mismatch", float("inf")
    if not np.allclose(low.energy_window_kev, high.energy_window_kev, atol=1e-6, rtol=0):
        return "energy window mismatch", float("inf")
    if not np.array_equal(low.radius_valid, high.radius_valid):
        return "radius-validity mismatch", float("inf")
    if not np.all(low.radius_valid):
        return "radius unavailable", float("inf")
    radius_diff = float(np.max(np.abs(low.radii_mm - high.radii_mm)))
    if radius_diff > radius_tolerance_mm:
        return "radius mismatch", radius_diff
    return None, radius_diff


def verify_pair(
    low: GatedSPECTSample,
    high: GatedSPECTSample,
    provenance: PairProvenance,
    *,
    max_median_shift_bins: float = 1.0,
    max_p95_shift_bins: float = 2.0,
    radius_tolerance_mm: float = 2.0,
) -> PairResult:
    """Validate a pair and construct an independent count target when possible."""

    if low.patient_key != high.patient_key:
        return PairResult(False, provenance, "patient mismatch")

    low_counts = low.counts.astype(np.int64, copy=False)
    high_counts = high.counts.astype(np.int64, copy=False)
    low_e = low.frame_exposure_s.astype(np.float64, copy=False)
    high_e = high.frame_exposure_s.astype(np.float64, copy=False)

    if provenance is PairProvenance.NESTED:
        mismatch, radius_diff = _geometry_mismatch_reason(
            low, high, radius_tolerance_mm=0.0
        )
        if mismatch is not None:
            metrics = {"radius_max_diff_mm": radius_diff} if np.isfinite(radius_diff) else {}
            return PairResult(False, provenance, mismatch, metrics=metrics)
        difference = high_counts - low_counts
        exposure_difference = high_e - low_e
        if np.any(difference < 0):
            return PairResult(False, provenance, "nested high-low counts contain negative bins")
        if np.any(exposure_difference <= 0):
            return PairResult(False, provenance, "nested high-low exposure is not positive")
        return PairResult(
            True,
            provenance,
            "accepted",
            target_counts=difference.astype(np.int32),
            target_exposure_s=exposure_difference.astype(np.float32),
            metrics={"count_ratio": float(high_counts.sum() / max(low_counts.sum(), 1))},
        )

    if provenance is PairProvenance.INDEPENDENT:
        mismatch, radius_diff = _geometry_mismatch_reason(
            low, high, radius_tolerance_mm=radius_tolerance_mm
        )
        if mismatch is not None:
            metrics = {"radius_max_diff_mm": radius_diff} if np.isfinite(radius_diff) else {}
            return PairResult(False, provenance, mismatch, metrics=metrics)
        shifts: list[float] = []
        peaks: list[float] = []
        low_norm = low_counts.sum(axis=0) / np.maximum(low_e.sum(axis=0)[..., None, None], 1e-8)
        high_norm = high_counts.sum(axis=0) / np.maximum(high_e.sum(axis=0)[..., None, None], 1e-8)
        for view in range(low.num_views):
            dy, dx, peak = _phase_correlation_shift(low_norm[view], high_norm[view])
            shifts.append(float(np.hypot(dy, dx)))
            peaks.append(peak)
        median_shift = float(np.median(shifts))
        p95_shift = float(np.percentile(shifts, 95))
        metrics = {
            "radius_max_diff_mm": radius_diff,
            "median_shift_bins": median_shift,
            "p95_shift_bins": p95_shift,
            "median_phase_peak": float(np.median(peaks)),
        }
        accepted = median_shift <= max_median_shift_bins and p95_shift <= max_p95_shift_bins
        return PairResult(
            accepted,
            provenance,
            "accepted" if accepted else "motion/alignment threshold exceeded",
            target_counts=high.counts.copy() if accepted else None,
            target_exposure_s=high.frame_exposure_s.copy() if accepted else None,
            metrics=metrics,
        )

    if provenance is PairProvenance.SYNTHETIC:
        return PairResult(
            False,
            provenance,
            "synthetic pairs must be generated by synthesize_count_split",
        )
    raise ValueError(f"unsupported provenance {provenance!r}")


def synthesize_count_split(
    high_counts: np.ndarray,
    high_exposure_s: np.ndarray,
    low_exposure_s: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a low sample and independent complement from high counts."""

    high = np.asarray(high_counts)
    high_e = np.asarray(high_exposure_s, dtype=np.float64)
    low_e = np.asarray(low_exposure_s, dtype=np.float64)
    if high.ndim != 4 or high_e.shape != high.shape[:2] or low_e.shape != high.shape[:2]:
        raise ValueError("high counts and exposures have incompatible shapes")
    if np.any(high < 0) or np.any(low_e <= 0) or np.any(high_e <= low_e):
        raise ValueError("synthetic split requires nonnegative counts and 0<low exposure<high exposure")
    p = low_e / high_e
    low = rng.binomial(high.astype(np.int64), p[..., None, None]).astype(np.int32)
    return low, (high.astype(np.int64) - low.astype(np.int64)).astype(np.int32)
