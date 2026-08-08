"""Typed contracts shared by ingestion and later model code.

The canonical data layer intentionally uses NumPy arrays. Raw counts stay outside
Torch until a dataset batch is constructed, which prevents accidental in-place
normalization of the statistical observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class DataContractError(ValueError):
    """Raised when a canonical record violates an invariant."""


class PairProvenance(StrEnum):
    NESTED = "nested"
    INDEPENDENT = "independent"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class ReaderConfig:
    """Explicit reader fallbacks.

    Geometry values are optional by design. If the DICOM does not provide a
    required value and no fallback is configured, the reader rejects the file
    instead of guessing scanner geometry.
    """

    allowed_modalities: tuple[str, ...] = ("NM",)
    default_start_angle_deg: float | None = None
    default_angular_step_deg: float | None = None
    default_rotation_direction: str | None = None
    default_view_dwell_s: float | None = None
    default_detector_spacing_mm: tuple[float, float] | None = None
    default_radius_mm: float | None = None
    vendor_exposure_keyword: str | None = None
    default_energy_window_kev: tuple[float, float] | None = None
    default_collimator_id: str = "unknown"
    default_stress_rest: str = "unknown"
    allow_ungated: bool = True
    institution_id: str = "unknown"
    hash_key: bytes | None = None
    allow_unhashed_for_test: bool = False
    require_complete_grid: bool = True
    count_q_clip: tuple[float, float] = (0.02, 1.25)
    low_information_q: float = 0.20


@dataclass(frozen=True)
class GatedSPECTSample:
    """Canonical complete or investigational gated acquisition."""

    sample_id: str
    counts: np.ndarray
    frame_exposure_s: np.ndarray
    relative_exposure_q: np.ndarray
    valid_frames: np.ndarray
    angles_rad: np.ndarray
    radii_mm: np.ndarray
    radius_valid: np.ndarray
    detector_spacing_mm: tuple[float, float]
    view_dwell_s: np.ndarray
    gate_phase_rad: np.ndarray
    energy_window_kev: tuple[float, float]
    scanner_id: str
    collimator_id: str
    stress_rest: str
    detector_stream: str
    patient_key: str
    study_key: str
    geometry_hash: str
    exposure_source: str
    source_path: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts)
        if counts.ndim != 4:
            raise DataContractError(f"counts must be [G,V,H,W], got {counts.shape}")
        if not np.issubdtype(counts.dtype, np.integer):
            raise DataContractError(f"counts must be integer, got {counts.dtype}")
        if np.any(counts < 0):
            raise DataContractError("counts cannot contain negative values")

        g, v, _, _ = counts.shape
        for name, value, expected in (
            ("frame_exposure_s", self.frame_exposure_s, (g, v)),
            ("relative_exposure_q", self.relative_exposure_q, (g, v)),
            ("valid_frames", self.valid_frames, (g, v)),
            ("view_dwell_s", self.view_dwell_s, (v,)),
            ("angles_rad", self.angles_rad, (v,)),
            ("radii_mm", self.radii_mm, (v,)),
            ("radius_valid", self.radius_valid, (v,)),
            ("gate_phase_rad", self.gate_phase_rad, (g,)),
        ):
            if np.asarray(value).shape != expected:
                raise DataContractError(
                    f"{name} must have shape {expected}, got {np.asarray(value).shape}"
                )

        exposure = np.asarray(self.frame_exposure_s, dtype=np.float64)
        valid = np.asarray(self.valid_frames, dtype=bool)
        if np.any(~np.isfinite(exposure)) or np.any(exposure[valid] <= 0):
            raise DataContractError("valid frames must have finite positive exposure")
        if np.any(exposure[~valid] < 0):
            raise DataContractError("invalid frames cannot have negative exposure")

        q = np.asarray(self.relative_exposure_q, dtype=np.float64)
        if np.any(~np.isfinite(q)) or np.any(q < 0):
            raise DataContractError("relative exposure must be finite and nonnegative")
        radius_valid = np.asarray(self.radius_valid, dtype=bool)
        radii = np.asarray(self.radii_mm, dtype=np.float64)
        if np.any(radius_valid & (~np.isfinite(radii) | (radii <= 0))):
            raise DataContractError("valid radii must be finite and positive")
        if len(self.detector_spacing_mm) != 2 or any(
            not np.isfinite(float(x)) or float(x) <= 0 for x in self.detector_spacing_mm
        ):
            raise DataContractError("detector spacing must contain two positive finite values")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(int(x) for x in self.counts.shape)  # type: ignore[return-value]

    @property
    def num_gates(self) -> int:
        return self.shape[0]

    @property
    def num_views(self) -> int:
        return self.shape[1]

    @property
    def detector_shape(self) -> tuple[int, int]:
        return self.shape[2], self.shape[3]


@dataclass(frozen=True)
class ExposureResult:
    frame_exposure_s: np.ndarray
    relative_exposure_q: np.ndarray
    source: str
    stable_gates: tuple[int, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairResult:
    accepted: bool
    provenance: PairProvenance
    reason: str
    target_counts: np.ndarray | None = None
    target_exposure_s: np.ndarray | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
