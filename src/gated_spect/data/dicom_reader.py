"""Strict ingestion of binned raw nuclear-medicine projection DICOM."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pydicom
from pydicom.dataset import Dataset

from ..types import GatedSPECTSample, ReaderConfig
from .exposure import estimate_exposure
from .frame_indexing import (
    build_frame_index,
    construct_counts,
)


class DICOMReadError(ValueError):
    """Base error for unsupported or malformed DICOM input."""


class UnsupportedSOPClassError(DICOMReadError):
    pass


class UnsupportedDetectorLayoutError(DICOMReadError):
    pass


class PixelDecodeError(DICOMReadError):
    pass


class InvalidPixelValueError(DICOMReadError):
    pass


class InvalidExposureError(DICOMReadError):
    pass


class UnresolvedGeometryError(DICOMReadError):
    pass


def _first_value(ds: Dataset, names: Iterable[str]) -> Any | None:
    for name in names:
        if hasattr(ds, name):
            value = getattr(ds, name)
            if value is not None and not (isinstance(value, str) and value == ""):
                return value
    return None


def _nested_value(ds: Dataset, sequence_name: str, names: Iterable[str]) -> Any | None:
    sequence = getattr(ds, sequence_name, None)
    if not sequence:
        return None
    for item in sequence:
        value = _first_value(item, names)
        if value is not None:
            return value
    return None


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise UnresolvedGeometryError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise UnresolvedGeometryError(f"{name} is not finite")
    return result


def _vector(ds: Dataset, name: str) -> np.ndarray | None:
    if not hasattr(ds, name):
        return None
    value = getattr(ds, name)
    arr = np.asarray(value)
    return arr.reshape(-1)


def stable_identity_key(
    value: Any,
    *,
    label: str,
    config: ReaderConfig,
) -> str:
    if value is None or str(value) == "":
        return f"missing:{label}"
    if config.hash_key is None:
        if config.allow_unhashed_for_test:
            return f"test:{label}:{value}"
        raise DICOMReadError(
            f"hash_key is required for {label}; set allow_unhashed_for_test only in synthetic tests"
        )
    message = f"{config.institution_id}|{label}|{value}".encode("utf-8")
    return hmac.new(config.hash_key, message, hashlib.sha256).hexdigest()


_stable_key = stable_identity_key


def _canonical_geometry_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _get_start_step_direction(
    ds: Dataset,
    num_views: int,
    config: ReaderConfig,
) -> tuple[float, float, str, str]:
    start = _first_value(ds, ("StartAngle",))
    step = _first_value(ds, ("AngularStep",))
    direction = _first_value(ds, ("RotationDirection",))
    arc = _first_value(ds, ("ScanArc",))
    if start is None:
        start = _nested_value(ds, "RotationInformationSequence", ("StartAngle",))
    if step is None:
        step = _nested_value(ds, "RotationInformationSequence", ("AngularStep",))
    if direction is None:
        direction = _nested_value(ds, "RotationInformationSequence", ("RotationDirection",))
    if arc is None:
        arc = _nested_value(ds, "RotationInformationSequence", ("ScanArc",))

    start_source = "dicom" if start is not None else "config"
    step_source = "dicom" if step is not None else "config"
    if start is None:
        start = config.default_start_angle_deg
    if step is None:
        step = config.default_angular_step_deg
    if step is None and arc is not None:
        step = _finite_float(arc, "ScanArc") / num_views
        step_source = "scan_arc"
    if direction is None:
        direction = config.default_rotation_direction
    if start is None or step is None or direction is None:
        raise UnresolvedGeometryError(
            "start angle, angular step, and rotation direction require DICOM values or explicit config"
        )
    direction_text = str(direction).strip().upper()
    if direction_text not in {"CW", "CC", "CLOCKWISE", "COUNTERCLOCKWISE"}:
        raise UnresolvedGeometryError(f"unsupported rotation direction {direction!r}")
    return (
        _finite_float(start, "start angle"),
        _finite_float(step, "angular step"),
        direction_text,
        f"{start_source}+{step_source}",
    )


def _get_spacing(ds: Dataset, config: ReaderConfig) -> tuple[float, float]:
    value = _first_value(ds, ("PixelSpacing", "DetectorElementSpacing"))
    if value is None:
        value = config.default_detector_spacing_mm
    if value is None:
        raise UnresolvedGeometryError("detector spacing is missing")
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 2 or np.any(~np.isfinite(arr)) or np.any(arr <= 0):
        raise UnresolvedGeometryError(f"detector spacing must contain two positive values, got {value!r}")
    return float(arr[0]), float(arr[1])


def _get_radii(
    ds: Dataset, num_views: int, config: ReaderConfig
) -> tuple[np.ndarray, np.ndarray, str]:
    value = _first_value(ds, ("RadialPosition", "Radius"))
    source = "top_level"
    if value is None:
        value = _nested_value(ds, "DetectorInformationSequence", ("RadialPosition",))
        source = "detector_sequence"
    if value is None:
        value = _nested_value(ds, "RotationInformationSequence", ("RadialPosition",))
        source = "rotation_sequence"
    if value is not None:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            arr = np.repeat(arr, num_views)
        if arr.size != num_views:
            raise UnresolvedGeometryError(
                f"radius vector must have {num_views} values, got {arr.size}"
            )
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
            raise UnresolvedGeometryError("radius values must be finite and positive")
        return arr, np.ones(num_views, dtype=bool), source
    if config.default_radius_mm is not None and config.default_radius_mm > 0:
        return (
            np.full(num_views, float(config.default_radius_mm), dtype=np.float64),
            np.ones(num_views, dtype=bool),
            "config",
        )
    return np.full(num_views, np.nan, dtype=np.float64), np.zeros(num_views, dtype=bool), "missing"


def _get_view_dwell(ds: Dataset, num_views: int, config: ReaderConfig) -> np.ndarray:
    value = _first_value(ds, ("AngularViewDwellTime", "ViewDwellTime"))
    from_total_stop = False
    if value is None:
        value = _first_value(ds, ("AngularStopDuration",))
        from_total_stop = value is not None
    from_rotation_duration_ms = False
    if value is None:
        value = _nested_value(ds, "RotationInformationSequence", ("ActualFrameDuration",))
        from_rotation_duration_ms = value is not None
    if value is None:
        if config.default_view_dwell_s is None or config.default_view_dwell_s <= 0:
            raise UnresolvedGeometryError("view dwell time is missing")
        return np.full(num_views, float(config.default_view_dwell_s), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        if from_total_stop:
            arr = arr / num_views
        if from_rotation_duration_ms:
            arr = arr / 1000.0 / num_views
        arr = np.repeat(arr, num_views)
    if arr.size != num_views:
        raise UnresolvedGeometryError(f"view dwell must have {num_views} values, got {arr.size}")
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
        raise UnresolvedGeometryError("view dwell values must be finite and positive")
    # DICOM exports encountered in this project use seconds for AngularStopDuration.
    # Unit conversion is intentionally not guessed for arbitrary private keywords.
    return arr


def _get_energy_window(ds: Dataset, config: ReaderConfig) -> tuple[float, float]:
    lower = _first_value(ds, ("EnergyWindowLowerLimit",))
    upper = _first_value(ds, ("EnergyWindowUpperLimit",))
    if lower is None or upper is None:
        sequence = getattr(ds, "EnergyWindowInformationSequence", None)
        if sequence and getattr(sequence[0], "EnergyWindowRangeSequence", None):
            range_item = sequence[0].EnergyWindowRangeSequence[0]
            lower = _first_value(range_item, ("EnergyWindowLowerLimit",))
            upper = _first_value(range_item, ("EnergyWindowUpperLimit",))
    if lower is None or upper is None:
        if config.default_energy_window_kev is None:
            return float("nan"), float("nan")
        return config.default_energy_window_kev
    low, high = _finite_float(lower, "energy lower"), _finite_float(upper, "energy upper")
    if low < 0 or high <= low:
        raise DICOMReadError("energy window must have 0 <= lower < upper")
    return low, high


def _get_gate_phase(ds: Dataset, num_gates: int) -> tuple[np.ndarray, str]:
    rr = _first_value(ds, ("NominalInterval", "NominalRRInterval"))
    if rr is not None:
        rr_ms = _finite_float(rr, "nominal R-R interval")
        if rr_ms > 0:
            # Gate labels are ordered by TimeSlotVector; trigger timing is not
            # reliably available in all binned exports, so use slot centers.
            return (
                (2.0 * np.pi * (np.arange(num_gates, dtype=np.float64) + 0.5) / num_gates),
                "slot_center_rr",
            )
    return (
        2.0 * np.pi * (np.arange(num_gates, dtype=np.float64) + 0.5) / num_gates,
        "slot_center",
    )


def read_nm_dicom(path: str | Path, config: ReaderConfig = ReaderConfig()) -> GatedSPECTSample:
    """Read one complete binned raw gated SPECT acquisition."""

    source = Path(path)
    try:
        ds = pydicom.dcmread(str(source), force=False)
    except Exception as exc:  # pydicom exposes several decoder-specific exceptions
        raise PixelDecodeError(f"cannot read DICOM {source}") from exc
    modality = str(getattr(ds, "Modality", "")).upper()
    if modality not in config.allowed_modalities:
        raise UnsupportedSOPClassError(f"unsupported modality {modality!r}")
    if not hasattr(ds, "pixel_array"):
        raise PixelDecodeError("DICOM has no decodable pixel_array")
    try:
        pixels = np.asarray(ds.pixel_array)
    except Exception as exc:
        raise PixelDecodeError("pixel_array decoding failed") from exc
    if pixels.ndim == 2:
        pixels = pixels[None, ...]
    if pixels.ndim != 3:
        raise PixelDecodeError(f"pixel_array must be [F,H,W], got {pixels.shape}")
    if np.any(~np.isfinite(pixels)) or np.any(pixels < 0):
        raise InvalidPixelValueError("pixel_array contains negative or non-finite values")

    frames = int(pixels.shape[0])
    gate_vector = _vector(ds, "TimeSlotVector")
    view_vector = _vector(ds, "AngularViewVector")
    gating_mode = "gated"
    if view_vector is None:
        raise DICOMReadError("AngularViewVector is required")
    if gate_vector is None:
        if not config.allow_ungated:
            raise DICOMReadError("TimeSlotVector is absent and ungated input is disabled")
        gate_vector = np.ones(frames, dtype=np.int64)
        gating_mode = "ungated"
    frame_index = build_frame_index(gate_vector, view_vector)
    detector_vector = _vector(ds, "DetectorVector")
    if detector_vector is not None and np.unique(detector_vector).size > 1:
        raise UnsupportedDetectorLayoutError("version 1 supports one detector stream only")
    if frame_index.num_frames != frames:
        raise DICOMReadError("frame-index vectors do not cover pixel_array frames")
    counts = construct_counts(pixels, frame_index)
    if counts.dtype != np.int32:
        counts = counts.astype(np.int32, copy=False)

    start_deg, step_deg, direction, angle_source = _get_start_step_direction(
        ds, frame_index.num_views, config
    )
    sign = -1.0 if direction in {"CW", "CLOCKWISE"} else 1.0
    angles_rad = np.deg2rad(start_deg + sign * step_deg * np.arange(frame_index.num_views))
    spacing = _get_spacing(ds, config)
    radii_mm, radius_valid, radius_source = _get_radii(ds, frame_index.num_views, config)
    view_dwell_s = _get_view_dwell(ds, frame_index.num_views, config)
    phase, phase_source = _get_gate_phase(ds, frame_index.num_gates)

    valid = np.ones((frame_index.num_gates, frame_index.num_views), dtype=bool)
    vendor_exposure = None
    if config.vendor_exposure_keyword:
        raw_exposure = _vector(ds, config.vendor_exposure_keyword)
        if raw_exposure is None:
            raise InvalidExposureError(
                f"configured vendor exposure keyword {config.vendor_exposure_keyword!r} is absent"
            )
        if raw_exposure.size != frame_index.num_frames:
            raise InvalidExposureError("vendor exposure vector length does not equal frame count")
        vendor_exposure = np.zeros((frame_index.num_gates, frame_index.num_views), dtype=np.float64)
        occupied = np.zeros_like(valid)
        for value, gate, view in zip(raw_exposure, frame_index.gate_indices, frame_index.view_indices):
            if not np.isfinite(float(value)) or float(value) <= 0:
                raise InvalidExposureError("vendor exposure contains nonpositive values")
            if occupied[gate, view]:
                raise InvalidExposureError("duplicate vendor exposure frame")
            vendor_exposure[gate, view] = float(value)
            occupied[gate, view] = True

    exposure_result = estimate_exposure(
        counts,
        view_dwell_s,
        vendor_exposure_s=vendor_exposure,
        valid_frames=valid,
    )

    patient_source = _first_value(ds, ("PatientID",))
    study_source = _first_value(ds, ("StudyInstanceUID",))
    patient_key = _stable_key(patient_source, label="patient", config=config)
    study_key = _stable_key(study_source, label="study", config=config)
    sample_source = _first_value(ds, ("SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID"))
    sample_id = str(sample_source or source.stem)
    manufacturer = str(_first_value(ds, ("Manufacturer",)) or "unknown")
    model = str(_first_value(ds, ("ManufacturerModelName",)) or "unknown")
    scanner_id = f"{manufacturer}|{model}"
    stress_hint = str(
        _first_value(ds, ("StressDescription", "StudyID", "SeriesDescription"))
        or config.default_stress_rest
    ).lower()
    if "stress" in stress_hint:
        stress_rest = "stress"
    elif "rest" in stress_hint:
        stress_rest = "rest"
    else:
        stress_rest = config.default_stress_rest
    collimator_id = str(
        _first_value(ds, ("CollimatorGridName",))
        or _nested_value(ds, "DetectorInformationSequence", ("CollimatorGridName",))
        or config.default_collimator_id
    )
    energy_window = _get_energy_window(ds, config)
    geometry_payload = {
        "schema": 1,
        "shape": list(counts.shape),
        "angles_rad": np.round(angles_rad, 6).tolist(),
        "radii_mm": [
            round(float(radius), 3) if valid else None
            for radius, valid in zip(radii_mm, radius_valid)
        ],
        "radius_valid": radius_valid.tolist(),
        "spacing_mm": [round(float(x), 3) for x in spacing],
        "detector_stream": "detector_0",
        "direction": direction,
        "energy_window_kev": [
            round(float(x), 3) if math.isfinite(float(x)) else None for x in energy_window
        ],
        "collimator_id": collimator_id,
    }
    geometry_hash = _canonical_geometry_hash(geometry_payload)
    metadata = {
        "angle_source": angle_source,
        "phase_source": phase_source,
        "gating_mode": gating_mode,
        "radius_source": radius_source,
        "radius_missing_views": np.flatnonzero(~radius_valid).astype(int).tolist(),
        "geometry_verified": False,
        "num_frames": frames,
        "total_counts": int(counts.astype(np.int64).sum()),
        "zero_fraction": float(np.mean(counts == 0)),
        "bits_stored": int(getattr(ds, "BitsStored", 0) or 0),
        "exposure_warnings": list(exposure_result.warnings),
    }
    return GatedSPECTSample(
        sample_id=sample_id,
        counts=counts,
        frame_exposure_s=exposure_result.frame_exposure_s,
        relative_exposure_q=exposure_result.relative_exposure_q,
        valid_frames=valid,
        angles_rad=angles_rad.astype(np.float32),
        radii_mm=radii_mm.astype(np.float32),
        radius_valid=radius_valid,
        detector_spacing_mm=spacing,
        view_dwell_s=view_dwell_s.astype(np.float32),
        gate_phase_rad=phase.astype(np.float32),
        energy_window_kev=energy_window,
        scanner_id=scanner_id,
        collimator_id=collimator_id,
        stress_rest=stress_rest,
        detector_stream="detector_0",
        patient_key=patient_key,
        study_key=study_key,
        geometry_hash=geometry_hash,
        exposure_source=exposure_result.source,
        source_path=str(source),
        metadata=metadata,
    )
