"""Full parsed acquisition manifests and deterministic split artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from ..types import ReaderConfig
from .dicom_reader import read_nm_dicom, stable_identity_key
from .splits import SplitConfig, make_dataset1_splits, make_dataset2_splits


def _root_id(root: Path) -> str:
    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_acquisition_manifest(
    root: str | Path,
    *,
    dataset_id: str,
    config: ReaderConfig,
) -> list[dict[str, Any]]:
    """Parse every DICOM and emit one de-identified row per acquisition."""

    if dataset_id not in {"dataset1", "dataset2"}:
        raise ValueError("dataset_id must be dataset1 or dataset2")
    root_path = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(root_path.rglob("*.dcm")):
        relative_path = str(path.relative_to(root_path))
        base: dict[str, Any] = {
            "dataset_id": dataset_id,
            "relative_path": relative_path,
            "source_root_id": _root_id(root_path),
            "parse_status": "error",
            "training_eligible": False,
        }
        try:
            sample = read_nm_dicom(path, config)
            header = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
            total_counts = int(sample.counts.astype(np.int64).sum())
            expected_value = getattr(header, "CountsAccumulated", None)
            if expected_value is None:
                count_status = "absent"
            else:
                count_status = "exact" if int(expected_value) == total_counts else "mismatch"
            valid_frames = np.asarray(sample.valid_frames, dtype=bool)
            q = np.asarray(sample.relative_exposure_q, dtype=np.float64)
            base.update(
                {
                    "parse_status": "parsed",
                    "parse_error": "",
                    "sample_key": stable_identity_key(sample.sample_id, label="sample", config=config),
                    "patient_key": sample.patient_key,
                    "study_key": sample.study_key,
                    "sample_id": sample.sample_id,
                    "shape": list(sample.shape),
                    "total_counts": total_counts,
                    "frame_exposure_total_s": float(np.asarray(sample.frame_exposure_s).sum()),
                    "relative_exposure_q_min": float(q[valid_frames].min()) if valid_frames.any() else 0.0,
                    "low_information_frame_count": int(np.count_nonzero(valid_frames & (q < 0.20))),
                    "valid_frame_fraction": float(valid_frames.mean()),
                    "count_status": count_status,
                    "training_eligible": bool(count_status != "mismatch" and valid_frames.all()),
                    "geometry_hash": sample.geometry_hash,
                    "geometry_verified": bool(sample.metadata.get("geometry_verified", False)),
                    "radius_valid_fraction": float(np.asarray(sample.radius_valid, dtype=bool).mean()),
                    "radius_source": str(sample.metadata.get("radius_source", "unknown")),
                    "gating_mode": str(sample.metadata.get("gating_mode", "unknown")),
                    "exposure_source": sample.exposure_source,
                    "energy_window_kev": list(sample.energy_window_kev),
                    "scanner_id": sample.scanner_id,
                    "collimator_id": sample.collimator_id,
                    "stress_rest": sample.stress_rest,
                    "detector_stream": sample.detector_stream,
                }
            )
        except Exception as exc:
            base.update({"parse_error": type(exc).__name__})
        rows.append({key: _json_safe(value) for key, value in base.items()})
    return rows


def write_manifest(rows: Iterable[Mapping[str, Any]], output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: str(row.get("relative_path", "")))
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ordered:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
    return destination


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_locked_splits(
    dataset1_rows: Iterable[Mapping[str, Any]],
    dataset2_rows: Iterable[Mapping[str, Any]],
    accepted_pair_rows: Iterable[Mapping[str, Any]],
    *,
    output: str | Path,
    config: SplitConfig = SplitConfig(),
) -> Path:
    """Write patient-level splits using only accepted Dataset 2 pair patients."""

    accepted_patients = {
        str(row["patient_key"])
        for row in accepted_pair_rows
        if bool(row.get("accepted")) and row.get("patient_key")
    }
    dataset2 = [
        row
        for row in dataset2_rows
        if bool(row.get("training_eligible")) and str(row.get("patient_key")) in accepted_patients
    ]
    if not dataset2:
        raise ValueError("no eligible accepted Dataset 2 acquisitions remain for splitting")
    dataset2_splits = make_dataset2_splits(dataset2, config=config)
    validation_test_patients = {
        patient for patient, split in dataset2_splits.items() if split in {"validation", "test"}
    }
    dataset1 = [row for row in dataset1_rows if bool(row.get("training_eligible"))]
    dataset1_splits = make_dataset1_splits(
        dataset1,
        excluded_patient_keys=validation_test_patients,
        config=config,
    )
    payload = {
        "algorithm": "grouped-stratified-v1",
        "seed": config.seed,
        "dataset2": dataset2_splits,
        "dataset1": dataset1_splits,
        "accepted_dataset2_patient_count": len(accepted_patients),
        "excluded_from_dataset1": sorted(validation_test_patients),
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return destination

