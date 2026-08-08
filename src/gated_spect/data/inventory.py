"""De-identified, header-first DICOM inventory generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pydicom
from pydicom.dataset import Dataset

from ..types import ReaderConfig
from .dicom_reader import stable_identity_key


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(roots: Iterable[str | Path]) -> list[tuple[Path, Path]]:
    result: list[tuple[Path, Path]] = []
    for root_value in roots:
        root = Path(root_value).resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_file():
            result.append((root, root.parent))
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            result.append((path, root))
    return result


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "|".join(_safe_text(item) for item in value)
    return str(value)


def _root_id(root: Path) -> str:
    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]


def inventory_headers(
    roots: Iterable[str | Path],
    *,
    dataset_id: str,
    config: ReaderConfig,
) -> list[dict[str, Any]]:
    """Inventory candidate DICOM files without decoding pixel data."""

    if dataset_id not in {"dataset1", "dataset2"}:
        raise ValueError("dataset_id must be dataset1 or dataset2")
    rows: list[dict[str, Any]] = []
    for path, relative_root in _source_files(roots):
        try:
            ds: Dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
        except Exception as exc:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "relative_path": str(path.relative_to(relative_root)),
                    "parse_status": "unreadable",
                    "parse_error": type(exc).__name__,
                    "source_sha256": _file_sha256(path),
                    "source_root_id": _root_id(relative_root),
                }
            )
            continue
        patient_key = stable_identity_key(
            getattr(ds, "PatientID", None),
            label="patient",
            config=config,
        )
        study_key = stable_identity_key(
            getattr(ds, "StudyInstanceUID", None),
            label="study",
            config=config,
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "relative_path": str(path.relative_to(relative_root)),
                "source_sha256": _file_sha256(path),
                "parse_status": "header_ok",
                "parse_error": "",
                "patient_key": patient_key,
                "study_key": study_key,
                "sop_class_uid": _safe_text(getattr(ds, "SOPClassUID", None)),
                "sop_instance_uid": _safe_text(getattr(ds, "SOPInstanceUID", None)),
                "series_instance_uid": _safe_text(getattr(ds, "SeriesInstanceUID", None)),
                "modality": _safe_text(getattr(ds, "Modality", None)),
                "number_of_frames": int(getattr(ds, "NumberOfFrames", 1) or 1),
                "rows": int(getattr(ds, "Rows", 0) or 0),
                "columns": int(getattr(ds, "Columns", 0) or 0),
                "scanner_id": "|".join(
                    (
                        _safe_text(getattr(ds, "Manufacturer", None)) or "unknown",
                        _safe_text(getattr(ds, "ManufacturerModelName", None)) or "unknown",
                    )
                ),
                "stress_rest": _safe_text(getattr(ds, "StressDescription", None)).lower() or "unknown",
                "collimator_id": _safe_text(getattr(ds, "CollimatorGridName", None)) or "unknown",
                "source_root_id": _root_id(relative_root),
            }
        )
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], output: str | Path) -> Path:
    """Write a deterministic JSONL manifest without absolute patient paths."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (str(row.get("dataset_id")), str(row.get("relative_path"))))
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ordered:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return destination
