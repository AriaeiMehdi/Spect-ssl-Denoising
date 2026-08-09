"""Explicit v1→v2 exposure migration analysis.

The migration tool intentionally reads legacy artifacts without feeding them
back into version-2 training.  It recomputes both exposure interpretations from
the raw DICOM and writes a quantitative compatibility decision.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pydicom

from ..types import ReaderConfig
from .dicom_reader import _first_value, _nested_value, read_nm_dicom
from .exposure import estimate_exposure


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats_payload(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stats = payload.get("stats", payload)
    return {
        "count_log_mean": float(stats["count_log_mean"]),
        "count_log_std": float(stats["count_log_std"]),
        "log_exposure_mean": float(stats["log_exposure_mean"]),
        "log_exposure_std": float(stats["log_exposure_std"]),
        "radius_mean_mm": float(stats["radius_mean_mm"]),
        "radius_std_mm": float(stats["radius_std_mm"]),
    }


def _legacy_view_dwell(path: Path, sample_views: int, corrected_view_dwell: np.ndarray) -> np.ndarray:
    """Reproduce only the legacy ActualFrameDuration interpretation."""

    dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    value = _first_value(dataset, ("AngularViewDwellTime", "ViewDwellTime"))
    if value is not None:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            return np.repeat(arr, sample_views)
        if arr.size == sample_views:
            return arr
    value = _first_value(dataset, ("AngularStopDuration",))
    if value is not None:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            return np.repeat(arr / sample_views, sample_views)
        return arr
    value = _nested_value(dataset, "RotationInformationSequence", ("ActualFrameDuration",))
    if value is not None:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            return np.repeat(arr / 1000.0 / sample_views, sample_views)
        return arr / 1000.0
    return np.asarray(corrected_view_dwell, dtype=np.float64)


def _z_values(exposure: np.ndarray, stats: Mapping[str, float], valid: np.ndarray) -> np.ndarray:
    values = np.log(np.clip(np.asarray(exposure, dtype=np.float64), 1.0e-6, None))
    z = (values - float(stats["log_exposure_mean"])) / float(stats["log_exposure_std"])
    return z[np.asarray(valid, dtype=bool)]


def build_exposure_migration_report(
    datasets: Iterable[tuple[str, str | Path, Iterable[Mapping[str, Any]]]],
    *,
    config: ReaderConfig,
    old_normalization: str | Path,
    new_normalization: str | Path,
    new_manifests: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    old_splits: Mapping[str, Any] | None = None,
    new_splits: Mapping[str, Any] | None = None,
    legacy_checkpoint: str | Path | None = None,
    old_dataset1_manifest: str | Path | None = None,
    new_dataset1_manifest: str | Path | None = None,
    old_dataset2_manifest: str | Path | None = None,
    new_dataset2_manifest: str | Path | None = None,
    new_dataset2_pairs_manifest: str | Path | None = None,
    old_splits_path: str | Path | None = None,
    new_splits_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare legacy and corrected exposure values for raw-DICOM samples."""

    old_stats = _stats_payload(old_normalization)
    new_stats = _stats_payload(new_normalization)
    rows: list[dict[str, Any]] = []
    all_delta: list[float] = []
    geometry_same = True
    for dataset_id, root_value, records in datasets:
        root = Path(root_value).resolve()
        new_by_sample = {
            str(row.get("sample_id")): row
            for row in (new_manifests or {}).get(dataset_id, [])
            if row.get("sample_id")
        }
        for record in records:
            if record.get("parse_status") not in (None, "parsed"):
                continue
            relative = str(record.get("relative_path", ""))
            if not relative:
                continue
            path = root / relative
            try:
                sample = read_nm_dicom(path, config)
                legacy_dwell = _legacy_view_dwell(path, sample.num_views, sample.view_dwell_s)
                legacy_exposure = estimate_exposure(
                    sample.counts,
                    legacy_dwell,
                    valid_frames=sample.valid_frames,
                ).frame_exposure_s
                valid = np.asarray(sample.valid_frames, dtype=bool)
                old_z = _z_values(legacy_exposure, old_stats, valid)
                new_z = _z_values(sample.frame_exposure_s, new_stats, valid)
                delta = new_z - old_z
                all_delta.extend(np.abs(delta).tolist())
                sample_key = str(record.get("sample_id", sample.sample_id))
                new_record = new_by_sample.get(sample_key)
                if new_record is not None and record.get("geometry_hash") is not None:
                    geometry_same &= str(record.get("geometry_hash")) == str(new_record.get("geometry_hash"))
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "sample_id": sample.sample_id,
                        "old_median_dwell_s": float(np.median(legacy_dwell)),
                        "new_median_dwell_s": float(np.median(sample.view_dwell_s)),
                        "actual_frame_duration_ms": sample.metadata.get("actual_frame_duration_ms"),
                        "dwell_ratio_new_over_old": float(
                            np.median(sample.view_dwell_s) / max(np.median(legacy_dwell), 1.0e-12)
                        ),
                        "old_mean_frame_exposure_s": float(old_z.size and np.mean(legacy_exposure[valid])),
                        "new_mean_frame_exposure_s": float(np.mean(sample.frame_exposure_s[valid])),
                        "old_standardized_log_exposure_mean": float(np.mean(old_z)),
                        "new_standardized_log_exposure_mean": float(np.mean(new_z)),
                        "mean_abs_delta_z": float(np.mean(np.abs(delta))),
                        "p99_abs_delta_z": float(np.percentile(np.abs(delta), 99)),
                        "view_dwell_source_v2": str(sample.metadata.get("view_dwell_source", "unknown")),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "sample_id": str(record.get("sample_id", relative)),
                        "migration_error": type(exc).__name__,
                    }
                )

    delta_array = np.asarray(all_delta, dtype=np.float64)
    count_norm_equal = all(
        np.isclose(old_stats[key], new_stats[key], rtol=0.0, atol=1.0e-12)
        for key in ("count_log_mean", "count_log_std")
    )
    mean_abs_delta = float(delta_array.mean()) if delta_array.size else float("inf")
    p99_abs_delta = float(np.percentile(delta_array, 99)) if delta_array.size else float("inf")
    split_summary: dict[str, Any] = {
        "provided": old_splits is not None and new_splits is not None,
        "unchanged": None,
        "changed_assignments": None,
    }
    if old_splits is not None and new_splits is not None:
        changed = 0
        compared = 0
        for dataset_id in ("dataset1", "dataset2"):
            old_map = old_splits.get(dataset_id, {})
            new_map = new_splits.get(dataset_id, {})
            for patient_key, old_assignment in old_map.items():
                if patient_key in new_map:
                    compared += 1
                    changed += int(new_map[patient_key] != old_assignment)
        split_summary["unchanged"] = changed == 0
        split_summary["changed_assignments"] = changed
        split_summary["compared_assignments"] = compared

    artifact_hashes = {
        "legacy_checkpoint_sha256": _sha256_file(legacy_checkpoint)
        if legacy_checkpoint is not None
        else None,
        "old_dataset1_manifest_sha256": _sha256_file(old_dataset1_manifest)
        if old_dataset1_manifest is not None
        else None,
        "new_dataset1_manifest_sha256": _sha256_file(new_dataset1_manifest)
        if new_dataset1_manifest is not None
        else None,
        "old_dataset2_manifest_sha256": _sha256_file(old_dataset2_manifest)
        if old_dataset2_manifest is not None
        else None,
        "new_dataset2_manifest_sha256": _sha256_file(new_dataset2_manifest)
        if new_dataset2_manifest is not None
        else None,
        # Dataset2 training consumes the derived pair manifest.  Keep this
        # optional for compatibility with the normative acquisition-manifest
        # schema, but include it whenever the migration audit is intended to
        # authorize a denoising run.
        "new_dataset2_pairs_sha256": _sha256_file(new_dataset2_pairs_manifest)
        if new_dataset2_pairs_manifest is not None
        else None,
        "old_splits_sha256": _sha256_file(old_splits_path)
        if old_splits_path is not None
        else None,
        "new_splits_sha256": _sha256_file(new_splits_path)
        if new_splits_path is not None
        else None,
        "old_normalization_sha256": _sha256_file(old_normalization),
        "new_normalization_sha256": _sha256_file(new_normalization),
    }
    required_artifact_keys = (
        "legacy_checkpoint_sha256",
        "old_dataset1_manifest_sha256",
        "new_dataset1_manifest_sha256",
        "old_dataset2_manifest_sha256",
        "new_dataset2_manifest_sha256",
        "old_splits_sha256",
        "new_splits_sha256",
        "old_normalization_sha256",
        "new_normalization_sha256",
    )
    artifact_hashes_complete = all(
        artifact_hashes[key] is not None for key in required_artifact_keys
    )
    compatibility = bool(
        count_norm_equal
        and geometry_same
        and mean_abs_delta < 1.0e-3
        and p99_abs_delta < 1.0e-2
        and artifact_hashes_complete
        and split_summary["unchanged"] is True
    )
    return {
        "migration_schema_version": 2,
        "scientific_semantics_version_old": 1,
        "scientific_semantics_version_new": 2,
        "exposure_semantics_version_old": 1,
        "exposure_semantics_version_new": 2,
        "legacy_backbone_compatibility": compatibility,
        "legacy_backbone_compatibility_reason": (
            "standardized exposure inputs and count/geometry normalization satisfy the v2 gate"
            if compatibility
            else "one or more standardized exposure, count normalization, or geometry gates failed"
        ),
        "count_normalization_unchanged": count_norm_equal,
        "geometry_inputs_unchanged": geometry_same,
        "mean_abs_delta_z": mean_abs_delta,
        "p95_abs_delta_z": float(np.percentile(delta_array, 95)) if delta_array.size else float("inf"),
        "p99_abs_delta_z": p99_abs_delta,
        "max_abs_delta_z": float(delta_array.max()) if delta_array.size else float("inf"),
        "legacy_rate_decoder_physically_calibrated": False,
        "patient_split_migration": split_summary,
        "artifact_hashes_complete": artifact_hashes_complete,
        **artifact_hashes,
        "rows": rows,
    }


def write_exposure_migration_report(
    report: Mapping[str, Any], json_path: str | Path, csv_path: str | Path
) -> None:
    """Write the machine-readable JSON and tabular migration reports."""

    json_destination = Path(json_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(report.get("rows", []))
    csv_destination = Path(csv_path)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with csv_destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
