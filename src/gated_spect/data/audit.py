"""Reproducible aggregate audit over a DICOM root."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pydicom

from ..types import ReaderConfig
from .dicom_reader import read_nm_dicom


def audit_root(root: str | Path, *, dataset_id: str, config: ReaderConfig) -> dict[str, Any]:
    """Parse every DICOM and return aggregate-only QC results.

    No patient/study identifiers or source paths are written to the result.
    """

    root_path = Path(root).resolve()
    paths = sorted(root_path.rglob("*.dcm"))
    errors: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    gating_modes: Counter[str] = Counter()
    radius_sources: Counter[str] = Counter()
    exposure_sources: Counter[str] = Counter()
    energy_windows: Counter[str] = Counter()
    counts_accumulated = Counter()
    counts_accumulated_mismatch_files = 0
    total_counts = 0
    for path in paths:
        try:
            sample = read_nm_dicom(path, config)
            shape_key = "x".join(str(x) for x in sample.shape)
            shapes[shape_key] += 1
            gating_modes[str(sample.metadata["gating_mode"])] += 1
            radius_sources[str(sample.metadata["radius_source"])] += 1
            exposure_sources[sample.exposure_source] += 1
            energy_windows[str(tuple(float(x) for x in sample.energy_window_kev))] += 1
            current_total = int(sample.counts.astype("int64").sum())
            total_counts += current_total
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
            expected = getattr(ds, "CountsAccumulated", None)
            if expected is None:
                counts_accumulated["absent"] += 1
            elif int(expected) == current_total:
                counts_accumulated["exact"] += 1
            else:
                counts_accumulated["mismatch"] += 1
                counts_accumulated_mismatch_files += 1
        except Exception as exc:
            errors[type(exc).__name__] += 1
    parsed = len(paths) - sum(errors.values())
    return {
        "dataset_id": dataset_id,
        "root_name": root_path.name,
        "files_seen": len(paths),
        "files_parsed": parsed,
        "training_eligible_files": parsed - counts_accumulated_mismatch_files,
        "parse_success_fraction": parsed / len(paths) if paths else 0.0,
        "errors": dict(sorted(errors.items())),
        "shapes": dict(sorted(shapes.items())),
        "gating_modes": dict(sorted(gating_modes.items())),
        "radius_sources": dict(sorted(radius_sources.items())),
        "exposure_sources": dict(sorted(exposure_sources.items())),
        "energy_windows": dict(sorted(energy_windows.items())),
        "counts_accumulated": dict(sorted(counts_accumulated.items())),
        "total_counts": total_counts,
    }
