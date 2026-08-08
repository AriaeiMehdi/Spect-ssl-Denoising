"""Atomic, de-identified NumPy cache for canonical acquisitions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..types import GatedSPECTSample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")


def write_sample_cache(sample: GatedSPECTSample, root: str | Path) -> Path:
    """Write an accepted sample atomically and return its final directory."""

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    final = root_path / sample.sample_id
    if final.exists():
        existing_checksums = final / "checksums.json"
        if not existing_checksums.exists():
            raise FileExistsError(f"cache directory exists without checksums: {final}")
        expected = json.loads(existing_checksums.read_text(encoding="utf-8"))
        for name, checksum in expected.items():
            if not (final / name).exists() or _sha256(final / name) != checksum:
                raise IOError(f"cache checksum mismatch for {final / name}")
        return final

    temp_name = tempfile.mkdtemp(prefix=f".{sample.sample_id}.", dir=root_path)
    temp = Path(temp_name)
    try:
        np.save(temp / "counts.npy", np.asarray(sample.counts, dtype=np.int32))
        np.save(temp / "frame_exposure_s.npy", np.asarray(sample.frame_exposure_s, dtype=np.float32))
        np.save(temp / "relative_exposure_q.npy", np.asarray(sample.relative_exposure_q, dtype=np.float32))
        np.save(temp / "valid_frames.npy", np.asarray(sample.valid_frames, dtype=np.bool_))
        np.save(temp / "radius_valid.npy", np.asarray(sample.radius_valid, dtype=np.bool_))
        np.savez(
            temp / "geometry.npz",
            angles_rad=np.asarray(sample.angles_rad, dtype=np.float32),
            radii_mm=np.asarray(sample.radii_mm, dtype=np.float32),
            view_dwell_s=np.asarray(sample.view_dwell_s, dtype=np.float32),
            gate_phase_rad=np.asarray(sample.gate_phase_rad, dtype=np.float32),
            detector_spacing_mm=np.asarray(sample.detector_spacing_mm, dtype=np.float32),
        )
        metadata = {
            "sample_id": sample.sample_id,
            "patient_key": sample.patient_key,
            "study_key": sample.study_key,
            "energy_window_kev": list(sample.energy_window_kev),
            "scanner_id": sample.scanner_id,
            "collimator_id": sample.collimator_id,
            "stress_rest": sample.stress_rest,
            "detector_stream": sample.detector_stream,
            "geometry_hash": sample.geometry_hash,
            "exposure_source": sample.exposure_source,
            "source_path": sample.source_path,
            "metadata": dict(sample.metadata),
        }
        _write_json(temp / "metadata.json", metadata)
        checksums = {
            name: _sha256(temp / name)
            for name in (
                "counts.npy",
                "frame_exposure_s.npy",
                "relative_exposure_q.npy",
                "valid_frames.npy",
                "radius_valid.npy",
                "geometry.npz",
                "metadata.json",
            )
        }
        _write_json(temp / "checksums.json", checksums)
        os.replace(temp, final)
    except Exception:
        for child in temp.iterdir():
            child.unlink()
        temp.rmdir()
        raise
    return final


def read_sample_cache(path: str | Path, *, mmap_counts: bool = True) -> GatedSPECTSample:
    """Read and verify a canonical cache directory."""

    directory = Path(path)
    checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
    for name, expected in checksums.items():
        actual_path = directory / name
        if not actual_path.exists() or _sha256(actual_path) != expected:
            raise IOError(f"cache checksum mismatch for {actual_path}")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    geometry = np.load(directory / "geometry.npz")
    return GatedSPECTSample(
        sample_id=str(metadata["sample_id"]),
        counts=np.load(directory / "counts.npy", mmap_mode="r" if mmap_counts else None),
        frame_exposure_s=np.load(directory / "frame_exposure_s.npy"),
        relative_exposure_q=np.load(directory / "relative_exposure_q.npy"),
        valid_frames=np.load(directory / "valid_frames.npy"),
        radius_valid=np.load(directory / "radius_valid.npy"),
        angles_rad=geometry["angles_rad"],
        radii_mm=geometry["radii_mm"],
        detector_spacing_mm=tuple(float(x) for x in geometry["detector_spacing_mm"]),
        view_dwell_s=geometry["view_dwell_s"],
        gate_phase_rad=geometry["gate_phase_rad"],
        energy_window_kev=tuple(float(x) for x in metadata["energy_window_kev"]),
        scanner_id=str(metadata["scanner_id"]),
        collimator_id=str(metadata["collimator_id"]),
        stress_rest=str(metadata["stress_rest"]),
        detector_stream=str(metadata["detector_stream"]),
        patient_key=str(metadata["patient_key"]),
        study_key=str(metadata["study_key"]),
        geometry_hash=str(metadata["geometry_hash"]),
        exposure_source=str(metadata["exposure_source"]),
        source_path=str(metadata["source_path"]),
        metadata=metadata.get("metadata", {}),
    )
