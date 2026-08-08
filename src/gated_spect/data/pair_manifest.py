"""Filename-pair discovery and de-identified pair-manifest generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..types import PairProvenance, ReaderConfig
from .dicom_reader import read_nm_dicom, stable_identity_key
from .pairing import verify_pair


@dataclass(frozen=True)
class DurationPair:
    pair_key: str
    low_path: Path
    high_path: Path


def discover_duration_pairs(root: str | Path) -> tuple[list[DurationPair], list[str]]:
    """Discover exact 10sec/30sec filename pairs without reading identifiers."""

    root_path = Path(root).resolve()
    by_key: dict[str, dict[str, Path]] = {}
    for path in sorted(root_path.glob("*.dcm")):
        if path.name.endswith(" 10sec.dcm"):
            key, kind = path.name[: -len(" 10sec.dcm")], "low"
        elif path.name.endswith(" 30sec.dcm"):
            key, kind = path.name[: -len(" 30sec.dcm")], "high"
        else:
            continue
        by_key.setdefault(key, {})[kind] = path
    pairs: list[DurationPair] = []
    incomplete: list[str] = []
    for key in sorted(by_key):
        value = by_key[key]
        if set(value) != {"low", "high"}:
            incomplete.append(key)
            continue
        pairs.append(DurationPair(key, value["low"], value["high"]))
    return pairs, incomplete


def audit_duration_pairs(
    root: str | Path,
    *,
    config: ReaderConfig,
    radius_tolerance_mm: float = 2.0,
) -> list[dict[str, Any]]:
    """Parse and verify all discovered pairs, returning de-identified rows."""

    root_path = Path(root).resolve()
    pairs, incomplete = discover_duration_pairs(root_path)
    rows: list[dict[str, Any]] = []
    for key in incomplete:
        rows.append(
            {
                "pair_id": stable_identity_key(key, label="pair", config=config),
                "accepted": False,
                "provenance": PairProvenance.INDEPENDENT.value,
                "reason": "incomplete_filename_pair",
                "metrics": {},
            }
        )
    for pair in pairs:
        try:
            low = read_nm_dicom(pair.low_path, config)
            high = read_nm_dicom(pair.high_path, config)
            result = verify_pair(
                low,
                high,
                PairProvenance.INDEPENDENT,
                radius_tolerance_mm=radius_tolerance_mm,
            )
            rows.append(
                {
                    "pair_id": stable_identity_key(pair.pair_key, label="pair", config=config),
                    "patient_key": low.patient_key,
                    "low_sample_key": stable_identity_key(low.sample_id, label="sample", config=config),
                    "high_sample_key": stable_identity_key(high.sample_id, label="sample", config=config),
                    "low_relative_path": str(pair.low_path.relative_to(root_path)),
                    "high_relative_path": str(pair.high_path.relative_to(root_path)),
                    "accepted": result.accepted,
                    "provenance": result.provenance.value,
                    "reason": result.reason,
                    "metrics": dict(result.metrics),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "pair_id": stable_identity_key(pair.pair_key, label="pair", config=config),
                    "accepted": False,
                    "provenance": PairProvenance.INDEPENDENT.value,
                    "reason": f"parse_error:{type(exc).__name__}",
                    "metrics": {},
                }
            )
    return sorted(rows, key=lambda row: str(row["pair_id"]))


def write_pair_manifest(rows: Iterable[dict[str, Any]], output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return destination
