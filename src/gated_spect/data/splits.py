"""Deterministic grouped patient-level split construction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


class SplitError(ValueError):
    """Raised when grouped split inputs are invalid."""


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 20260806
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    minimum_patients_per_split: int = 10
    dataset1_validation_fraction: float = 0.05
    dataset1_minimum_validation_patients: int = 10

    def __post_init__(self) -> None:
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(x <= 0 for x in fractions) or not np.isclose(sum(fractions), 1.0):
            raise SplitError("Dataset 2 fractions must be positive and sum to one")


def _largest_remainder(n: int, fractions: tuple[float, ...]) -> list[int]:
    raw = np.asarray(fractions, dtype=np.float64) * n
    result = np.floor(raw).astype(int)
    remainder = int(n - result.sum())
    order = np.argsort(-(raw - result), kind="stable")
    for index in order[:remainder]:
        result[int(index)] += 1
    return [int(x) for x in result]


def _balanced_slots(n: int, fractions: tuple[float, ...]) -> list[int]:
    """Spread exact global targets through an ordered patient list."""

    target = _largest_remainder(n, fractions)
    assigned = [0] * len(fractions)
    slots: list[int] = []
    for position in range(n):
        scores = [
            (position + 1) * fractions[i] - assigned[i] if assigned[i] < target[i] else -np.inf
            for i in range(len(fractions))
        ]
        chosen = int(np.argmax(scores))
        assigned[chosen] += 1
        slots.append(chosen)
    if assigned != target:
        raise SplitError(f"internal slot allocation error: {assigned} != {target}")
    return slots


def _stress_value(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text if text in {"stress", "rest", "mixed"} else "unknown"


def _patient_groups(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        patient = str(record.get("patient_key", ""))
        if not patient:
            raise SplitError("every record needs a nonempty patient_key")
        dataset = str(record.get("dataset_id", ""))
        if dataset not in {"dataset1", "dataset2"}:
            raise SplitError(f"unsupported dataset_id {dataset!r}")
        group = groups.setdefault(
            patient,
            {
                "patient_key": patient,
                "datasets": set(),
                "sample_ids": [],
                "scanner_ids": [],
                "stress_values": [],
                "total_counts": [],
            },
        )
        group["datasets"].add(dataset)
        group["sample_ids"].append(str(record.get("sample_id", "")))
        group["scanner_ids"].append(str(record.get("scanner_id", "unknown")) or "unknown")
        group["stress_values"].append(_stress_value(record.get("stress_rest")))
        group["total_counts"].append(float(record.get("total_counts", 0.0)))
    return groups


def _stratum_keys(groups: dict[str, dict[str, Any]]) -> dict[str, tuple[str, str, int]]:
    totals = np.asarray(
        [max(float(np.sum(group["total_counts"])), 0.0) for group in groups.values()],
        dtype=np.float64,
    )
    if totals.size == 0:
        raise SplitError("no patients found")
    boundaries = np.quantile(totals, [0.25, 0.50, 0.75], method="linear")
    result: dict[str, tuple[str, str, int]] = {}
    for patient, group in groups.items():
        scanner_values = [x for x in group["scanner_ids"] if x]
        scanner = sorted(set(scanner_values))[0] if scanner_values else "unknown"
        stress = "mixed" if len(set(group["stress_values"])) > 1 else group["stress_values"][0]
        total = max(float(np.sum(group["total_counts"])), 0.0)
        quartile = int(np.searchsorted(boundaries, total, side="right"))
        result[patient] = (scanner, stress, quartile)
    return result


def make_dataset2_splits(
    records: Iterable[Mapping[str, Any]],
    *,
    config: SplitConfig = SplitConfig(),
) -> dict[str, str]:
    """Return patient_key -> train/validation/test using deterministic grouped strata."""

    groups = _patient_groups(records)
    eligible = {p: g for p, g in groups.items() if "dataset2" in g["datasets"]}
    if len(eligible) < config.minimum_patients_per_split * 3:
        raise SplitError(
            f"Dataset 2 has {len(eligible)} patients; at least "
            f"{config.minimum_patients_per_split * 3} are required for configured split minimums"
        )
    keys = _stratum_keys(eligible)
    by_stratum: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for patient, stratum in keys.items():
        by_stratum[stratum].append(patient)
    rng = np.random.default_rng(config.seed)
    assigned: dict[str, str] = {}
    split_names = ("train", "validation", "test")
    ordered: list[str] = []
    # Round-robin across strata prevents rare strata from being consumed by
    # train before validation/test slots are assigned.
    buckets = {stratum: sorted(patients) for stratum, patients in by_stratum.items()}
    for patients in buckets.values():
        rng.shuffle(patients)
    while any(buckets.values()):
        for stratum in sorted(buckets):
            if buckets[stratum]:
                ordered.append(buckets[stratum].pop())
    slot_indices = _balanced_slots(
        len(ordered),
        (config.train_fraction, config.validation_fraction, config.test_fraction),
    )
    for patient, slot in zip(ordered, slot_indices):
        assigned[patient] = split_names[slot]

    counts = {name: sum(split == name for split in assigned.values()) for name in split_names}
    if any(counts[name] < config.minimum_patients_per_split for name in split_names):
        raise SplitError(f"split minimums not met: {counts}")
    return assigned


def make_dataset1_splits(
    records: Iterable[Mapping[str, Any]],
    *,
    excluded_patient_keys: set[str],
    config: SplitConfig = SplitConfig(),
) -> dict[str, str]:
    """Split Dataset 1 after excluding validation/test Dataset 2 patients."""

    groups = _patient_groups(records)
    eligible = {
        patient: group
        for patient, group in groups.items()
        if "dataset1" in group["datasets"] and patient not in excluded_patient_keys
    }
    n = len(eligible)
    validation_count = max(
        config.dataset1_minimum_validation_patients,
        int(round(config.dataset1_validation_fraction * n)),
    )
    if n < max(20, config.dataset1_minimum_validation_patients + 1):
        raise SplitError(f"Dataset 1 has {n} eligible patients; at least 20 are required")
    validation_count = min(validation_count, n - 1)
    keys = _stratum_keys(eligible)
    strata: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for patient, stratum in keys.items():
        strata[stratum].append(patient)
    rng = np.random.default_rng(config.seed)
    ordered: list[str] = []
    for stratum in sorted(strata):
        patients = sorted(strata[stratum])
        rng.shuffle(patients)
        ordered.extend(patients)
    validation = set(ordered[:validation_count])
    return {
        patient: ("validation" if patient in validation else "pretrain")
        for patient in eligible
    }
