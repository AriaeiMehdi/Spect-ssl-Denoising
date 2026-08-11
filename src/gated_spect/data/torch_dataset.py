"""PyTorch datasets that preserve the SPECT count/exposure contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..losses import binomial_count_split, count_split_seed, make_mask_plan
from ..types import PairProvenance, ReaderConfig
from .dicom_reader import read_nm_dicom


@dataclass(frozen=True)
class NormalizationStats:
    """Streaming statistics frozen in a training manifest/checkpoint."""

    count_log_mean: float = 0.0
    count_log_std: float = 1.0
    log_exposure_mean: float = 0.0
    log_exposure_std: float = 1.0
    radius_mean_mm: float = 0.0
    radius_std_mm: float = 1.0

    def __post_init__(self) -> None:
        if self.count_log_std <= 0 or self.log_exposure_std <= 0 or self.radius_std_mm <= 0:
            raise ValueError("normalization standard deviations must be positive")

    def transform_counts(self, counts: Tensor) -> Tensor:
        transformed = torch.log1p(counts.to(dtype=torch.float32).clamp_min(0.0))
        transformed = (transformed - self.count_log_mean) / self.count_log_std
        return transformed.clamp(-8.0, 8.0)

    def to_dict(self) -> dict[str, float]:
        return {
            "count_log_mean": self.count_log_mean,
            "count_log_std": self.count_log_std,
            "log_exposure_mean": self.log_exposure_mean,
            "log_exposure_std": self.log_exposure_std,
            "radius_mean_mm": self.radius_mean_mm,
            "radius_std_mm": self.radius_std_mm,
        }

    @classmethod
    def from_json(cls, path: str | Path) -> "NormalizationStats":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        from ..versioning import require_artifact_versions

        require_artifact_versions(payload, artifact_name=f"normalization {path}")
        if "stats" in payload:
            payload = payload["stats"]
        return cls(**payload)


class _Welford:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return
        count = int(flat.size)
        mean = float(flat.mean(dtype=np.float64))
        centered = flat - mean
        m2 = float(np.dot(centered, centered))
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean, m2
            return
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def finalize(self) -> tuple[float, float, int]:
        if self.count < 2:
            raise ValueError("at least two observations are required for normalization")
        variance = max(self.m2 / self.count, 1.0e-12)
        return self.mean, float(np.sqrt(variance)), self.count


def compute_normalization_stats(
    root: str | Path,
    records: Sequence[Mapping[str, Any]] | str | Path,
    *,
    config: ReaderConfig,
    patient_splits: Mapping[str, str] | None = None,
) -> tuple[NormalizationStats, dict[str, int]]:
    """Compute streaming log-count/exposure/radius statistics from eligible data."""

    rows = _read_records(records)
    selected = [
        row
        for row in rows
        if bool(row.get("training_eligible", True))
        and (patient_splits is None or patient_splits.get(str(row.get("patient_key"))) == "pretrain")
    ]
    if not selected:
        raise ValueError("no eligible records selected for normalization")
    root_path = Path(root).resolve()
    count_stats, exposure_stats, radius_stats = _Welford(), _Welford(), _Welford()
    files = 0
    for row in selected:
        sample = read_nm_dicom(root_path / str(row["relative_path"]), config)
        count_stats.update(np.log1p(np.asarray(sample.counts, dtype=np.float64)))
        valid = np.asarray(sample.valid_frames, dtype=bool)
        exposure = np.asarray(sample.frame_exposure_s, dtype=np.float64)
        exposure_stats.update(np.log(np.clip(exposure[valid], 1.0e-6, None)))
        radius_valid = np.asarray(sample.radius_valid, dtype=bool)
        radii = np.asarray(sample.radii_mm, dtype=np.float64)
        radius_stats.update(radii[radius_valid])
        files += 1
    count_mean, count_std, count_elements = count_stats.finalize()
    exposure_mean, exposure_std, exposure_elements = exposure_stats.finalize()
    radius_mean, radius_std, radius_elements = radius_stats.finalize()
    return (
        NormalizationStats(
            count_log_mean=count_mean,
            count_log_std=count_std,
            log_exposure_mean=exposure_mean,
            log_exposure_std=exposure_std,
            radius_mean_mm=radius_mean,
            radius_std_mm=radius_std,
        ),
        {
            "files": files,
            "count_elements": count_elements,
            "exposure_elements": exposure_elements,
            "radius_elements": radius_elements,
        },
    )


def _vocabulary(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    values = sorted({str(row.get(field, "unknown") or "unknown") for row in records})
    # Index zero is reserved for unknown categories.
    return {value: index + 1 for index, value in enumerate(values) if value != "unknown"}


def _category_index(vocabulary: Mapping[str, int], value: str) -> int:
    return int(vocabulary.get(str(value or "unknown"), 0))


def _read_records(records: Sequence[Mapping[str, Any]] | str | Path) -> list[dict[str, Any]]:
    if isinstance(records, (str, Path)):
        with Path(records).open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return [dict(row) for row in records]


class AcquisitionDataset(Dataset[dict[str, Any]]):
    """Read complete canonical acquisitions and construct model input tensors."""

    def __init__(
        self,
        root: str | Path,
        records: Sequence[Mapping[str, Any]] | str | Path,
        *,
        config: ReaderConfig,
        normalization: NormalizationStats = NormalizationStats(),
        training_only: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        loaded = _read_records(records)
        self.records = [
            row for row in loaded if not training_only or bool(row.get("training_eligible", True))
        ]
        if not self.records:
            raise ValueError("acquisition dataset contains no records")
        self.config = config
        self.normalization = normalization
        self.scanner_vocab = _vocabulary(self.records, "scanner_id")
        self.collimator_vocab = _vocabulary(self.records, "collimator_id")
        self.stress_vocab = _vocabulary(self.records, "stress_rest")

    def __len__(self) -> int:
        return len(self.records)

    def _read(self, index: int) -> tuple[dict[str, Any], Any]:
        record = self.records[index]
        relative_path = str(record.get("relative_path", ""))
        if not relative_path:
            raise ValueError("manifest record has no relative_path")
        sample = read_nm_dicom(self.root / relative_path, self.config)
        raw_counts = torch.from_numpy(np.asarray(sample.counts, dtype=np.int64))
        valid_frames = torch.from_numpy(np.asarray(sample.valid_frames, dtype=np.bool_))
        exposure = torch.from_numpy(np.asarray(sample.frame_exposure_s, dtype=np.float32))
        q = torch.from_numpy(np.asarray(sample.relative_exposure_q, dtype=np.float32))
        h, w = sample.detector_shape
        visibility = valid_frames[:, :, None, None].expand(*valid_frames.shape, h, w).clone()
        item = {
            "counts_raw": raw_counts,
            "counts_input": self.normalization.transform_counts(raw_counts)[:, :, None],
            "input_exposure_s": exposure,
            "relative_exposure_q": q,
            "visibility": visibility,
            "valid_frames": valid_frames,
            "angles_rad": torch.from_numpy(np.asarray(sample.angles_rad, dtype=np.float32)),
            "radii_mm": torch.from_numpy(np.asarray(sample.radii_mm, dtype=np.float32)),
            "gate_phase_rad": torch.from_numpy(np.asarray(sample.gate_phase_rad, dtype=np.float32)),
            "split_fraction": torch.zeros(1, dtype=torch.float32),
            "scanner_index": torch.tensor(
                _category_index(self.scanner_vocab, sample.scanner_id), dtype=torch.long
            ),
            "collimator_index": torch.tensor(
                _category_index(self.collimator_vocab, sample.collimator_id), dtype=torch.long
            ),
            "stress_rest_index": torch.tensor(
                _category_index(self.stress_vocab, sample.stress_rest), dtype=torch.long
            ),
            "meta": {
                "sample_id": sample.sample_id,
                "patient_key": sample.patient_key,
                "study_key": sample.study_key,
                "relative_path": relative_path,
                "geometry_hash": sample.geometry_hash,
                "shape": sample.shape,
            },
        }
        return item, sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        item, _ = self._read(index)
        return item


class PretrainingDataset(Dataset[dict[str, Any]]):
    """Deterministic per-epoch binomial thinning and masking for Dataset 1."""

    def __init__(
        self,
        base: AcquisitionDataset,
        *,
        global_seed: int = 20260806,
        epoch: int = 0,
    ) -> None:
        self.base = base
        self.global_seed = int(global_seed)
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_view(self, index: int, *, seed_offset: int = 0, teacher: bool = False) -> dict[str, Any]:
        item, sample = self.base._read(index)
        sample_id = str(item["meta"]["sample_id"])
        stream = "teacher" if teacher else ("student_a" if int(seed_offset) == 0 else "student_b")
        seed = count_split_seed(
            self.global_seed, self.epoch, sample_id, stream
        )
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        p = float(0.4 + 0.3 * torch.rand((), generator=rng))
        first, second = binomial_count_split(item["counts_raw"], p, seed=seed ^ 0x5DEECE66D)
        if teacher:
            # The EMA teacher sees an independent complement without an
            # artificial gate/view mask.  Invalid clinical frames remain invalid.
            mask_visibility = item["visibility"]
            masked_first = second
            mask_task = "teacher_unmasked"
            masked_target = torch.zeros_like(mask_visibility)
            visible_target = torch.zeros_like(mask_visibility)
        else:
            mask = make_mask_plan(
                item["valid_frames"],
                item["relative_exposure_q"],
                height=sample.detector_shape[0],
                width=sample.detector_shape[1],
                seed=seed ^ 0x9E3779B9,
            )
            mask_visibility = mask.visibility
            masked_first = first.masked_fill(~mask_visibility, 0)
            masked_target = mask.masked_target
            visible_target = mask.visible_target
            mask_task = mask.task
        input_fraction = (1.0 - p) if teacher else p
        input_exposure = item["input_exposure_s"] * input_fraction
        target_exposure = item["input_exposure_s"] * (1.0 - input_fraction)
        target_weight = (
            item["valid_frames"]
            & (item["relative_exposure_q"] >= 0.20)
            & (target_exposure > 0)
        ).to(dtype=torch.float32)
        target_counts = first if teacher else second
        item.update(
            {
                "counts_input": self.base.normalization.transform_counts(masked_first)[:, :, None],
                "input_exposure_s": input_exposure,
                "target_counts": target_counts,
                "input_counts_raw": second if teacher else first,
                "target_exposure_s": target_exposure,
                "target_weight": target_weight,
                "masked_target": masked_target,
                "visible_target": visible_target,
                "split_fraction": torch.tensor([input_fraction], dtype=torch.float32),
                "split_probability": input_fraction,
                "mask_task": mask_task,
            }
        )
        item["visibility"] = mask_visibility
        item["masked_target"] = masked_target
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._build_view(index)


class TwoViewPretrainingDataset(Dataset[dict[str, Any]]):
    """Return two independent student views and an unmasked EMA-teacher view."""

    def __init__(
        self,
        base: AcquisitionDataset,
        *,
        global_seed: int = 20260806,
        epoch: int = 0,
    ) -> None:
        self.base = base
        self.global_seed = int(global_seed)
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _dataset(self) -> PretrainingDataset:
        return PretrainingDataset(self.base, global_seed=self.global_seed, epoch=self.epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset = self._dataset()
        return {
            "student_a": dataset._build_view(index, seed_offset=0),
            "student_b": dataset._build_view(index, seed_offset=1),
            "teacher": dataset._build_view(index, seed_offset=2, teacher=True),
        }


class PairDataset(Dataset[dict[str, Any]]):
    """Construct supervised low/target acquisitions from an accepted pair manifest."""

    def __init__(
        self,
        root: str | Path,
        pair_records: Sequence[Mapping[str, Any]] | str | Path,
        *,
        config: ReaderConfig,
        normalization: NormalizationStats = NormalizationStats(),
        scanner_vocab: Mapping[str, int] | None = None,
        collimator_vocab: Mapping[str, int] | None = None,
        stress_rest_vocab: Mapping[str, int] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        rows = _read_records(pair_records)
        self.records = [row for row in rows if bool(row.get("accepted"))]
        if not self.records:
            raise ValueError("pair dataset contains no accepted pairs")
        self.config = config
        self.normalization = normalization
        self.scanner_vocab = dict(scanner_vocab) if scanner_vocab is not None else _vocabulary(self.records, "scanner_id")
        self.collimator_vocab = (
            dict(collimator_vocab)
            if collimator_vocab is not None
            else _vocabulary(self.records, "collimator_id")
        )
        self.stress_vocab = (
            dict(stress_rest_vocab)
            if stress_rest_vocab is not None
            else _vocabulary(self.records, "stress_rest")
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        low = read_nm_dicom(self.root / str(row["low_relative_path"]), self.config)
        high = read_nm_dicom(self.root / str(row["high_relative_path"]), self.config)
        provenance = PairProvenance(str(row.get("provenance", PairProvenance.INDEPENDENT.value)))
        if low.shape != high.shape:
            raise ValueError("accepted pair has incompatible shapes")
        if provenance is PairProvenance.NESTED:
            target_counts = high.counts.astype(np.int64) - low.counts.astype(np.int64)
            target_exposure = high.frame_exposure_s - low.frame_exposure_s
            target_q = target_exposure / np.asarray(low.view_dwell_s, dtype=np.float32)[None, :]
        elif provenance is PairProvenance.INDEPENDENT:
            target_counts = high.counts.astype(np.int64)
            target_exposure = high.frame_exposure_s
            target_q = high.relative_exposure_q
        else:
            raise ValueError("synthetic pair rows require an explicit synthetic generator")
        if np.any(target_counts < 0) or np.any(target_exposure <= 0):
            raise ValueError("accepted pair produced a negative target or invalid exposure")
        raw_low = torch.from_numpy(low.counts.astype(np.int64))
        target = torch.from_numpy(target_counts)
        exposure = torch.from_numpy(low.frame_exposure_s.astype(np.float32))
        target_exposure_tensor = torch.from_numpy(target_exposure.astype(np.float32))
        valid = torch.from_numpy(np.asarray(low.valid_frames & high.valid_frames, dtype=np.bool_))
        q = torch.from_numpy(np.asarray(target_q, dtype=np.float32))
        h, w = low.detector_shape
        return {
            "counts_raw": raw_low,
            "counts_input": self.normalization.transform_counts(raw_low)[:, :, None],
            "input_exposure_s": exposure,
            "target_counts": target,
            "target_exposure_s": target_exposure_tensor,
            "target_weight": (valid & (q >= 0.20)).to(dtype=torch.float32),
            "relative_exposure_q": q,
            "visibility": valid[:, :, None, None].expand(*valid.shape, h, w).clone(),
            "valid_frames": valid,
            "angles_rad": torch.from_numpy(low.angles_rad.astype(np.float32)),
            "radii_mm": torch.from_numpy(low.radii_mm.astype(np.float32)),
            "gate_phase_rad": torch.from_numpy(low.gate_phase_rad.astype(np.float32)),
            "split_fraction": torch.zeros(1, dtype=torch.float32),
            "scanner_index": torch.tensor(
                _category_index(self.scanner_vocab, low.scanner_id), dtype=torch.long
            ),
            "collimator_index": torch.tensor(
                _category_index(self.collimator_vocab, low.collimator_id), dtype=torch.long
            ),
            "stress_rest_index": torch.tensor(
                _category_index(self.stress_vocab, low.stress_rest), dtype=torch.long
            ),
            "meta": {
                "pair_id": str(row.get("pair_id", "")),
                "patient_key": str(row.get("patient_key", low.patient_key)),
                "provenance": provenance.value,
                "low_relative_path": str(row["low_relative_path"]),
                "high_relative_path": str(row["high_relative_path"]),
            },
        }


def _pad_tensor_batch(items: Sequence[Mapping[str, Any]], key: str, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
    return torch.zeros((len(items), *shape), dtype=dtype)


def collate_acquisitions(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad G/V axes while preserving explicit validity masks.

    Detector H/W must match within a batch; the model contract rejects silent
    resizing.  Missing gates/views are zero-padded and marked invalid.
    """

    if not items:
        raise ValueError("cannot collate an empty batch")
    h, w = map(int, items[0]["counts_raw"].shape[-2:])
    if any(tuple(item["counts_raw"].shape[-2:]) != (h, w) for item in items):
        raise ValueError("all detector H/W must match within a batch")
    b = len(items)
    max_g = max(int(item["valid_frames"].shape[0]) for item in items)
    max_v = max(int(item["valid_frames"].shape[1]) for item in items)
    batch: dict[str, Any] = {
        "counts_raw": torch.zeros(b, max_g, max_v, h, w, dtype=torch.int64),
        "counts_input": torch.zeros(b, max_g, max_v, 1, h, w, dtype=torch.float32),
        "input_exposure_s": torch.zeros(b, max_g, max_v, dtype=torch.float32),
        "relative_exposure_q": torch.zeros(b, max_g, max_v, dtype=torch.float32),
        "visibility": torch.zeros(b, max_g, max_v, h, w, dtype=torch.bool),
        "valid_frames": torch.zeros(b, max_g, max_v, dtype=torch.bool),
        "angles_rad": torch.zeros(b, max_v, dtype=torch.float32),
        "radii_mm": torch.full((b, max_v), float("nan"), dtype=torch.float32),
        "gate_phase_rad": torch.zeros(b, max_g, dtype=torch.float32),
        "split_fraction": torch.zeros(b, 1, dtype=torch.float32),
        "scanner_index": torch.zeros(b, dtype=torch.long),
        "collimator_index": torch.zeros(b, dtype=torch.long),
        "stress_rest_index": torch.zeros(b, dtype=torch.long),
        "meta": [],
    }
    optional_specs = {
        "input_counts_raw": torch.int64,
        "target_counts": torch.int64,
        "target_exposure_s": torch.float32,
        "target_weight": torch.float32,
        "masked_target": torch.bool,
        "visible_target": torch.bool,
    }
    for key, dtype in optional_specs.items():
        if key in items[0]:
            shape = (
                (max_g, max_v, h, w)
                if key in {"input_counts_raw", "target_counts", "masked_target", "visible_target"}
                else (max_g, max_v)
            )
            batch[key] = torch.zeros((b, *shape), dtype=dtype)
    for row, item in enumerate(items):
        g, v = item["valid_frames"].shape
        batch["counts_input"][row, :g, :v] = item["counts_input"]
        batch["counts_raw"][row, :g, :v] = item["counts_raw"]
        batch["input_exposure_s"][row, :g, :v] = item["input_exposure_s"]
        batch["relative_exposure_q"][row, :g, :v] = item["relative_exposure_q"]
        batch["visibility"][row, :g, :v] = item["visibility"]
        batch["valid_frames"][row, :g, :v] = item["valid_frames"]
        batch["angles_rad"][row, :v] = item["angles_rad"]
        batch["radii_mm"][row, :v] = item["radii_mm"]
        batch["gate_phase_rad"][row, :g] = item["gate_phase_rad"]
        batch["split_fraction"][row] = item["split_fraction"]
        for key in ("scanner_index", "collimator_index", "stress_rest_index"):
            batch[key][row] = item[key]
        for key in optional_specs:
            if key in batch and key in item:
                if key in {"target_counts", "masked_target", "visible_target"}:
                    batch[key][row, :g, :v] = item[key]
                else:
                    batch[key][row, :g, :v] = item[key]
        batch["meta"].append(item.get("meta", {}))
    return batch


def collate_two_views(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate nested two-student/teacher samples."""

    if not items:
        raise ValueError("cannot collate an empty two-view batch")
    return {
        "student_a": collate_acquisitions([item["student_a"] for item in items]),
        "student_b": collate_acquisitions([item["student_b"] for item in items]),
        "teacher": collate_acquisitions([item["teacher"] for item in items]),
    }


def model_batch(
    batch: Mapping[str, Any],
    *,
    include_counts_raw: bool = False,
) -> dict[str, Tensor]:
    """Select tensor fields accepted by the backbone and optional rate baseline."""

    names = (
        "counts_input",
        "input_exposure_s",
        "visibility",
        "valid_frames",
        "angles_rad",
        "radii_mm",
        "gate_phase_rad",
        "split_fraction",
        "scanner_index",
        "collimator_index",
        "stress_rest_index",
    )
    selected = {name: batch[name] for name in names}
    if include_counts_raw:
        selected["counts_raw"] = batch["counts_raw"]
    return selected
