"""Data ingestion, exposure, pairing, and split utilities."""

from .cache import StaleCacheError, read_sample_cache, write_sample_cache
from .audit import audit_root
from .dicom_reader import read_nm_dicom
from .dicom_export import write_nm_projection_dicom
from .exposure import estimate_exposure
from .frame_indexing import FrameIndex, build_frame_index, construct_counts
from .inventory import inventory_headers, write_jsonl
from .manifest import build_acquisition_manifest, read_jsonl, write_locked_splits, write_manifest
from .torch_dataset import (
    AcquisitionDataset,
    NormalizationStats,
    PairDataset,
    PretrainingDataset,
    TwoViewPretrainingDataset,
    collate_acquisitions,
    collate_two_views,
    compute_normalization_stats,
    model_batch,
)
from .pairing import verify_pair
from .pair_manifest import audit_duration_pairs, discover_duration_pairs, write_pair_manifest
from .splits import make_dataset1_splits, make_dataset2_splits

__all__ = [
    "FrameIndex",
    "audit_root",
    "audit_duration_pairs",
    "discover_duration_pairs",
    "build_frame_index",
    "construct_counts",
    "estimate_exposure",
    "inventory_headers",
    "make_dataset1_splits",
    "make_dataset2_splits",
    "read_nm_dicom",
    "write_nm_projection_dicom",
    "read_sample_cache",
    "StaleCacheError",
    "verify_pair",
    "write_pair_manifest",
    "write_jsonl",
    "write_sample_cache",
    "build_acquisition_manifest",
    "read_jsonl",
    "write_locked_splits",
    "write_manifest",
    "AcquisitionDataset",
    "NormalizationStats",
    "PairDataset",
    "PretrainingDataset",
    "TwoViewPretrainingDataset",
    "collate_acquisitions",
    "collate_two_views",
    "compute_normalization_stats",
    "model_batch",
]
