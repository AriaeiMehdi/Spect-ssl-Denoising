"""Validated DICOM frame-to-gate/view indexing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class FrameIndexError(ValueError):
    """Base error for malformed frame index vectors."""


class MissingFrameIndexError(FrameIndexError):
    """A required index vector is absent or has the wrong length."""


class NonContiguousFrameIndexError(FrameIndexError):
    """DICOM labels do not form the expected one-based contiguous set."""


class DuplicateFrameError(FrameIndexError):
    """More than one frame maps to one gate/view pair."""


class MissingFrameError(FrameIndexError):
    """At least one gate/view pair is absent."""


@dataclass(frozen=True)
class FrameIndex:
    gate_indices: np.ndarray
    view_indices: np.ndarray
    num_gates: int
    num_views: int

    @property
    def num_frames(self) -> int:
        return int(self.gate_indices.size)


def _validated_labels(values: Sequence[int] | np.ndarray, n_frames: int, name: str) -> np.ndarray:
    if values is None:
        raise MissingFrameIndexError(f"{name} is missing")
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size != n_frames:
        raise MissingFrameIndexError(
            f"{name} must be a vector with {n_frames} entries, got {arr.shape}"
        )
    if np.issubdtype(arr.dtype, np.floating):
        if np.any(~np.isfinite(arr)) or np.any(arr != np.floor(arr)):
            raise NonContiguousFrameIndexError(f"{name} contains non-integer values")
    try:
        labels = arr.astype(np.int64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NonContiguousFrameIndexError(f"{name} is not integer-valued") from exc
    if np.any(labels < 1):
        raise NonContiguousFrameIndexError(f"{name} must use positive one-based labels")
    expected = np.arange(1, int(labels.max()) + 1, dtype=np.int64)
    if not np.array_equal(np.unique(labels), expected):
        raise NonContiguousFrameIndexError(
            f"{name} labels must be contiguous 1..max; got {np.unique(labels).tolist()}"
        )
    return labels - 1


def build_frame_index(
    time_slot_vector: Sequence[int] | np.ndarray,
    angular_view_vector: Sequence[int] | np.ndarray,
) -> FrameIndex:
    """Validate DICOM vectors and return zero-based gate/view indices."""

    gate_raw = np.asarray(time_slot_vector)
    view_raw = np.asarray(angular_view_vector)
    if gate_raw.ndim != 1 or view_raw.ndim != 1 or gate_raw.size != view_raw.size:
        raise MissingFrameIndexError(
            f"gate/view vectors must be one-dimensional and equal length, "
            f"got {gate_raw.shape} and {view_raw.shape}"
        )
    gates = _validated_labels(gate_raw, int(gate_raw.size), "TimeSlotVector")
    views = _validated_labels(view_raw, int(view_raw.size), "AngularViewVector")
    return FrameIndex(
        gate_indices=gates,
        view_indices=views,
        num_gates=int(gates.max()) + 1,
        num_views=int(views.max()) + 1,
    )


def construct_counts(pixels: np.ndarray, frame_index: FrameIndex) -> np.ndarray:
    """Place shuffled frames into [G,V,H,W], rejecting duplicates/missing pairs."""

    arr = np.asarray(pixels)
    if arr.ndim == 2 and frame_index.num_frames == 1:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[0] != frame_index.num_frames:
        raise MissingFrameIndexError(
            f"pixels must be [F,H,W] with F={frame_index.num_frames}, got {arr.shape}"
        )
    if not np.issubdtype(arr.dtype, np.integer):
        if np.any(~np.isfinite(arr)) or np.any(arr != np.floor(arr)):
            raise ValueError("projection pixels must be nonnegative integer-valued")
        arr = arr.astype(np.int32)
    if np.any(arr < 0):
        raise ValueError("projection pixels cannot be negative")

    h, w = int(arr.shape[1]), int(arr.shape[2])
    result = np.zeros((frame_index.num_gates, frame_index.num_views, h, w), dtype=np.int32)
    occupied = np.zeros((frame_index.num_gates, frame_index.num_views), dtype=bool)
    for frame, gate, view in zip(arr, frame_index.gate_indices, frame_index.view_indices):
        gate_i, view_i = int(gate), int(view)
        if occupied[gate_i, view_i]:
            raise DuplicateFrameError(f"duplicate frame at gate={gate_i}, view={view_i}")
        result[gate_i, view_i] = frame.astype(np.int32, copy=False)
        occupied[gate_i, view_i] = True
    if not bool(occupied.all()):
        missing = np.argwhere(~occupied).tolist()
        raise MissingFrameError(f"missing gate/view pairs: {missing}")
    return result
