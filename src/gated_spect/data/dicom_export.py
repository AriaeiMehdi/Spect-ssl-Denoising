"""Derived multi-frame Nuclear Medicine DICOM export.

The denoiser operates on projection frames, so its portable output is a
multi-frame NM projection object rather than an anatomical slice volume.  This
module preserves the source acquisition geometry and frame order so a
commercial SPECT application can import the exported projections and perform
its own reconstruction.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Sequence

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.tag import Tag
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


_FRAME_INCREMENT_POINTER = [
    Tag(0x00540010),  # Energy Window Vector
    Tag(0x00540020),  # Detector Vector
    Tag(0x00540050),  # Rotation Vector
    Tag(0x00540090),  # Angular View Vector
]
_PRIVATE_CREATOR = "GATED_SPECT_EXPORT"


def _as_uint16_counts(counts: np.ndarray) -> np.ndarray:
    """Round expected counts without silently changing their scale."""

    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("counts must have shape [G,V,H,W]")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("counts must be finite and nonnegative")
    largest = float(values.max(initial=0.0))
    if largest > np.iinfo(np.uint16).max:
        raise ValueError(
            "exported counts exceed the 16-bit NM DICOM range; refusing to clip "
            f"the maximum value {largest:.3f}"
        )
    return np.rint(values).astype(np.uint16, copy=False)


def _set_private_text(dataset: Dataset, element: int, value: str) -> None:
    """Store a short export provenance value in a private LO element."""

    dataset.add_new((0x0011, 0x0010), "LO", _PRIVATE_CREATOR)
    dataset.add_new((0x0011, element), "LO", str(value))


def write_nm_projection_dicom(
    source_path: str | Path,
    output_path: str | Path,
    counts: np.ndarray,
    *,
    frame_exposure_s: np.ndarray,
    series_description: str,
    derivation_description: str,
    series_number: int,
    expected_counts: bool,
    provenance: str,
) -> Path:
    """Write a derived multi-frame NM projection DICOM.

    Parameters
    ----------
    source_path:
        Source DICOM whose patient, study, detector, and rotation metadata are
        copied.
    counts:
        Projection counts with shape ``[G,V,H,W]`` in gate-major/view-minor
        order.  Values are rounded to unsigned 16-bit integer counts.
    frame_exposure_s:
        Exposure per ``[G,V]`` frame in seconds.  It is recorded in the
        provenance tags and used to update the source rotation duration.
    expected_counts:
        Whether the values are model-estimated expected counts rather than
        directly acquired counts.
    """

    source = pydicom.dcmread(str(source_path), force=False)
    values = _as_uint16_counts(counts)
    exposure = np.asarray(frame_exposure_s, dtype=np.float64)
    if exposure.shape != values.shape[:2]:
        raise ValueError(
            "frame_exposure_s must have shape [G,V], "
            f"got {exposure.shape} for counts {values.shape}"
        )
    if np.any(~np.isfinite(exposure)) or np.any(exposure <= 0):
        raise ValueError("frame_exposure_s must be finite and positive")

    g, v, h, w = (int(x) for x in values.shape)
    output = deepcopy(source)
    output.file_meta = deepcopy(source.file_meta)
    output.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    sop_instance_uid = generate_uid()
    series_instance_uid = generate_uid()
    output.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    output.file_meta.ImplementationVersionName = "GATED_SPECT_1"
    output.SOPInstanceUID = sop_instance_uid
    output.SeriesInstanceUID = series_instance_uid
    output.ImageType = ["DERIVED", "SECONDARY", "TOMO", "EMISSION"]
    output.SeriesDescription = str(series_description)[:64]
    output.DerivationDescription = str(derivation_description)[:1024]
    output.SeriesNumber = int(series_number)
    output.InstanceNumber = 1
    output.Modality = "NM"
    output.SamplesPerPixel = 1
    output.PhotometricInterpretation = "MONOCHROME2"
    output.Rows = h
    output.Columns = w
    output.NumberOfFrames = g * v
    output.BitsAllocated = 16
    output.BitsStored = 16
    output.HighBit = 15
    output.PixelRepresentation = 0
    output.RescaleIntercept = 0
    output.RescaleSlope = 1
    output.Units = "CNTS"
    output.PixelSpacing = list(getattr(source, "PixelSpacing", [1.0, 1.0]))
    output.SmallestImagePixelValue = int(values.min(initial=0))
    output.LargestImagePixelValue = int(values.max(initial=0))
    # pydicom keeps these dictionary entries as ambiguous US/SS elements;
    # PixelRepresentation=0 makes the unsigned interpretation explicit for
    # Explicit VR Little Endian output.
    output[0x00280106].VR = "US"
    output[0x00280107].VR = "US"
    output.CountsAccumulated = str(int(values.astype(np.int64).sum()))
    output.FrameIncrementPointer = list(_FRAME_INCREMENT_POINTER)

    # DICOM vectors follow the same gate-major/view-minor order as the pixel
    # array.  The validation data are ungated (G=1), while this remains valid
    # for future gated exports as well.
    output.EnergyWindowVector = [1] * (g * v)
    output.DetectorVector = [1] * (g * v)
    output.RotationVector = [1] * (g * v)
    output.AngularViewVector = [view + 1 for _gate in range(g) for view in range(v)]
    if g > 1:
        output.NumberOfTimeSlots = g
        output.TimeSlotVector = [gate + 1 for gate in range(g) for _view in range(v)]
    elif hasattr(output, "TimeSlotVector"):
        del output.TimeSlotVector
        if hasattr(output, "NumberOfTimeSlots"):
            del output.NumberOfTimeSlots

    rotation_sequence = getattr(output, "RotationInformationSequence", None)
    if rotation_sequence:
        rotation = rotation_sequence[0]
        rotation.NumberOfFramesInRotation = v
        # The reader and source files treat this value as total series dwell;
        # keep that convention so the exported object round-trips correctly.
        rotation.ActualFrameDuration = int(round(float(exposure.sum()) * 1000.0))

    _set_private_text(output, 0x1001, f"shape={g}x{v}x{h}x{w}")
    _set_private_text(output, 0x1002, "frame_order=gate_then_view")
    _set_private_text(output, 0x1003, "count_semantics=expected_counts" if expected_counts else "count_semantics=acquired_counts")
    _set_private_text(output, 0x1004, f"provenance={provenance}")
    _set_private_text(output, 0x1005, f"exposure_seconds_total={float(exposure.sum()):.9g}")

    output.PixelData = values.reshape(g * v, h, w).tobytes(order="C")
    output.is_little_endian = True
    output.is_implicit_VR = False
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save_as(str(destination), write_like_original=False)
    return destination


__all__: Sequence[str] = ("write_nm_projection_dicom",)
