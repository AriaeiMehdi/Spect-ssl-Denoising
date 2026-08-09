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
from pydicom.sequence import Sequence
from pydicom.tag import Tag
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from ..versioning import SCIENTIFIC_SEMANTICS_VERSION


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
    view_dwell_s: np.ndarray,
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
    view_dwell_s:
        Acquisition time per angular position in seconds, with shape
        ``[G,V]`` (or ``[V]`` for ungated ``G=1``).  The validated exporter currently supports only ``G=1``;
        the median per-view dwell is written to DICOM ``ActualFrameDuration``
        (milliseconds), as required by the NM IOD.
    expected_counts:
        Whether the values are model-estimated expected counts rather than
        directly acquired counts.
    """

    source = pydicom.dcmread(str(source_path), force=False)
    values = _as_uint16_counts(counts)
    dwell = np.asarray(view_dwell_s, dtype=np.float64)
    if dwell.ndim == 1 and values.shape[0] == 1:
        dwell = dwell[None, :]
    if dwell.shape != values.shape[:2]:
        raise ValueError(
            "view_dwell_s must have shape [G,V], "
            f"got {dwell.shape} for counts {values.shape}"
        )
    if np.any(~np.isfinite(dwell)) or np.any(dwell <= 0):
        raise ValueError("view_dwell_s must be finite and positive")

    g, v, h, w = (int(x) for x in values.shape)
    if g != 1:
        raise NotImplementedError(
            "validated DICOM export currently supports only ungated G=1 projections"
        )
    dwell_view = dwell.reshape(-1)
    dwell_median = float(np.median(dwell_view))
    if not np.allclose(dwell_view, dwell_median, rtol=0.01, atol=1e-6):
        raise ValueError(
            "view_dwell_s must be uniform within 1% for a single NM "
            "ActualFrameDuration value"
        )
    output = deepcopy(source)
    output.file_meta = deepcopy(source.file_meta)
    output.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    sop_instance_uid = generate_uid()
    series_instance_uid = generate_uid()
    output.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    output.file_meta.ImplementationVersionName = f"GATED_SPECT_{SCIENTIFIC_SEMANTICS_VERSION}"
    output.SOPInstanceUID = sop_instance_uid
    output.SeriesInstanceUID = series_instance_uid
    output.ImageType = ["DERIVED", "PRIMARY", "TOMO", "EMISSION"]
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
    # Expected model output is not an acquired count total.  Leave the
    # acquisition-only field empty rather than presenting a derived estimate
    # as a scanner accumulation.
    output.CountsAccumulated = "" if expected_counts else str(int(values.astype(np.int64).sum()))
    output.FrameIncrementPointer = list(_FRAME_INCREMENT_POINTER)

    # DICOM vectors follow the view order of the ungated projection array.
    output.EnergyWindowVector = [1] * (g * v)
    output.DetectorVector = [1] * (g * v)
    output.RotationVector = [1] * (g * v)
    output.AngularViewVector = [view + 1 for _gate in range(g) for view in range(v)]
    if hasattr(output, "TimeSlotVector"):
        del output.TimeSlotVector
        if hasattr(output, "NumberOfTimeSlots"):
            del output.NumberOfTimeSlots

    rotation_sequence = getattr(output, "RotationInformationSequence", None)
    if rotation_sequence:
        rotation = rotation_sequence[0]
        rotation.NumberOfFramesInRotation = v
        # ActualFrameDuration is the acquisition time for one angular
        # position, not the total duration of the rotation.
        rotation.ActualFrameDuration = int(round(dwell_median * 1000.0))
    else:
        rotation = Dataset()
        rotation.NumberOfFramesInRotation = v
        rotation.ActualFrameDuration = int(round(dwell_median * 1000.0))
        output.RotationInformationSequence = Sequence([rotation])

    source_reference = Dataset()
    source_reference.ReferencedSOPClassUID = str(source.SOPClassUID)
    source_reference.ReferencedSOPInstanceUID = str(source.SOPInstanceUID)
    source_reference.SpatialLocationsPreserved = "YES"
    output.SourceImageSequence = Sequence([source_reference])

    _set_private_text(output, 0x1001, f"shape={g}x{v}x{h}x{w}")
    _set_private_text(output, 0x1002, "frame_order=gate_then_view")
    _set_private_text(output, 0x1003, "count_semantics=expected_counts" if expected_counts else "count_semantics=acquired_counts")
    _set_private_text(output, 0x1004, f"provenance={provenance}")
    _set_private_text(output, 0x1005, f"view_dwell_seconds={dwell_median:.9g}")
    _set_private_text(output, 0x1006, "actual_frame_duration_semantics=per_angular_position")
    _set_private_text(output, 0x1007, f"scientific_semantics_version={SCIENTIFIC_SEMANTICS_VERSION}")

    output.PixelData = values.reshape(g * v, h, w).tobytes(order="C")
    output.is_little_endian = True
    output.is_implicit_VR = False
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save_as(str(destination), write_like_original=False)
    return destination


__all__: Sequence[str] = ("write_nm_projection_dicom",)
