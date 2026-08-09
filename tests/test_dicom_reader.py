from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from gated_spect.data.dicom_reader import read_nm_dicom
from gated_spect.types import ReaderConfig


def write_fixture(path: Path, *, actual_frame_duration_ms: int | None = None, views: int = 2) -> np.ndarray:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.Modality = "NM"
    ds.PatientID = "patient-1"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.Manufacturer = "fixture"
    ds.ManufacturerModelName = "fixture-model"
    ds.Rows = 4
    ds.Columns = 4
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.NumberOfFrames = views * 2
    ds.PixelSpacing = [6.4, 6.4]
    ds.RadialPosition = [180.0 + index for index in range(views)]
    ds.StartAngle = 225.0
    ds.AngularStep = 5.625
    ds.RotationDirection = "CW"
    if views == 2 and actual_frame_duration_ms is None:
        ds.TimeSlotVector = [2, 1, 2, 1]
        ds.AngularViewVector = [2, 1, 1, 2]
    else:
        ds.TimeSlotVector = [2, 1] * views
        ds.AngularViewVector = [view + 1 for view in range(views) for _gate in range(2)]
    ds.DetectorVector = [1] * (views * 2)
    ds.EnergyWindowLowerLimit = 126.45
    ds.EnergyWindowUpperLimit = 154.55
    if actual_frame_duration_ms is not None:
        rotation = pydicom.dataset.Dataset()
        rotation.ActualFrameDuration = actual_frame_duration_ms
        ds.RotationInformationSequence = pydicom.sequence.Sequence([rotation])
    pixels = np.arange(views * 2 * 4 * 4, dtype=np.uint16).reshape(views * 2, 4, 4)
    ds.PixelData = pixels.tobytes()
    ds.save_as(str(path), write_like_original=False)
    return pixels


class DICOMReaderTests(unittest.TestCase):
    def test_shuffled_multiframe_is_canonicalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.dcm"
            pixels = write_fixture(path)
            sample = read_nm_dicom(
                path,
                ReaderConfig(hash_key=b"test-secret", default_view_dwell_s=1.0),
            )
            self.assertEqual(sample.shape, (2, 2, 4, 4))
            self.assertEqual(sample.detector_stream, "detector_0")
            self.assertEqual(sample.exposure_source, "count_based_gate_occupancy")
            self.assertEqual(int(sample.counts.sum()), int(pixels.sum()))
            np.testing.assert_array_equal(sample.counts[1, 1], pixels[0])
            np.testing.assert_array_equal(sample.counts[0, 0], pixels[1])
            np.testing.assert_array_equal(sample.counts[1, 0], pixels[2])
            np.testing.assert_array_equal(sample.counts[0, 1], pixels[3])
            np.testing.assert_allclose(sample.angles_rad, np.deg2rad([225.0, 219.375]))
            self.assertEqual(sample.energy_window_kev, (126.45, 154.55))

    def test_actual_frame_duration_is_per_angular_position(self) -> None:
        for duration_ms, expected_seconds in ((10_000, 10.0), (30_000, 30.0)):
            with self.subTest(duration_ms=duration_ms):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "duration_fixture.dcm"
                    write_fixture(path, actual_frame_duration_ms=duration_ms, views=32)
                    sample = read_nm_dicom(path, ReaderConfig(hash_key=b"test-secret"))
                    self.assertEqual(sample.shape, (2, 32, 4, 4))
                    np.testing.assert_allclose(
                        sample.view_dwell_s,
                        np.full(32, expected_seconds, dtype=np.float32),
                    )
                    self.assertEqual(sample.metadata["view_dwell_source"], "dicom_actual_frame_duration")
                    self.assertEqual(sample.metadata["actual_frame_duration_ms"], duration_ms)


if __name__ == "__main__":
    unittest.main()
