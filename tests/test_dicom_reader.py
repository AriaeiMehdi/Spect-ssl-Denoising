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


def write_fixture(path: Path) -> np.ndarray:
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
    ds.NumberOfFrames = 4
    ds.PixelSpacing = [6.4, 6.4]
    ds.RadialPosition = [180.0, 181.0]
    ds.StartAngle = 225.0
    ds.AngularStep = 5.625
    ds.RotationDirection = "CW"
    ds.TimeSlotVector = [2, 1, 2, 1]
    ds.AngularViewVector = [2, 1, 1, 2]
    ds.DetectorVector = [1, 1, 1, 1]
    ds.EnergyWindowLowerLimit = 126.45
    ds.EnergyWindowUpperLimit = 154.55
    pixels = np.arange(4 * 4 * 4, dtype=np.uint16).reshape(4, 4, 4)
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
            self.assertEqual(sample.exposure_source, "derived")
            self.assertEqual(int(sample.counts.sum()), int(pixels.sum()))
            np.testing.assert_array_equal(sample.counts[1, 1], pixels[0])
            np.testing.assert_array_equal(sample.counts[0, 0], pixels[1])
            np.testing.assert_array_equal(sample.counts[1, 0], pixels[2])
            np.testing.assert_array_equal(sample.counts[0, 1], pixels[3])
            np.testing.assert_allclose(sample.angles_rad, np.deg2rad([225.0, 219.375]))
            self.assertEqual(sample.energy_window_kev, (126.45, 154.55))


if __name__ == "__main__":
    unittest.main()
