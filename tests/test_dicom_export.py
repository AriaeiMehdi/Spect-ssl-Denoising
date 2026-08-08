from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, NuclearMedicineImageStorage, generate_uid

from gated_spect.data.dicom_export import write_nm_projection_dicom


class DicomExportTests(unittest.TestCase):
    def test_multiframe_projection_export_preserves_shape_and_count_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.dcm"
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = NuclearMedicineImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.ImplementationClassUID = generate_uid()
            source = FileDataset(str(source_path), {}, file_meta=meta, preamble=b"\0" * 128)
            source.SOPClassUID = NuclearMedicineImageStorage
            source.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            source.StudyInstanceUID = generate_uid()
            source.SeriesInstanceUID = generate_uid()
            source.Modality = "NM"
            source.Rows = 2
            source.Columns = 3
            source.NumberOfFrames = 2
            source.SamplesPerPixel = 1
            source.PhotometricInterpretation = "MONOCHROME2"
            source.BitsAllocated = 16
            source.BitsStored = 16
            source.HighBit = 15
            source.PixelRepresentation = 0
            source.PixelSpacing = [6.4, 6.4]
            source.AngularViewVector = [1, 2]
            source.EnergyWindowVector = [1, 1]
            source.DetectorVector = [1, 1]
            source.RotationVector = [1, 1]
            source.FrameIncrementPointer = [(0x0054, 0x0010), (0x0054, 0x0020)]
            rotation = Dataset()
            rotation.NumberOfFramesInRotation = 2
            rotation.ActualFrameDuration = 2000
            source.RotationInformationSequence = Sequence([rotation])
            source.PixelData = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16).tobytes()
            source.save_as(source_path, write_like_original=False)

            counts = np.array(
                [
                    [
                        [[1.2, 2.4, 3.6], [4.8, 5.0, 6.0]],
                        [[7.2, 8.4, 9.6], [10.8, 11.0, 12.0]],
                    ]
                ],
                dtype=np.float32,
            )
            destination = write_nm_projection_dicom(
                source_path,
                root / "derived.dcm",
                counts,
                frame_exposure_s=np.array([[1.0, 1.0]], dtype=np.float32),
                series_description="derived",
                derivation_description="test",
                series_number=901,
                expected_counts=True,
                provenance="independent",
            )
            derived = pydicom.dcmread(destination)
            self.assertEqual(derived.SOPClassUID, NuclearMedicineImageStorage)
            self.assertNotEqual(derived.SOPInstanceUID, source.SOPInstanceUID)
            self.assertEqual(int(derived.NumberOfFrames), 2)
            self.assertEqual((derived.Rows, derived.Columns), (2, 3))
            self.assertEqual(derived.pixel_array.dtype, np.uint16)
            np.testing.assert_array_equal(
                derived.pixel_array,
                np.array(
                    [
                        [[1, 2, 4], [5, 5, 6]],
                        [[7, 8, 10], [11, 11, 12]],
                    ],
                    dtype=np.uint16,
                ),
            )
            self.assertEqual(derived.Units, "CNTS")
            self.assertEqual(derived[0x0011, 0x1003].value, "count_semantics=expected_counts")


if __name__ == "__main__":
    unittest.main()
