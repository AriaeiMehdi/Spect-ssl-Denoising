from __future__ import annotations

import unittest

import numpy as np

from gated_spect.data.frame_indexing import (
    DuplicateFrameError,
    MissingFrameError,
    NonContiguousFrameIndexError,
    build_frame_index,
    construct_counts,
)


class FrameIndexingTests(unittest.TestCase):
    def test_shuffled_frames_are_reordered_and_counts_preserved(self) -> None:
        # Source order: (gate, view) = (1,1), (0,0), (1,0), (0,1)
        gate = np.array([2, 1, 2, 1])
        view = np.array([2, 1, 1, 2])
        pixels = np.arange(4 * 2 * 3, dtype=np.uint16).reshape(4, 2, 3)
        index = build_frame_index(gate, view)
        counts = construct_counts(pixels, index)
        self.assertEqual(counts.shape, (2, 2, 2, 3))
        self.assertEqual(int(counts.sum()), int(pixels.sum()))
        np.testing.assert_array_equal(counts[1, 1], pixels[0])
        np.testing.assert_array_equal(counts[0, 0], pixels[1])

    def test_noncontiguous_labels_are_rejected(self) -> None:
        with self.assertRaises(NonContiguousFrameIndexError):
            build_frame_index([1, 3], [1, 2])

    def test_duplicate_and_missing_pairs_are_rejected(self) -> None:
        index = build_frame_index([1, 1], [1, 1])
        pixels = np.ones((2, 2, 2), dtype=np.int32)
        with self.assertRaises(DuplicateFrameError):
            construct_counts(pixels, index)
        missing_index = build_frame_index([1, 2], [1, 2])
        with self.assertRaises(MissingFrameError):
            construct_counts(pixels, missing_index)


if __name__ == "__main__":
    unittest.main()
