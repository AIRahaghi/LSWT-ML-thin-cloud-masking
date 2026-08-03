from __future__ import annotations

import unittest

import numpy as np

from lswt_cloud_masking.features import compute_spectral_indices
from lswt_cloud_masking.qa import decode_qa_pixel, fmask_class_layer, fmask_clear_water


class FeatureTests(unittest.TestCase):
    def test_spectral_indices_match_notebook_formulas(self) -> None:
        arrays = {
            "bluemap": np.array([[0.20]], dtype="float32"),
            "greenmap": np.array([[0.30]], dtype="float32"),
            "redmap": np.array([[0.10]], dtype="float32"),
            "nirmap": np.array([[0.50]], dtype="float32"),
            "swirmap1": np.array([[0.25]], dtype="float32"),
            "swirmap2": np.array([[0.20]], dtype="float32"),
        }
        out = compute_spectral_indices(arrays)
        self.assertAlmostEqual(float(out["hotmap"][0, 0]), 0.15, places=6)
        self.assertAlmostEqual(float(out["ndvimap"][0, 0]), 0.6666667, places=6)
        self.assertAlmostEqual(float(out["brtestmap1"][0, 0]), 0.5, places=6)


class QaTests(unittest.TestCase):
    def test_fmask_clear_water_and_cloud_classes(self) -> None:
        qa = np.array(
            [
                [128, 136],
                [8, 0],
            ],
            dtype="uint16",
        )
        lake = np.ones_like(qa, dtype=bool)
        decoded = decode_qa_pixel(qa)
        clear = fmask_clear_water(decoded, lake)
        layer = fmask_class_layer(qa, lake)

        self.assertTrue(clear[0, 0])
        self.assertFalse(clear[0, 1])
        self.assertEqual(int(layer[0, 0]), 1)
        self.assertEqual(int(layer[0, 1]), 2)
        self.assertEqual(int(layer[1, 0]), 2)
        self.assertEqual(int(layer[1, 1]), 3)


if __name__ == "__main__":
    unittest.main()
