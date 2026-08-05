from __future__ import annotations

import unittest

import numpy as np

from lswt_cloud_masking.lake_tables import (
    Station,
    filter_lst_with_classes,
    lake_mask_from_lonlat,
    parse_scene_info,
    window_stats,
)


class LakeTableTests(unittest.TestCase):
    def test_parse_scene_info_uses_sensor_date_and_pathrow(self) -> None:
        info = parse_scene_info("LC09_L1TP_195028_20240705_20240712_02_T1")

        self.assertEqual(info.sensor, "LANDSAT_9")
        self.assertEqual(info.time_utc, "20240705")
        self.assertEqual(info.pathrow, "195028")
        self.assertEqual(info.match_key, ("LANDSAT_9", "20240705", "195028"))

    def test_window_stats_centered_on_nearest_station_pixel(self) -> None:
        lon = np.tile(np.arange(3, dtype="float32"), (3, 1))
        lat = lon.T
        values = np.arange(9, dtype="float32").reshape(3, 3)

        stats = window_stats(lon, lat, values, Station("center", 1.0, 1.0), 3)

        self.assertAlmostEqual(stats["median"], 4.0)
        self.assertAlmostEqual(stats["mean"], 4.0)
        self.assertAlmostEqual(stats["std"], float(np.std(values)))

    def test_filter_lst_with_classes_masks_selected_ml_labels(self) -> None:
        lst = np.array([[10.0, 11.0], [12.0, 13.0]], dtype="float32")
        classes = np.array([[1.0, 2.0], [3.0, np.nan]], dtype="float32")

        out = filter_lst_with_classes(lst, classes, mask_cloud_classes=(1, 2))

        self.assertTrue(np.isnan(out[0, 0]))
        self.assertTrue(np.isnan(out[0, 1]))
        self.assertAlmostEqual(float(out[1, 0]), 12.0)
        self.assertAlmostEqual(float(out[1, 1]), 13.0)

    def test_lake_mask_from_lonlat_handles_polygon(self) -> None:
        geometry = {
            "type": "Polygon",
            "coordinates": [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]],
        }
        lon = np.array([[-1.0, 1.0, 3.0]], dtype="float32")
        lat = np.array([[1.0, 1.0, 1.0]], dtype="float32")

        mask = lake_mask_from_lonlat(geometry, lon, lat)

        self.assertEqual(mask.tolist(), [[False, True, False]])


if __name__ == "__main__":
    unittest.main()
