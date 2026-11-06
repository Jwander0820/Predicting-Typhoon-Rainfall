import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from typhoon_rainfall_realtime.ra import GridSpec, encode_labels, generate_ra, land_mask, mainland_boundary


def rectangle(west=120, south=23, east=122, north=25):
    return [[west, south], [east, south], [east, north], [west, north], [west, south]]


def boundary(rings=None):
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {
        "type": "Polygon", "coordinates": rings or [rectangle()]}, "properties": {}}]}


class RATests(unittest.TestCase):
    def test_grid_uses_north_first_cell_centers_not_edges(self):
        lon, lat = GridSpec(120, 23, 122, 25, 2, 2).coordinates()
        np.testing.assert_array_equal(lon, [[120.5, 121.5], [120.5, 121.5]])
        np.testing.assert_array_equal(lat, [[24.5, 24.5], [23.5, 23.5]])

    def test_land_mask_respects_sea_holes_and_multipolygon(self):
        lon, lat = GridSpec(120, 23, 122, 25, 2, 2).coordinates()
        geography = boundary([rectangle(), rectangle(120.4, 24.4, 120.6, 24.6)])
        np.testing.assert_array_equal(land_mask(lon, lat, geography), [[False, True], [True, True]])
        geography["features"][0]["geometry"] = {"type": "MultiPolygon", "coordinates": [
            [rectangle(120.4, 24.4, 120.6, 24.6)], [rectangle(121.4, 23.4, 121.6, 23.6)]]}
        np.testing.assert_array_equal(land_mask(lon, lat, geography), [[True, False], [False, True]])

    def test_label_zero_remains_valid_and_nodata_is_distinct(self):
        np.testing.assert_array_equal(encode_labels(np.array([[0, 0.5, 1.49, 1.5, 105, np.nan, -1]])),
                                      [[0, 1, 1, 2, 100, 255, 255]])

    def test_observation_to_ra_exports_values_labels_mask_and_method(self):
        records = [{"station_id": "A", "longitude_wgs84": 120.5, "latitude_wgs84": 24.5,
                    "rainfall_mm": 2.6, "interval_start": "2026-10-31T07:00:00+08:00", "interval_end": "2026-10-31T08:00:00+08:00"},
                   {"station_id": "B", "longitude_wgs84": 121.5, "latitude_wgs84": 23.5,
                    "rainfall_mm": 0, "interval_start": "2026-10-31T07:00:00+08:00", "interval_end": "2026-10-31T08:00:00+08:00"}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            source = path / "observations.json"
            source.write_text(json.dumps(records))
            summary = generate_ra(source, path / "ra", boundary(), {"source": "fixture"},
                                  grid=GridSpec(120, 23, 122, 25, 2, 2), max_distance_km=250)
            directory = Path(summary["maps"][0]["directory"])
            values = np.loadtxt(directory / "RA_mm.csv", delimiter=",")
            self.assertAlmostEqual(values[0, 0], 2.6)
            self.assertEqual(values[1, 1], 0)
            with Image.open(directory / "RA_labels.png") as image:
                self.assertEqual(image.mode, "L")
                self.assertEqual(image.getpixel((0, 0)), 3)
                self.assertEqual(image.getpixel((1, 1)), 0)
            with Image.open(directory / "valid_mask.png") as image:
                self.assertEqual(image.getpixel((1, 1)), 255)
            self.assertTrue((directory / "RA_map.png").is_file())
            self.assertFalse(summary["thesis_grid_compatibility_verified"])
            self.assertEqual(summary["maps"][0]["interval_end"], records[0]["interval_end"])
            self.assertIn("thesis_confirmed_N5_u2", summary["method"])

    def test_mainland_filter_excludes_offshore_counties_and_small_island_components(self):
        geography = boundary()
        geography["features"][0]["geometry"] = {"type": "MultiPolygon", "coordinates": [
            [rectangle()], [rectangle(120.5, 22.5, 120.6, 22.6)]]}
        offshore = boundary()["features"][0]
        offshore["properties"]["countyname"] = "澎湖縣"
        geography["features"].append(offshore)
        result = mainland_boundary(geography)
        self.assertEqual(len(result["features"]), 1)
        lon, lat = np.array([121, 120.55]), np.array([24, 22.55])
        np.testing.assert_array_equal(land_mask(lon, lat, result), [True, False])

    def test_interval_and_empty_station_data_fail_instead_of_fake_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            source = path / "observations.json"
            source.write_text("[]")
            with self.assertRaisesRegex(ValueError, "No hourly"):
                generate_ra(source, path / "ra", boundary(), {})
            source.write_text(json.dumps([{"interval_start": "2026-10-31T07:00:00+08:00",
                                          "interval_end": "2026-10-31T09:00:00+08:00"}]))
            with self.assertRaisesRegex(ValueError, "one-hour"):
                generate_ra(source, path / "ra", boundary(), {})


if __name__ == "__main__":
    unittest.main()
