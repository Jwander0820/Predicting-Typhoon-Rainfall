import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from typhoon_rainfall_realtime.geometry import GridSpec, SOURCE_GRIDS, SourceImageGrid, reproject_image
from typhoon_rainfall_realtime.pipeline import prepare_images
from typhoon_rainfall_realtime.alignment import ASSETS, training_land_mask


class GeometryTests(unittest.TestCase):
    def test_radar_graticules_use_pixel_centers_and_north_up(self):
        x, y = SOURCE_GRIDS["RD"].world_to_pixel(np.array([119, 122]), np.array([26, 22]))
        np.testing.assert_allclose(x, [599.5, 2399.5], atol=1e-8)
        np.testing.assert_allclose(y, [299.5, 2699.5], atol=1e-8)

    def test_infrared_holdout_graticules_and_independent_coast_points(self):
        evidence = json.loads((ASSETS / "georeference_validation.json").read_text())
        points = np.asarray(evidence["graticule_points_lon_lat_pixel_x_y"])
        held = points[evidence["calibration"]["holdout_indices"]]
        x, y = SOURCE_GRIDS["IR"].world_to_pixel(held[:, 0], held[:, 1])
        errors = np.hypot(x - held[:, 2], y - held[:, 3])
        self.assertLess(float(errors.max()), 0.6)
        self.assertLess(float(np.sqrt(np.mean(errors ** 2))), 0.45)
        for point in evidence["independent_coast_checks"]:
            x, y = SOURCE_GRIDS["IR"].world_to_pixel(point["lon"], point["lat"])
            expected = point["nearest_white_coast_xy"]
            self.assertLess(float(np.hypot(x - expected[0], y - expected[1])), 0.65)

    def test_cell_integration_preserves_coordinate_gradient_without_axis_flip(self):
        source = SourceImageGrid("fixture", "fixture", 8, 8, "EPSG:4326", (1, -0.5, -1, 7.5),
                                 (0, 0, 8, 8), "hand-calculated fixture")
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(8)[None, :] * 20
        image[:, :, 1] = np.arange(8)[:, None] * 20
        target = GridSpec(2, 2, 6, 6, 4, 4)
        result, metadata = reproject_image(image, source, target)
        np.testing.assert_allclose(result[:, :, :2], image[2:6, 2:6, :2], atol=1)
        self.assertIn("INTER_AREA", metadata["resampling"])

    def test_source_border_annotation_is_not_mixed_into_mainland_view(self):
        source = SourceImageGrid("fixture", "fixture", 16, 16, "EPSG:4326", (1, -0.5, -1, 15.5),
                                 (2, 2, 14, 14), "fixture")
        image = np.full((16, 16), 25, dtype=np.uint8)
        image[:2] = image[-2:] = 250
        image[:, :2] = image[:, -2:] = 250
        result, _ = reproject_image(image, source, GridSpec(4, 4, 12, 12, 8, 8))
        np.testing.assert_array_equal(result, 25)
        with self.assertRaisesRegex(ValueError, "outside"):
            reproject_image(image, source, GridSpec(0, 0, 16, 16, 8, 8))

    def test_roi_half_cell_sampling_never_blends_annotations(self):
        source = SourceImageGrid("fixture", "fixture", 8, 8, "EPSG:4326", (1, -0.5, -1, 7.5),
                                 (2, 2, 6, 6), "fixture")
        image = np.full((8, 8), 250, dtype=np.uint8)
        image[2:6, 2:6] = 25
        result, _ = reproject_image(image, source, GridSpec(2, 2, 6, 6, 4, 4))
        np.testing.assert_array_equal(result, 25)

    def test_16bit_rgb_png_is_rejected_even_if_pillow_calls_it_rgb(self):
        import cv2
        ok, data = cv2.imencode(".png", np.full((8, 8, 3), 8000, dtype=np.uint16))
        self.assertTrue(ok)
        rd = SimpleNamespace(product_id="O-A0058-003", content=data.tobytes())
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "8-bit RGB"):
                prepare_images({"RD": rd}, Path(temporary))

    def test_target_grid_matches_training_graticules_not_the_old_crawler_guess(self):
        evidence = json.loads((ASSETS / "training_grid_validation.json").read_text())
        grid = GridSpec()
        x120, x122, y24, y22 = evidence["line_pixels"]
        dx, dy = (grid.east - grid.west) / 128, (grid.north - grid.south) / 128
        np.testing.assert_allclose([grid.west + (x120 + 0.5) * dx, grid.west + (x122 + 0.5) * dx],
                                    [120, 122], atol=1e-9)
        np.testing.assert_allclose([grid.north - (y24 + 0.5) * dy, grid.north - (y22 + 0.5) * dy],
                                    [24, 22], atol=1e-9)

    def test_radar_gray_and_rgba_are_rejected_instead_of_silent_color_conversion(self):
        for mode, color in (("L", 20), ("RGBA", (20, 20, 20, 255))):
            buffer = io.BytesIO()
            Image.new(mode, (3600, 3600), color).save(buffer, format="PNG")
            rd = SimpleNamespace(product_id="O-A0058-003", content=buffer.getvalue())
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, "encoding"):
                    prepare_images({"RD": rd}, Path(temporary))

    def test_radar_and_ir_maps_cover_the_same_shared_grid(self):
        lon, lat = GridSpec().coordinates()
        for kind in ("RD", "IR"):
            x, y = SOURCE_GRIDS[kind].world_to_pixel(lon, lat)
            inverse_lon, inverse_lat = SOURCE_GRIDS[kind].pixel_to_world(x, y)
            np.testing.assert_allclose(inverse_lon, lon, atol=1e-8)
            np.testing.assert_allclose(inverse_lat, lat, atol=1e-8)
        self.assertTrue(np.all(np.diff(lat[:, 0]) < 0))
        self.assertEqual(int(training_land_mask().sum()), 4700)


if __name__ == "__main__":
    unittest.main()
