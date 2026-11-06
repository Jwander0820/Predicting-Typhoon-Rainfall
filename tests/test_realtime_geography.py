import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image

from typhoon_rainfall_realtime.geography import ARCHIVE_MEMBERS, export_geography
from typhoon_rainfall_realtime.geometry import GridSpec


def png_bytes(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def historical_images():
    longitude = np.broadcast_to(np.arange(128, dtype=np.uint8) * 2, (128, 128)).copy()
    latitude = longitude.T[::-1].copy()
    return {"LON": png_bytes(longitude), "LAT": png_bytes(latitude)}


def write_source(directory, images=None):
    images = images or historical_images()
    directory.mkdir()
    for kind, filename in (("LON", "lon.png"), ("LAT", "lat.png")):
        (directory / filename).write_bytes(images[kind])
    return images


class GeographyTests(unittest.TestCase):
    def test_original_bytes_and_normalized_intensity_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            original = write_source(base / "source")
            paths, metadata = export_geography(base / "inputs", base / "source")
            for kind in ("LON", "LAT"):
                self.assertEqual(paths[kind].read_bytes(), original[kind])
                self.assertEqual(metadata["files"][kind]["sha256"], hashlib.sha256(original[kind]).hexdigest())
            longitude = np.array(Image.open(paths["LON"]), dtype=np.float32) / 255.0
            latitude = np.array(Image.open(paths["LAT"]), dtype=np.float32) / 255.0
            self.assertAlmostEqual(float(longitude[0, -1]), 254 / 255, places=6)
            self.assertAlmostEqual(float(latitude[0, 0]), 254 / 255, places=6)
            self.assertEqual(float(longitude[0, 0]), 0)
            self.assertEqual(float(latitude[-1, -1]), 0)
            self.assertTrue(metadata["static"])
            self.assertEqual(metadata["grid"]["row_order"], "north_to_south")

    def test_default_zip_uses_exact_original_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "database.zip"
            originals = historical_images()
            with ZipFile(source, "w") as archive:
                for kind, member in ARCHIVE_MEMBERS.items():
                    archive.writestr(member, originals[kind])
                archive.writestr("unrelated/lon.png", b"not an image")
            with patch("typhoon_rainfall_realtime.geography.DEFAULT_SOURCE", source):
                paths, metadata = export_geography(base / "inputs")
            self.assertEqual(paths["LON"].read_bytes(), originals["LON"])
            self.assertEqual(metadata["files"]["LAT"]["source_image"], ARCHIVE_MEMBERS["LAT"])
            self.assertFalse((base / "database").exists())

    def test_swapped_or_reversed_coordinates_are_rejected_before_writing(self):
        for change in ("swap", "reverse", "reverse_latitude"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                images = historical_images()
                if change == "swap":
                    images = {"LON": images["LAT"], "LAT": images["LON"]}
                elif change == "reverse":
                    lon = np.array(Image.open(io.BytesIO(images["LON"])))
                    images["LON"] = png_bytes(lon[:, ::-1])
                else:
                    lat = np.array(Image.open(io.BytesIO(images["LAT"])))
                    images["LAT"] = png_bytes(lat[::-1])
                write_source(base / "source", images)
                with self.assertRaisesRegex(ValueError, "latitude" if change == "reverse_latitude" else "longitude"):
                    export_geography(base / "inputs", base / "source")
                self.assertFalse((base / "inputs").exists())

    def test_wrong_size_and_color_do_not_silently_convert(self):
        for bad in (np.zeros((64, 64), dtype=np.uint8), np.zeros((128, 128, 3), dtype=np.uint8)):
            with self.subTest(shape=bad.shape), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                images = historical_images()
                images["LON"] = png_bytes(bad)
                write_source(base / "source", images)
                with self.assertRaisesRegex(ValueError, "128x128 grayscale PNG"):
                    export_geography(base / "inputs", base / "source")

    def test_non_png_does_not_get_silently_reencoded(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            images = historical_images()
            stream = io.BytesIO()
            Image.open(io.BytesIO(images["LON"])).save(stream, format="JPEG")
            images["LON"] = stream.getvalue()
            write_source(base / "source", images)
            with self.assertRaisesRegex(ValueError, "grayscale PNG"):
                export_geography(base / "inputs", base / "source")

    def test_missing_source_and_other_grid_size_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            with self.assertRaisesRegex(FileNotFoundError, "GI source not found"):
                export_geography(base / "inputs", base / "missing")
            with self.assertRaisesRegex(ValueError, "128x128"):
                export_geography(base / "inputs", base / "missing", GridSpec(width=256))


if __name__ == "__main__":
    unittest.main()
