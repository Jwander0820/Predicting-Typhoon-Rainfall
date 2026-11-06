import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from typhoon_rainfall_realtime.training import load_training_data, validation_baselines


class MultimodalTrainingDataTests(unittest.TestCase):
    def fixture(self, root, train_pair=("201308271130.png", "201308271230.png")):
        annotation = root / "N-4" / "t2t+1"
        annotation.mkdir(parents=True)
        annotations = {"train": train_pair,
                       "val": ("201509281130.png", "201509281230.png")}
        for split, (source, target) in annotations.items():
            (annotation / (split + ".txt")).write_text(source + ";" + target + "\n", encoding="utf8")
            for kind, folder, mode, value in (
                ("RD", "train_data_RD", "RGB", 51), ("IR", "train_data_IR", "L", 102),
                ("RA", "train_data_rainfallmap/EWB01", "L", 100),
            ):
                directory = root / folder / split
                directory.mkdir(parents=True)
                Image.new(mode, (128, 128), value if mode == "L" else (value, value, value)).save(directory / source)
                if kind == "RA":
                    Image.new("L", (128, 128), 20).save(directory / target)
        gi = root / "train_data_GI"
        gi.mkdir()
        Image.new("L", (128, 128), 204).save(gi / "lon.png")
        Image.new("L", (128, 128), 153).save(gi / "lat.png")
        coast = np.zeros((128, 128))
        coast[0, 0] = 30300
        np.savetxt(root / "taiwan_coastline_128x128_full.csv", coast, delimiter=",")

    def test_real_modalities_keep_label100_and_normalize_in_core_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            data = load_training_data(root)
            self.assertEqual(data.train_inputs.shape, (1, 128, 128, 9))
            np.testing.assert_allclose(data.train_inputs[0, 10, 10],
                                       [0.2, 0.2, 0.2, 0.4, 100/101, 100/101, 100/101, 0.8, 0.6])
            self.assertEqual(int(data.val_targets[0, 10, 10]), 20)
            self.assertFalse(data.land_mask[0, 0])
            self.assertEqual(data.provenance["land_pixels"], 16383)
            baseline = validation_baselines(data)
            self.assertAlmostEqual(baseline["zero_land_rmse"], 20)
            self.assertAlmostEqual(baseline["persistence_land_rmse"], 80)

    def test_overlapping_train_validation_timestamps_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            annotation = root / "N-4/t2t+1"
            (annotation / "val.txt").write_text((annotation / "train.txt").read_text(), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "overlapping"):
                load_training_data(root)

    def test_non_hourly_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            (root / "N-4/t2t+1/train.txt").write_text(
                "201308271130.png;201308271330.png\n", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "exactly one-hour"):
                load_training_data(root)

    def test_missing_or_out_of_contract_rainfall_is_not_silently_clipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            Image.new("L", (128, 128), 255).save(
                root / "train_data_rainfallmap/EWB01/train/201308271130.png")
            with self.assertRaisesRegex(ValueError, "outside 0..100"):
                load_training_data(root)

    def test_known_original_calendar_overflow_keeps_files_and_records_anomaly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root, train_pair=("201909302330.png", "201909310030.png"))
            data = load_training_data(root)
            self.assertEqual(len(data.train_inputs), 1)
            self.assertEqual(data.provenance["calendar_overflow_filenames"], ["201909310030.png"])
            self.assertTrue((root / "train_data_rainfallmap/EWB01/train/201909310030.png").is_file())


if __name__ == "__main__":
    unittest.main()
