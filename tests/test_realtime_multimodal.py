import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image

from test_realtime_pipeline import sample_images
from typhoon_rainfall_realtime.model_contract import validate_checkpoint
from typhoon_rainfall_realtime.pipeline import run_images


TZ = timezone(timedelta(hours=8))
END = datetime(2026, 11, 6, 18, tzinfo=TZ)


def weights_file(path, channels=9, classes=101):
    with h5py.File(path, "w") as file:
        file.attrs["layer_names"] = ["first", "last"]
        for name, shape in (("first", (3, 3, channels, 2)), ("last", (1, 1, 2, classes))):
            group = file.create_group(name)
            group.attrs["weight_names"] = ["kernel:0"]
            group.create_dataset("kernel:0", shape=shape)


class MultimodalWorkflowTests(unittest.TestCase):
    def test_stale_training_metadata_is_rejected_before_downloading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "weights.h5"
            weights_file(checkpoint)
            checkpoint.with_suffix(".metadata.json").write_text(json.dumps({"checkpoint_sha256": "old-weights"}))
            config = root / "config.json"
            config.write_text(json.dumps({"data": {"use_ra": True, "use_gi": True},
                                          "model": {"num_classes": 101, "checkpoint_path": str(checkpoint)}}))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                run_images(config_path=config, client=SimpleNamespace())

    def test_wrong_trained_channels_are_rejected_before_downloading(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "weights.h5"
            weights_file(checkpoint, channels=4, classes=100)
            config = Path(temporary) / "config.json"
            config.write_text(json.dumps({"data": {"use_ra": True, "use_gi": True},
                                          "model": {"num_classes": 101, "checkpoint_path": str(checkpoint)}}))
            with self.assertRaisesRegex(ValueError, "4 channels / 100 classes"):
                run_images(config_path=config, client=SimpleNamespace())

    def test_real_idw_and_gi_flow_into_nine_channel_predictor(self):
        snapshots = sample_images()
        for snapshot in snapshots.values():
            snapshot.observed_at = END - timedelta(minutes=30)
        captured = {}

        def get_hour(start, end, **kwargs):
            captured["image_window"] = (start, end)
            return snapshots

        def collect(client, output_dir, **kwargs):
            output_dir.mkdir(parents=True)
            path = output_dir / "rainfall_hourly.json"
            rows = [{"interval_start": (END - timedelta(hours=1)).isoformat(),
                     "interval_end": END.isoformat(), "longitude_wgs84": lon,
                     "latitude_wgs84": lat, "rainfall_mm": mm}
                    for lon, lat, mm in ((121.0, 23.0, 3), (121.5, 24.0, 4))]
            path.write_text(json.dumps(rows), encoding="utf-8")
            return {"files": {"hourly_json": str(path)}, "hours": 1}

        class Predictor:
            def __init__(self, data, model, predict):
                self.output = predict.output_dir

            def predict_array(self, merged):
                captured["merged"] = merged.copy()
                return np.full((128, 128), 2, dtype=np.uint8)

            def save_prediction(self, labels, stem):
                csv = self.output / (stem + ".csv")
                png = self.output / (stem + ".png")
                np.savetxt(csv, labels, delimiter=",")
                Image.fromarray(labels).save(png)
                return SimpleNamespace(csv_path=csv, png_path=png)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "weights.h5"
            weights_file(checkpoint)
            config = root / "config.json"
            config.write_text(json.dumps({"data": {"use_ra": True, "use_gi": True},
                                          "model": {"name": "Unet3", "num_classes": 101,
                                                    "checkpoint_path": str(checkpoint)}}))
            gi = root / "gi"
            gi.mkdir()
            lon = np.broadcast_to(np.arange(128, dtype=np.uint8) * 2, (128, 128)).copy()
            Image.fromarray(lon).save(gi / "lon.png")
            Image.fromarray(lon.T[::-1].copy()).save(gi / "lat.png")
            boundary = root / "boundary.json"
            boundary.write_text(json.dumps({"type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {"countyname": "臺北市"},
                 "geometry": {"type": "Polygon", "coordinates": [
                     [[119, 21], [123, 21], [123, 26], [119, 26], [119, 21]]]}}]}))
            with patch("typhoon_rainfall_realtime.rainfall.collect_rainfall", side_effect=collect):
                output = run_images(config_path=config, output_root=root / "runs",
                                    client=SimpleNamespace(get_images_for_hour=get_hour),
                                    predictor_factory=Predictor, geography_source=gi,
                                    boundary_file=boundary, now=END + timedelta(minutes=20))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(captured["image_window"], (END - timedelta(hours=1), END))
            self.assertEqual(manifest["prediction"]["input_shape"], [128, 128, 9])
            self.assertEqual(manifest["prediction"]["target_interval"],
                             [END.isoformat(), (END + timedelta(hours=1)).isoformat()])
            self.assertEqual(manifest["model"]["modalities"], ["RD", "IR", "RA", "GI"])
            self.assertTrue((output / "forecast.png").is_file())
            self.assertTrue((output / "prediction_mm.csv").is_file())
            merged = captured["merged"]
            self.assertEqual(merged.shape, (128, 128, 9))
            np.testing.assert_allclose(merged[0, 0, :4], np.array([17, 83, 161, 85]) / 255)
            self.assertGreater(float(merged[..., 4].min()), 0)
            np.testing.assert_array_equal(merged[..., 4], merged[..., 5])
            np.testing.assert_array_equal(merged[..., 5], merged[..., 6])
            self.assertAlmostEqual(float(merged[0, -1, 7]), 254 / 255)
            self.assertAlmostEqual(float(merged[0, 0, 8]), 254 / 255)
            self.assertEqual(float(merged[-1, 0, 8]), 0)

    def test_checkpoint_contract_reads_channels_and_output_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "weights.h5"
            weights_file(checkpoint)
            details = validate_checkpoint(checkpoint, (128, 128, 9), 101)
            self.assertEqual(details["input_channels"], 9)
            self.assertEqual(details["num_classes"], 101)


if __name__ == "__main__":
    unittest.main()
