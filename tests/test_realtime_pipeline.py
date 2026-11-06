import hashlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from typhoon_rainfall_realtime.pipeline import load_prediction_config, prepare_images, run_images


def sample_images():
    timestamp = datetime(2026, 10, 31, 8, 0, tzinfo=timezone(timedelta(hours=8)))
    snapshots = {}
    for kind, size, mode, color, fmt in (
        ("RD", (3600, 3600), "RGB", (17, 83, 161), "PNG"),
        ("IR", (800, 800), "RGB", (85, 85, 85), "JPEG"),
    ):
        buffer = io.BytesIO()
        Image.new(mode, size, color).save(buffer, format=fmt)
        snapshots[kind] = SimpleNamespace(
            content=buffer.getvalue(), product_id="O-A0058-003" if kind == "RD" else "O-C0042-004",
            image_url="https://example.invalid/" + kind,
            observed_at=timestamp, published_at=timestamp, acquired_at=timestamp)
    return snapshots


class PipelineTests(unittest.TestCase):
    def test_live_images_keep_thesis_rgb_gray_and_four_channel_contract(self):
        from typhoon_rainfall.config import DataConfig
        from typhoon_rainfall.data.contracts import InputModes
        from typhoon_rainfall.data.dataset import merge_modal_arrays
        with tempfile.TemporaryDirectory() as temporary:
            paths = prepare_images(sample_images(), Path(temporary))
            with Image.open(paths["RD"]) as rd, Image.open(paths["IR"]) as ir:
                self.assertEqual(rd.mode, "RGB")
                self.assertEqual(ir.mode, "L")
                self.assertEqual(rd.size, (128, 128))
                merged = merge_modal_arrays(InputModes.from_data_config(DataConfig(use_ra=False, use_gi=False)), 100,
                                            rd_image=rd, ir_image=ir)
            self.assertEqual(merged.shape, (128, 128, 4))
            np.testing.assert_allclose(merged[0, 0], np.array([17, 83, 161, 85]) / 255, atol=1e-6)

    def test_reject_product_size_change_before_silent_bad_crop(self):
        snapshots = sample_images()
        buffer = io.BytesIO()
        Image.new("RGB", (800, 800)).save(buffer, format="PNG")
        snapshots["RD"].content = buffer.getvalue()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "dimensions"):
                prepare_images(snapshots, Path(temporary))

    def test_reject_truncated_jpeg_that_still_has_a_valid_image_header(self):
        snapshots = sample_images()
        noise = np.random.default_rng(42).integers(0, 256, (800, 800, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(noise, "RGB").save(buffer, format="JPEG", quality=90)
        content = buffer.getvalue()
        snapshots["IR"].content = content[:len(content) // 2]
        # The header still passes the old format/mode checks, unlike a random file.
        with Image.open(io.BytesIO(snapshots["IR"].content)) as image:
            self.assertEqual(image.size, (800, 800))
            self.assertEqual(image.format, "JPEG")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Incomplete or invalid IR"):
                prepare_images(snapshots, Path(temporary))
            self.assertFalse((Path(temporary) / "inputs" / "IR.png").exists())

    def test_fetch_records_provenance_and_does_not_load_predictor(self):
        snapshots = sample_images()
        client = SimpleNamespace(get_latest_images=lambda **kwargs: snapshots)
        with tempfile.TemporaryDirectory() as temporary:
            directory = run_images(output_root=Path(temporary), fetch_only=True, client=client,
                                   predictor_factory=lambda *args: self.fail("fetch loaded the predictor"))
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "fetched")
            self.assertEqual(manifest["sources"]["RD"]["sha256"], hashlib.sha256(snapshots["RD"].content).hexdigest())
            self.assertEqual(manifest["prediction"]["target_time"], "2026-10-31T09:00:00+08:00")
            self.assertFalse((directory / "prediction.png").exists())

    def test_failed_download_persists_failure_not_success(self):
        def failed(**kwargs):
            raise ValueError("fixture unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "unavailable"):
                run_images(output_root=Path(temporary), fetch_only=True,
                           client=SimpleNamespace(get_latest_images=failed))
            manifests = list(Path(temporary).glob("*/manifest.json"))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(json.loads(manifests[0].read_text())["status"], "failed")

    def test_ra_encoding_and_unrelated_checkpoint_fallback_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({"data": {"use_ra": True}, "model": {"num_classes": 100}}))
            with self.assertRaisesRegex(ValueError, "101-class"):
                load_prediction_config(path)
            (Path(temporary) / "OtherModel.h5").touch()
            path.write_text(json.dumps({"model": {"name": "Unet3", "checkpoint_dir": temporary}}))
            with self.assertRaises(FileNotFoundError):
                load_prediction_config(path)

    def test_missing_bundled_weights_requests_restore_instead_of_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            missing = Path(temporary) / "Unet3_val_min_RMSE.h5"
            path.write_text(json.dumps({"model": {"checkpoint_path": str(missing)}}))
            with self.assertRaises(FileNotFoundError) as error:
                load_prediction_config(path)
            self.assertIn("Restore the pretrained checkpoint", str(error.exception))
            self.assertNotIn("train_realtime.py", str(error.exception))


if __name__ == "__main__":
    unittest.main()
