import tempfile
import unittest
from pathlib import Path

from typhoon_rainfall.inference.checkpoints import preferred_checkpoint_path, resolve_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_preferred_checkpoint_path(self):
        path = preferred_checkpoint_path(Path("checkpoint"), "Unet5")
        self.assertEqual(path, Path("checkpoint") / "Unet5_val_min_RMSE.h5")

    def test_resolve_checkpoint_prefers_named_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            preferred = checkpoint_dir / "Unet5_val_min_RMSE.h5"
            preferred.write_text("x", encoding="utf-8")
            other = checkpoint_dir / "other.h5"
            other.write_text("y", encoding="utf-8")
            self.assertEqual(resolve_checkpoint(checkpoint_dir, "Unet5"), preferred)

    def test_resolve_checkpoint_rejects_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            first = checkpoint_dir / "a.h5"
            second = checkpoint_dir / "b.h5"
            first.write_text("a", encoding="utf-8")
            second.write_text("b", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "Checkpoint for MissingModel not found"):
                resolve_checkpoint(checkpoint_dir, "MissingModel")

    def test_explicit_checkpoint_takes_precedence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            preferred = checkpoint_dir / "Unet3_val_min_RMSE.h5"
            preferred.write_text("preferred", encoding="utf-8")
            explicit = checkpoint_dir / "custom.h5"
            explicit.write_text("explicit", encoding="utf-8")
            self.assertEqual(resolve_checkpoint(checkpoint_dir, "Unet3", explicit), explicit)

    def test_directory_is_not_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            preferred = checkpoint_dir / "Unet3_val_min_RMSE.h5"
            preferred.mkdir()
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint(checkpoint_dir, "Unet3")
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint(checkpoint_dir, "Unet3", preferred)


if __name__ == "__main__":
    unittest.main()
