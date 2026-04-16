import tempfile
import time
import unittest
from pathlib import Path

from typhoon_rainfall.inference.checkpoints import preferred_checkpoint_path, resolve_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_preferred_checkpoint_path(self):
        path = preferred_checkpoint_path(Path("checkpoint"), "Unet5")
        self.assertEqual(str(path), "checkpoint/Unet5_val_min_RMSE.h5")

    def test_resolve_checkpoint_prefers_named_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            preferred = checkpoint_dir / "Unet5_val_min_RMSE.h5"
            preferred.write_text("x", encoding="utf-8")
            other = checkpoint_dir / "other.h5"
            other.write_text("y", encoding="utf-8")
            self.assertEqual(resolve_checkpoint(checkpoint_dir, "Unet5"), preferred)

    def test_resolve_checkpoint_falls_back_to_latest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            first = checkpoint_dir / "a.h5"
            second = checkpoint_dir / "b.h5"
            first.write_text("a", encoding="utf-8")
            time.sleep(0.01)
            second.write_text("b", encoding="utf-8")
            self.assertEqual(resolve_checkpoint(checkpoint_dir, "MissingModel"), second)


if __name__ == "__main__":
    unittest.main()
