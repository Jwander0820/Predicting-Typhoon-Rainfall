import tempfile
import unittest
from pathlib import Path

from typhoon_rainfall.config import ProjectConfig, save_project_config


class ConfigTests(unittest.TestCase):
    def test_project_config_roundtrip_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "config.json"
            config = ProjectConfig.defaults()
            save_project_config(config, output_path)
            loaded = ProjectConfig.from_json(output_path)
            self.assertEqual(loaded.data.shift, 1)
            self.assertEqual(loaded.model.name, "Unet5")
            self.assertEqual(str(loaded.model.checkpoint_dir), "checkpoint")


if __name__ == "__main__":
    unittest.main()
