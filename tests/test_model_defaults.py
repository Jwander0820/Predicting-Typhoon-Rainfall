import unittest
from pathlib import Path
from types import SimpleNamespace

from typhoon_rainfall.config import ModelConfig, ProjectConfig, apply_overrides
from typhoon_rainfall.data.contracts import InputModes, compute_input_shape
from typhoon_rainfall_realtime.model_contract import checkpoint_channels
from typhoon_rainfall_realtime.pipeline import DEFAULT_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CHECKPOINT = Path("checkpoint") / "Unet3_val_min_RMSE.h5"


class BundledModelDefaultsTests(unittest.TestCase):
    def test_public_configs_and_dataclasses_share_nine_channel_contract(self):
        configs = [ProjectConfig.defaults()]
        configs.extend(ProjectConfig.from_json(PROJECT_ROOT / "configs" / name)
                       for name in ("default.json", "realtime.json"))
        for config in configs:
            with self.subTest(config=config):
                modes = InputModes.from_data_config(config.data)
                self.assertEqual(compute_input_shape(config.data.image_size, modes), (128, 128, 9))
                self.assertEqual(config.model.name, "Unet3")
                self.assertEqual(config.model.num_classes, 101)
        self.assertEqual(compute_input_shape((128, 128), InputModes()), (128, 128, 9))
        self.assertEqual(DEFAULT_CONFIG, PROJECT_ROOT / "configs" / "realtime.json")

    def test_public_configs_point_to_the_replaced_tracked_checkpoint(self):
        for name in ("default.json", "realtime.json"):
            config = ProjectConfig.from_json(PROJECT_ROOT / "configs" / name)
            self.assertEqual(config.model.checkpoint_path, CANONICAL_CHECKPOINT)
        details = checkpoint_channels(PROJECT_ROOT / CANONICAL_CHECKPOINT)
        self.assertEqual(details["input_channels"], 9)
        self.assertEqual(details["num_classes"], 101)
        self.assertEqual(details["convolution_layers"], 12)

    def test_custom_model_does_not_inherit_unet3_explicit_checkpoint(self):
        config = ProjectConfig.from_json(PROJECT_ROOT / "configs" / "default.json")
        args = SimpleNamespace(interval=None, shift=None, model_name="Unet5", checkpoint=None,
                               epochs=None, batch_size=None, learning_rate=None, limit=None)
        changed = apply_overrides(config, args)
        self.assertEqual(changed.model.name, "Unet5")
        self.assertIsNone(changed.model.checkpoint_path)
        self.assertIsNone(ModelConfig(name="Unet5").checkpoint_path)
        self.assertEqual(config.model.checkpoint_path, CANONICAL_CHECKPOINT)

    def test_custom_model_keeps_explicit_user_checkpoint(self):
        config = ProjectConfig.from_json(PROJECT_ROOT / "configs" / "default.json")
        explicit = Path("custom") / "Unet5.h5"
        args = SimpleNamespace(interval=None, shift=None, model_name="Unet5", checkpoint=explicit,
                               epochs=None, batch_size=None, learning_rate=None, limit=None)
        changed = apply_overrides(config, args)
        self.assertEqual(changed.model.checkpoint_path, explicit)


if __name__ == "__main__":
    unittest.main()
