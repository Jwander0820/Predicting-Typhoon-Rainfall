import unittest
from pathlib import Path
from unittest.mock import patch

from core.predictor import Predictor
from utils.model_output import Unet
from typhoon_rainfall.data.contracts import InputModes, compute_input_shape


class LegacyModelDefaultsTests(unittest.TestCase):
    def test_default_facades_pass_bundled_multimodal_contract(self):
        for constructor, target in ((Predictor, "core.predictor.RainfallPredictor"),
                                    (Unet, "utils.model_output.RainfallPredictor")):
            with self.subTest(constructor=constructor.__name__), patch(target) as predictor:
                constructor()
                data, model, _ = predictor.call_args.args
                shape = compute_input_shape(data.image_size, InputModes.from_data_config(data))
                self.assertEqual(shape, (128, 128, 9))
                self.assertEqual(model.name, "Unet3")
                self.assertEqual(model.num_classes, 101)
                self.assertIsNone(model.checkpoint_path)

    def test_explicit_single_modality_and_custom_weight_remain_available(self):
        for constructor, target, path_argument in (
            (Predictor, "core.predictor.RainfallPredictor", "checkpoint_path"),
            (Unet, "utils.model_output.RainfallPredictor", "model_path"),
        ):
            with self.subTest(constructor=constructor.__name__), patch(target) as predictor:
                constructor(RD=False, IR=False, RA=True, GI=False,
                            num_classes=18, **{path_argument: "custom/ra-only.h5"})
                data, model, _ = predictor.call_args.args
                shape = compute_input_shape(data.image_size, InputModes.from_data_config(data))
                self.assertEqual(shape, (128, 128, 3))
                self.assertEqual(model.num_classes, 18)
                self.assertEqual(model.checkpoint_path, Path("custom/ra-only.h5"))


if __name__ == "__main__":
    unittest.main()
