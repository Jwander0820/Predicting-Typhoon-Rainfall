"""Backward-compatible predictor wrapper.

舊版程式會從 `core.predictor` import `Predictor`。重構後主要邏輯已移到
`typhoon_rainfall.inference`，但保留這個檔案可讓既有 notebook 或腳本
不必一次全部改寫。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from typhoon_rainfall.config import DataConfig, ModelConfig, PredictConfig
from typhoon_rainfall.inference.predictor import RainfallPredictor


class Predictor:
    """Legacy facade around `RainfallPredictor`."""

    def __init__(
        self,
        RD: bool = False,
        IR: bool = False,
        RA: bool = False,
        GI: bool = False,
        shift: int = 1,
        choose_model: str = "Unet3",
        interval: str = "EWB01",
        num_classes: int = 100,
        blend: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """Translate legacy constructor flags into the new config objects."""
        data_config = DataConfig(
            interval=interval,
            shift=shift,
            use_rd=RD,
            use_ir=IR,
            use_ra=RA,
            use_gi=GI,
        )
        model_config = ModelConfig(
            name=choose_model,
            num_classes=num_classes,
            checkpoint_dir=Path("checkpoint"),
            checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        )
        predict_config = PredictConfig(blend=blend)
        self.predictor = RainfallPredictor(data_config, model_config, predict_config)

    def predict(self, RD_image, IR_image, RA_image):
        """Run prediction using legacy RD/IR/RA positional arguments."""
        normalized_ra = None
        if RA_image is not None and not (isinstance(RA_image, (int, float)) and RA_image == 0):
            normalized_ra = RA_image
        return self.predictor.predict_modal_images(
            rd_image=RD_image,
            ir_image=IR_image,
            ra_image=normalized_ra,
        )

    def save_prediction(self, r_label, filename):
        """Save prediction labels as integer CSV, matching old behavior."""
        np.savetxt(filename, r_label, delimiter=",", fmt="%d")
