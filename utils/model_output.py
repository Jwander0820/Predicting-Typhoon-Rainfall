"""Backward-compatible model output wrapper.

原本 notebook 會使用 `utils.model_output.Unet` 做模型載入與推論。
現在正式實作在 `typhoon_rainfall.inference.predictor`，此檔案只保留
舊 API 形狀，降低重構對既有研究紀錄的衝擊。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from typhoon_rainfall.config import DataConfig, ModelConfig, PredictConfig
from typhoon_rainfall.inference.predictor import RainfallPredictor


class Unet:
    """Legacy class name kept for existing notebooks and scripts."""

    def __init__(
        self,
        model_path=None,
        model_image_size=None,
        num_classes=18,
        blend=None,
        RD=None,
        IR=None,
        GI=None,
        RA=None,
        choose_model=None,
        backbone="resnet50",
        downsample_factor=8,
    ):
        """Convert historical Unet wrapper arguments into new configs."""
        del model_image_size, backbone, downsample_factor
        self.data_config = DataConfig(
            interval="EWB01",
            shift=1,
            use_rd=bool(RD),
            use_ir=bool(IR),
            use_ra=bool(RA),
            use_gi=bool(GI),
        )
        self.model_config = ModelConfig(
            name=choose_model or "Unet3",
            num_classes=num_classes,
            checkpoint_dir=Path("checkpoint"),
            checkpoint_path=Path(model_path) if model_path else None,
        )
        self.predict_config = PredictConfig(blend=bool(blend))
        self.predictor = RainfallPredictor(self.data_config, self.model_config, self.predict_config)

    def detect_image_merge_label(self, RD_image, IR_image, RA_image):
        """Return a 2-D label array for merged RD/IR/RA inputs."""
        ra_image = None
        if RA_image is not None and not (isinstance(RA_image, (int, float)) and RA_image == 0):
            ra_image = RA_image
        return self.predictor.predict_modal_images(rd_image=RD_image, ir_image=IR_image, ra_image=ra_image)

    def detect_image(self, image):
        """Compatibility method for single-image segmentation style output."""
        prediction = self.predictor.predict_modal_images(rd_image=image, ir_image=None, ra_image=None)
        return Image.fromarray(np.uint8(prediction))
