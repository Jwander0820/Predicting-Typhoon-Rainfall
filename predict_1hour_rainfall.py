# -*- coding: utf-8 -*-
"""Legacy-style realtime prediction script.

此檔案保留原本「抓即時 IR/RD 圖後丟進模型」的研究測試流程。
它依賴外部氣象網站，因此比離線資料推論更不穩定；正式研究驗證
仍建議使用 `run_predict_sample.py` 或 CLI 的 `predict/evaluate`。
"""
from pathlib import Path

from tensorflow.keras import backend as K

from typhoon_rainfall.config import DataConfig, ModelConfig, PredictConfig, ProjectConfig
from typhoon_rainfall.experimental.crawler import WeatherCrawler
from typhoon_rainfall.inference.runner import run_single_prediction


def main() -> None:
    """Download current weather images, run prediction, and save CSV/PNG."""
    # 以既有研究設定為預設：RD + IR 預測 t+1 的雨量標籤分布。
    K.clear_session()

    crawler = WeatherCrawler(output_dir=Path("img"))
    formatted = crawler.format_time()
    ir_exists, rd_exists = crawler.get_weather_images()
    if not (ir_exists and rd_exists):
        raise RuntimeError("Unable to download both IR and RD images for prediction.")

    compact = formatted.compact
    project_config = ProjectConfig(
        data=DataConfig(
            interval="EWB01",
            shift=1,
            use_rd=rd_exists,
            use_ir=ir_exists,
            use_ra=False,
            use_gi=False,
        ),
        model=ModelConfig(name="Unet3", num_classes=100, checkpoint_dir=Path("checkpoint")),
        predict=PredictConfig(
            output_dir=Path("img"),
            prediction_array_dir=Path("img"),
            prediction_plot_dir=Path("img"),
            blend=False,
        ),
    )
    run_single_prediction(
        config=project_config,
        stem=f"{compact}_predict",
        rd_image_path=Path("img") / f"{compact}_RD_crop.png",
        ir_image_path=Path("img") / f"{compact}_IR_crop.png",
        ra_image_path=None,
    )


if __name__ == "__main__":
    main()
