"""Convenience runners for prediction commands and PyCharm scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from PIL import Image

from typhoon_rainfall.config import ProjectConfig
from typhoon_rainfall.data.contracts import parse_annotation_file
from typhoon_rainfall.inference.predictor import RainfallPredictor


def run_prediction(config: ProjectConfig) -> Dict[str, Path]:
    """Run prediction for pairs in the configured split.

    `limit` in PredictConfig is useful for smoke tests because it lets us check
    the full prediction path without processing the entire validation set.
    """

    predictor = RainfallPredictor(config.data, config.model, config.predict)
    pairs = parse_annotation_file(config.data.train_list_dir() / f"{config.predict.source_split}.txt")
    if config.predict.limit is not None:
        pairs = pairs[: config.predict.limit]

    last_artifacts = None
    prediction_count = 0
    for pair in pairs:
        prediction = predictor.predict_dataset_pair(config.predict.source_split, pair)
        last_artifacts = predictor.save_prediction(prediction, pair.prediction_stem(config.data.shift))
        prediction_count += 1
    if last_artifacts is None:
        raise ValueError("No samples were available for prediction.")
    return {
        "prediction_count": prediction_count,
        "prediction_array_dir": config.predict.prediction_array_dir,
        "prediction_plot_dir": config.predict.prediction_plot_dir,
        "last_csv_path": last_artifacts.csv_path,
        "last_png_path": last_artifacts.png_path,
    }


def run_single_prediction(
    config: ProjectConfig,
    stem: str,
    rd_image_path: Optional[Path] = None,
    ir_image_path: Optional[Path] = None,
    ra_image_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Run one manual prediction from explicitly provided image files."""
    predictor = RainfallPredictor(config.data, config.model, config.predict)
    rd_image = Image.open(rd_image_path) if rd_image_path else None
    ir_image = Image.open(ir_image_path) if ir_image_path else None
    ra_image = Image.open(ra_image_path) if ra_image_path else None
    prediction = predictor.predict_modal_images(rd_image=rd_image, ir_image=ir_image, ra_image=ra_image)
    artifacts = predictor.save_prediction(prediction, stem)
    return {
        "prediction_count": 1,
        "prediction_array_dir": config.predict.prediction_array_dir,
        "prediction_plot_dir": config.predict.prediction_plot_dir,
        "csv_path": artifacts.csv_path,
        "png_path": artifacts.png_path,
    }
