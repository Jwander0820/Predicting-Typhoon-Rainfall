"""Validation-set evaluation workflow."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from typhoon_rainfall.config import ProjectConfig
from typhoon_rainfall.data.contracts import parse_annotation_file
from typhoon_rainfall.data.dataset import build_dataset_paths, load_target_labels
from typhoon_rainfall.inference.predictor import RainfallPredictor


def run_evaluation(config: ProjectConfig) -> Dict[str, Path]:
    """Predict the validation split and write a CSV summary.

    The RMSE here is computed on rainfall label indices, matching the original
    notebook's quick validation style. Additional true-value CSV evaluation can
    be added later without changing the prediction pipeline.
    """

    predictor = RainfallPredictor(config.data, config.model, config.predict)
    dataset_paths = build_dataset_paths(config.data)
    pairs = parse_annotation_file(config.data.train_list_dir() / "val.txt")
    if config.predict.limit is not None:
        pairs = pairs[: config.predict.limit]

    metrics_rows: List[Dict[str, float]] = []
    for pair in pairs:
        prediction = predictor.predict_dataset_pair("val", pair)
        target = load_target_labels(
            dataset_paths=dataset_paths,
            split="val",
            filename=pair.target_name,
            num_classes=config.model.num_classes,
        )
        label_rmse = float(np.sqrt(np.mean((prediction.astype(np.float32) - target.astype(np.float32)) ** 2)))
        stem = pair.prediction_stem(config.data.shift)
        predictor.save_prediction(prediction, stem)
        metrics_rows.append(
            {
                "source_name": pair.source_name,
                "target_name": pair.target_name,
                "prediction_stem": stem,
                "label_rmse": label_rmse,
            }
        )

    config.predict.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        config.predict.output_dir
        / f"evaluation_t2t+{config.data.shift}_{config.data.interval}_{config.model.name}.csv"
    )
    with summary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(metrics_rows[0].keys()) if metrics_rows else [])
        if metrics_rows:
            writer.writeheader()
            writer.writerows(metrics_rows)
    return {"summary_csv": summary_path}
