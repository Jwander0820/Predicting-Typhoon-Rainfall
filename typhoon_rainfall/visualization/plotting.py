"""Plotting helpers for rainfall predictions and training history."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np


class Plotting:
    """Color-map wrapper compatible with the original plotting utility."""

    def __init__(self) -> None:
        self.nws_precip_colors = [
            "#fdfdfd",
            "#c1c1c1",
            "#99ffff",
            "#00ccff",
            "#0099ff",
            "#0166ff",
            "#329900",
            "#33ff00",
            "#ffff00",
            "#ffcc00",
            "#fe9900",
            "#fe0000",
            "#cc0001",
            "#990000",
            "#990099",
            "#cb00cc",
            "#ff00fe",
            "#feccff",
        ]
        self.precip_colormap = matplotlib.colors.ListedColormap(self.nws_precip_colors)
        epsilon = 1e-7
        clevels = [
            0,
            0.1,
            1 + epsilon,
            2 + epsilon,
            6 + epsilon,
            10 + epsilon,
            15 + epsilon,
            20 + epsilon,
            30 + epsilon,
            40 + epsilon,
            50 + epsilon,
            70 + epsilon,
            90 + epsilon,
            110 + epsilon,
            130 + epsilon,
            150 + epsilon,
            200 + epsilon,
            300 + epsilon,
            500 + epsilon,
        ]
        self.norm = matplotlib.colors.BoundaryNorm(clevels, 18)

    def plot_predict(self, filename: str) -> None:
        """Load a prediction CSV and save its PNG next to the CSV file."""
        predict = np.genfromtxt(filename, delimiter=",")
        save_prediction_plot(predict, Path(filename).with_suffix(".png"))


def save_prediction_array(prediction: np.ndarray, output_path: Path) -> None:
    """Save a 2-D label prediction as integer CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, prediction, delimiter=",", fmt="%d")


def save_prediction_plot(prediction: np.ndarray, output_path: Path) -> None:
    """Save a 2-D prediction label map as a colorized rainfall PNG."""
    plotter = Plotting()
    plt.figure(figsize=(1, 1))
    plt.imshow(prediction, cmap=plotter.precip_colormap, alpha=1, norm=plotter.norm)
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=128, pad_inches=0.0)
    plt.close()


def save_training_history_plot(history_rows, output_path: Path) -> None:
    """Save the standard multi-panel training history figure."""
    metric_groups = [
        ("loss", "val_loss"),
        ("_Iou_score", "val__Iou_score"),
        ("_f_score", "val__f_score"),
        ("metric_precision", "val_metric_precision"),
        ("metric_recall", "val_metric_recall"),
        ("_RMSE", "val__RMSE"),
        ("acc", "val_acc"),
    ]
    if not history_rows:
        return
    plt.figure(figsize=(7, 24), dpi=60)
    history_columns = {key: [row.get(key) for row in history_rows] for key in history_rows[0]}
    for index, (train_metric, val_metric) in enumerate(metric_groups, start=1):
        plt.subplot(len(metric_groups), 1, index)
        if train_metric in history_columns:
            plt.plot(history_columns[train_metric], label="train")
        if val_metric in history_columns:
            plt.plot(history_columns[val_metric], label="val")
        plt.title(train_metric.replace("_", ""))
        plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
