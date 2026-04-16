"""Training workflow for the research model.

本檔案把 notebook 中原本分散的訓練流程集中成可重複呼叫的函數。
它負責建立模型、編譯 loss/metrics、建立 generator、設定 callback，
並輸出訓練歷程；模型本身與資料讀取規則則分別交給其他模組。
"""

from __future__ import annotations

import csv
from pathlib import Path

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

from typhoon_rainfall.config import ProjectConfig
from typhoon_rainfall.data.contracts import InputModes, compute_input_shape, parse_annotation_file
from typhoon_rainfall.data.dataset import MergedDataGenerator, build_dataset_paths
from typhoon_rainfall.models.factory import build_model
from typhoon_rainfall.training.losses import build_loss
from typhoon_rainfall.training.metrics import default_metrics
from typhoon_rainfall.inference.checkpoints import preferred_checkpoint_path
from typhoon_rainfall.visualization.plotting import save_training_history_plot


def run_training(config: ProjectConfig):
    """Run the full training flow using a ProjectConfig.

    回傳 dict 是為了讓 notebook 或測試可以拿到 history、模型物件、
    checkpoint 與輸出檔路徑。CLI 使用時通常不需要讀取回傳值。
    """

    K.clear_session()
    dataset_paths = build_dataset_paths(config.data)
    input_modes = InputModes.from_data_config(config.data)
    input_shape = compute_input_shape(config.data.image_size, input_modes)

    train_pairs = parse_annotation_file(config.data.train_list_dir() / "train.txt")
    val_pairs = parse_annotation_file(config.data.train_list_dir() / "val.txt")

    model = build_model(
        name=config.model.name,
        input_shape=input_shape,
        num_classes=config.model.num_classes,
    )
    model.compile(
        loss=build_loss(
            use_focal_loss=config.train.use_focal_loss,
            num_classes=config.model.num_classes,
        ),
        optimizer=Adam(learning_rate=config.train.learning_rate),
        metrics=default_metrics(),
    )

    config.train.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.train.logs_dir.mkdir(parents=True, exist_ok=True)
    config.train.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = preferred_checkpoint_path(
        checkpoint_dir=config.train.checkpoint_dir,
        model_name=config.model.name,
    )
    # Monitor name follows the historical Keras metric function name `_RMSE`.
    # Keeping this name helps preserve output/checkpoint behavior from notebooks.
    callbacks = [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val__RMSE",
            save_weights_only=True,
            save_best_only=True,
            verbose=1,
            mode="min",
            save_freq="epoch",
        ),
        TensorBoard(log_dir=str(config.train.logs_dir)),
        EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=15,
            verbose=1,
            mode="min",
            restore_best_weights=True,
        ),
    ]

    train_generator = MergedDataGenerator(
        batch_size=config.train.batch_size,
        pairs=train_pairs,
        dataset_paths=dataset_paths,
        split="train",
        input_modes=input_modes,
        num_classes=config.model.num_classes,
        shuffle_each_epoch=True,
    ).iter_batches()
    val_generator = MergedDataGenerator(
        batch_size=config.train.validation_batch_size,
        pairs=val_pairs,
        dataset_paths=dataset_paths,
        split="val",
        input_modes=input_modes,
        num_classes=config.model.num_classes,
        shuffle_each_epoch=False,
    ).iter_batches()

    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, len(train_pairs) // config.train.batch_size),
        validation_data=val_generator,
        validation_steps=len(val_pairs),
        epochs=config.train.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    history_csv_path = (
        config.train.output_dir
        / f"model_history_t2t+{config.data.shift}_{config.data.interval}_{config.model.name}.csv"
    )
    history_png_path = history_csv_path.with_suffix(".png")
    history_rows = export_history(history.history)
    save_history_csv(history_rows, history_csv_path)
    save_training_history_plot(history_rows, history_png_path)
    return {
        "history": history,
        "model": model,
        "history_csv": history_csv_path,
        "history_png": history_png_path,
        "checkpoint_path": checkpoint_path,
    }


def export_history(history_dict):
    """Convert Keras History.history into row dictionaries for CSV writing."""
    epochs = len(next(iter(history_dict.values()), []))
    rows = []
    for epoch_index in range(epochs):
        row = {"epoch": epoch_index}
        for key, values in history_dict.items():
            row[key] = values[epoch_index]
        rows.append(row)
    return rows


def save_history_csv(rows, output_path: Path) -> None:
    """Save training history rows as UTF-8 CSV."""
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
