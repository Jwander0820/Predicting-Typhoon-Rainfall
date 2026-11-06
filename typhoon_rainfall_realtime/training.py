"""Train the existing small Unet3 with real RD/IR/RA/GI source-target pairs.

This is an optional preparation command. Realtime prediction never trains a
model, and this module does not change the original training pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from typhoon_rainfall.data.contracts import InputModes, parse_annotation_file
from typhoon_rainfall.data.dataset import merge_modal_arrays


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "output" / "4_unet-keras_模型建立" / "database"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoint" / "Unet3_val_min_RMSE.h5"
TAIPEI = timezone(timedelta(hours=8))
NUM_CLASSES = 101
INPUT_SHAPE = (128, 128, 9)
MODES = InputModes(use_rd=True, use_ir=True, use_ra=True, use_gi=True)


@dataclass
class TrainingData:
    train_inputs: np.ndarray
    train_targets: np.ndarray
    val_inputs: np.ndarray
    val_targets: np.ndarray
    land_mask: np.ndarray
    provenance: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image(path: Path, mode: str) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != mode or image.size != (128, 128):
            raise ValueError("Expected 128x128 " + mode + " image: " + str(path))
        return np.asarray(image).copy()


def _pairs(root: Path, split: str, limit):
    path = root / "N-4" / "t2t+1" / (split + ".txt")
    pairs = parse_annotation_file(path)
    if not pairs:
        raise ValueError("Empty source-target split: " + str(path))
    if len({(p.source_name, p.target_name) for p in pairs}) != len(pairs):
        raise ValueError("Duplicate source-target pairs in " + str(path))
    for pair in pairs:
        for name in (pair.source_name, pair.target_name):
            if Path(name).name != name:
                raise ValueError("Annotation filenames must not contain directories")
        start = _timestamp(pair.source_name)
        end = _timestamp(pair.target_name)
        if end - start != timedelta(hours=1):
            raise ValueError("Training annotations must have exactly one-hour leads")
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("Split limits must be positive integers")
        pairs = pairs[:limit]
    return path, pairs


def _timestamp(filename):
    # The original Mitag archive calls Oct 1 "20190931". Keep the filenames
    # untouched and record this known anomaly; normalize only to verify leads.
    if filename.startswith("20190931"):
        filename = "20191001" + filename[8:]
    return datetime.strptime(filename, "%Y%m%d%H%M.png")


def load_training_data(dataset_root: Path = DEFAULT_DATASET, *, limit_train=None,
                       limit_val=None) -> TrainingData:
    """Load the archive's explicit storm splits and original grayscale GI maps."""
    root = Path(dataset_root).resolve()
    lists = {split: _pairs(root, split, limit) for split, limit in
             (("train", limit_train), ("val", limit_val))}
    names = {split: {_timestamp(name) for pair in pairs for name in (pair.source_name, pair.target_name)}
             for split, (_, pairs) in lists.items()}
    if names["train"] & names["val"]:
        raise ValueError("Training and validation contain overlapping timestamps")
    lon_path, lat_path = root / "train_data_GI" / "lon.png", root / "train_data_GI" / "lat.png"
    longitude, latitude = _image(lon_path, "L"), _image(lat_path, "L")
    mask_path = root / "taiwan_coastline_128x128_full.csv"
    coast = np.loadtxt(mask_path, delimiter=",")
    if coast.shape != (128, 128) or not np.all(np.isfinite(coast)):
        raise ValueError("Expected a finite 128x128 historical coastline mask")
    mask = coast == 0
    if not np.any(mask):
        raise ValueError("Historical land mask is empty")
    loaded = {}
    metadata = {}
    for split, (list_path, pairs) in lists.items():
        inputs = np.empty((len(pairs),) + INPUT_SHAPE, dtype=np.float32)
        targets = np.empty((len(pairs), 128, 128), dtype=np.uint8)
        for index, pair in enumerate(pairs):
            rd = _image(root / "train_data_RD" / split / pair.source_name, "RGB")
            ir = _image(root / "train_data_IR" / split / pair.source_name, "L")
            ra = _image(root / "train_data_rainfallmap" / "EWB01" / split / pair.source_name, "L")
            target = _image(root / "train_data_rainfallmap" / "EWB01" / split / pair.target_name, "L")
            if np.max(ra) >= NUM_CLASSES or np.max(target) >= NUM_CLASSES:
                raise ValueError("EWB01 data has a label outside 0..100")
            inputs[index] = merge_modal_arrays(
                MODES, NUM_CLASSES, rd_image=rd, ir_image=ir, ra_image=ra,
                lon_image=longitude, lat_image=latitude)
            targets[index] = target
        loaded[split] = (inputs, targets)
        metadata[split] = {"pairs": len(pairs), "annotations": str(list_path),
                           "annotations_sha256": _sha256(list_path),
                           "first_source": pairs[0].source_name, "last_source": pairs[-1].source_name}
    overflow_names = sorted({name for _, pairs in lists.values() for pair in pairs
                             for name in (pair.source_name, pair.target_name)
                             if name.startswith("20190931")})
    provenance = {"dataset_root": str(root), "splits": metadata,
                  "gi": {"longitude_sha256": _sha256(lon_path), "latitude_sha256": _sha256(lat_path)},
                  "land_mask_sha256": _sha256(mask_path), "land_pixels": int(mask.sum()),
                  "limited_training": limit_train is not None or limit_val is not None,
                  "split_policy": "Original explicit train/val storm lists; no timestamp overlap",
                  "calendar_overflow_filenames": overflow_names,
                  "calendar_overflow_note": "Known original 20190931 filenames retained; interpreted as20191001 only for lead/overlap checks"}
    return TrainingData(*loaded["train"], *loaded["val"], mask, provenance)


def validation_baselines(data: TrainingData) -> dict:
    """Report label-mm errors for zero rainfall and unchanged observed rainfall."""
    target = data.val_targets[:, data.land_mask].astype(np.float64)
    persisted = np.rint(data.val_inputs[..., 4][:, data.land_mask] * NUM_CLASSES)
    return {"zero_land_rmse": float(np.sqrt(np.mean(target ** 2))),
            "persistence_land_rmse": float(np.sqrt(np.mean((persisted - target) ** 2))),
            "target_mean_mm": float(np.mean(target)), "target_maximum_mm": int(np.max(target))}


def sparse_focal_loss(y_true, y_pred):
    """Categorical focal variant: alpha=label+1, gamma=2; return per-pixel loss.

    This avoids allocating full one-hot training labels. It shares the thesis
    focal weighting concept, rather than claiming the identical legacy binary
    one-vs-rest loss implementation.
    """
    import tensorflow as tf
    labels = tf.cast(y_true, tf.int32)
    if labels.shape.rank == y_pred.shape.rank:
        labels = tf.squeeze(labels, axis=-1)
    probabilities = tf.cast(y_pred, tf.float32)
    probabilities = tf.clip_by_value(probabilities, 1e-7, 1.0 - 1e-7)
    chosen = tf.gather(probabilities, labels, axis=-1, batch_dims=3)
    return -(tf.cast(labels, tf.float32) + 1.0) * tf.square(1.0 - chosen) * tf.math.log(chosen)


def train_model(*, dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                output_root=None, epochs=20, batch_size=8, learning_rate=1e-4,
                patience=8, seed=42, threads=4, loss="sparse_focal",
                limit_train=None, limit_val=None, overwrite=False) -> Path:
    """Train real nine-channel Unet3 and save best validation-RMSE weights."""
    from typhoon_rainfall_realtime.runtime import configure_runtime
    configure_runtime()
    import tensorflow as tf
    from nets.unet import Unet3

    if epochs < 1 or batch_size < 1 or patience < 1 or threads < 1:
        raise ValueError("epochs, batch-size, patience and threads must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning-rate must be positive and finite")
    if loss not in ("sparse_focal", "sparse_ce"):
        raise ValueError("Unsupported training loss")
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.exists() and not overwrite:
        raise FileExistsError("Checkpoint already exists; choose another path or explicitly pass --overwrite")
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(min(threads, 2))
    except RuntimeError:
        pass  # A caller may already have initialized TensorFlow.
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(seed)
    data = load_training_data(dataset_root, limit_train=limit_train, limit_val=limit_val)
    output_root = Path(output_root or PROJECT_ROOT / "output" / "realtime-training")
    directory = output_root.resolve() / datetime.now(TAIPEI).strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    contract = {"model": "Unet3", "input_shape": list(INPUT_SHAPE), "num_classes": NUM_CLASSES,
                "lead_hours": 1, "modalities": ["RD", "IR", "RA", "GI"],
                "channel_order": ["RD_R", "RD_G", "RD_B", "IR", "RA", "RA", "RA", "longitude", "latitude"],
                "normalization": {"RD_IR_GI_divisor": 255, "RA_divisor": NUM_CLASSES},
                "label_encoding": "EWB01 integer millimeters, labels 0..100; class100 means >=100mm",
                "GI_encoding": "Original 128x128 lon.png/lat.png grayscale values divided by255",
                "time_basis": "Source filenames are hourly midpoint; target is next hourly midpoint"}
    report = {"status": "training", "created_at": datetime.now(TAIPEI).isoformat(),
              "checkpoint": str(checkpoint), "data_contract": contract, "data": data.provenance,
              "training": {"epochs_requested": epochs, "batch_size": batch_size,
                           "learning_rate": learning_rate, "loss": loss, "seed": seed,
                           "threads": threads, "patience": patience,
                           "mask": "Loss and validation RMSE evaluated on historical mainland mask"},
              "baselines": validation_baselines(data), "history": [],
              "tensorflow_version": tf.__version__,
              "devices": [device.name for device in tf.config.list_physical_devices()]}

    def write_report():
        (directory / "training.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    write_report()
    mask_weights = data.land_mask.astype(np.float32) / np.mean(data.land_mask)
    dataset = tf.data.Dataset.from_tensor_slices((data.train_inputs, data.train_targets))
    dataset = dataset.shuffle(len(data.train_targets), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda x, y: (x, y, mask_weights), num_parallel_calls=1)
    dataset = dataset.batch(batch_size).prefetch(1)
    options = tf.data.Options()
    options.threading.private_threadpool_size = min(threads, 2)
    dataset = dataset.with_options(options)
    model = Unet3(INPUT_SHAPE, NUM_CLASSES)
    report["parameters"] = model.count_params()
    objective = sparse_focal_loss if loss == "sparse_focal" else "sparse_categorical_crossentropy"
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=objective)

    class ValidationCheckpoint(tf.keras.callbacks.Callback):
        best = float("inf")
        wait = 0

        def on_epoch_end(self, epoch, logs=None):
            squared_error = 0.0
            count = 0
            nonzero = 0
            max_label = 0
            for first in range(0, len(data.val_inputs), batch_size):
                inputs = data.val_inputs[first:first + batch_size]
                probabilities = self.model(inputs, training=False).numpy()
                if not np.all(np.isfinite(probabilities)):
                    raise ValueError("Non-finite validation probabilities")
                prediction = probabilities.argmax(axis=-1)[:, data.land_mask]
                truth = data.val_targets[first:first + batch_size, data.land_mask]
                squared_error += float(np.sum((prediction.astype(np.float64) - truth) ** 2))
                count += prediction.size
                nonzero += int(np.count_nonzero(prediction))
                max_label = max(max_label, int(np.max(prediction)))
            rmse = float(np.sqrt(squared_error / count))
            item = {"epoch": epoch + 1, "loss": float(logs.get("loss", 0)),
                    "val_land_rmse": rmse, "val_nonzero_pixels": nonzero,
                    "val_maximum_label": max_label}
            if not np.isfinite(item["loss"]):
                raise ValueError("Non-finite training loss")
            report["history"].append(item)
            print(f"epoch {epoch + 1}: val_land_rmse={rmse:.6f}, "
                  f"nonzero={nonzero}/{count}, maximum={max_label}", flush=True)
            if rmse < self.best:
                self.best, self.wait = rmse, 0
                temporary = checkpoint.with_name(checkpoint.stem + ".pending.h5")
                self.model.save_weights(str(temporary))
                temporary.replace(checkpoint)
                report["best"] = item.copy()
            else:
                self.wait += 1
                if self.wait >= patience:
                    self.model.stop_training = True
            write_report()

    print("Training Unet3: " + str(len(data.train_inputs)) + " train / "
          + str(len(data.val_inputs)) + " held-out validation pairs; 128x128x9 -> 101 classes", flush=True)
    try:
        model.fit(dataset, epochs=epochs,
                  callbacks=[tf.keras.callbacks.TerminateOnNaN(), ValidationCheckpoint()], verbose=2)
        if not checkpoint.is_file() or "best" not in report:
            raise ValueError("Training did not produce a valid checkpoint")
        report.update(status="completed", finished_at=datetime.now(TAIPEI).isoformat(),
                      checkpoint_sha256=_sha256(checkpoint), checkpoint_bytes=checkpoint.stat().st_size)
        report["quality_note"] = "Held-out historical label-RMSE is not validation of current weather accuracy."
        write_report()
        metadata = dict(contract, checkpoint_sha256=report["checkpoint_sha256"],
                        training_report=str((directory / "training.json").resolve()),
                        validation=report["best"], baselines=report["baselines"],
                        source_train_count=len(data.train_inputs), source_val_count=len(data.val_inputs),
                        selection_metric="val_land_rmse", dataset=data.provenance)
        checkpoint.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return directory
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        write_report()
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare real nine-channel Unet3 weights for realtime prediction")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--loss", choices=("sparse_focal", "sparse_ce"), default="sparse_focal")
    parser.add_argument("--limit-train", type=int, help="Optional benchmark subset, recorded in provenance")
    parser.add_argument("--limit-val", type=int, help="Optional benchmark subset, recorded in provenance")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    directory = train_model(**vars(args))
    print(json.dumps({"training_report": str(directory / "training.json"),
                      "checkpoint": str(args.checkpoint.resolve())}, ensure_ascii=False, indent=2))
    return 0
