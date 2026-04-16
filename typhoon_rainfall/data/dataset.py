"""Image loading and batch generation utilities.

這個模組負責把論文資料夾中的 RD/IR/RA/GI 影像轉成模型可吃的
NumPy array。訓練和推論都走同一套融合規則，避免 notebook 與
正式程式對同一筆資料產生不同 channel 排列。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from PIL import Image

from typhoon_rainfall.config import DataConfig
from typhoon_rainfall.data.contracts import DatasetPaths, InputModes, SamplePair


@dataclass(frozen=True)
class LoadedSample:
    """A fully loaded training/evaluation sample."""

    merged_input: np.ndarray
    target_one_hot: np.ndarray
    target_labels: np.ndarray


def load_grayscale(path: Path) -> np.ndarray:
    """Load an image as a float32 grayscale array."""
    return np.array(Image.open(path), dtype=np.float32)


def load_color(path: Path) -> np.ndarray:
    """Load an image as a float32 RGB/RGBA-style array as stored on disk."""
    return np.array(Image.open(path), dtype=np.float32)


def merge_modalities(
    dataset_paths: DatasetPaths,
    split: str,
    sample_name: str,
    input_modes: InputModes,
    num_classes: int,
) -> np.ndarray:
    """Load and merge modalities from dataset paths into one input tensor.

    This is used by dataset-based prediction/training. It reads image files
    from disk, normalizes them, and concatenates channels in the fixed research
    order: RD, IR, RA, then GI.
    """

    channels: List[np.ndarray] = []

    if input_modes.use_rd:
        rd = load_color(dataset_paths.rd_path(split, sample_name)) / 255.0
        channels.append(rd)

    if input_modes.use_ir:
        ir = load_grayscale(dataset_paths.ir_path(split, sample_name)) / 255.0
        if input_modes.use_rd or input_modes.use_ra:
            channels.append(ir[..., np.newaxis])
        else:
            channels.append(np.stack([ir, ir, ir], axis=-1))

    if input_modes.use_ra:
        ra = load_grayscale(dataset_paths.ra_path(split, sample_name)) / float(num_classes)
        channels.append(np.stack([ra, ra, ra], axis=-1))

    if input_modes.use_gi:
        lon = load_grayscale(dataset_paths.lon_path()) / 255.0
        lat = load_grayscale(dataset_paths.lat_path()) / 255.0
        channels.append(lon[..., np.newaxis])
        channels.append(lat[..., np.newaxis])

    normalized_channels: List[np.ndarray] = []
    for channel in channels:
        if channel.ndim == 2:
            normalized_channels.append(channel[..., np.newaxis])
        else:
            normalized_channels.append(channel)
    return np.concatenate(normalized_channels, axis=-1).astype(np.float32)


def merge_modal_arrays(
    input_modes: InputModes,
    num_classes: int,
    rd_image=None,
    ir_image=None,
    ra_image=None,
    lon_image=None,
    lat_image=None,
) -> np.ndarray:
    """Merge already-open images or arrays into one model input tensor.

    This path is used by single-image prediction and the legacy crawler script.
    It intentionally mirrors `merge_modalities` so online and offline inference
    use the same channel rules.
    """

    channels: List[np.ndarray] = []

    if input_modes.use_rd:
        rd = np.array(rd_image, dtype=np.float32) / 255.0
        channels.append(rd)

    if input_modes.use_ir:
        ir = np.array(ir_image, dtype=np.float32) / 255.0
        if input_modes.use_rd or input_modes.use_ra:
            channels.append(ir[..., np.newaxis] if ir.ndim == 2 else ir)
        else:
            if ir.ndim == 2:
                channels.append(np.stack([ir, ir, ir], axis=-1))
            else:
                channels.append(ir)

    if input_modes.use_ra:
        ra = np.array(ra_image, dtype=np.float32) / float(num_classes)
        if ra.ndim == 2:
            channels.append(np.stack([ra, ra, ra], axis=-1))
        else:
            channels.append(ra)

    if input_modes.use_gi:
        lon = np.array(lon_image, dtype=np.float32) / 255.0
        lat = np.array(lat_image, dtype=np.float32) / 255.0
        channels.append(lon[..., np.newaxis] if lon.ndim == 2 else lon)
        channels.append(lat[..., np.newaxis] if lat.ndim == 2 else lat)

    normalized_channels: List[np.ndarray] = []
    for channel in channels:
        if channel.ndim == 2:
            normalized_channels.append(channel[..., np.newaxis])
        else:
            normalized_channels.append(channel)
    return np.concatenate(normalized_channels, axis=-1).astype(np.float32)


def load_target_labels(
    dataset_paths: DatasetPaths,
    split: str,
    filename: str,
    num_classes: int,
) -> np.ndarray:
    """Load a rainfall label image and clip labels into valid class range."""
    labels = load_grayscale(dataset_paths.ra_path(split, filename)).astype(np.int32)
    labels[labels >= num_classes] = num_classes - 1
    return labels


def to_one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert a 2-D label map into a 3-D one-hot segmentation target."""
    one_hot = np.eye(num_classes, dtype=np.float32)[labels.reshape([-1])]
    return one_hot.reshape((labels.shape[0], labels.shape[1], num_classes))


def load_sample(
    dataset_paths: DatasetPaths,
    split: str,
    pair: SamplePair,
    input_modes: InputModes,
    num_classes: int,
) -> LoadedSample:
    """Load one source-target pair for training or validation."""
    merged = merge_modalities(
        dataset_paths=dataset_paths,
        split=split,
        sample_name=pair.source_name,
        input_modes=input_modes,
        num_classes=num_classes,
    )
    labels = load_target_labels(
        dataset_paths=dataset_paths,
        split=split,
        filename=pair.target_name,
        num_classes=num_classes,
    )
    return LoadedSample(
        merged_input=merged,
        target_one_hot=to_one_hot(labels, num_classes=num_classes),
        target_labels=labels,
    )


class MergedDataGenerator:
    """Keras-compatible infinite generator for merged image batches.

    The generator keeps the original research style of yielding NumPy batches
    to `model.fit`. It shuffles only the training split by default and leaves
    validation order stable so evaluation summaries are easier to trace.
    """

    def __init__(
        self,
        batch_size: int,
        pairs: Sequence[SamplePair],
        dataset_paths: DatasetPaths,
        split: str,
        input_modes: InputModes,
        num_classes: int,
        shuffle_each_epoch: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.pairs = list(pairs)
        self.dataset_paths = dataset_paths
        self.split = split
        self.input_modes = input_modes
        self.num_classes = num_classes
        self.shuffle_each_epoch = shuffle_each_epoch

    def iter_batches(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield `(inputs, one_hot_targets)` forever for Keras training."""
        pairs = list(self.pairs)
        while True:
            if self.shuffle_each_epoch:
                random.shuffle(pairs)
            batch_inputs: List[np.ndarray] = []
            batch_targets: List[np.ndarray] = []
            for pair in pairs:
                sample = load_sample(
                    dataset_paths=self.dataset_paths,
                    split=self.split,
                    pair=pair,
                    input_modes=self.input_modes,
                    num_classes=self.num_classes,
                )
                batch_inputs.append(sample.merged_input)
                batch_targets.append(sample.target_one_hot)
                if len(batch_targets) == self.batch_size:
                    yield np.array(batch_inputs), np.array(batch_targets)
                    batch_inputs = []
                    batch_targets = []


def build_dataset_paths(config: DataConfig) -> DatasetPaths:
    """Build a DatasetPaths helper from DataConfig."""
    return DatasetPaths(root=config.dataset_root, interval=config.interval)
