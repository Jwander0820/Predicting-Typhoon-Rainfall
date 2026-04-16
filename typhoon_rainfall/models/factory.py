"""Factory for supported neural network architectures."""

from __future__ import annotations

from typing import Sequence

from nets.FCN8 import FCN8
from nets.pspnet import pspnet
from nets.unet import Unet3, Unet4, Unet5, Unet6


def build_model(name: str, input_shape: Sequence[int], num_classes: int):
    """Create a model by its research name.

    Keeping model selection in one function makes train/evaluate/predict share
    the same architecture lookup. The names intentionally match historical
    notebook values such as `Unet3` and `Unet5`.
    """

    if name == "Unet6":
        return Unet6(tuple(input_shape), num_classes)
    if name == "Unet5":
        return Unet5(tuple(input_shape), num_classes)
    if name == "Unet4":
        return Unet4(tuple(input_shape), num_classes)
    if name == "Unet3":
        return Unet3(tuple(input_shape), num_classes)
    if name == "FCN8":
        return FCN8(tuple(input_shape), num_classes)
    if name == "pspnet":
        return pspnet(tuple(input_shape), num_classes)
    raise ValueError(f"Unsupported model name: {name}")
