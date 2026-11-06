"""Check trained input/output channels before collecting live observations."""

import hashlib
from pathlib import Path


def checkpoint_channels(path):
    """Read Keras HDF5 kernels in layer order without importing TensorFlow."""
    import h5py

    kernels = []
    with h5py.File(path, "r") as file:
        group = file["model_weights"] if "model_weights" in file else file
        if "layer_names" not in group.attrs:
            raise ValueError("Checkpoint is not a Keras HDF5 weights file")
        for name in group.attrs["layer_names"]:
            name = name.decode() if isinstance(name, bytes) else name
            layer = group[name]
            for weight in layer.attrs.get("weight_names", []):
                weight = weight.decode() if isinstance(weight, bytes) else weight
                tensor = layer[weight]
                if len(tensor.shape) == 4 and "kernel" in weight:
                    kernels.append(tuple(tensor.shape))
    if not kernels or kernels[-1][:2] != (1, 1):
        raise ValueError("Checkpoint has no supported convolutional input/output kernels")
    return {"input_channels": kernels[0][2], "num_classes": kernels[-1][3],
            "convolution_layers": len(kernels), "size_bytes": Path(path).stat().st_size}


def validate_checkpoint(path, input_shape, num_classes):
    details = checkpoint_channels(path)
    if details["input_channels"] != input_shape[-1] or details["num_classes"] != num_classes:
        raise ValueError(
            "Checkpoint contract mismatch: weights accept "
            f"{details['input_channels']} channels / {details['num_classes']} classes, "
            f"configuration requires {input_shape[-1]} channels / {num_classes} classes. "
            "Use matching trained weights; changing modality flags does not convert a model."
        )
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    details["sha256"] = digest.hexdigest()
    return details
