"""Enable an optional project-local CUDA runtime before importing TensorFlow."""

from __future__ import annotations

import os
from pathlib import Path


CUDA_DIRECTORY = Path(__file__).resolve().parents[1] / ".cache" / "realtime-gpu" / "bin"
_DLL_HANDLES = {}


def configure_runtime():
    """Register local CUDA DLLs for this process, or leave CPU/system setup alone.

    Call this before the first TensorFlow import. Directory handles remain alive
    for the process lifetime, and repeated calls do not duplicate PATH entries.
    There are no machine/user environment changes and no TensorFlow import here.
    Actual GPU availability is determined by TensorFlow after initialization.
    """
    directory = CUDA_DIRECTORY.resolve()
    if os.name != "nt" or not directory.is_dir():
        return {"local_cuda_registered": False, "dll_directory": None}
    key = str(directory)
    if key not in _DLL_HANDLES:
        _DLL_HANDLES[key] = os.add_dll_directory(key)
    current = os.environ.get("PATH", "")
    entries = {os.path.normcase(item) for item in current.split(os.pathsep)}
    if os.path.normcase(key) not in entries:
        os.environ["PATH"] = key + os.pathsep + current
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    return {"local_cuda_registered": True, "dll_directory": key}
