"""Checkpoint selection rules.

The project historically saved weights as `{model}_val_min_RMSE.h5`.
These helpers use that named file or an explicit path. Missing weights never
fall back to an unrelated model just because its file was modified recently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def preferred_checkpoint_path(checkpoint_dir: Path, model_name: str) -> Path:
    """Return the conventional best-checkpoint filename for a model."""
    return checkpoint_dir / f"{model_name}_val_min_RMSE.h5"


def resolve_checkpoint(
    checkpoint_dir: Path,
    model_name: str,
    explicit_path: Optional[Path] = None,
) -> Path:
    """Resolve which checkpoint should be loaded for prediction/evaluation."""
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {explicit_path}")
        return explicit_path

    preferred = preferred_checkpoint_path(checkpoint_dir, model_name)
    if preferred.is_file():
        return preferred
    raise FileNotFoundError(
        f"Checkpoint for {model_name} not found: {preferred}. "
        "Supply an explicit checkpoint trained for this model and input configuration."
    )
