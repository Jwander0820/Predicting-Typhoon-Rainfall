"""Checkpoint selection rules.

The project historically saved weights as `{model}_val_min_RMSE.h5`.
These helpers make that convention explicit and provide a deterministic
fallback when the preferred file is not present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


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
        if not explicit_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {explicit_path}")
        return explicit_path

    preferred = preferred_checkpoint_path(checkpoint_dir, model_name)
    if preferred.exists():
        return preferred

    candidates = sorted(checkpoint_dir.glob("*.h5"), key=lambda path: (path.stat().st_mtime, path.name))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found under: {checkpoint_dir}")
    return candidates[-1]
