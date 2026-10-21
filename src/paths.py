from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRETRAINING_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "base"
DEFAULT_FINETUNING_ROOT_DIR = PROJECT_ROOT / "checkpoints_finetuning"


def resolve_project_path(value: str | os.PathLike[str] | None, default: Path) -> Path:
    """Resolve paths identically on Windows, Linux and Docker.

    Relative paths are always anchored at the repository root instead of the
    process working directory. Empty values select the supplied default.
    """
    raw = os.path.expandvars(os.fspath(value or "").strip())
    if not raw:
        return default.resolve()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_pretraining_checkpoint_dir(value: str | os.PathLike[str] | None) -> Path:
    """Resolve CHECKPOINT_DIR; an empty value always means checkpoints/base."""
    raw = os.fspath(value or "").strip()
    if raw:
        return resolve_project_path(raw, DEFAULT_PRETRAINING_CHECKPOINT_DIR)
    return DEFAULT_PRETRAINING_CHECKPOINT_DIR.resolve()


def resolve_finetuning_source_dir(
    value: str | os.PathLike[str] | None,
    pretraining_value: str | os.PathLike[str] | None,
) -> Path:
    """An empty SFT source follows the effective pre-training checkpoint."""
    raw = os.fspath(value or "").strip()
    if raw:
        return resolve_project_path(raw, resolve_pretraining_checkpoint_dir(pretraining_value))
    return resolve_pretraining_checkpoint_dir(pretraining_value)


def resolve_finetuning_root_dir(value: str | os.PathLike[str] | None) -> Path:
    return resolve_project_path(value, DEFAULT_FINETUNING_ROOT_DIR)
