"""Portable paths for the local reproduction tools and saved run metadata.

Stored paths are relative to the repository, independent of the caller's cwd.
The logical ``reproduction/data`` prefix may be relocated with an environment
variable. Absolute paths supplied explicitly by a caller are still supported.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PREFIX = Path("reproduction/data")


def repo_path(value: str | Path, *, root: Path = REPO_ROOT) -> Path:
  """Expand a path against the checkout without dereferencing model symlinks."""
  expanded = Path(os.path.expandvars(os.fspath(value))).expanduser()
  return Path(os.path.abspath(root / expanded))


def data_root(*, root: Path = REPO_ROOT) -> Path:
  return repo_path(os.environ.get("SUPERMIMIC_DATA_DIR", str(DATA_PREFIX)), root=root)


def resolve_motion_path(value: str | Path, *, root: Path = REPO_ROOT) -> Path:
  expanded = Path(os.path.expandvars(os.fspath(value))).expanduser()
  if not expanded.is_absolute() and expanded.is_relative_to(DATA_PREFIX):
    return repo_path(
      data_root(root=root) / expanded.relative_to(DATA_PREFIX), root=root
    )
  return repo_path(expanded, root=root)


def portable_path(value: str | Path, *, root: Path = REPO_ROOT) -> str:
  """Serialize external data through its logical prefix, other paths relatively."""
  path = repo_path(value, root=root)
  storage = data_root(root=root)
  if path.is_relative_to(storage):
    return (DATA_PREFIX / path.relative_to(storage)).as_posix()
  return Path(os.path.relpath(path, root)).as_posix()


def portable_tree(value: Any, *, root: Path = REPO_ROOT) -> Any:
  """Convert path values when saving JSON/configs, without mutating runtime state."""
  if isinstance(value, Path):
    return portable_path(value, root=root)
  if isinstance(value, str) and Path(value).is_absolute():
    return portable_path(value, root=root)
  if isinstance(value, dict):
    return {k: portable_tree(v, root=root) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [portable_tree(v, root=root) for v in value]
  return value


def legacy_motion_reference(value: str | Path, *, root: Path = REPO_ROOT) -> str:
  """Rebase known data layouts in old checkpoints; retain the full dataset suffix.

  This does not compare only basenames: filtered and unfiltered data remain
  distinct. Old binary checkpoints can be loaded without rewriting their weights.
  """
  path = Path(os.path.expandvars(os.fspath(value))).expanduser()
  if path.is_absolute():
    # Prefer the configured storage location before recognizing historical layouts.
    storage = data_root(root=root)
    if path.is_relative_to(storage):
      return portable_path(path, root=root)
    parts = path.parts
    for i in range(len(parts) - 1):
      if parts[i : i + 2] in (
        ("official_reproduction", "data"),
        ("reproduction", "data"),
      ):
        return (DATA_PREFIX / Path(*parts[i + 2 :])).as_posix()
  return portable_path(resolve_motion_path(path, root=root), root=root)


def same_motion_source(
  old: str | Path, new: str | Path, *, root: Path = REPO_ROOT
) -> bool:
  return legacy_motion_reference(old, root=root) == legacy_motion_reference(
    new, root=root
  )
