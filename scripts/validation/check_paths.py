"""Audit versioned paths without training, downloading, or allocating a GPU."""

import argparse
import ast
import json
import os
import re
import subprocess
from pathlib import Path

from supermimic.utils.reproduction_paths import REPO_ROOT, resolve_motion_path


def absolute_values(value, key=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from absolute_values(v, f"{key}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from absolute_values(v, f"{key}[{i}]")
    elif isinstance(value, str) and (
        Path(value).is_absolute() or value.startswith("file:///")
    ):
        yield key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    files = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, text=True
    ).split("\0")
    errors = []
    for name in filter(None, files):
        path = REPO_ROOT / name
        if path.is_symlink():
            if os.path.isabs(os.readlink(path)):
                errors.append(f"{name}: absolute symlink")
            continue
        if path.suffix == ".json":
            for key in absolute_values(json.loads(path.read_text())):
                errors.append(f"{name}{key}: absolute serialized path")
        if path.suffix == ".py":
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if re.match(
                        r"^/(?:home|root|Users|mnt|workspace|scratch|opt)/", node.value
                    ):
                        errors.append(
                            f"{name}:{node.lineno}: host-specific path literal"
                        )
        if path.suffix in {".md", ".sh", ".txt", ".toml", ".yml", ".yaml"}:
            if re.search(
                r"(?<![\w:])/(?:home|root|Users)/[^\s`\"']+", path.read_text()
            ):
                errors.append(f"{name}: host-specific text path")
    if args.check_data:
        for name in ("lafan1", "bones_unfiltered"):
            config = json.loads(
                (
                    REPO_ROOT / "reproduction/experiments" / name / "config.json"
                ).read_text()
            )
            if not resolve_motion_path(config["args"]["motion_file"]).exists():
                errors.append(
                    f"{name}: data not found; prepare data or set SUPERMIMIC_DATA_DIR"
                )
    if errors:
        raise SystemExit("\n".join(errors))
    print("PASS: relative stored paths, portable scripts and relative symlinks.")
    if args.check_data:
        print("PASS: current experiment data resolves on this server.")


if __name__ == "__main__":
    main()
