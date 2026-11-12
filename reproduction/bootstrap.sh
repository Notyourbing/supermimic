#!/usr/bin/env bash
# Install only; never launches training or downloads gated motion data.
set -euo pipefail
prep_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname -- "$prep_dir")"
prefix="${SUPERMIMIC_ENV_PREFIX:-${HOME}/miniconda3/envs/supermimic-official}"
dependency="${SUPERMIMIC_MJLAB_DIR:-$(dirname -- "$repo_dir")/mjlab_research}"
revision=dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd
offline=()
conda_flags=()
if [[ -n "${1:-}" && "${1:-}" != --offline ]]; then echo 'Usage: bootstrap.sh [--offline]' >&2; exit 2; fi
if [[ "${1:-}" == --offline ]]; then offline=(--offline --no-index); fi
if (( ${#offline[@]} )); then conda_flags=(--offline); fi
if [[ ! -x "$prefix/bin/python" ]]; then
  conda_bin="${CONDA_EXE:-$(command -v conda || true)}"
  if [[ -z "$conda_bin" && -x "$HOME/miniconda3/bin/conda" ]]; then conda_bin="$HOME/miniconda3/bin/conda"; fi
  if [[ -z "$conda_bin" ]]; then
    echo 'Install Miniconda first: https://www.anaconda.com/docs/getting-started/miniconda/install' >&2
    exit 1
  fi
  "$conda_bin" create -p "$prefix" "${conda_flags[@]}" --override-channels -c conda-forge python=3.13.18 uv=0.13.7 pip -y
fi
if [[ ! -x "$prefix/bin/uv" ]]; then echo "uv missing in $prefix; install uv=0.13.7 with conda." >&2; exit 1; fi
"$prefix/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,13), "Python 3.13 required"'
if [[ ! -d "$dependency" ]]; then
  if (( ${#offline[@]} )); then echo 'Offline mode requires an existing pinned mjlab_research checkout.' >&2; exit 1; fi
  git clone https://github.com/HybridRobotics/mjlab_research.git "$dependency"
  git -C "$dependency" checkout --detach "$revision"
fi
if [[ "$(git -C "$dependency" rev-parse HEAD)" != "$revision" ]]; then
  echo "Expected mjlab_research $revision at $dependency. Existing checkout preserved." >&2
  exit 1
fi
links=()
if [[ -d "$prep_dir/setup/wheels" ]]; then links=(--find-links "$prep_dir/setup/wheels"); fi
"$prefix/bin/uv" pip install --python "$prefix/bin/python" "${offline[@]}" "${links[@]}" \
  -r "$prep_dir/setup/resolved_constraints.txt"
# Both distributions install into the mujoco namespace. Only the 3.16 package is used.
"$prefix/bin/uv" pip uninstall --python "$prefix/bin/python" mujoco-core
"$prefix/bin/uv" pip install --python "$prefix/bin/python" "${offline[@]}" "${links[@]}" \
  --reinstall-package mujoco --no-sources -c "$prep_dir/setup/resolved_constraints.txt" \
  -e "$dependency" -e "$repo_dir" mujoco==3.16.0
"$prefix/bin/uv" pip check --python "$prefix/bin/python"
echo 'Environment ready. See reproduction/README.md for validation and data preparation. No training started.'
