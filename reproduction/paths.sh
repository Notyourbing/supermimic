#!/usr/bin/env bash
# Source this file: all configured relative paths use the repository as base.
prep_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname -- "$prep_dir")"
if [[ -f "$prep_dir/local_paths.sh" ]]; then
  source "$prep_dir/local_paths.sh"
fi
prefix="${SUPERMIMIC_ENV_PREFIX:-${HOME}/miniconda3/envs/supermimic-official}"
dependency="${SUPERMIMIC_MJLAB_DIR:-../mjlab_research}"
data_dir="${SUPERMIMIC_DATA_DIR:-reproduction/data}"
[[ "$prefix" = /* ]] || prefix="$repo_dir/$prefix"
[[ "$dependency" = /* ]] || dependency="$repo_dir/$dependency"
[[ "$data_dir" = /* ]] || data_dir="$repo_dir/$data_dir"
export SUPERMIMIC_ENV_PREFIX="$prefix"
export SUPERMIMIC_MJLAB_DIR="$dependency"
export SUPERMIMIC_DATA_DIR="$data_dir"
