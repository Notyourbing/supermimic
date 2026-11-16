#!/usr/bin/env bash
set -euo pipefail
prep_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run=(bash "$prep_dir/run_env.sh")
case "${1:-}" in
  lafan)
    "${run[@]}" python "$prep_dir/../scripts/data/download_lafan.py"
    CUDA_VISIBLE_DEVICES='' "${run[@]}" python "$prep_dir/../scripts/data/convert_lafan.py"
    CUDA_VISIBLE_DEVICES='' "${run[@]}" python -m mjlab.scripts.pack_motions \
      --input "$prep_dir/data/lafan1/npz/*.npz" --output "$prep_dir/data/lafan1/packed" --workers 8
    ;;
  bones)
    "${run[@]}" python "$prep_dir/../scripts/data/check_bones_access.py"
    "${run[@]}" python "$prep_dir/../scripts/data/download_bones_ranges.py"
    CUDA_VISIBLE_DEVICES='' "${run[@]}" python "$prep_dir/../scripts/data/convert_bones_stream.py" --workers "${BONES_WORKERS:-24}"
    CUDA_VISIBLE_DEVICES='' "${run[@]}" python "$prep_dir/../scripts/data/finalize_bones.py"
    ;;
  *) echo 'Usage: bash reproduction/prepare_data.sh lafan|bones' >&2; exit 2 ;;
esac
