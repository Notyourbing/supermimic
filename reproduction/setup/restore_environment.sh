#!/usr/bin/env bash
set -euo pipefail
prep_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$prep_dir/bootstrap.sh" --offline
