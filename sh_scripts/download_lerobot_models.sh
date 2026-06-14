#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mlic/mingukang/lerobot}"
cd "$REPO_ROOT"
source .venv/bin/activate

python scripts/download_lerobot_models.py "$@"
