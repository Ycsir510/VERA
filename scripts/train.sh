#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERA_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$VERA_ROOT"
export PYTHONPATH="$VERA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
CONFIG="${CONFIG:-$VERA_ROOT/config/wikimel.yaml}"
PYTHON="${PYTHON:-python}"
exec "$PYTHON" -u "$VERA_ROOT/codes/main.py" --config "$CONFIG" "$@"
