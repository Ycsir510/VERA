#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERA_ROOT="$(cd -- "$SCRIPT_DIR" && pwd)"
DATASET="${DATASET:-wikimel}"
case "$DATASET" in
  wikimel) CONFIG="${CONFIG:-$VERA_ROOT/config/wikimel.yaml}" ;;
  wikidiverse) CONFIG="${CONFIG:-$VERA_ROOT/config/wikidiverse.yaml}" ;;
  *) echo "DATASET must be wikimel or wikidiverse" >&2; exit 2 ;;
esac
exec env CONFIG="$CONFIG" "$VERA_ROOT/scripts/train.sh" "$@"
