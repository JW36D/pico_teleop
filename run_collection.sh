#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/app/paths.sh"
exec "$PICO_CONTROL_PYTHON" "$PROJECT_DIR/app/collection_tmux.py" "$@"
