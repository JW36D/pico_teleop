#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"
exec "$PICO_CONTROL_PYTHON" "$PROJECT_DIR/app/raw_dataset.py" "$@"
