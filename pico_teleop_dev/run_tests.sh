#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"
exec "$PICO_CONTROL_PYTHON" -m unittest discover -s "$DEV_DIR/tests" "$@"
