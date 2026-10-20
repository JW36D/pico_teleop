#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"
exec "$PICO_CONTROL_PYTHON" "$DEV_DIR/tools/maintenance/dual_teleop.py" "$@"
