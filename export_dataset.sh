#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/app/paths.sh"
FORMAT=
CHECK_ONLY=false
PREVIOUS=
for ARGUMENT in "$@"; do
    if [[ "$PREVIOUS" == --format ]]; then FORMAT="$ARGUMENT"; fi
    if [[ "$ARGUMENT" == --format=* ]]; then FORMAT="${ARGUMENT#--format=}"; fi
    if [[ "$ARGUMENT" == --check-only || "$ARGUMENT" == --help || "$ARGUMENT" == -h ]]; then CHECK_ONLY=true; fi
    PREVIOUS="$ARGUMENT"
done
if $CHECK_ONLY; then
    exec "$PICO_CONTROL_PYTHON" "$PROJECT_DIR/app/export_dataset.py" "$@"
fi
case "$FORMAT" in
    v2.1) EXPORT_PYTHON=${PICO_EXPORT_V21_PYTHON:-} ;;
    v3.0) EXPORT_PYTHON=${PICO_EXPORT_V30_PYTHON:-} ;;
    *) echo '请指定 --format v2.1 或 v3.0' >&2; exit 2 ;;
esac
if [[ -n "$EXPORT_PYTHON" && -x "$EXPORT_PYTHON" ]]; then
    export HF_HUB_OFFLINE=1
    exec "$EXPORT_PYTHON" "$PROJECT_DIR/app/export_dataset.py" "$@"
fi
exec "$PICO_CONTROL_PYTHON" "$PROJECT_DIR/app/run_export_container.py" "$@"
