#!/usr/bin/env bash
DEV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$DEV_DIR/../app/paths.sh"
export PYTHONPATH="$PROJECT_DIR/app:$DEV_DIR:$DEV_DIR/tools/maintenance:$DEV_DIR/tools/validation:$DEV_DIR/environment/receiver${PYTHONPATH:+:$PYTHONPATH}"
