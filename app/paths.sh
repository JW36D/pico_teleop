#!/usr/bin/env bash
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PICO_CONTROL_PYTHON=${PICO_CONTROL_PYTHON:-$PROJECT_DIR/../RM/.venv/bin/python}
export PICO_CONTROL_PYTHON
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_DIR/app${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -x "$PICO_CONTROL_PYTHON" ]]; then
    echo '找不到遥操Python环境，请通过 PICO_CONTROL_PYTHON 指定已安装机械臂SDK的Python。' >&2
    exit 1
fi
