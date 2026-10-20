#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"
RECEIVER_STATE=$("$PICO_CONTROL_PYTHON" -c 'from runtime_paths import RECEIVER_STATE_DIR; print(RECEIVER_STATE_DIR)')
mkdir -p "$RECEIVER_STATE"
if sudo -n docker container inspect rm-pico-xr >/dev/null 2>&1; then
    MOUNT_SOURCE=$(sudo -n docker inspect --format '{{range .Mounts}}{{if eq .Destination "/state"}}{{.Source}}{{end}}{{end}}' rm-pico-xr)
    if [[ "$MOUNT_SOURCE" != "$RECEIVER_STATE" ]]; then
        echo "接收服务目录不匹配，请先核对配置：$MOUNT_SOURCE；当前配置：$RECEIVER_STATE" >&2
        exit 1
    fi
    exec sudo -n docker start rm-pico-xr
fi
exec sudo -n docker run -d --name rm-pico-xr --network host \
    --restart unless-stopped --cap-drop ALL --security-opt no-new-privileges \
    --log-opt max-size=10m --log-opt max-file=3 \
    --mount "type=bind,src=$RECEIVER_STATE,dst=/state" \
    "${PICO_RECEIVER_IMAGE:-rm-pico-xr:collection-v1}"
