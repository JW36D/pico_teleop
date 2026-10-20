# Independent CPU export environment. No robot or USB devices are mounted.
FROM rm-pico-xr:1.0.0-arm64 AS dependencies
USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python3-venv python3-dev git ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/export \
    && /opt/export/bin/pip install --no-cache-dir 'pip==25.0.1' 'setuptools==75.8.2' \
    && /opt/export/bin/pip install --no-cache-dir 'torch==2.7.1' 'torchvision==0.22.1' 'numpy==1.26.4'
FROM dependencies
ARG LEROBOT_COMMIT
ARG DATASET_FORMAT=v3.0
COPY pico_teleop_dev/environment/export/ /opt/export-locks/
COPY pico_teleop_dev/environment/export/vendor/lerobot-${DATASET_FORMAT}.tar.gz /opt/lerobot-source.tar.gz
RUN test -n "$LEROBOT_COMMIT" \
    && /opt/export/bin/python -c "import json,hashlib; p=json.load(open('/opt/export-locks/source-pins.json'))['${DATASET_FORMAT}']; assert p['commit']=='${LEROBOT_COMMIT}'; assert hashlib.sha256(open('/opt/lerobot-source.tar.gz','rb').read()).hexdigest()==p['sha256']" \
    && if test -f "/opt/export-locks/${DATASET_FORMAT}-resolved.txt"; then \
         /opt/export/bin/pip install --no-cache-dir --no-deps -r "/opt/export-locks/${DATASET_FORMAT}-resolved.txt"; \
       else \
         /opt/export/bin/pip install --no-cache-dir -c "/opt/export-locks/${DATASET_FORMAT}.txt" datasets av huggingface-hub 'jsonlines==4.0.0' 'packaging==24.2' 'draccus==0.10.0' 'accelerate==1.10.1' opencv-python-headless \
         && /opt/export/bin/pip install --no-cache-dir --no-deps /opt/lerobot-source.tar.gz; \
       fi \
    && /opt/export/bin/pip freeze > /opt/export/requirements-resolved.txt
ENV HF_HUB_OFFLINE=1
ENV PYNPUT_BACKEND=dummy
ENTRYPOINT ["/opt/export/bin/python", "/workspace/app/export_dataset.py"]
WORKDIR /workspace
