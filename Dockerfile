FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libgl1 libglib2.0-0 ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --retries 10 --timeout 120 torch torchvision --index-url ${TORCH_INDEX}

# torchreid's setup.py imports the package to read __version__, so its deps
# must exist before the build and isolation must be off.
RUN pip install --no-cache-dir --retries 10 --timeout 120 numpy Cython setuptools scipy h5py Pillow six \
      opencv-python-headless matplotlib gdown yacs imageio chardet future \
      tabulate tensorboard \
 && pip install --no-cache-dir --retries 10 --timeout 120 --no-build-isolation --no-deps \
      git+https://github.com/KaiyangZhou/deep-person-reid.git

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r /tmp/requirements.txt

ENV YOLO_CONFIG_DIR=/tmp/ultralytics \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app
