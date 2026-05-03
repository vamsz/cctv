# NVIDIA CUDA base — required for GPU inference. Use nvidia-container-toolkit
# on the host. For CPU-only deployment, swap base for python:3.11-slim and
# install paddlepaddle (CPU build) instead of paddlepaddle-gpu.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Default command runs the pipeline. The compose file overrides this for
# the API service.
CMD ["python", "scripts/run_pipeline.py"]
