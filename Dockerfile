# PRAHARI — GPU-enabled CCTV enforcement platform
# Requires nvidia-container-toolkit on the host for GPU pass-through.
# For CPU-only: swap base image for python:3.11-slim (remove CUDA deps).
#
# Build:  docker build -t prahari:latest .
# Dev:    docker compose up
# Prod:   docker compose -f docker-compose.yml -f docker-compose.prod.yml up

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Cache pip layer separately from source code
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Runtime directories (overridden by compose volumes in production)
RUN mkdir -p data/evidence data/samples models

# Ports: 8000 = API/dashboard, 9090 = Prometheus metrics
EXPOSE 8000 9090

# Liveness probe for docker compose healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["python", "scripts/run_pipeline.py"]
