"""Combined launcher — pipeline + API in one process.

Runs detection pipeline threads and the FastAPI review console in the same
Python process so they share _live_frames and _crowd_state without IPC.

Usage:
    python scripts/run_all.py

Then open http://localhost:8000 in a browser.
Login: admin@local / admin
"""
import os

# ── Performance + log-noise environment (must be set BEFORE any import that
# triggers numpy / torch / onnxruntime / opencv) ──────────────────────────
_cpu_count = os.cpu_count() or 8

# Use ALL physical+logical cores for every BLAS / OpenMP / OpenCV worker.
# i9-12700H = 14 cores / 20 threads; we set 20.
os.environ.setdefault("OMP_NUM_THREADS",          str(_cpu_count))
os.environ.setdefault("MKL_NUM_THREADS",          str(_cpu_count))
os.environ.setdefault("OPENBLAS_NUM_THREADS",     str(_cpu_count))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS",   str(_cpu_count))
os.environ.setdefault("NUMEXPR_NUM_THREADS",      str(_cpu_count))
# torch's intra-op pool — runner overrides this further but the env var
# affects threads created BEFORE the runner can set them.
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

# Suppress the spammy TensorRT-DLL-missing warnings. We don't have TensorRT
# installed and don't need it — onnxruntime falls back to CUDA cleanly.
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL",   "3")    # 0=verbose ... 3=error
os.environ.setdefault("ORT_DISABLE_TENSORRT",     "1")    # honored by some EPs
# Paddle / PaddleOCR boot warnings (ccache / oneDNN banner)
os.environ.setdefault("FLAGS_call_stack_level",   "1")
os.environ.setdefault("GLOG_minloglevel",         "2")
# Hugging Face symlink warning
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import logging
import sys
import threading
import time
import warnings
from pathlib import Path

# Hide UserWarnings from transformers/paddleocr that we can't fix
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=UserWarning, module="paddle.*")
warnings.filterwarnings("ignore", message=".*ccache.*")
warnings.filterwarnings("ignore", message=".*Could not find files.*")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Silence onnxruntime's Python-side logger
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
log = logging.getLogger("run_all")

# Configure OpenCV to use every core. Must come AFTER the env var setup
# so the OpenCV worker pool sizes itself correctly.
import cv2  # noqa: E402
cv2.setNumThreads(_cpu_count)
cv2.setUseOptimized(True)
log.info("OpenCV threads: %d (optimised=%s)", cv2.getNumThreads(), cv2.useOptimized())

from config.settings import settings  # noqa: E402
from src.pipeline.runner import PipelineOrchestrator  # noqa: E402
import uvicorn  # noqa: E402

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("CCTV Enforcement System starting up")
    log.info("Device: %s | CPU cores: %d", settings.device, _cpu_count)
    log.info("=" * 60)

    orch = PipelineOrchestrator()
    t = threading.Thread(target=orch.run, daemon=True, name="pipeline")
    t.start()

    log.info("Pipeline thread started — waiting 3 s for models to warm up ...")
    time.sleep(3)

    log.info("Opening review console on http://localhost:%d", settings.api_port)
    log.info("Login: %s / %s", settings.bootstrap_admin_email,
             settings.bootstrap_admin_password.get_secret_value())

    uvicorn.run(
        "src.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="warning",
        access_log=False,
        # Use uvloop equivalent + h11 with bigger buffers for WS bursts.
        ws_max_queue=128,
        timeout_keep_alive=30,
    )
