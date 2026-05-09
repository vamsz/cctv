"""Combined launcher — pipeline + API in one process.

Runs detection pipeline threads and the FastAPI review console in the same
Python process so they share _live_frames and _crowd_state without IPC.

Usage:
    python scripts/run_all.py

Then open http://localhost:8000 in a browser.
Login: admin@local / admin
"""
import logging
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("run_all")

from config.settings import settings  # noqa: E402
from src.pipeline.runner import PipelineOrchestrator  # noqa: E402
import uvicorn  # noqa: E402

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("CCTV Enforcement System starting up")
    log.info("Device: %s", settings.device)
    log.info("=" * 60)

    orch = PipelineOrchestrator()
    t = threading.Thread(target=orch.run, daemon=True, name="pipeline")
    t.start()

    # Give the pipeline 3 s to initialise models and open video streams
    # before the API starts accepting requests. This avoids "no frames yet"
    # errors on first page load.
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
        log_level="warning",   # suppress per-request noise; pipeline logs are enough
        access_log=False,
    )
