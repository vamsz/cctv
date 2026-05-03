"""Entry point for the CCTV inference pipeline.

Usage:
    python scripts/run_pipeline.py
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.runner import PipelineOrchestrator  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    PipelineOrchestrator().run()
