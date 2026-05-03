"""Entry point for the FastAPI review console."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from config.settings import settings  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "src.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="info",
    )
