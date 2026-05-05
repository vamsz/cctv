"""Centralised settings, loaded from environment / .env.

Production posture:
  - Fail loud on missing critical secrets in production environment.
  - Coerce paths, validate ranges, expose computed properties.
  - Do NOT print secrets in __repr__.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # environment
    environment: Literal["development", "staging", "production"] = "development"
    deployment_id: str = "local"
    timezone: str = "Asia/Kolkata"

    # storage
    evidence_dir: Path = Field(default=ROOT / "data" / "evidence")
    evidence_backend: Literal["local", "s3"] = "local"
    s3_bucket: str = ""
    s3_region: str = "ap-south-1"
    s3_prefix: str = "evidence/"
    aws_access_key_id: SecretStr = SecretStr("")
    aws_secret_access_key: SecretStr = SecretStr("")

    # database
    database_url: str = Field(default=f"sqlite:///{ROOT / 'data' / 'cctv.db'}")
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # redis
    redis_url: str = "redis://localhost:6379/0"

    # weights
    detector_weights: Path = Field(default=ROOT / "models" / "yolo11n.pt")
    helmet_weights: Path = Field(default=ROOT / "models" / "helmet.pt")
    plate_weights: Path = Field(default=ROOT / "models" / "plate.pt")
    ocr_lang: str = "en"

    # inference — auto-detects CUDA if not set in env
    device: str = ""
    max_fps: int = 15
    det_conf: float = 0.35
    ocr_auto_accept: float = 0.80

    # auth
    jwt_secret: SecretStr = SecretStr("dev-only-not-for-production")
    jwt_issuer: str = "cctv-enforcement"
    jwt_audience: str = "cctv-operators"
    jwt_ttl_minutes: int = 480
    bootstrap_admin_email: str = "admin@local"
    bootstrap_admin_password: SecretStr = SecretStr("admin")

    # evidence integrity
    evidence_signing_key_path: Path = Field(default=ROOT / "data" / "evidence_signing.key")
    evidence_signing_pubkey_path: Path = Field(default=ROOT / "data" / "evidence_signing.pub")

    # observability
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"
    sentry_dsn: str = ""
    prometheus_port: int = 9090
    metrics_auth_token: SecretStr = SecretStr("")

    # api
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    allowed_origins: str = "*"
    session_cookie_secure: bool = True
    rate_limit_per_minute: int = 120

    # retention
    evidence_retention_days: int = 180
    archive_s3_bucket: str = ""

    # reid
    reid_enabled: bool = True
    reid_threshold: float = 0.85         # cosine similarity threshold for cross-camera match
    reid_max_age_seconds: float = 300.0  # how long to keep embeddings in memory
    reid_extract_every_n: int = 15       # extract embedding every N frames per track

    # plate detection
    sahi_plate_enabled: bool = False     # sliced plate inference (adds ~10ms per frame)

    # chalana
    chalana_api_url: str = ""
    chalana_api_key: SecretStr = SecretStr("")
    chalana_push_enabled: bool = False

    # alerting
    alert_webhook_url: str = ""
    alert_camera_offline_seconds: int = 120

    # config files
    cameras_yaml: Path = Field(default=ROOT / "config" / "cameras.yaml")
    rules_yaml: Path = Field(default=ROOT / "config" / "rules.yaml")

    # ------------------------------------------------------------------
    @field_validator("evidence_dir", "evidence_signing_key_path", "evidence_signing_pubkey_path", mode="before")
    @classmethod
    def _expand(cls, v):
        return Path(v).expanduser() if v else v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def origins_list(self) -> list[str]:
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    def assert_production_ready(self) -> None:
        """Hard-fail at startup if production secrets aren't real."""
        if not self.is_production:
            return
        problems = []
        if self.jwt_secret.get_secret_value() in ("", "dev-only-not-for-production") or len(self.jwt_secret.get_secret_value()) < 32:
            problems.append("JWT_SECRET must be at least 32 chars in production")
        if self.bootstrap_admin_password.get_secret_value() in ("", "admin", "CHANGE_ME_STRONG"):
            problems.append("BOOTSTRAP_ADMIN_PASSWORD must be changed in production")
        if self.database_url.startswith("sqlite"):
            problems.append("Production must use Postgres, not SQLite")
        if self.evidence_backend == "s3" and not self.s3_bucket:
            problems.append("EVIDENCE_BACKEND=s3 but S3_BUCKET is empty")
        if not self.evidence_signing_key_path.exists():
            problems.append(f"Evidence signing key missing: {self.evidence_signing_key_path}")
        if problems:
            raise RuntimeError("Production preflight failed:\n  - " + "\n  - ".join(problems))


def _auto_device(requested: str) -> str:
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


settings = Settings()
settings.device = _auto_device(settings.device)

# Best-effort path setup for local/dev; production paths are owned by the OS package.
for p in (settings.evidence_dir, settings.evidence_signing_key_path.parent):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        pass
