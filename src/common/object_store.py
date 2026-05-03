"""Pluggable object store: local filesystem or S3-compatible.

Production deployments should use S3 (or MinIO on-prem) so:
  - the inference host is stateless and disposable
  - evidence survives node failure
  - retention/archival can be policy-driven on the bucket

The local backend is for dev and air-gapped sites.
"""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path

from config.settings import settings


class ObjectStore(ABC):
    @abstractmethod
    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...


class LocalObjectStore(ObjectStore):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _abs(self, key: str) -> Path:
        full = (self.root / key).resolve()
        # path traversal guard
        if not str(full).startswith(str(self.root.resolve())):
            raise ValueError(f"path traversal: {key}")
        return full

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        full = self._abs(key)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)
        return key

    def get_bytes(self, key: str) -> bytes:
        return self._abs(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._abs(key).exists()

    def delete(self, key: str) -> None:
        full = self._abs(key)
        if full.exists():
            full.unlink()


class S3ObjectStore(ObjectStore):
    def __init__(self, bucket: str, region: str, prefix: str = ""):
        import boto3  # local import so dev installs without boto3 issues
        self.bucket = bucket
        self.prefix = prefix.lstrip("/")
        self.client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=settings.aws_access_key_id.get_secret_value() or None,
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() or None,
        )

    def _full(self, key: str) -> str:
        return f"{self.prefix}{key}".lstrip("/")

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._full(key),
            Body=data,
            ContentType=content_type,
            ServerSideEncryption="AES256",
        )
        return key

    def get_bytes(self, key: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=self._full(key))
        return obj["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._full(key))
            return True
        except Exception:
            return False

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._full(key))


def get_object_store() -> ObjectStore:
    if settings.evidence_backend == "s3":
        if not settings.s3_bucket:
            raise RuntimeError("EVIDENCE_BACKEND=s3 but S3_BUCKET is empty")
        return S3ObjectStore(settings.s3_bucket, settings.s3_region, settings.s3_prefix)
    return LocalObjectStore(settings.evidence_dir)
