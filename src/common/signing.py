"""Evidence integrity layer.

Two guarantees:

1. Per-record Ed25519 signature
   For each violation we sign a canonical JSON of its identifying
   fields PLUS the SHA-256 of the captured frame bytes. The signature
   is stored alongside the row. Verifying a record only needs the public
   key, so the public key can be distributed to courts/auditors.

2. Hash chain across records
   Each row stores `chain_hash = SHA-256(prev_chain_hash || sha256)`.
   Any tampering with a past record breaks the chain at that point;
   the verify-chain CLI will pinpoint the first divergence.

The signing key is loaded once at process start. Generate with:

    openssl genpkey -algorithm Ed25519 -out evidence_signing.key
    openssl pkey -in evidence_signing.key -pubout -out evidence_signing.pub
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from config.settings import settings


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=1)
def _private_key() -> Ed25519PrivateKey:
    path = settings.evidence_signing_key_path
    if not path.exists():
        # Auto-generate in dev so the system runs end-to-end OOTB.
        if settings.is_production:
            raise RuntimeError(f"Evidence signing key not found: {path}")
        key = Ed25519PrivateKey.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        pub_path = settings.evidence_signing_pubkey_path
        pub_path.parent.mkdir(parents=True, exist_ok=True)
        pub_path.write_bytes(
            key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        return key
    pem = path.read_bytes()
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise RuntimeError("Signing key is not Ed25519")
    return key


@lru_cache(maxsize=1)
def public_key() -> Ed25519PublicKey:
    return _private_key().public_key()


def canonical_json(payload: dict) -> bytes:
    """Deterministic JSON encoding for signature stability."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign(payload: dict, sha256_hex: str) -> bytes:
    msg = canonical_json({"payload": payload, "sha256": sha256_hex})
    return _private_key().sign(msg)


def verify(payload: dict, sha256_hex: str, signature: bytes) -> bool:
    try:
        msg = canonical_json({"payload": payload, "sha256": sha256_hex})
        public_key().verify(signature, msg)
        return True
    except Exception:
        return False


def chain_hash(prev_chain_hash: str | None, sha256_hex: str) -> str:
    h = hashlib.sha256()
    h.update((prev_chain_hash or "").encode())
    h.update(sha256_hex.encode())
    return h.hexdigest()
