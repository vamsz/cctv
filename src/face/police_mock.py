"""Mock police records database — drop-in for a real face-search provider.

API:
    db = PoliceRecordsMock(records_csv, embed_dim=384, threshold=0.55)
    match = db.search(embedding)        # → MatchResult | None
    record = db.get(record_id)          # → dict | None
    all_records = db.list_all()

The mock seeds a CSV at data/police_records_mock.csv on first run with
a small set of fake suspects (Indian names + plausible charges). For
each record it generates a *deterministic* random 384-d unit
embedding (seeded by record_id) so the matcher returns stable results
across restarts.

Replace with a real provider by implementing the same three methods
(search, get, list_all) — the rest of the system never imports this
module directly; it goes through ``policedb_for(device)`` factory.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("face.police_mock")


@dataclass
class PoliceRecord:
    record_id: str
    name: str
    age: int
    gender: str
    charges: str
    risk_level: str          # low | medium | high | critical
    last_known_address: str
    notes: str = ""


@dataclass
class MatchResult:
    record: PoliceRecord
    similarity: float


# ---------------------------------------------------------------------------
# Seed data — fake records, used to bootstrap the mock CSV on first run
# ---------------------------------------------------------------------------
_SEED_RECORDS: list[PoliceRecord] = [
    PoliceRecord("PR-1001", "Ravi Kumar Reddy", 34, "M", "Robbery, Assault", "high",
                 "Hyderabad, Banjara Hills",
                 "Wanted in 2 cases, last seen 2025-11-12"),
    PoliceRecord("PR-1002", "Suresh Naidu",      29, "M", "Chain snatching",   "medium",
                 "Vijayawada, Patamata",
                 "Habitual offender — 3 prior arrests"),
    PoliceRecord("PR-1003", "Pradeep Goud",      41, "M", "Vehicle theft, Mugging", "high",
                 "Hyderabad, LB Nagar",
                 "Operates with 2 known accomplices"),
    PoliceRecord("PR-1004", "Anil Verma",        37, "M", "Burglary",          "medium",
                 "Secunderabad, Trimulgherry", ""),
    PoliceRecord("PR-1005", "Kiran Yadav",       26, "M", "Pickpocketing",      "low",
                 "Hyderabad, Charminar",
                 "Petty theft repeat offender"),
    PoliceRecord("PR-1006", "Mahesh Babu Singh", 45, "M", "Assault, Public nuisance", "medium",
                 "Warangal, Hanamkonda", ""),
    PoliceRecord("PR-1007", "Ramesh Choudhary",  31, "M", "Drug trafficking",   "critical",
                 "Vizag, Dwaraka Nagar",
                 "Inter-state operative — DO NOT APPROACH ALONE"),
    PoliceRecord("PR-1008", "Vinod Kumar",       28, "M", "Theft, Trespassing", "low",
                 "Hyderabad, Kukatpally", ""),
    PoliceRecord("PR-1009", "Sandeep Rao",       33, "M", "Extortion, Threatening", "high",
                 "Vijayawada, Governorpet",
                 "Linked to local protection ring"),
    PoliceRecord("PR-1010", "Deepak Sharma",     39, "M", "Murder (cold case)", "critical",
                 "Hyderabad, Ameerpet",
                 "Fled jurisdiction in 2023 — high priority"),
    PoliceRecord("PR-1011", "Manoj Tiwari",      25, "M", "Rioting",            "medium",
                 "Karimnagar", ""),
    PoliceRecord("PR-1012", "Lakshmi Priya",     30, "F", "Cybercrime, Fraud",  "medium",
                 "Hyderabad, Madhapur",
                 "Online romance scam ring"),
]


def _embedding_for(record_id: str, dim: int) -> np.ndarray:
    """Deterministic unit-norm embedding seeded from the record id.

    Using a hash for the seed guarantees the same record gets the same
    fake embedding across restarts — so a face that matches it once
    will keep matching it.
    """
    seed = int(hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = np.linalg.norm(v)
    return v / (n if n > 0 else 1.0)


# ---------------------------------------------------------------------------
# Storage layer
# ---------------------------------------------------------------------------

class PoliceRecordsMock:
    def __init__(
        self,
        records_csv: Path,
        embed_dim: int = 384,
        # Below this similarity, a face is reported as "no match".
        # 0.55 is intentionally permissive for a mock — random embeddings
        # rarely cross 0.20, so honest hits land above 0.55.
        threshold: float = 0.55,
    ):
        self.records_csv = Path(records_csv)
        self.embed_dim = embed_dim
        self.threshold = threshold
        self._lock = threading.RLock()
        self._records: dict[str, PoliceRecord] = {}
        self._embeddings: Optional[np.ndarray] = None       # (N, dim) float32
        self._record_ids: list[str] = []                     # parallel index → record_id

        self._bootstrap_csv()
        self._load()
        log.info("Police mock DB: %d records, threshold=%.2f, dim=%d",
                 len(self._records), threshold, embed_dim)

    # ---- public API ----------------------------------------------------

    def search(self, embedding: np.ndarray) -> Optional[MatchResult]:
        """Return the highest-similarity record above threshold, else None."""
        if embedding is None or self._embeddings is None or len(self._record_ids) == 0:
            return None
        if embedding.shape[0] != self._embeddings.shape[1]:
            log.warning("embedding dim mismatch: face=%d police=%d",
                        embedding.shape[0], self._embeddings.shape[1])
            return None

        # Cosine similarity (both already unit-norm) = dot product
        sims = self._embeddings @ embedding.astype(np.float32)
        idx = int(np.argmax(sims))
        sim = float(sims[idx])
        if sim < self.threshold:
            return None
        rid = self._record_ids[idx]
        with self._lock:
            rec = self._records.get(rid)
        if rec is None:
            return None
        return MatchResult(record=rec, similarity=sim)

    def get(self, record_id: str) -> Optional[PoliceRecord]:
        with self._lock:
            return self._records.get(record_id)

    def list_all(self) -> list[PoliceRecord]:
        with self._lock:
            return list(self._records.values())

    # ---- internals -----------------------------------------------------

    def _bootstrap_csv(self) -> None:
        if self.records_csv.exists():
            return
        self.records_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.records_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["record_id", "name", "age", "gender", "charges",
                        "risk_level", "last_known_address", "notes"])
            for r in _SEED_RECORDS:
                w.writerow([r.record_id, r.name, r.age, r.gender, r.charges,
                            r.risk_level, r.last_known_address, r.notes])
        log.info("Seeded police mock CSV at %s with %d records",
                 self.records_csv, len(_SEED_RECORDS))

    def _load(self) -> None:
        records: dict[str, PoliceRecord] = {}
        with open(self.records_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row["record_id"].strip()
                if not rid:
                    continue
                try:
                    age = int(row.get("age", "0") or 0)
                except ValueError:
                    age = 0
                records[rid] = PoliceRecord(
                    record_id=rid,
                    name=row.get("name", "").strip(),
                    age=age,
                    gender=row.get("gender", "").strip(),
                    charges=row.get("charges", "").strip(),
                    risk_level=row.get("risk_level", "low").strip().lower(),
                    last_known_address=row.get("last_known_address", "").strip(),
                    notes=row.get("notes", "").strip(),
                )

        ids = sorted(records.keys())
        if ids:
            mat = np.stack([_embedding_for(rid, self.embed_dim) for rid in ids], axis=0)
        else:
            mat = np.zeros((0, self.embed_dim), dtype=np.float32)

        with self._lock:
            self._records = records
            self._record_ids = ids
            self._embeddings = mat


# ---------------------------------------------------------------------------
# Factory — keeps a single shared instance so the matrix isn't rebuilt per
# camera pipeline.
# ---------------------------------------------------------------------------

_singleton: Optional[PoliceRecordsMock] = None
_singleton_lock = threading.Lock()


def get_police_db(records_csv: Path, embed_dim: int = 384, threshold: float = 0.55) -> PoliceRecordsMock:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = PoliceRecordsMock(records_csv, embed_dim=embed_dim, threshold=threshold)
        return _singleton


def record_to_dict(r: PoliceRecord) -> dict:
    return asdict(r)
