"""Hard-reset all runtime data — DB rows, evidence files, camera state.

Keeps: signing keys, model weights, config files, sample videos.
Deletes: all violations, users (re-bootstraps admin), watchlist, reid subjects,
         all evidence images in data/evidence/.

Usage:
    python scripts/reset_data.py          # full reset (prompts)
    python scripts/reset_data.py --yes    # full reset, no prompt
    python scripts/reset_data.py --seed-only --yes  # delete only seeded demo data
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def confirm(msg: str) -> bool:
    ans = input(f"{msg} [y/N] ").strip().lower()
    return ans in ("y", "yes")


# Track IDs used by seed_demo.py start at 200
_SEED_TRACK_ID_START = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="skip confirmation")
    parser.add_argument("--seed-only", action="store_true",
                        help="delete only demo-seeded rows (track_id >= 200), keep real pipeline data")
    args = parser.parse_args()

    from src.common.db import session_scope
    from src.evidence.models import Base, ReidSubject, User, Violation, WatchlistEntry
    from src.common.db import engine as get_engine

    if args.seed_only:
        # Delete only the synthetic seeded rows (track_id >= 200), leave real pipeline data
        if not args.yes:
            print("This will delete only demo-seeded violations (track_id >= 200),")
            print("watchlist entries, and reid subjects added by seed_demo.py.")
            print("Real pipeline violations are kept.")
            if not confirm("Proceed?"):
                print("Aborted.")
                return

        from sqlalchemy import select
        deleted_frames = []
        with session_scope() as s:
            rows = s.scalars(
                select(Violation).where(Violation.track_id >= _SEED_TRACK_ID_START)
            ).all()
            for row in rows:
                for path_attr in ("frame_path", "annotated_path", "plate_crop_path"):
                    p = getattr(row, path_attr, None)
                    if p:
                        deleted_frames.append(p)
                s.delete(row)
        print(f"Deleted {len(deleted_frames)} seeded violation rows + evidence files")

        evidence_dir = ROOT / "data" / "evidence"
        removed = 0
        for rel in deleted_frames:
            full = evidence_dir / rel
            if full.exists():
                full.unlink()
                removed += 1
        print(f"Removed {removed} evidence files from disk")

        with session_scope() as s:
            wl_rows = s.scalars(select(WatchlistEntry)).all()
            for w in wl_rows:
                s.delete(w)
            reid_rows = s.scalars(select(ReidSubject)).all()
            for r in reid_rows:
                s.delete(r)
        print(f"Deleted {len(wl_rows)} watchlist entries, {len(reid_rows)} reid subjects")

        # Remove reviewer demo user
        with session_scope() as s:
            rev = s.scalar(select(User).where(User.email == "reviewer@demo.local"))
            if rev:
                s.delete(rev)
                print("Removed demo reviewer user")

        print("\nSeed-only reset complete. Real pipeline violations preserved.")
        return

    # ---- Full reset --------------------------------------------------------
    if not args.yes:
        print("This will delete ALL violations, users, watchlist entries,")
        print("reid subjects, and evidence images from data/evidence/.")
        print("Model weights and config files are NOT touched.")
        if not confirm("Proceed?"):
            print("Aborted.")
            return

    # -- 1. Wipe evidence directory ----------------------------------------
    evidence_dir = ROOT / "data" / "evidence"
    if evidence_dir.exists():
        n = sum(1 for _ in evidence_dir.rglob("*.jpg"))
        shutil.rmtree(evidence_dir)
        print(f"Deleted {n} evidence images from {evidence_dir}")
    else:
        print("data/evidence/ — already empty, skipping")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # -- 2. Drop + recreate all DB tables -----------------------------------
    print("Resetting database...")
    eng = get_engine()
    Base.metadata.drop_all(bind=eng)
    Base.metadata.create_all(bind=eng)
    print("Database tables dropped and recreated.")

    # -- 3. Re-bootstrap admin user -----------------------------------------
    from src.common.security import hash_password
    from config.settings import settings

    with session_scope() as s:
        s.add(User(
            email=settings.bootstrap_admin_email,
            full_name="System Admin",
            password_hash=hash_password(settings.bootstrap_admin_password.get_secret_value()),
            role="admin",
            is_active=True,
        ))
    print(f"Admin user recreated: {settings.bootstrap_admin_email}")

    print("\nReset complete.")


if __name__ == "__main__":
    main()
