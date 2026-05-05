"""Seed the database with realistic synthetic violations.

Generates photo-realistic synthetic frames for every violation type
so you can demo the full dashboard without a live camera feed.

Covers all 7 violation types, 3 cameras, 50+ violations over 24h,
realistic Indian plates, vehicle colours, ReID cross-camera subjects,
watchlist hits, and mixed review statuses.

Usage:
    python scripts/seed_demo.py
    python scripts/seed_demo.py --count 100   # generate more
    python scripts/seed_demo.py --reset       # wipe first then seed
"""
from __future__ import annotations

import argparse
import random
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.db import engine, session_scope
from src.common.security import hash_password
from src.evidence.models import Base, CameraHealth, ReidSubject, User, WatchlistEntry
from src.evidence.store import EvidenceStore, ReviewStatus
from src.rules.types import ViolationCode, ViolationEvent

# --------------------------------------------------------------------------
# Realistic Indian plate data
# --------------------------------------------------------------------------
STATE_CODES = [
    "TS", "AP", "MH", "KA", "DL", "TN", "GJ", "RJ", "UP", "HR",
    "WB", "PB", "KL", "OD", "MP", "JH", "UK", "CG", "GA", "HP",
]

DISTRICT_CODES = {
    "TS": ["09", "10", "11", "13", "14"],
    "AP": ["05", "09", "31", "37"],
    "MH": ["02", "04", "12", "14", "43"],
    "KA": ["01", "02", "05", "51"],
    "DL": ["01", "02", "03", "08"],
    "TN": ["09", "38", "72", "77"],
    "GJ": ["01", "07", "18", "27"],
    "RJ": ["14", "19", "20", "45"],
    "UP": ["12", "14", "16", "32", "80"],
    "HR": ["26", "55", "66"],
    "MH": ["02", "04", "12", "14", "43"],
}

SERIES_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"


def random_plate() -> str:
    state = random.choice(STATE_CODES)
    districts = DISTRICT_CODES.get(state, ["01", "02", "09"])
    district = random.choice(districts)
    series = random.choice(SERIES_LETTERS) + random.choice(SERIES_LETTERS)
    number = str(random.randint(1000, 9999))
    return f"{state}{district}{series}{number}"


# --------------------------------------------------------------------------
# Vehicle colours & types with realistic India distribution
# --------------------------------------------------------------------------
VEHICLE_COLORS = [
    ("white", (240, 240, 240)),
    ("silver", (180, 180, 190)),
    ("gray", (130, 130, 140)),
    ("black", (30, 30, 30)),
    ("red", (30, 30, 180)),          # BGR
    ("blue", (180, 80, 30)),
    ("yellow", (20, 200, 220)),
    ("green", (40, 130, 40)),
    ("orange", (20, 130, 200)),
    ("brown", (40, 70, 110)),
]

VEHICLE_TYPES = ["car", "two_wheeler", "bus", "truck"]
TYPE_WEIGHTS = [0.40, 0.45, 0.08, 0.07]


# --------------------------------------------------------------------------
# Synthetic frame generators
# --------------------------------------------------------------------------
def _draw_road(frame: np.ndarray) -> None:
    H, W = frame.shape[:2]
    # Sky gradient
    frame[:H//3] = (60, 50, 40)
    for y in range(H//3):
        brightness = int(40 + y * 0.3)
        frame[y] = (brightness, brightness-5, brightness-15)
    # Road surface
    road_color = (55, 55, 60)
    frame[H//3:] = road_color
    # Lane markings
    for x in range(0, W, 120):
        cv2.rectangle(frame, (x, H//2 + 30), (x + 60, H//2 + 38), (220, 220, 160), -1)
    # Kerb
    cv2.rectangle(frame, (0, H//3), (W, H//3 + 8), (100, 100, 100), -1)


def _draw_vehicle(
    frame: np.ndarray,
    x: int, y: int,
    vehicle_type: str,
    color_bgr: tuple,
    plate_text: str | None = None,
) -> tuple[int, int, int, int]:
    """Draw a vehicle and return its bounding box (x1,y1,x2,y2)."""
    if vehicle_type == "two_wheeler":
        w, h = 120, 160
    elif vehicle_type == "bus":
        w, h = 340, 220
    elif vehicle_type == "truck":
        w, h = 300, 200
    else:  # car
        w, h = 240, 140

    x1, y1, x2, y2 = x, y, x + w, y + h

    # Body
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), 1)

    if vehicle_type == "two_wheeler":
        # Wheels
        cv2.circle(frame, (x1 + w//2, y2 - 15), 18, (20, 20, 20), -1)
        cv2.circle(frame, (x1 + w//2, y1 + 15), 14, (20, 20, 20), -1)
        # Rider silhouette
        rider_color = (60, 80, 80)
        cv2.ellipse(frame, (x1 + w//2, y1 - 25), (18, 22), 0, 0, 360, rider_color, -1)
        cv2.rectangle(frame, (x1 + w//2 - 15, y1 - 5), (x1 + w//2 + 15, y1 + 50), rider_color, -1)
    elif vehicle_type in ("car",):
        # Windshield
        cv2.rectangle(frame, (x1 + 30, y1 + 10), (x2 - 30, y1 + 55), (30, 60, 80), -1)
        # Wheels
        for wx in [x1 + 25, x2 - 25]:
            cv2.ellipse(frame, (wx, y2), (18, 10), 0, 0, 360, (20, 20, 20), -1)
    elif vehicle_type == "bus":
        # Windows row
        for i in range(4):
            wx = x1 + 30 + i * 70
            cv2.rectangle(frame, (wx, y1 + 15), (wx + 50, y1 + 60), (30, 60, 80), -1)
        # Destination board
        cv2.rectangle(frame, (x1 + 20, y1 + 5), (x2 - 20, y1 + 28), (10, 10, 10), -1)

    # Number plate
    if plate_text:
        pw, ph = 140, 36
        px = x1 + (w - pw) // 2
        py = y2 - ph - 4
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 255, 255), -1)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 0), 2)
        # IND header bar (blue)
        cv2.rectangle(frame, (px, py), (px + 22, py + ph), (160, 40, 0), -1)
        cv2.putText(frame, "IND", (px + 1, py + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
        # Plate text — format with space for readability
        display = plate_text[:2] + " " + plate_text[2:4] + " " + plate_text[4:6] + " " + plate_text[6:]
        font_scale = 0.52 if len(display) <= 12 else 0.44
        cv2.putText(frame, display, (px + 26, py + 26),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)

    return x1, y1, x2, y2


def _draw_stop_line(frame: np.ndarray, y: int = 540) -> None:
    H, W = frame.shape[:2]
    cv2.line(frame, (0, y), (W, y), (0, 0, 255), 3)
    cv2.putText(frame, "STOP", (20, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)


def _draw_signal(frame: np.ndarray, state: str, x: int = 20, y: int = 60) -> None:
    colors = {"RED": (0, 0, 200), "GREEN": (0, 200, 0), "YELLOW": (0, 200, 220)}
    cv2.rectangle(frame, (x, y), (x + 30, y + 90), (30, 30, 30), -1)
    for i, sig in enumerate(["RED", "YELLOW", "GREEN"]):
        cx, cy = x + 15, y + 15 + i * 30
        c = colors[sig] if sig == state else (40, 40, 40)
        cv2.circle(frame, (cx, cy), 10, c, -1)


def _add_camera_overlay(frame: np.ndarray, cam_id: str, ts: datetime) -> None:
    ts_str = ts.strftime("%d/%m/%Y %H:%M:%S")
    cv2.rectangle(frame, (0, 0), (400, 24), (0, 0, 0), -1)
    cv2.putText(frame, f"{cam_id}  |  {ts_str}", (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 200), 1, cv2.LINE_AA)


def _add_noise(frame: np.ndarray, level: float = 0.03) -> np.ndarray:
    noise = (np.random.randn(*frame.shape) * level * 255).astype(np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_frame(
    vehicle_type: str,
    color_bgr: tuple,
    plate_text: str | None,
    violation_code: str,
    camera_id: str,
    ts: datetime,
    veh_x: int = 500,
    veh_y: int = 380,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Returns (raw_frame, annotated_frame, plate_crop_or_None)."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    _draw_road(frame)

    # Signal state based on violation
    signal = "RED" if violation_code == "red_light_jump" else "GREEN"
    _draw_signal(frame, signal)
    _draw_stop_line(frame)
    _add_camera_overlay(frame, camera_id, ts)

    # Draw background vehicles for realism
    for bx, by, btype, bcol in [
        (120, 420, "car", (100, 100, 180)),
        (820, 410, "two_wheeler", (60, 160, 60)),
        (950, 400, "car", (200, 200, 200)),
    ]:
        _draw_vehicle(frame, bx, by, btype, bcol)

    # Primary offending vehicle
    bbox = _draw_vehicle(frame, veh_x, veh_y, vehicle_type, color_bgr, plate_text)
    x1, y1, x2, y2 = bbox

    # Extra overlays per violation type
    if violation_code == "no_helmet":
        # Red circle on rider head (no helmet)
        hx, hy = (x1 + x2) // 2, y1 - 20
        cv2.circle(frame, (hx, hy), 22, (0, 0, 200), 2)
        cv2.putText(frame, "NO HELMET", (hx - 45, hy - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)

    elif violation_code == "triple_riding":
        # Three rider silhouettes stacked
        for i in range(3):
            rx = x1 + 20 + i * 30
            ry = y1 - 10
            cv2.circle(frame, (rx, ry - 18), 12, (60, 60, 80), -1)
            cv2.rectangle(frame, (rx - 10, ry - 5), (rx + 10, ry + 30), (60, 80, 80), -1)
        cv2.putText(frame, "3 RIDERS", (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 160, 220), 1)

    elif violation_code == "wrong_way":
        # Arrow pointing wrong direction
        arrow_y = (y1 + y2) // 2
        cv2.arrowedLine(frame, (x2 + 20, arrow_y), (x2 + 100, arrow_y),
                        (0, 0, 230), 3, tipLength=0.4)
        cv2.putText(frame, "WRONG WAY", (x1, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 230), 2)

    elif violation_code == "overspeed":
        speed_val = random.randint(72, 120)
        cv2.putText(frame, f"{speed_val} km/h", (x1, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 220), 2)

    elif violation_code == "watchlist_hit":
        cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (180, 0, 180), 3)
        cv2.putText(frame, "! WATCHLIST", (x1, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 0, 200), 2)

    # Plate crop (before annotating)
    plate_crop = None
    if plate_text:
        # The plate is at the bottom of the vehicle bbox
        pw, ph = 140, 36
        px = x1 + ((x2 - x1) - pw) // 2
        py = y2 - ph - 4
        plate_region = frame[py:py + ph, px:px + pw]
        if plate_region.size > 0:
            plate_crop = plate_region.copy()

    frame = _add_noise(frame, 0.025)

    # Annotated version — add violation box overlay
    annotated = frame.copy()
    viol_color = {
        "no_helmet":       (0, 50, 220),
        "red_light_jump":  (0, 0, 220),
        "plate_unreadable": (200, 120, 0),
        "wrong_way":       (0, 0, 255),
        "triple_riding":   (20, 100, 220),
        "overspeed":       (0, 120, 220),
        "watchlist_hit":   (200, 0, 220),
    }.get(violation_code, (0, 0, 200))
    cv2.rectangle(annotated, (x1, y1), (x2, y2), viol_color, 3)
    label = violation_code.upper().replace("_", " ")
    cv2.rectangle(annotated, (x1, y1 - 26), (x1 + len(label) * 11 + 8, y1), viol_color, -1)
    cv2.putText(annotated, label, (x1 + 4, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if plate_text:
        pw, ph = 140, 36
        px2 = x1 + ((x2 - x1) - pw) // 2
        py2 = y2 - ph - 4
        cv2.rectangle(annotated, (px2 - 2, py2 - 2), (px2 + pw + 2, py2 + ph + 2), (0, 220, 220), 2)

    return frame, annotated, plate_crop


# --------------------------------------------------------------------------
# Violation distribution
# --------------------------------------------------------------------------
VIOLATION_DISTRIBUTION = [
    # (code, weight, vehicle_types)
    (ViolationCode.NO_HELMET,       0.30, ["two_wheeler"]),
    (ViolationCode.RED_LIGHT_JUMP,  0.25, ["car", "two_wheeler", "truck"]),
    (ViolationCode.TRIPLE_RIDING,   0.15, ["two_wheeler"]),
    (ViolationCode.PLATE_UNREADABLE,0.12, ["car", "two_wheeler", "truck", "bus"]),
    (ViolationCode.WRONG_WAY,       0.08, ["car", "two_wheeler"]),
    (ViolationCode.OVERSPEED,       0.06, ["car", "truck", "bus"]),
    (ViolationCode.WATCHLIST_HIT,   0.04, ["car", "two_wheeler"]),
]

CAMERAS = [
    {"id": "CAM01", "name": "Signal Free Road - Junction A",    "fps": 15.0},
    {"id": "CAM02", "name": "NH48 Highway Entry",               "fps": 25.0},
    {"id": "CAM03", "name": "Market Circle",                    "fps": 15.0},
]

# 5 watchlisted plates
WATCHLIST_PLATES = [
    ("TS09EA1234", "Stolen vehicle reported 2026-04-12"),
    ("MH12CD9999", "Linked to robbery suspect"),
    ("DL01AB0001", "Warrant pending — district court"),
    ("AP31XY5678", "Overloading repeat offender"),
    ("KA05MJ4444", "Insurance lapsed, multiple priors"),
]


def _weighted_choice(distribution):
    codes, weights, vtypes_list = zip(*[(c, w, v) for c, w, v in distribution])
    total = sum(weights)
    r = random.random() * total
    cumulative = 0
    for code, weight, vtypes in zip(codes, weights, vtypes_list):
        cumulative += weight
        if r <= cumulative:
            return code, vtypes
    return codes[-1], vtypes_list[-1]


def _build_extras(code: ViolationCode, vtype: str, color: str, global_id: str | None) -> dict:
    extras = {
        "vehicle_color": color,
        "vehicle_type": vtype,
    }
    if global_id:
        extras["reid_global_id"] = global_id
    if code == ViolationCode.OVERSPEED:
        extras["speed_kmh"] = float(random.randint(72, 128))
    if code == ViolationCode.TRIPLE_RIDING:
        extras["rider_count"] = random.randint(3, 4)
    if code == ViolationCode.WATCHLIST_HIT:
        extras["watchlist_reason"] = random.choice([r for _, r in WATCHLIST_PLATES])
    return extras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=52, help="Number of violations to generate")
    parser.add_argument("--reset", action="store_true", help="Wipe data before seeding")
    args = parser.parse_args()

    if args.reset:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "reset_data.py"), "--yes"], check=True)

    print("Setting up database tables...")
    Base.metadata.create_all(bind=engine())

    # -- Bootstrap admin if missing ----------------------------------------
    from config.settings import settings
    with session_scope() as s:
        from sqlalchemy import select
        if not s.scalar(select(User).where(User.email == settings.bootstrap_admin_email)):
            s.add(User(
                email=settings.bootstrap_admin_email,
                full_name="System Admin",
                password_hash=hash_password(settings.bootstrap_admin_password.get_secret_value()),
                role="admin", is_active=True,
            ))
            print(f"Admin created: {settings.bootstrap_admin_email}")

    # -- Add reviewer user for demo ----------------------------------------
    with session_scope() as s:
        from sqlalchemy import select
        if not s.scalar(select(User).where(User.email == "reviewer@demo.local")):
            s.add(User(
                email="reviewer@demo.local",
                full_name="Demo Reviewer",
                badge_number="CID-2045",
                password_hash=hash_password("reviewer123"),
                role="reviewer", is_active=True,
            ))
            print("Reviewer user: reviewer@demo.local / reviewer123")

    # -- Watchlist entries -------------------------------------------------
    print("Seeding watchlist...")
    with session_scope() as s:
        from sqlalchemy import select
        for plate, reason in WATCHLIST_PLATES:
            if not s.scalar(select(WatchlistEntry).where(WatchlistEntry.plate_pattern == plate)):
                s.add(WatchlistEntry(plate_pattern=plate, reason=reason, is_active=True))
    print(f"  {len(WATCHLIST_PLATES)} watchlist entries added")

    # -- Camera health records ---------------------------------------------
    with session_scope() as s:
        for cam in CAMERAS:
            row = s.get(CameraHealth, cam["id"])
            if row is None:
                s.add(CameraHealth(
                    camera_id=cam["id"],
                    last_frame_at=datetime.utcnow(),
                    fps_observed=cam["fps"],
                    is_up=True,
                ))

    # -- Generate violations -----------------------------------------------
    store = EvidenceStore()
    now = time.time()
    hour = 3600

    # Assign 2 cross-camera ReID subjects (same vehicle seen on 2 cameras)
    reid_subjects = {
        "subject_A": (str(uuid.uuid4()), "TS09EA1234"),   # global_id, plate
        "subject_B": (str(uuid.uuid4()), "MH02CD5678"),
    }

    # Persist reid subjects
    with session_scope() as s:
        for name, (gid, plate) in reid_subjects.items():
            s.add(ReidSubject(
                global_id=gid,
                first_seen_at=datetime.utcfromtimestamp(now - 2 * hour),
                last_seen_at=datetime.utcfromtimestamp(now - 30 * 60),
                camera_ids=["CAM01", "CAM02"],
                plate_text=plate,
                vehicle_color=random.choice([c for c, _ in VEHICLE_COLORS]),
                vehicle_type="car",
                match_count=3,
            ))

    watchlist_plates_set = {p for p, _ in WATCHLIST_PLATES}

    print(f"\nGenerating {args.count} violations...")
    saved_ids = []

    for i in range(args.count):
        # Time spread: last 24 hours, heavier in rush hours
        hour_offset = random.choices(
            range(24),
            weights=[2,1,1,1,1,2, 5,8,7,5,4,4, 6,5,4,5,8,9, 7,5,3,2,2,1],
            k=1
        )[0]
        minute_offset = random.randint(0, 59)
        ts_epoch = now - (24 - hour_offset) * 3600 - minute_offset * 60 + random.randint(0, 59)
        ts_dt = datetime.utcfromtimestamp(ts_epoch)

        # Camera selection — CAM01 busiest
        cam = random.choices(CAMERAS, weights=[0.50, 0.30, 0.20], k=1)[0]

        # Violation type
        code, vtypes = _weighted_choice(VIOLATION_DISTRIBUTION)
        vtype = random.choice(vtypes)

        # Vehicle appearance
        color_name, color_bgr = random.choice(VEHICLE_COLORS)
        color_bgr = tuple(int(c + random.randint(-15, 15)) for c in color_bgr)
        color_bgr = tuple(max(0, min(255, c)) for c in color_bgr)

        # Plate
        if code == ViolationCode.WATCHLIST_HIT:
            plate = random.choice(WATCHLIST_PLATES)[0]
        elif code == ViolationCode.PLATE_UNREADABLE:
            plate = None
        else:
            plate = random_plate()

        # ReID cross-camera logic
        global_id = None
        if plate in (reid_subjects["subject_A"][1], reid_subjects["subject_B"][1]):
            key = "subject_A" if plate == reid_subjects["subject_A"][1] else "subject_B"
            global_id = reid_subjects[key][0]

        # Vehicle position — vary slightly
        vx = random.randint(400, 650)
        vy = random.randint(350, 430)

        # Confidence scores
        rule_conf = round(random.uniform(0.72, 0.98), 3)
        ocr_conf = round(random.uniform(0.65, 0.97), 3) if plate else round(random.uniform(0.10, 0.35), 3)

        # Generate synthetic frame
        frame, annotated, plate_crop = make_frame(
            vtype, color_bgr, plate, code.value,
            cam["id"], ts_dt, vx, vy,
        )

        # Extras
        extras = _build_extras(code, vtype, color_name, global_id)
        if code == ViolationCode.OVERSPEED:
            rule_conf = round(random.uniform(0.82, 0.99), 3)

        event = ViolationEvent(
            code=code,
            camera_id=cam["id"],
            track_id=200 + i,
            timestamp=ts_epoch,
            frame_idx=5000 + i * 15,
            confidence=rule_conf,
            plate_text=plate,
            plate_ocr_confidence=ocr_conf,
            bbox=(float(vx), float(vy), float(vx + 240), float(vy + 180)),
            extras=extras,
        )

        # Pass frame at half resolution so seed preview images load quickly
        small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        small_ann = cv2.resize(annotated, (640, 360), interpolation=cv2.INTER_AREA)
        vid = store.save(event, frame=small, annotated=small_ann, plate_crop=plate_crop)
        saved_ids.append(vid)

        # Print progress
        plate_display = plate or "—"
        print(f"  [{i+1:02d}/{args.count}] id={vid}  {cam['id']}  {code.value:20s}  {plate_display:14s}  {color_name}")

    # -- Set realistic review statuses ------------------------------------
    print("\nApplying review statuses...")
    with session_scope() as s:
        from sqlalchemy import select
        from src.evidence.models import Violation
        rows = s.scalars(select(Violation).where(Violation.id.in_(saved_ids))).all()

        # 30% approved, 15% rejected, 55% pending
        random.shuffle(rows)
        n = len(rows)
        approve_n = int(n * 0.30)
        reject_n  = int(n * 0.15)

        admin = s.scalar(select(User).where(User.role == "admin"))
        reviewer = s.scalar(select(User).where(User.email == "reviewer@demo.local"))

        for i, row in enumerate(rows):
            if i < approve_n:
                row.status = ReviewStatus.APPROVED.value
                row.reviewed_at = datetime.utcfromtimestamp(
                    time.time() - random.randint(0, 7200)
                )
                row.reviewer_id = reviewer.id if reviewer else None
                row.review_notes = random.choice([None, "Confirmed — challan issued", "Clear violation"])
            elif i < approve_n + reject_n:
                row.status = ReviewStatus.REJECTED.value
                row.reviewed_at = datetime.utcfromtimestamp(
                    time.time() - random.randint(0, 3600)
                )
                row.reviewer_id = admin.id if admin else None
                row.review_notes = random.choice([
                    "Image unclear", "False positive — road works",
                    "Plate partial match only", "Wrong camera angle",
                ])

    # -- Final stats -------------------------------------------------------
    with session_scope() as s:
        from sqlalchemy import select, func
        from src.evidence.models import Violation
        total = s.scalar(select(func.count(Violation.id)))
        pending = s.scalar(select(func.count(Violation.id)).where(Violation.status == "pending"))
        approved = s.scalar(select(func.count(Violation.id)).where(Violation.status == "approved"))
        rejected = s.scalar(select(func.count(Violation.id)).where(Violation.status == "rejected"))

        by_code = dict(s.execute(
            select(Violation.code, func.count(Violation.id)).group_by(Violation.code)
        ).all())

    print(f"""
{'='*60}
Seeding complete!

  Total violations : {total}
  Pending          : {pending}
  Approved         : {approved}
  Rejected         : {rejected}

  By type:""")
    for code, count in sorted(by_code.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"    {code:22s} {count:3d}  {bar}")

    print(f"""
  Watchlist entries: {len(WATCHLIST_PLATES)}
  ReID subjects    : 2 cross-camera vehicles

{'='*60}
Start the review console:

    python scripts/run_api.py

Then open http://localhost:8000
Login: admin@local / admin
       reviewer@demo.local / reviewer123
{'='*60}""")


if __name__ == "__main__":
    main()
