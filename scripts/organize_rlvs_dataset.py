"""Organize the RLVS violence dataset into data/violence_ft/ for training.

The RLVS (Real Life Violence Situations) dataset from Kaggle contains
two folders: Violence and NonViolence, each with ~1000 .avi clips.

This script:
  1. Finds the downloaded RLVS folder (auto-detects common locations)
  2. Converts .avi clips to .mp4 (our trainer expects mp4)
  3. Copies them into data/violence_ft/train/{Fight,NonFight}/
  4. Creates a proper 80/20 train/val split

Usage:
  # First download from: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset
  # Extract the zip file somewhere (e.g., Downloads)
  # Then run:
  python scripts/organize_rlvs_dataset.py --source "C:/Users/raghu/Downloads/Real Life Violence Dataset"
"""
import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("organize_rlvs")

TRAIN_DIR = ROOT / "data" / "violence_ft" / "train"
VAL_DIR = ROOT / "data" / "violence_ft" / "val"

# Common download locations to auto-detect
AUTO_SEARCH = [
    Path.home() / "Downloads",
    Path.home() / "Desktop",
    ROOT,
]

RLVS_FOLDER_NAMES = [
    "Real Life Violence Dataset",
    "real-life-violence-situations-dataset",
    "Real Life Violence Situations Dataset",
]


def find_rlvs_folder(hint: str = None) -> Path:
    """Auto-detect the RLVS dataset folder."""
    if hint:
        p = Path(hint)
        if p.exists():
            return p

    for search_dir in AUTO_SEARCH:
        for name in RLVS_FOLDER_NAMES:
            candidate = search_dir / name
            if candidate.exists():
                log.info("Auto-detected RLVS at %s", candidate)
                return candidate

    raise FileNotFoundError(
        "Could not find the RLVS dataset. Download it from:\n"
        "  https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset\n"
        "Then run: python scripts/organize_rlvs_dataset.py --source <path_to_extracted_folder>"
    )


def copy_clips(src_dir: Path, label: str, train_frac: float = 0.8):
    """Copy clips from src_dir into train/val splits."""
    fight_name = "Fight" if label == "violence" else "NonFight"
    clips = sorted(list(src_dir.glob("*.avi")) + list(src_dir.glob("*.mp4")))

    if not clips:
        log.warning("No clips found in %s", src_dir)
        return 0

    random.shuffle(clips)
    n_train = int(len(clips) * train_frac)
    train_clips = clips[:n_train]
    val_clips = clips[n_train:]

    train_out = TRAIN_DIR / fight_name
    val_out = VAL_DIR / fight_name
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    total = 0
    for split_clips, out_dir in [(train_clips, train_out), (val_clips, val_out)]:
        for clip in split_clips:
            # Always save as .mp4 (our trainer expects it)
            dest = out_dir / f"rlvs_{clip.stem}.mp4"
            if dest.exists():
                continue

            if clip.suffix.lower() == ".avi":
                # Convert AVI to MP4 using OpenCV
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(clip))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(dest), fourcc, fps, (w, h))
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        writer.write(frame)
                    cap.release()
                    writer.release()
                    total += 1
                except Exception as e:
                    log.warning("Failed to convert %s: %s", clip.name, e)
            else:
                shutil.copy2(clip, dest)
                total += 1

            if total % 100 == 0:
                log.info("Processed %d clips...", total)

    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default=None,
                    help="Path to the extracted RLVS dataset folder")
    args = ap.parse_args()

    rlvs_dir = find_rlvs_folder(args.source)
    log.info("RLVS dataset found at: %s", rlvs_dir)

    # RLVS structure: Violence/ and NonViolence/ subfolders
    violence_dir = None
    nonviolence_dir = None

    for child in rlvs_dir.iterdir():
        name_lower = child.name.lower()
        if child.is_dir():
            if "nonviolence" in name_lower or "non_violence" in name_lower or "non-violence" in name_lower:
                nonviolence_dir = child
            elif "violence" in name_lower:
                violence_dir = child

    if violence_dir is None or nonviolence_dir is None:
        # Try looking one level deeper
        for child in rlvs_dir.iterdir():
            if child.is_dir():
                for grandchild in child.iterdir():
                    name_lower = grandchild.name.lower()
                    if grandchild.is_dir():
                        if "nonviolence" in name_lower:
                            nonviolence_dir = grandchild
                        elif "violence" in name_lower:
                            violence_dir = grandchild

    if violence_dir is None or nonviolence_dir is None:
        log.error("Could not find Violence/ and NonViolence/ subdirectories in %s", rlvs_dir)
        log.error("Contents: %s", [c.name for c in rlvs_dir.iterdir()])
        sys.exit(1)

    log.info("Violence folder: %s", violence_dir)
    log.info("NonViolence folder: %s", nonviolence_dir)

    n_fight = copy_clips(violence_dir, "violence")
    n_nonfight = copy_clips(nonviolence_dir, "nonviolence")

    log.info("=" * 60)
    log.info("Done! Added %d Fight + %d NonFight clips from RLVS", n_fight, n_nonfight)
    log.info("")
    log.info("NEXT STEP: retrain the violence classifier:")
    log.info("  python scripts/train_violence_classifier.py")


if __name__ == "__main__":
    main()
