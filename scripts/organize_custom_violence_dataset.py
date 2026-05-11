"""Organize the custom violence dataset found in Downloads into data/violence_ft/.

Source: C:/Users/raghu/Downloads/A-Dataset-for-Automatic-Violence-Detection-in-Videos-master/A-Dataset-for-Automatic-Violence-Detection-in-Videos-master/violence-detection-dataset
Structure:
  violent/cam1/*.mp4
  violent/cam2/*.mp4
  non-violent/cam1/*.mp4
  non-violent/cam2/*.mp4

We will split them 80/20 into train/val.
"""
import logging
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("organize_custom")

SRC_DIR = Path("C:/Users/raghu/Downloads/A-Dataset-for-Automatic-Violence-Detection-in-Videos-master/A-Dataset-for-Automatic-Violence-Detection-in-Videos-master/violence-detection-dataset")
TRAIN_DIR = ROOT / "data" / "violence_ft" / "train"
VAL_DIR = ROOT / "data" / "violence_ft" / "val"

def process_folder(src_sub, dest_name, train_frac=0.8):
    violent_dir = SRC_DIR / src_sub
    clips = []
    for cam in ["cam1", "cam2"]:
        cam_dir = violent_dir / cam
        if cam_dir.exists():
            clips.extend(list(cam_dir.glob("*.mp4")))
    
    if not clips:
        log.warning("No clips found in %s", violent_dir)
        return
        
    random.shuffle(clips)
    n_train = int(len(clips) * train_frac)
    train_clips = clips[:n_train]
    val_clips = clips[n_train:]
    
    for split_clips, out_root in [(train_clips, TRAIN_DIR), (val_clips, VAL_DIR)]:
        out_dir = out_root / dest_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for clip in split_clips:
            dest = out_dir / f"custom_{clip.parent.name}_{clip.name}"
            if not dest.exists():
                shutil.copy2(clip, dest)
                
    log.info("Processed %d clips for %s", len(clips), dest_name)

def main():
    if not SRC_DIR.exists():
        log.error("Source directory not found: %s", SRC_DIR)
        return
        
    process_folder("violent", "Fight")
    process_folder("non-violent", "NonFight")
    
    log.info("Done organizing custom dataset.")

if __name__ == "__main__":
    main()
