"""Merge Kaggle (Pascal VOC XML) + Roboflow Indian (YOLOv8) helmet datasets.

Output layout  →  data/helmet_combined/
  train/images/  train/labels/
  val/images/    val/labels/
  data.yaml

Unified classes:
  0  helmet
  1  no_helmet
  2  license_plate   (Roboflow only — ignored by our helmet rule)
"""
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import os

ROOT        = Path(__file__).resolve().parent.parent
KAGGLE_IMG  = ROOT / "Kaggle_Helmet" / "images"
KAGGLE_ANN  = ROOT / "Kaggle_Helmet" / "annotations"
ROBO_ROOT   = ROOT / "helmet-lincense-plate-detection.v10i.yolov8"
OUT         = ROOT / "data" / "helmet_combined"

CLASSES = ["helmet", "no_helmet", "license_plate"]

# Roboflow original: 0=LP, 1=helmet, 2=no_helmet  → remap to our schema
ROBO_REMAP = {0: 2, 1: 0, 2: 1}

# Kaggle class name → our class id
KAGGLE_MAP = {
    "with helmet":    0,
    "helmet":         0,
    "without helmet": 1,
    "no helmet":      1,
}


def xml_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """Convert a Pascal VOC XML annotation to YOLO TXT lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        cls_id = KAGGLE_MAP.get(name)
        if cls_id is None:
            continue
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def get_img_size(img_path: Path):
    from PIL import Image
    with Image.open(img_path) as im:
        return im.width, im.height


def copy_roboflow(split: str, out_split: str):
    src_img = ROBO_ROOT / split / "images"
    src_lbl = ROBO_ROOT / split / "labels"
    dst_img = OUT / out_split / "images"
    dst_lbl = OUT / out_split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    count = 0
    for lbl_file in src_lbl.glob("*.txt"):
        # Remap class IDs
        lines = lbl_file.read_text().splitlines()
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            old_cls = int(parts[0])
            new_cls = ROBO_REMAP.get(old_cls, old_cls)
            new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
        # Copy image
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            img_file = src_img / (lbl_file.stem + ext)
            if img_file.exists():
                shutil.copy2(img_file, dst_img / img_file.name)
                break
        (dst_lbl / lbl_file.name).write_text("\n".join(new_lines))
        count += 1

    print(f"  Roboflow {split} → {out_split}: {count} samples")


def convert_kaggle(val_ratio: float = 0.18):
    xml_files = sorted(KAGGLE_ANN.glob("*.xml"))
    random.seed(42)
    random.shuffle(xml_files)
    n_val = max(1, int(len(xml_files) * val_ratio))
    splits = {"val": xml_files[:n_val], "train": xml_files[n_val:]}

    for split_name, files in splits.items():
        dst_img = OUT / split_name / "images"
        dst_lbl = OUT / split_name / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        count = 0
        for xml_path in files:
            # Find matching image
            img_path = None
            for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
                candidate = KAGGLE_IMG / (xml_path.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue
            w, h = get_img_size(img_path)
            lines = xml_to_yolo(xml_path, w, h)
            if not lines:
                continue
            shutil.copy2(img_path, dst_img / img_path.name)
            (dst_lbl / (xml_path.stem + ".txt")).write_text("\n".join(lines))
            count += 1
        print(f"  Kaggle {split_name}: {count} samples")


def write_yaml():
    yaml_content = f"""path: {OUT.as_posix()}
train: train/images
val: val/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    (OUT / "data.yaml").write_text(yaml_content)
    print(f"\nWrote {OUT / 'data.yaml'}")


def stats():
    for split in ("train", "val"):
        n_img = len(list((OUT / split / "images").glob("*")))
        n_lbl = len(list((OUT / split / "labels").glob("*.txt")))
        print(f"  {split}: {n_img} images, {n_lbl} labels")


if __name__ == "__main__":
    if OUT.exists():
        shutil.rmtree(OUT)

    print("=== Copying Roboflow Indian dataset ===")
    copy_roboflow("train", "train")
    copy_roboflow("valid", "val")

    print("\n=== Converting Kaggle Pascal VOC dataset ===")
    convert_kaggle()

    write_yaml()

    print("\n=== Final dataset stats ===")
    stats()
    print("\nDataset ready at:", OUT)
