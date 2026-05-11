# Training Guide — Custom Models for the CCTV System

You have three fine-tuneable pieces in the pipeline. This guide tells
you which dataset to use, how to prepare it, and the exact commands to
run on your **RTX 3070 Ti Laptop GPU**.

| What | Dataset you already have | Public datasets to augment | Script |
|---|---|---|---|
| Helmet detection | `data/helmet_combined/` (8 757 train, 482 val) | `Helmet Detection Dataset` (Roboflow Universe) | `train_helmet_detector.py` |
| Plate detection (bbox) | `data/plate_ft/` (6 675 train, 250 val) | `Indian License Plate Detection` (Roboflow), `CCPD2019` | `train_plate_detector.py` |
| **Plate recognition (text)** | none — must label | Indian License Plates with Labels (Kaggle), CCPD2019 | `extract_plate_crops_for_labelling.py` + `train_plate_recognizer.py` |
| **Violence (binary)** | `data/violence_ft/` (115 + 60 train, 115 + 60 val) | **RWF-2000**, **RLVS** | `train_violence_classifier.py` |

---

## 1. Violence fine-tuning (easiest — data already available)

Your `data/violence_ft/` has labelled `Fight/` and `NonFight/` clips
ready to go. The trainer uses X3D-M (3.8 M params, Kinetics-400
pretrained) and freezes the backbone — only the last block + the new
binary head trains. This avoids overfitting on the small dataset.

```powershell
python scripts/train_violence_classifier.py
```

- **Runtime**: ~25 min on RTX 3070 Ti (50 epochs, batch 4)
- **Output**: `models/violence_ft.pt`
- **Expected val accuracy**: 85-92 % on the existing dataset

### To push accuracy further

Add public clips into the same folders:

| Dataset | Size | License | Where |
|---|---|---|---|
| **RWF-2000** | 2 000 clips (1 000 fight / 1 000 non-fight) | MIT (request from Duke) | https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection |
| **RLVS** | 2 000 clips | Kaggle CC0 | https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset |
| **Hockey Fight** | 1 000 clips (hockey-specific) | Academic | https://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89 |

Place clips under `data/violence_ft/extra_train/Fight/*.mp4` and
`.../NonFight/*.mp4`. The trainer's `list_clips()` picks them up
automatically (extend the path glob if needed).

Adding RWF-2000 + RLVS to your existing 175 clips brings you to
~4 200 train clips — typically **+8 % val accuracy**.

---

## 2. Helmet fine-tuning (data already prepared)

```powershell
python scripts/train_helmet_detector.py
```

- **Runtime**: ~45 min on RTX 3070 Ti (80 epochs, batch 16, imgsz 640)
- **Output**: `models/helmet_ft.pt` (replaces existing weights)
- **Then**: edit `config/rules.yaml` to set `helmet.enabled: true`

The dataset has 3 classes (helmet, no_helmet, license_plate). The
license_plate class acts as a hard negative for the helmet head —
prevents the model from confusing low-res plates with helmets at
distance.

To improve recall further, you can mix in:
- **Helmet Detection Dataset** (Roboflow Universe):
  https://universe.roboflow.com/search?q=helmet

---

## 3. Custom plate OCR (highest impact, needs labelling)

Off-the-shelf OCRs (fast-plate-ocr, PaddleOCR) average 75-85 % char
accuracy on your Indian-camera footage. A CRNN trained on 500-1000
hand-labelled plates from **your** cameras typically hits 95-98 %.

### Step 3a — Extract crops

```powershell
python scripts/extract_plate_crops_for_labelling.py
```

This scans every `.mp4` under `data/samples/`, detects plates with the
existing pipeline, and saves each unique crop to
`data/plate_ocr_ft/raw/` named after the OCR's best guess —
e.g. `test2_001520_AP02NN9091.jpg`.

CLI options:
- `--source data/samples/test2.mp4` — single video
- `--max-per-video 100` — cap extraction
- `--frame-stride 5` — process every 5th frame (faster scan)

Aim for 300-1000 crops covering different cameras, distances, angles.

### Step 3b — Hand-label

1. Open `data/plate_ocr_ft/raw/` in Windows Explorer
2. For each image, look at the **actual** plate text
3. If the filename is correct → leave it
4. If wrong → rename: `test2_001520_AP02NN9091.jpg` →
   `test2_001520_AP02MN9091.jpg` (just edit the text portion)
5. Delete crops that are too blurry / partial to read
6. Move (or copy) all correctly-named files to `data/plate_ocr_ft/labelled/`

The trainer strips the prefix (`test2_001520_`) and the trailing
`_<digit>` automatically, so duplicates of the same plate are fine.

### Step 3c — Train

```powershell
python scripts/train_plate_recognizer.py
```

- **Runtime**: ~20-30 min on RTX 3070 Ti (60 epochs, batch 32)
- **Output**: `models/plate_ocr_ft.pt`
- **Expected val plate-accuracy**: 90-98 % depending on label count

The model is a MobileNetV3-Small CRNN + BiLSTM + CTC, ~2 M params.
Inference latency: ~5 ms per plate on RTX 3070 Ti, ~25 ms on CPU.

### To skip hand-labelling

Two public Indian datasets with text labels:

| Dataset | Size | Where |
|---|---|---|
| **Indian License Plates with Labels** | ~1 500 plates | https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels |
| **ALPR License Plates** | ~25 k crops | https://www.kaggle.com/datasets/raj4126/alpr-license-plates |
| **CCPD2019** (Chinese, similar format) | 250 k labelled | https://github.com/detectRecog/CCPD |

After downloading, place crops named `<TEXT>.jpg` under
`data/plate_ocr_ft/labelled/<dataset_name>/` and rerun the trainer.
CCPD plates have different fonts so pretraining on it then fine-tuning
on your 500 hand-labelled Indian crops gives the best of both worlds.

---

## Hardware and runtime cheatsheet

All three trainers default to RTX 3070 Ti Laptop (8 GB VRAM) friendly
batch sizes. If you hit OOM:

| Trainer | OOM fix |
|---|---|
| `train_violence_classifier.py` | `batch_size=4 → 2`, or `CLIP_LEN=16 → 8` |
| `train_helmet_detector.py` | `batch=16 → 8`, or `imgsz=640 → 512` |
| `train_plate_recognizer.py` | `batch=32 → 16` |

Recommended order: **violence first** (data ready, fastest), then
**plate OCR labelling + training** (highest end-user impact), then
add public datasets and **retrain everything** with the bigger corpus.

---

## After every training

The runtime pipeline auto-loads the fine-tuned weights when they
exist. Just restart:

```powershell
.\scripts\reset_run.ps1
```

and the boot log will show which `_ft.pt` files were picked up.
