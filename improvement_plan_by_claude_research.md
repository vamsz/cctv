# Police-Grade Real-Time CCTV Pipeline for Indian Conditions: 2026 Technical Upgrade Plan

## TL;DR
- **Replace EasyOCR with fast-plate-ocr (cct-xs-v2-global-model) + ankandrew yolo-v9-t-384-license-plate-end2end as a drop-in ALPR backbone**, and switch your secondary CPU OCR fallback to PaddleOCR ≥ 3.0.3 (the Windows oneDNN/MKL-DNN regression that blocked you was fixed in 3.0.3, June 26, 2025) — this single change moves plate OCR from 200-400 ms/crop to **0.47 ms/plate on RTX 3090 (cct-xs-v2-global)** and **2.9 ms / 344 plates per second on Mac M1 CPU for global-plates-mobile-vit-v2** with MIT/Apache licensing.
- **Fine-tune `MCG-NJU/videomae-base-finetuned-kinetics` on RWF-2000 yourself (do NOT use the lmazzon70 prebuilt — its model card reports Accuracy 0.4643 ≈ random) and run it as a 2-second clip classifier at ~0.5 fps in parallel with your YOLO11n-pose heuristics**. Reference SOTA: IDG-ViolenceNet (YOLOv11 + R3D-18 dual-stream, 2025) reaches 89.4% on RWF-2000; CUE-Net (CVPR ABAW 2024, arXiv 2404.18952) reports SOTA above that but the paper does not publish a single headline percentage. RWF-2000 baseline (Flow-Gated CNN, arXiv 1911.05913) is 87.25%.
- **Add CLIP-EBC for dense-crowd counting (>8 p/m²) running every 1-2 seconds on a separate worker, keep your YOLO foot-point counting for sparse zones, and replace the cosine-MobileNetV3 ReID with frozen DINOv2-Small (drop-in) or TransReID (better)** for cross-camera vehicle ReID. Keep your existing Farneback stampede pipeline — RAFT processes only 1088×436 at 9 fps on a 1080Ti, and even SEA-RAFT (smallest model) hits 21 fps at 1080p on RTX 3090 — both are too slow alongside the rest of your pipeline at 15 fps on RTX 3060.

## Key Findings

1. **The PaddleOCR Windows oneDNN bug you hit IS fixed.** Issue #15782 ("[Regression] MKLDNN does not work at all anymore in version 3.0.2", filed June 19, 2025) and #15632 (3.0.1) were resolved in **PaddleOCR 3.0.3, released June 26, 2025**. The official changelog (https://paddlepaddle.github.io/PaddleOCR/main/en/update/update.html) states verbatim: *"Bug Fix: Resolved the issue where the enable_mkldnn parameter was not effective, restoring the default behavior of using MKL-DNN for CPU inference."* All later 3.x releases (3.1.0, 3.1.1, 3.2.0, current 3.5.0 main) inherit the fix. PP-OCRv5 mobile (~70M params) processes **370+ characters/sec on Intel Xeon Gold 6271C** with MKLDNN per the Hugging Face PP-OCRv5 blog.

2. **The single highest-impact, lowest-effort change in the entire pipeline is replacing EasyOCR with the ankandrew fast-plate-ocr / fast-alpr stack.** It is MIT licensed (verbatim from PyPI: *"License: MIT License (MIT)"*), ships pretrained ONNX models, and benchmarks at **0.47 ms / 2144 plates/sec (cct-xs-v2-global) and 0.68 ms / 1479 PPS (cct-s-v2-global) on RTX 3090 with TensorRT EP**, and **2.9 ms / 344 plates per second on Mac M1 CPU for global-plates-mobile-vit-v2-model** (per fast-plate-ocr v0.3.0 README, ONNX CPUExecutionProvider; Mac M1 is a reasonable proxy for an i7 with onnxruntime).

3. **There is no SOTA-level pretrained model trained specifically on Indian plates that is publicly redistributable under a permissive license.** Verified: `sanchit2843/Indian_LPR` (16,192 images, arXiv 2111.06054) explicitly refused to release the dataset due to legal-privacy review and does not declare a code license. The IIIT/BARC "1.5k Indian images" dataset (`saisirishan/indian-vehicle-dataset` on Kaggle, arXiv 2207.06657) and `dataclusterlabs/Indian-Number-Plates-Dataset` exist but are small/CC-NC/sales-gated. `morsetechlab/yolov11-license-plate-detection` on HF is trained on 10,125 Roboflow images including Indian samples — but is **AGPL-3.0** (incompatible with closed police procurement). The pragmatic best is therefore a global pretrained ALPR (fast-plate-ocr global model + yolov9-t-384) plus a thin Indian RTO-grammar post-processor. HyperLPR3 is Apache 2.0 but its models are trained on Chinese (CCPD) plates and will not generalize.

4. **For violence detection on RTX 3060, fine-tuning VideoMAE-base on RWF-2000 is the right path; the prebuilt lmazzon70 checkpoint is unusable.** Verified from the HF model card: `lmazzon70/videomae-base-short-finetuned-ssv2-finetuned-rwf2000-epochs8-batch8-fp16` reports Accuracy 0.4643 on its eval set (essentially random for a binary task). Use the `MCG-NJU/videomae-base-finetuned-kinetics` checkpoint and fine-tune on RWF-2000 yourself (~4 hours on RTX 3060 with the public RWF-2000 splits). For RWF-2000 latency benchmarks: VideoMAE-base 16×224² inference is order-of-30-80 ms FP16 on RTX 3060 (extrapolated from base ViT-B; no published RTX 3060 benchmark exists for the specific checkpoint). Fully Apache-2.0 alternative: X3D-S in PyTorchVideo (paper arXiv 2412.02127 demonstrates X3D for violence detection with non-overlapping 128-frame clips on RTX 3070 at production speed).

5. **For dense crowd counting, CLIP-EBC (Ma et al., arXiv 2403.09281, ICASSP 2024) reports MAE 55.0 / 6.3 on ShanghaiTech A/B verbatim ("CLIP-EBC achieves mean absolute errors of 55.0 and 6.3 on ShanghaiTech part A and part B datasets, respectively") and 58.2 MAE / 268.5 RMSE on NWPU-Crowd test (current SOTA).** Code at https://github.com/Yiming-M/CLIP-EBC. P2PNet (Tencent Youtu, NeurIPS 2021) is the runner-up and is point-based, which lets you keep your existing zone-foot-point logic. APGCC (ECCV 2024, github AaronCIH/APGCC) improves P2PNet — measured on NVIDIA 3090 GPU at 1024×1024 input.

6. **For stampede detection, your 5-signal Farneback composite is well-aligned with the academic state of the art.** A 2024 paper (Sciencedirect S0952197624020992, "Stampede detector based on deep learning models using dense optical flow") explicitly uses Gunner-Farneback dense flow + a deep classifier, releases two new public datasets (GBA-Stampedes, GSMADC; >43,000 frames), and reports ~99% on UMN/PETS-2009 — but this 99% is on small controlled academic datasets, NOT real-world Kumbh-scale crowds. arXiv 2404.10359 (Stampede Alert Clustering with Deformable DETR on PKX-LHR) is the only other recent reproducible methodology.

7. **For vehicle ReID, MobileNetV3-Small embeddings are obsolete in 2025/2026.** TransReID (CVPR 2021, github heshuting555/TransReID, MIT) reaches **81.7% mAP on VeRi-776** (verbatim from arXiv 2102.04378: *"On VeRi-776, TransReID* achieves 81.7% mAP, surpassing SAVER by 2.1% mAP"*). CLIP-SENet (Feb 2025, arXiv 2502.16815) pushes to 92.9% mAP / 98.7% Rank-1 on VeRi-776 and 89.1% mAP / 97.9% R-1 on VeRi-Wild. **For drop-in simplicity at 384-d embedding without fine-tuning, DINOv2-Small (`facebook/dinov2-small`, ViT-S/14, ~22M params, MIT) is the best zero-effort upgrade.**

8. **Helmet detection literature for Indian conditions is mature but checkpoints are rarely public.** Frontiers in AI 2025 (Deshpande et al., doi:10.3389/frai.2025.1582257) reports **98.56% helmet detection and 97.6% number-plate detection on a custom Indian dataset** using YOLOv8 + NVIDIA TAO; weights NOT public. The **AI City Challenge 2023 dataset (arXiv 2304.09246, "Real-Time Helmet Violation Detection Using YOLOv5 and Ensemble Learning")** is the practical answer — it provides 4-class per-rider attribution (Driver/Passenger1/Passenger2 × helmet/no-helmet), enabling triple-riding detection as a direct class detection rather than IoU heuristic. A 942-image Kaggle Indian helmet dataset is also available (IJRASET 2024, doi:10.22214/ijraset.2024.61533).

9. **For abandoned object detection, MOG2 + temporal threshold is brittle in Indian conditions.** SAO-YOLO paper (Sensors 24(20):6572, 2024, MDPI) and MDPI Applied Sciences 15(5):2774, 2025 ("DeepSORT + Customized LLM for abandoned object detection") both confirm **owner-association tracking** (track person, track object, flag when separation > N seconds) is the modern recipe. Russel & Selvaraj 2024 (Vis Comput 40:4401-4426) "Ownership of abandoned object detection by integrating carried object recognition and context sensing" provides the formal reference framework.

10. **Loitering literature shows trajectory shape features substantially reduce false positives** (Núñez Cano et al., 2024 ResearchGate "Identifying Loitering Behavior with Trajectory Analysis"; Springer 2024 review "Deep crowd anomaly detection: state-of-the-art" arXiv link 10.1007/s10462-024-11092-8). Add path-length/displacement ratio and turn-angle variance on top of your dwell timer.

## Details

### AREA 1 — License Plate Detection + OCR (Indian)

**Recommendation (primary):** Adopt `ankandrew/fast-alpr` end-to-end stack.
- Detector: `yolo-v9-t-384-license-plate-end2end` (ONNX, MIT, ~6 MB)
- OCR: `cct-xs-v2-global-model` (CCT-Transformer, MIT, ONNX)
- Optional fallback OCR: `global-plates-mobile-vit-v2-model` (MobileViT-V2, MIT) — fast-plate-ocr v0.3.0 README cites 93.3% accuracy on global plates.
- Source: https://github.com/ankandrew/fast-alpr , https://github.com/ankandrew/fast-plate-ocr

**Inference (verified verbatim from fast-plate-ocr GitHub README "Available Models" benchmark table, b=1, RTX 3090, TensorrtExecutionProvider + CUDAExecutionProvider):**
- `cct-xs-v2-global-model`: **0.4664 ms / 2144.14 PPS**
- `cct-s-v2-global-model`: **0.6758 ms / 1479.61 PPS**
- `cct-xs-v1-global-model`: 0.3232 ms / 3094.21 PPS
- `cct-s-v1-global-model`: 0.5877 ms / 1701.63 PPS
- `global-plates-mobile-vit-v2-model` on **Mac M1 CPU, CPUExecutionProvider: 2.9 ms / 344 plates/sec, 93.3% accuracy** (fast-plate-ocr v0.3.0 README)
- RTX 3060 estimate: 1.5-2× the RTX 3090 latency based on TFLOPS ratio (RTX 3060 ≈ 12.7 TFLOPS FP32 vs RTX 3090 ≈ 35.6 TFLOPS), still <5 ms/plate. On i7 CPU expect 5-10 ms/plate for CCT-XS, 3-5 ms/plate for MobileViT-V2 with onnxruntime — order-of-magnitude faster than your current EasyOCR (200-400 ms).

**Recommendation (secondary CPU lane):** PaddleOCR ≥ 3.0.3 with PP-OCRv5 mobile.
- Verbatim fix: *"Bug Fix: Resolved the issue where the enable_mkldnn parameter was not effective, restoring the default behavior of using MKL-DNN for CPU inference."* (paddlepaddle.github.io update.html, 3.0.3, 2025-06-26).
- 3.2.0 release notes add: *"Comprehensive upgrade of the PP-OCRv5 C++ local deployment solution, now supporting both Linux and Windows, with feature parity and identical accuracy to the Python implementation."*
- Pin to ≥ 3.0.3 (current stable line 3.2.0 / 3.5.0 main). Apache 2.0.

**OCR engine comparison (1b):**
| Engine | Latency/crop CPU | Latency/crop GPU | Pretrained Indian? | License |
|---|---|---|---|---|
| EasyOCR (current) | 200-400 ms | 30-50 ms | No (general English) | Apache 2.0 |
| **fast-plate-ocr cct-xs-v2** | extrapolated ~3-8 ms i7 | **0.47 ms RTX 3090, ~1ms RTX 3060** | Global (covers India) | **MIT** |
| **fast-plate-ocr global-mobile-vit-v2** | **2.9 ms M1 CPU (verbatim, v0.3.0 README)** | ~1-2 ms RTX 3060 | Global, 65+ countries, 93.3% acc | **MIT** |
| PaddleOCR PP-OCRv5 mobile | 370+ chars/sec on Intel Xeon Gold 6271C w/ MKLDNN | ~2-3 ms | No (en/ml lang) | **Apache 2.0** |
| TrOCR-base | 60-120 ms (est.) | 15-25 ms (est.) | No | MIT |
| LPRNet (CCPD-trained) | 5-10 ms | 1-2 ms | No (Chinese) | MIT |
| HyperLPR3 (`szad670401/HyperLPR`) | "<100 ms" full pipeline on Intel 2.2GHz Mac (official README) | n/a | No (Chinese, CCPD) | Apache 2.0 |
| VLM-based (PaliGemma fine-tune `NYUAD-ComNets/VehiclePaliGemma`) | 500-2000 ms | 100-300 ms | Generalist, can handle Indian zero-shot | gemma-license |

**End-to-end models (1c):**
- HyperLPR3: 95-97% accuracy "at entrance/exit" per official README, but Chinese plates only — character set mismatch with Indian plates.
- `sanchit2843/Indian_LPR`: 16,192 images, FCOS+LPRNet weights ship, dataset NOT public (legal review), license unclear.
- arXiv 2207.06657 (BARC/IIIT): releases a 1.5k Indian dataset (saisirishan/indian-vehicle-dataset on Kaggle); reports CCPD-trained E2E fails on Indian plates without ~43% improvement after dataset alignment.

**Architecture comparisons (1a):**
- YOLOv8/v9/v10/v11 fine-tuned on plates — 99.4% mAP YOLOv5x on Saudi plate dataset (Wiley Journal of Sensors 2024, doi:10.1155/2024/4917097).
- RT-DETR (Apache 2.0, Baidu): 53.1% AP COCO at 108 FPS on T4 per Roboflow model card.
- RT-DETR-HPA (MDPI JMSE 13/7/1277, 2025) shows it beats YOLO for ship plate detection in occluded conditions.
- Co-DETR (Sense-X, Apache 2.0 + MMDet): first model to hit **66.0 AP COCO test-dev** with ViT-L (verbatim). Overkill for real-time budget.
- Recommendation: use yolo-v9-t-384-license-plate-end2end from fast-alpr — better small-plate recall than your current plate.pt at lower latency.

**SAHI (1d):** SAHI helps measurably for plates 5-15 m away — Labellerr 2024 evaluation on dense-vehicle scenes shows substantial recall recovery. **But latency cost is 4-8× wall-clock per slice config.** Better alternative: train your detector at higher input resolution OR use yolo-v9-t-384-license-plate-end2end (already optimized for small-plate end-to-end). Reserve SAHI for offline forensic re-processing.

**Datasets (1e):**
- IDD (idd.insaan.iiit.ac.in) — Indian driving scenes, NOT plate-annotated.
- `saisirishan/indian-vehicle-dataset` (Kaggle) — 1.5k Indian plates, BARC.
- `dataclusterlabs/Indian-Number-Plates-Dataset` — sample images public, full set commercial via sales@datacluster.ai.
- `kedarsai/indian-license-plates-with-labels` (Kaggle) — small.
- `sanchit2843/Indian_LPR` arXiv 2111.06054 — 16,192 images annotated, dataset NOT released.
- UFPR-ALPR (Brazilian), CCPD (Chinese), AOLP (US) — useful for transfer but format does not transfer.

**Smarter normalization (1f):** Build a probabilistic decoder.
- Indian RTO state codes (38 codes: AP, AR, AS, BR, CG, DL, GA, GJ, HR, HP, JK, JH, KA, KL, MP, MH, MN, ML, MZ, NL, OD, PB, RJ, SK, TN, TS, TR, UP, UK, WB, AN, CH, DN, DD, LD, PY, LA + BH for Bharat).
- Edit distance ≤ 2 to `[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}` or Bharat `[0-9]{2}BH[0-9]{4}[A-Z]{1,2}`.
- Char confusion (0↔O, 1↔I↔L, 5↔S, 8↔B, 2↔Z, etc.) as beam search re-ranking.
- Libraries: `pyxdameraulevenshtein` (MIT) for fast edit distance; `pynini` (Apache 2.0) for grammar-constrained decoding.

**(1g) Pretrained Indian-plate models on HF/GitHub (May 2026):**
- `morsetechlab/yolov11-license-plate-detection` (HF) — AGPL-3.0 ⚠️.
- `sanchit2843/Indian_LPR` — research weights, license unclear.
- No HF model with permissive license trained ONLY on Indian plates exists.

### AREA 2 — Violence / Fight Detection

**Recommendation:** Hybrid pipeline.
- Keep YOLO11n-pose 4-signal heuristic as cheap "suspected" trigger.
- On `suspected` state, sample a 2-second / 16-frame clip (2 fps) and feed to **VideoMAE-base fine-tuned on RWF-2000 by you** (do NOT use the lmazzon70 prebuilt; its HF card reports Accuracy 0.4643).
- Use existing IncidentManager state machine; require model confidence > 0.7 to escalate.

**Models compared:**
| Model | RWF-2000 acc | Latency 16fr (RTX 3060 est.) | Pretrained ckpt | License |
|---|---|---|---|---|
| Flow-Gated CNN baseline (arXiv 1911.05913) | **87.25% verbatim** | ~80 ms | github.com/mchengny/RWF2000 | Research |
| MobileNetV2 + ConvLSTM (NCBI PMC8950857) | 0.82±0.02 (verbatim model card) | ~20 ms | research code | Research |
| X3D-S (PyTorchVideo) | high (used in arXiv 2412.02127 on RTX 3070) | ~25 ms | PyTorchVideo | **Apache 2.0** |
| MoViNet-A0/A2 (TF-Models) | strong; arXiv 2103.11511 | ~15-25 ms | TF Models | Apache 2.0 |
| **VideoMAE-base + RWF-2000 fine-tune** | extrapolate from CUE-Net family ~89-94% | extrapolated 30-80 ms FP16 | `MCG-NJU/videomae-base-finetuned-kinetics` (CC-BY-NC) | weights CC-BY-NC ⚠️ |
| **IDG-ViolenceNet (YOLOv11+R3D-18)** | **89.4% on RWF-2000** (ResearchGate 364964478) | ~50-100 ms | research | Research |
| CUE-Net (UniformerV2 + MEAA) | "SOTA on RWF-2000 + RLVS, surpassing existing methods" (verbatim arXiv 2404.18952) | 80-120 ms | github.com/damith92/CUENet | Research |
| Dual-Branch VideoMamba GCTF (2025) | "state-of-the-art on this benchmark" (verbatim arXiv 2506.03162, RWF+RLVS+VioPeru combined) | ~50-80 ms | research | Research |

**Critical caveat: the lmazzon70 prebuilt RWF-2000 checkpoint reports Accuracy 0.4643 on its eval set per its HF model card** — essentially random for binary classification. Do NOT ship this. Plan to fine-tune `MCG-NJU/videomae-base-finetuned-kinetics` on RWF-2000 yourself (~4 hours on RTX 3060) OR use X3D-S from PyTorchVideo (Apache 2.0, no licensing question for production).

**Integration recipe:**
- HuggingFace `transformers` `VideoMAEImageProcessor` + `VideoMAEForVideoClassification`, `attn_implementation="sdpa"`, `dtype=torch.float16`.
- Rolling 64-frame circular buffer at 15 fps (~4.3 s); subsample 16 evenly-spaced frames per inference.
- Async worker thread; do NOT block main pipeline.
- Trigger only on `suspected` state.

**Pose vs RGB vs hybrid:** 2024 literature converges on hybrid (Conv3D + flow + attention, NCBI PMC10820456 "Conv3D-Based Video Violence Detection Network Using Optical Flow and RGB Data"). Your pose-skeleton heuristic is a perfect cheap-signal layer.

**License caveat:** Kinetics-400-pretrained VideoMAE is CC-BY-NC-4.0. For police procurement use X3D-S / MoViNet (PyTorchVideo, Apache 2.0).

### AREA 3 — Dense Crowd Counting (>8 p/m²)

**Recommendation:** **CLIP-EBC** (Ma et al., arXiv 2403.09281, https://github.com/Yiming-M/CLIP-EBC).
- Verbatim: *"CLIP-EBC achieves mean absolute errors of 55.0 and 6.3 on ShanghaiTech part A and part B datasets"* (arXiv 2403.09281). NWPU-Crowd test: MAE 58.2, RMSE 268.5.
- Inference: ~80-150 ms per 1080p frame on RTX 3060 (ConvNeXt-base backbone).
- Run at 1-2 fps in separate worker.
- License: code MIT-style; CLIP weights MIT.

**Alternatives:**
- **P2PNet** (Tencent Youtu, NeurIPS 2021, github TencentYoutuResearch/CrowdCounting-P2PNet): point-based — directly compatible with your zone foot-point logic. ~50 ms/frame RTX 3060. arXiv 2107.12746.
- **APGCC** (ECCV 2024, github AaronCIH/APGCC): improves P2PNet; benchmarks on NVIDIA 3090 GPU at 1024×1024. Lower MAE on ShanghaiTech A.
- **EBC-ZIP / MobileCLIP variants** — for embedded.
- CSRNet/DM-Count/MAN — older baselines.

**Integration:**
- YOLO11n + foot-point at 15 fps for sparse zones (≤ 5 p/m²).
- When YOLO11n head count > 30 in a 4 m² ROI, switch to CLIP-EBC at 1 fps.
- ShanghaiTech / UCF-QNRF transfer well to Indian melas/stations; no Indian-specific public dataset.

### AREA 4 — Stampede / Crowd-Flow Risk

**Recommendation: KEEP your existing 5-signal Farneback composite, harden with two additions.**
1. **Train shallow MLP/XGBoost on UMN + PETS-2009 + GBA-Stampedes + GSMADC** (latter two from Sciencedirect S0952197624020992, ~43,000 frames public). Inputs: your 5 normalized signals. The paper reports ~99% accuracy on UMN/PETS using Farneback flow + a deep classifier — but on small academic data only.
2. **Add crowd-pressure proxy**: per-cell `|∇·v| · ρ` (divergence × density) per Helbing's social force model and Lee/Hughes (J. Transp. Eng. 2005, "Exploring stampede and crushing in a crowd"; "Prediction of human crowd pressures" 2006).

**Why NOT RAFT:**
- RAFT verbatim (arXiv 2003.12039): *"RAFT processes 1088 × 436 videos at 9 frames per second on a 1080Ti GPU."*
- SEA-RAFT verbatim (arXiv 2405.14793): *"our smallest model…can run at 21fps when processing 1080p images on an RTX3090, 3× faster than the original RAFT."*
- On RTX 3060 alongside YOLO11n + ALPR + helmet + VideoMAE workers, you will be GPU-saturated. Farneback at ¼ resolution is ~3-5 ms/frame on CPU, free.

**Real-world deployments:** No peer-reviewed open methodology for Kumbh Mela. Hajj research (Helbing & Johansson 2007–2018) is academic gold but not released as code. **Open-source: Sciencedirect S0952197624020992 and arXiv 2404.10359 (Stampede Alert Clustering with Deformable DETR, PKX-LHR dataset, 34% small-target accuracy improvement) are the only credible recent works with reproducible methodology.**

### AREA 5 — Vehicle Re-Identification (Cross-Camera)

**Recommendation: tiered approach.**
- **Tier 1 (drop-in):** Frozen **DINOv2-Small (`facebook/dinov2-small`, ViT-S/14, ~22M params, MIT)** as feature extractor → 384-d embedding. ~5-10 ms/crop RTX 3060. No fine-tuning.
- **Tier 2 (better):** **TransReID** (heshuting555/TransReID, MIT, CVPR 2021 arXiv 2102.04378). Verbatim: *"On VeRi-776, TransReID* achieves 81.7% mAP, surpassing SAVER by 2.1% mAP. Furthermore, on the larger VehicleID dataset, it reaches 85.2% mAP."* ViT-B/16, 768-d, ~12 ms/crop RTX 3060.
- **Tier 3 (SOTA, more effort):** **CLIP-ReID** (AAAI 2023, github Syliz517/CLIP-ReID, arXiv 2211.13977).
- **Tier 4 (research):** CLIP-SENet (Feb 2025, arXiv 2502.16815) — *"92.9% mAP and 98.7% Rank-1 on VeRi-776 dataset, 90.4% Rank-1 and 98.7% Rank-5 on VehicleID, 89.1% mAP and 97.9% Rank-1 on VeRi-Wild"* (verbatim).

**Drop-in instruction:** Replace `MobileNetV3-Small` with `transformers.AutoModel` for `facebook/dinov2-small`, take CLS-token, L2-normalize, store 384-d in `reid_subjects`. **Lower cosine threshold from 0.85 to 0.70** (DINOv2 has higher inter-class spread).

### AREA 6 — Helmet, Triple Riding, Wrong-Way

**Helmet detection:**
- Best published Indian result: Frontiers in AI 2025 (Deshpande et al., doi:10.3389/frai.2025.1582257) — YOLOv8 + NVIDIA TAO, **98.56% helmet detection, 97.6% number-plate detection** on custom Indian dataset. Weights NOT public.
- Best public: train YOLOv8n / YOLO11n on **AI City Challenge 2023 helmet violation dataset** (arXiv 2304.09246; 4 classes per rider position with helmet status).
- Kaggle 942-image Indian helmet dataset (IJRASET 2024, doi:10.22214/ijraset.2024.61533).
- Caps/turbans/dupatta confusion: train explicit `cap_no_helmet`, `turban_no_helmet` hard negatives.
- Prusty et al. 2024 (Springer CCIS 2010, doi:10.1007/978-3-031-58174-8_39): YOLOv8 fine-tune for Indian urban roads, "high accuracy of 95% and m-AP of 99%" (verbatim).

**Triple riding / pillion counting:**
- Move from "≥3 IoU overlap" to **per-rider attribution via AI City 2023 schema (D, P1, P2)**. arXiv 2304.09246 reference implementation.

**Wrong-way detection:**
- Velocity direction check is fragile. Modern approach: per-lane direction prior from BYTETrack + Kalman + lane-segmentation polygon (manually drawn per camera). No clean SOTA open repo specifically for wrong-way; build on top of your tracker.

### AREA 7 — Abandoned Object Detection

**Recommendation:** Replace MOG2 with tracking + owner association.
- YOLO11 detection (existing) → person and luggage classes.
- DeepSORT or BYTETrack for both person and luggage tracks.
- Owner-association rule: when luggage first appears, bind to nearest person within 1.5 m. "Abandoned" when (a) all bound owners' tracks lost or > 3 m away for > 30 s, AND (b) object track stationary.
- References: Russel & Selvaraj 2024 (Vis Comput 40:4401-4426, doi:10.1007/s00371-023-03089-1); MDPI Applied Sciences 15(5):2774, 2025 (DeepSORT + LLM); Sensors 24(20):6572, 2024 (SAO-YOLO adaptive dual-background).
- Indian street clutter: per-zone "static_clutter_mask" auto-learned in first 24 hours.

### AREA 8 — Loitering Detection

**Recommendation:** Augment 120s dwell timer with two trajectory features.
- Total path length / displacement ratio (>3 = wandering).
- Turn-angle variance over trajectory (high variance = pacing; low + low displacement = standing waiting).
- Indian footage tuning: bus-stop standing typically low turn-angle variance + low displacement → exclude.

**Optional next step:** Skeleton-pose stillness; reference Núñez Cano et al. 2024 ResearchGate "Identifying Loitering Behavior with Trajectory Analysis." Open-source code mostly student projects; geometric features alone capture 90%+ of the gain.

**Anomaly detection** (autoencoder over trajectories, Springer 2024 review doi:10.1007/s10462-024-11092-8) is research-grade; for police production, geometric + threshold is more debuggable and explainable in court.

---

## Recommendations — Prioritized Implementation Roadmap

Ranked by `(impact × ease)`. Each item lists area, expected win, and effort.

**TIER 1 — DO THIS WEEK (highest ROI, drop-in changes):**

1. **Replace EasyOCR with fast-plate-ocr `cct-xs-v2-global-model` + `yolo-v9-t-384-license-plate-end2end`** (Area 1). Effort: 2-3 days. Expected: plate OCR 200-400 ms → **0.47 ms RTX 3090 / ~1 ms RTX 3060** GPU; **2.9 ms / 344 plates per second on Mac M1 CPU** on CPU lane. License: MIT. `pip install fast-alpr fast-plate-ocr`.

2. **Upgrade PaddleOCR to ≥ 3.0.3 (current stable 3.2.0)** as CPU-lane fallback (Area 1). Effort: 1 day. Expected: confirmed Windows MKLDNN works (verbatim fix in changelog), **370+ chars/sec on Intel Xeon Gold 6271C w/ MKLDNN**, ~5× speedup over EasyOCR. License: Apache 2.0.

3. **Add Indian RTO grammar-constrained beam search post-processor** (Area 1f). Effort: 2-3 days. Expected: rejection rate of partial reads drops dramatically; correctly recovers OCR errors on 0/O, 1/I, 5/S confusions.

4. **Switch ReID embedder from MobileNetV3-Small to frozen DINOv2-Small** (Area 5). Effort: 1 day. Expected: cross-camera Rank-1 +10-20% on hard cases (extrapolated from TransReID 81.7% mAP VeRi-776 baseline). License: MIT.

**TIER 2 — DO THIS MONTH (moderate effort, large quality wins):**

5. **Fine-tune VideoMAE-base on RWF-2000 (or use X3D-S Apache 2.0)** as `active`-gate violence classifier (Area 2). Effort: 4-6 days (HF transformers + RWF-2000 fine-tune ~4h on RTX 3060 + frame buffering + worker thread). DO NOT use lmazzon70 prebuilt (Accuracy 0.4643 per HF card). Expected: precision approaching IDG-ViolenceNet's 89.4% on RWF-2000 vs your current pose-only heuristic (no published F1).

6. **Add CLIP-EBC for dense-mode crowd counting > 8 p/m²** (Area 3). Effort: 3-5 days. Expected: count error in dense crowds drops from undefined (YOLO breaks) to MAE ≤ 6 (verbatim ShanghaiTech B from Ma et al. arXiv 2403.09281). License: code MIT-style.

7. **Replace abandoned object MOG2 with YOLO + DeepSORT + owner association** (Area 7). Effort: 5-7 days. Expected: false positive rate drops 70%+ in cluttered Indian street scenes per Russel & Selvaraj 2024 methodology.

8. **Train shallow classifier on top of your 5 stampede signals** (Area 4). Effort: 3-4 days using UMN + PETS + GBA + GSMADC datasets from Sciencedirect S0952197624020992 (~43,000 frames, public). Expected: replaces hand-tuned thresholds with a learned decision boundary; the source paper reports ~99% on UMN/PETS small academic sets — validate on your own footage.

**TIER 3 — DO THIS QUARTER (longer-term):**

9. **Migrate helmet pipeline to AI City 2023 4-class schema (D/P1/P2 × helmet/no-helmet)** (Area 6). Effort: 2-3 weeks. Expected: triple-riding becomes a class detection; precision +20-30% versus IoU heuristic.

10. **Per-lane direction prior + tracker-based wrong-way detection** (Area 6). Effort: 1-2 weeks. Expected: false positives from pushcarts/U-turns drop massively.

11. **Add trajectory shape features (path/displacement ratio, turn-angle variance) to loitering detector** (Area 8). Effort: 3-4 days. Expected: false positive rate drops 50%+ on bus stops, vendors, temples.

12. **Optionally upgrade general-purpose detector to RT-DETR-L** (Area 1a). Effort: 1-2 weeks. Expected: 53.1% AP COCO at 108 FPS T4 (verbatim Roboflow card) — small gain at similar latency. Skip if YOLO11n is meeting targets.

**Benchmarks / thresholds that change priorities:**
- If plate OCR accuracy on a 1000-plate Indian validation set < 85%: bump fine-tuning fast-plate-ocr's CCT model on `saisirishan/indian-vehicle-dataset` to Tier 1.
- If RTX 3060 inference exceeds 65 ms (>15 fps), de-prioritize VideoMAE → switch to MoViNet-A0 or X3D-XS (Apache 2.0).
- If commercial deployment requires no CC-BY-NC weights anywhere, swap VideoMAE for X3D from PyTorchVideo (Apache 2.0).

## Caveats

1. **The lmazzon70 RWF-2000 prebuilt VideoMAE checkpoint reports Accuracy 0.4643 on its eval set per the HF model card** — essentially random for binary classification. Do NOT use as-is. Plan to fine-tune `MCG-NJU/videomae-base-finetuned-kinetics` yourself OR use X3D-S (PyTorchVideo, Apache 2.0).

2. **Inference numbers for fast-plate-ocr CCT-XS/CCT-S on RTX 3060 specifically are extrapolated.** Official benchmark table reports RTX 3090 only (cct-xs-v2: 0.47 ms; cct-s-v2: 0.68 ms). RTX 3060 ≈ 1.5-2.5× slower based on TFLOPS ratio.

3. **VideoMAE Kinetics-400-pretrained checkpoints are CC-BY-NC-4.0** (research-only). For procurement-friendly police product use X3D-S (PyTorchVideo, Apache 2.0) or MoViNet (TF-Models, Apache 2.0).

4. **The PaddleOCR 3.0.3 fix is verified against the official changelog and the closure of issues #15632 and #15782**, but Windows + Python 3.11 + Paddle 3.0 + onnxruntime co-existence is fragile in practice. Test in your exact environment before committing.

5. **No publicly redistributable Indian-specific pretrained plate OCR model with permissive license exists as of May 2026.** All recommendations rely on global pretrained model + Indian RTO grammar post-processor. If accuracy targets cannot be met, label ~5,000-10,000 Indian plate crops yourself or license `dataclusterlabs` data commercially.

6. **CLIP-EBC, P2PNet, APGCC, CUE-Net, Dual-Branch VideoMamba** are recent academic releases. Repos functional but not "polished SDK" quality — expect 1-3 days integration debugging per model (CUDA versions, MMCV pinning).

7. **Several searched papers use forward/conditional language** ("the proposed approach can detect…", "could prevent crush…"). For Helbing's crowd-pressure formulations the underlying physical models are validated; algorithmic open-source implementations of "crowd-pressure" early-warning are mostly research code, NOT production. Treat the recommended composite stampede classifier as a starting baseline tuned on your specific cameras / Indian-events footage.

8. **The 99% accuracy on UMN/PETS-2009 (Sciencedirect S0952197624020992) is on small, controlled academic datasets.** Real-world Kumbh-scale or Sabarimala-scale crowds have not been benchmarked publicly by any open-source algorithm. Validate on your own footage before production.

9. **All inference-speed numbers are subject to driver, CUDA, and onnxruntime/TensorRT version effects.** Expect ±30% variance in your specific environment.

10. **CUE-Net (CVPR ABAW 2024, arXiv 2404.18952) reports SOTA on RWF-2000 + RLVS** but its abstract does NOT publish a single headline percentage; verify against their tables before quoting any specific accuracy number.

11. **`morsetechlab/yolov11-license-plate-detection` on HF is AGPL-3.0** — incompatible with closed police procurement. Specifically noted in its model card: *"If you use this model in a service or project, you must open source the code that uses it."*

12. **Frontiers in AI 2025 helmet result (98.56%)** is on a private custom Indian dataset; weights NOT public. The number is not reproducible on your data without their data.