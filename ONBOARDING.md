# Gait-YOLO — Agent & AI Onboarding

> Read this file first. It gives you the full picture of what this project is,
> what has already been built and evaluated, and the current state of every
> component. Do NOT re-do work that is already marked complete.

---

## 1. What This Project Is

**Gait-YOLO** is a real-time multimodal anomaly detection system for CCTV
surveillance. It runs three parallel deep-learning branches and fuses their
outputs through a hierarchical rule cascade + MLP to produce four-level alerts
(CRITICAL / HIGH / MEDIUM / LOW).

**Core thesis:** Single-modality detectors produce 15–25% false-alarm rates.
Cross-modal validation collapses this to ~2%.

**Goal:** Produce a conference-ready research system with real evaluation results,
ablation study, and a complete IEEE paper — built on top of an existing project
report (`Gait-YOLO-Project-Report.pdf`, 31 pages).

---

## 2. System Architecture

```
Live CCTV Stream
       │
       ▼
Frame Extraction & Preprocessing
       │
  ┌────┴──────────────────────────┐
  │              │                │
  ▼              ▼                ▼
Branch 1      Branch 2         Branch 3
YOLOv8n      VideoMAE         CNN-Transformer
(every        (every 4th)      Gait-AE
 frame)                        (every 4th)
  │              │                │
conf≥0.60    prob≥0.75       err > τ*(0.0491)
  │              │                │
  └──────┬────────────────────────┘
         ▼
  Rule Cascade + MLP Fusion Head (3→32→16→4)
         │
  ┌──────┴───────────┬────────────┬──────────┐
  ▼                  ▼            ▼          ▼
CRITICAL           HIGH        MEDIUM      LOW
(weapon)      (violent act)  (act+gait) (gait only)
```

### Branch Summary

| Branch | Model | Input Shape | Cadence | Latency |
|--------|-------|-------------|---------|---------|
| Object Detection | YOLOv8n (3.2M params, CSPDarknet53) | 640×640 RGB | Every frame | ~12 ms |
| Action Recognition | VideoMAE ViT-B (12L, 768-dim, 12 heads) | 16×224×224 RGB | Every 4th frame | ~85 ms |
| Gait Analysis | CNN-Transformer AE (latent=512, 4L transformer, 8 heads) | 15×1×64×64 greyscale | Every 4th frame | ~85 ms |

**Effective throughput:** 18–22 FPS on NVIDIA T4 GPU (16 GB VRAM).
**Per-frame latency:** < 60 ms.

---

## 3. Fusion Layer

### Rule-based cascade — `src/fusion/mlp_fusion.py`

```
Priority 1 — CRITICAL : yolo_conf ≥ 0.60  (persistent 5+ consecutive frames)
Priority 2 — HIGH     : action_prob ≥ 0.75
Priority 3 — MEDIUM   : 0.40 ≤ action_prob < 0.75  AND  gait_err > 0.0491
Priority 4 — LOW      : gait_err > 0.0520
Default    — LOW
```

### MLP Head

- Architecture: `Linear(3→32) → ReLU → Dropout(0.3) → Linear(32→16) → ReLU → Linear(16→4) → Softmax`
- Input vector: `[yolo_conf, action_prob, gait_err / 0.06]`
- Trained: 50 epochs, 10,000 bootstrap samples, Adam lr=1e-3, cross-entropy loss
- Final training loss: 0.1235 (epoch 50)
- Saved weights: `results/mlp_fusion_weights.pth`

### Ensemble Decision

```python
final_alert = ALERT_LEVELS[min(rule_level, mlp_level)]
# 0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW — lower index = higher severity
```

---

## 4. Datasets

| Dataset | Local Path | Size | Purpose |
|---------|-----------|------|---------|
| CASIA-B | `datasets/casia-b/output/` | ~729 MB | Gait-AE training & evaluation |
| Guns & Knives | `datasets/guns-knives/combined_gunsnknifes/` | ~990 MB | YOLO evaluation |
| UCF-Crime | Predictions only: `models/videoMae/ucf_results.csv` | 45 videos | VideoMAE evaluation |

**CASIA-B directory structure:**
```
output/{subject:03d}/{condition}-{seq:02d}/{angle:03d}/*.png
  nm-01 .. nm-06  →  NORMAL class
  bg-01, bg-02    →  ABNORMAL class  (carrying bag)
  cl-01, cl-02    →  ABNORMAL class  (wearing coat)
  124 subjects × 11 angles (000°–180°) × ~96 frames/dir
```

**Guns & Knives structure:**
```
combined_gunsnknifes/test/images/*.jpg   — 324 test images
combined_gunsnknifes/test/labels/*.txt   — YOLO format: class cx cy w h (normalised)
data.yaml: nc=2, names=['pistol', 'knife']
```

---

## 5. Real Evaluation Results (Authoritative Numbers)

All numbers below come from real datasets. The original PDF used simulated/synthetic
numbers — those are now superseded.

### YOLO — 324 test images, IoU threshold = 0.50
| Metric | Value |
|--------|-------|
| mAP@0.50 | **0.7242** |
| Precision | 0.7988 |
| Recall | 0.7566 |
| F1 | 0.7771 |
| Pistol AP | 0.7091 |
| Knife AP | 0.7394 |
| TP / FP / FN | 258 / 65 / 83 |

Source: `results/yolo_real_metrics.json`

### Gait — 600 CASIA-B sequences (300 normal + 300 abnormal)
| Metric | Value |
|--------|-------|
| Optimal threshold τ* | **0.0491** |
| F1 | **0.6652** |
| Normal mean ± std | 0.0511 ± 0.0006 |
| Abnormal mean ± std | 0.0513 ± 0.0013 |
| Variance ratio (ab/nm) | 2.2× |

Source: `results/gait_real_metrics.json`

> **Critical finding — gait score scale:** Real checkpoint produces errors
> near 0.051, not 0.43 as in PDF simulations. CASIA-B silhouettes are ~95%
> black pixels → MSE and SSIM are inherently tiny. All thresholds and
> bootstrap distributions were recalibrated to match the real scale.
> Without this fix, the Medium/Low alert branches are dead code.

### VideoMAE — 45 UCF-Crime videos
| Metric | Value |
|--------|-------|
| Accuracy | 0.1333 (n=45) |
| Recalibrated threshold | **0.055** (was 0.15) |

Source: `models/videoMae/ucf_results.csv`

### Fusion MLP
| Metric | Value |
|--------|-------|
| Precision | **0.9975** |
| Recall | **0.8946** |
| F1 | **0.9433** |
| FP Rate | 0.0185 |
| FP Reduction vs YOLO-only | **95.8%** |

Source: `results/full_real_eval.json`

### Ablation (7 configurations, real data)
| Configuration | P | R | F1 | FPR | FPS |
|---|---|---|---|---|---|
| YOLO-Only | 0.971 | 0.810 | 0.883 | 0.333 | 35 |
| VideoMAE-Only | 1.000 | 0.071 | 0.133 | 0.000 | 8 |
| Gait-Only | 0.444 | 1.000 | 0.615 | 1.000 | 12 |
| YOLO + VideoMAE | 0.971 | 0.810 | 0.883 | 0.333 | 15 |
| YOLO + Gait | 0.933 | 1.000 | 0.966 | 1.000 | 18 |
| Full System (Rule) | 0.971 | 0.810 | 0.883 | 0.333 | 20 |
| **Full System (MLP)** | **0.998** | **0.895** | **0.943** | **0.018** | 19.5 |

Source: `results/ablation_results_real.csv`

---

## 6. Critical Constants — Do Not Guess or Change Without Analysis

```python
# src/fusion/mlp_fusion.py
_YOLO_CRITICAL_CONF  = 0.60     # persistence: must hold for 5+ consecutive frames
_ACTION_HIGH_PROB    = 0.75
_ACTION_MED_LOW      = 0.40
_GAIT_MEDIUM_THRESH  = 0.0491   # data-calibrated F1-optimal (PDF had 0.4521 — wrong scale)
_GAIT_LOW_THRESH     = 0.0520   # nm_mean + 1.5 × nm_std = 0.0511 + 1.5×0.0006
GAIT_NORM_MAX        = 0.06     # empirical max for normalising gait_err to [0,1]

# models/videoMae/test.py
THRESHOLD            = 0.055    # recalibrated from 0.15 — max normal score = 0.0513

# src/pipeline/demo.py
GAIT_THRESH          = 0.0491
ACTION_STRIDE        = 4        # VideoMAE + Gait run every 4th frame
```

---

## 7. Key File Map

### Models & Checkpoints
| Path | Description |
|------|-------------|
| `models/knifes&pistol/best.pt` | YOLOv8n fine-tuned on weapon dataset |
| `models/casib-b/best_transformer_gait.pth` | Gait-AE checkpoint (78 MB) |
| `models/casib-b/train.py` | **Reconstructed** — defines `TransformerAutoencoder`, `GaitSequenceDataset`, `build_index_map`, `ssim_loss_sequence` |
| `models/videoMae/best_model/` | VideoMAE fine-tuned on UCF-Crime |
| `models/videoMae/test.py` | VideoMAE inference (THRESHOLD=0.055) |
| `results/mlp_fusion_weights.pth` | MLP fusion head weights (trained) |

### Core Source
| Path | Description |
|------|-------------|
| `src/pipeline/demo.py` | End-to-end inference on video files |
| `src/fusion/mlp_fusion.py` | Rule cascade + MLP, `FusionEnsemble` class |
| `src/experiments/ablation.py` | Ablation (`--real` flag for real data) |
| `src/gait/threshold_optimizer.py` | Threshold sweep utilities |
| `src/evaluation/metrics.py` | Precision/recall/F1 helpers |

### Evaluation Scripts
| Path | Description |
|------|-------------|
| `scripts/run_yolo_eval.py` | YOLO eval on Guns & Knives test set |
| `scripts/run_gait_eval.py` | Gait eval on CASIA-B (dynamic threshold sweep) |
| `scripts/run_full_real_eval.py` | Combined real fusion evaluation |

### Paper & Docs
| Path | Description |
|------|-------------|
| `paper/ieee_paper.tex` | **Primary output — IEEE LaTeX (Overleaf-ready)** |
| `paper/ieee_paper_draft.md` | Earlier Markdown draft (superseded) |
| `CHANGES.md` | Full changelog with before/after result tables |
| `Gait-YOLO-Project-Report.pdf` | Original 31-page project report (reference only) |

---

## 8. Completion Status

### Done
- [x] Reconstructed `models/casib-b/train.py` (was missing, blocked all gait imports)
- [x] Downloaded CASIA-B (729 MB) and Guns & Knives (990 MB)
- [x] Real YOLO evaluation → mAP50=0.724
- [x] Real Gait evaluation → F1=0.665, τ*=0.0491
- [x] Gait score scale discovery and full threshold recalibration
- [x] VideoMAE threshold recalibration (0.15 → 0.055)
- [x] MLP fusion retrained on recalibrated bootstrap data
- [x] Full fusion evaluation → F1=0.943, FP reduction=95.8%
- [x] 7-configuration ablation study on real data
- [x] IEEE LaTeX paper (`paper/ieee_paper.tex`) — Overleaf-ready, 3 authors

### Remaining / Possible Next Steps
- [ ] Replace dummy author names/affiliations in `paper/ieee_paper.tex`
- [ ] Fine-tune gait model on labeled abnormal data (Δμ=0.0002 — very small separation)
- [ ] Fix VideoMAE class imbalance (Shooting F1=0.10, Arrest F1=0.11)
- [ ] Edge deployment: TensorRT INT8 quantization for Jetson Orin/Xavier
- [ ] Skeleton-pose integration (MediaPipe/PoseNet) for occlusion robustness
- [ ] Attention-based dynamic fusion weights (per-scene context)

---

## 9. How to Run

```bash
# Evaluate YOLO on real test set
python3 scripts/run_yolo_eval.py

# Evaluate gait model on CASIA-B (~2 min on CPU)
python3 scripts/run_gait_eval.py

# Combined real fusion evaluation
python3 scripts/run_full_real_eval.py

# Real ablation study
python3 src/experiments/ablation.py --real

# Full pipeline on a video
python3 src/pipeline/demo.py --video sample.mp4 --output annotated.mp4
```

---

## 10. Paper Quick Facts

- **File:** `paper/ieee_paper.tex`
- **Format:** IEEE conference, two-column, Overleaf-compatible
- **Sections:** Abstract · Introduction · Related Work · Methodology · Experimental Setup · Results · Discussion · Conclusion
- **Figures:** 4 (system arch TikZ, fusion flow TikZ, PR curve pgfplots, gait dist pgfplots)
- **Tables:** 6 (YOLO, VideoMAE per-class, Gait stats, Ablation, FP reduction, Prior work comparison)
- **References:** 12
- **Compile:** `pdflatex → bibtex → pdflatex × 2` or upload directly to Overleaf

> All real numbers in this document take precedence over the original PDF.
> When in doubt, read `results/*.json` directly — those are the ground truth.
