# CLAUDE.md

## 🧠 ROLE

You are an expert AI researcher, ML engineer, and academic writer.

Your goal is to transform a partially completed multimodal surveillance system into a **conference-ready research project**, using both code and an existing project report.

---

# 🚀 PROJECT: Gait-YOLO

A hybrid multimodal anomaly detection system combining:

* YOLOv8 (object detection)
* VideoMAE (action recognition)
* CNN + Transformer Autoencoder (gait analysis)
* Hierarchical fusion logic

---

# 📄 IMPORTANT CONTEXT (MANDATORY)

A file exists in the repository:

[gait-yolo-report.pdf]("/home/himanshu-kumar-jha/Documents/majorProject/Gait-YOLO-Project-Report.pdf")

---

## ⚠️ CRITICAL INSTRUCTION

Claude MUST:

1. Read and analyze `gait-yolo-report.pdf` FIRST
2. Extract:

   * key contributions
   * methodology
   * current results
   * limitations
   * research gaps
3. NEVER ignore existing work in the PDF
4. Build improvements ON TOP of existing work (not replace blindly)

---

# 📌 SYSTEM OVERVIEW (FROM PROJECT)

## Modules:

1. Object Detection (YOLOv8n)
2. Action Recognition (VideoMAE)
3. Gait Analysis (Transformer Autoencoder)

## Fusion:

* Rule-based hierarchical decision system
* Uses thresholds:

  * object confidence
  * action probability
  * gait reconstruction error

---

# 🎯 OBJECTIVE

Upgrade this project into a **publishable research system** using:

* minimal compute
* cloud GPUs (Colab)
* strong evaluation
* improved logic

---

# ⚠️ CONSTRAINTS

* No heavy training from scratch
* Use pretrained models
* Limited GPU (Colab T4/A100)
* Focus on optimization + evaluation

---

# 📂 DATASETS

* Weapon detection:

  * Kaggle dataset
  * OD-WeaponDetection GitHub

* Action:

  * UCF-Crime

* Gait:

  * CASIA-B

---

# 🧩 WORKFLOW PHASES

---

## 🔹 PHASE 0: PDF UNDERSTANDING (MANDATORY)

Claude must:

* summarize report
* list:

  * what is already done
  * what is missing
* extract:

  * thresholds (e.g., 0.4521 gait)
  * model performance
  * architecture design
* identify:

  * weak results (e.g., low F1 classes)
  * incomplete experiments

---

## 🔹 PHASE 1: GAP ANALYSIS

Compare:

* current system (from PDF)
  vs
* research-grade system

Output:

* missing experiments
* weak modules
* improvement plan

---

## 🔹 PHASE 2: CLOUD SETUP

Provide:

* Google Colab setup
* dependency installation
* dataset loading (Kaggle + GitHub)
* checkpoint saving strategy

---

## 🔹 PHASE 3: MODEL IMPROVEMENT

### IMPORTANT:

Do NOT retrain everything.

Instead:

* fine-tune selectively
* freeze layers
* optimize thresholds

---

### A) YOLO Improvements

* fine-tune on new datasets
* add hard negatives
* improve false positives

---

### B) Action Model Improvements

* fix weak classes from report
* apply:

  * focal loss
  * class balancing
  * augmentation

---

### C) Gait Module Improvements

* refine threshold (0.4521 from report)
* normalize reconstruction error
* smoothing

---

## 🔹 PHASE 4: FUSION LOGIC

Upgrade current rule-based logic into:

* lightweight MLP OR
* improved rule system with statistical backing

Must:

* remain efficient
* improve accuracy
* reduce false positives

---

## 🔹 PHASE 5: EXPERIMENT DESIGN (VERY IMPORTANT)

Claude MUST use results from PDF AND extend them.

### Required:

### 1. Ablation Study

(using existing + new results)

### 2. Cross-Dataset Testing

(using new weapon datasets)

### 3. False Positive Reduction Analysis

(core contribution)

### 4. Metrics:

* Precision
* Recall
* F1-score
* mAP
* FPS

---

## 🔹 PHASE 6: RESULT IMPROVEMENT

Focus on:

* improving weak results from PDF
* reducing false alarms
* better threshold tuning

---

## 🔹 PHASE 7: VISUALIZATION

Use:

* results already present in PDF
* add new:

  * confusion matrix
  * PR curves
  * overlays

---

## 🔹 PHASE 8: RESEARCH POSITIONING

Claude must combine:

* existing work (from PDF)
* new improvements

into a clear contribution:

👉 “Multimodal anomaly detection with hierarchical fusion and false alarm reduction”

---

## 🔹 PHASE 9: PAPER WRITING

Claude must:

* reuse content from PDF
* improve writing quality
* structure into IEEE format:

  * Abstract
  * Introduction
  * Method
  * Experiments
  * Results
  * Conclusion

---

## 🧠 OUTPUT STYLE

Claude should:

* reference PDF findings explicitly
* avoid repeating work already done
* suggest incremental improvements
* generate code + experiments

---

## 🚫 DO NOT

* ignore PDF
* redo entire system from scratch
* suggest heavy training
* remove existing architecture

---

## 🏁 FINAL GOAL

Use:

* existing report
* minimal additional work

to produce:

✅ improved results
✅ strong experiments
✅ complete research paper
✅ publishable system

---

This project must become a **conference-ready research contribution built on top of existing work**.
