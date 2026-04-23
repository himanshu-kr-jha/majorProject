"""
Evaluate the fine-tuned VideoMAE model on UCF-Crime test videos.

Model:   models/videoMae/best_model  (14-class VideoMAE ViT-B fine-tuned on UCF-Crime)
Dataset: datasets/anomalydetectiondatasetucf
         Anomaly_Test.txt → 140 anomaly + 150 normal videos (290 total)

Anomaly score = 1 - P(Normal_Videos_Event)  [class id=7 per config.json]

Outputs saved to results/videomae_results/:
  metrics.json    — overall metrics (P/R/F1/Acc/AUC-ROC + optimal threshold)
  per_video.csv   — per-video scores and predictions
  per_class.csv   — per anomaly-category detection rate
  confusion.txt   — TP/FP/FN/TN summary

Usage:
    python3 scripts/run_videomae_ucf_eval.py
    python3 scripts/run_videomae_ucf_eval.py --dataset /path --model /path --results /path
"""

import sys, json, csv, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT        = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "anomalydetectiondatasetucf"
MODEL_DIR   = ROOT / "models" / "videoMae" / "best_model"
RESULTS_DIR = ROOT / "results" / "videomae_results"

NORMAL_CLASS_ID = 7   # "Normal_Videos_Event" — config.json id2label[7]
NUM_FRAMES      = 16  # VideoMAE ViT-B expects exactly 16 frames


# ── video loading ──────────────────────────────────────────────────────────────

def load_video_frames(video_path: Path, num_frames: int = NUM_FRAMES):
    """Uniformly sample num_frames RGB frames. Returns list of (H,W,3) arrays or None."""
    try:
        import cv2
        cap   = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < num_frames:
            cap.release()
            return None
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames  = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    except Exception:
        return None


# ── dataset discovery ──────────────────────────────────────────────────────────

def find_test_videos(dataset_dir: Path):
    """
    Parse Anomaly_Test.txt → (anomaly_list, normal_list).
    Each item: (Path, label:int, category:str).  label 1=anomaly, 0=normal.
    """
    test_file = dataset_dir / "Anomaly_Test.txt"
    if not test_file.exists():
        raise FileNotFoundError(f"Anomaly_Test.txt not found in {dataset_dir}")

    anomaly_videos, normal_videos = [], []
    with open(test_file) as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            full = dataset_dir / rel
            if not full.exists():
                continue
            if "Normal_Videos" in rel:
                normal_videos.append((full, 0, "Normal"))
            else:
                # extract category from parent dir, e.g. .../Abuse/Abuse028_x264.mp4 → Abuse
                cat = Path(rel).parts[-2] if len(Path(rel).parts) >= 2 else "Unknown"
                anomaly_videos.append((full, 1, cat))

    return anomaly_videos, normal_videos


# ── metrics ────────────────────────────────────────────────────────────────────

def threshold_sweep(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return the threshold that maximises F1 over 400 candidates."""
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.linspace(0.01, 0.99, 400):
        preds = (scores >= thresh).astype(int)
        tp    = int(((preds == 1) & (labels == 1)).sum())
        fp    = int(((preds == 1) & (labels == 0)).sum())
        fn    = int(((preds == 0) & (labels == 1)).sum())
        prec  = tp / (tp + fp + 1e-9)
        rec   = tp / (tp + fn + 1e-9)
        f1    = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)
    return best_thresh


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Trapezoidal AUC-ROC (no sklearn required)."""
    pos, neg = (labels == 1).sum(), (labels == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order  = np.argsort(-scores)
    l_sort = labels[order]
    tpr_pts, fpr_pts = [0.0], [0.0]
    tp = fp = 0
    for l in l_sort:
        if l == 1:
            tp += 1
        else:
            fp += 1
        tpr_pts.append(tp / pos)
        fpr_pts.append(fp / neg)
    tpr_pts.append(1.0); fpr_pts.append(1.0)
    return float(np.trapezoid(tpr_pts, fpr_pts))


# ── main ───────────────────────────────────────────────────────────────────────

def run_eval(args):
    import torch
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:  {device}")
    print(f"Model:   {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Results: {args.results}\n")

    processor = VideoMAEImageProcessor.from_pretrained(args.model)
    model     = VideoMAEForVideoClassification.from_pretrained(args.model)
    model.to(device).eval()
    print("Model loaded.\n")

    anomaly_vids, normal_vids = find_test_videos(Path(args.dataset))
    all_vids = anomaly_vids + normal_vids
    print(f"Test set: {len(anomaly_vids)} anomaly + {len(normal_vids)} normal = {len(all_vids)} total\n")

    rows, skipped = [], 0

    pbar = tqdm(all_vids, desc="Evaluating", unit="video",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for video_path, true_label, category in pbar:
        pbar.set_postfix({"file": video_path.name[:30], "skip": skipped})

        frames = load_video_frames(video_path)
        if frames is None:
            skipped += 1
            pbar.set_postfix({"file": video_path.name[:30], "skip": skipped})
            continue

        try:
            inputs = processor(list(frames), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            probs         = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            normal_prob   = float(probs[NORMAL_CLASS_ID])
            anomaly_score = 1.0 - normal_prob
            pred_class_id = int(probs.argmax())
            rows.append({
                "video":         video_path.name,
                "category":      category,
                "true_label":    true_label,
                "anomaly_score": round(anomaly_score, 6),
                "normal_prob":   round(normal_prob, 6),
                "pred_class_id": pred_class_id,
                "pred_class":    model.config.id2label.get(pred_class_id, str(pred_class_id)),
            })
        except Exception as e:
            tqdm.write(f"  [ERR] {video_path.name}: {e}")
            skipped += 1

    if not rows:
        print("No videos processed — check --dataset path.")
        return

    scores = np.array([r["anomaly_score"] for r in rows])
    labels = np.array([r["true_label"]    for r in rows])

    best_thresh = threshold_sweep(scores, labels)
    preds       = (scores >= best_thresh).astype(int)

    tp   = int(((preds == 1) & (labels == 1)).sum())
    fp   = int(((preds == 1) & (labels == 0)).sum())
    fn   = int(((preds == 0) & (labels == 1)).sum())
    tn   = int(((preds == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    acc  = (tp + tn) / len(rows)
    auc  = compute_auc(scores, labels)

    nm_sc = scores[labels == 0]
    an_sc = scores[labels == 1]

    # per-category true-positive rate (anomaly classes only)
    cat_stats = defaultdict(lambda: {"total": 0, "detected": 0})
    for r, pred in zip(rows, preds):
        cat = r["category"]
        cat_stats[cat]["total"] += 1
        if r["true_label"] == 1:
            cat_stats[cat]["detected"] += int(pred == 1)

    # ── console summary ────────────────────────────────────────────────────────
    print("\n" + "="*58)
    print("  VideoMAE  |  UCF-Crime Evaluation")
    print("="*58)
    print(f"  Threshold (τ*) : {best_thresh:.4f}")
    print(f"  Precision      : {prec:.4f}")
    print(f"  Recall         : {rec:.4f}")
    print(f"  F1-score       : {f1:.4f}")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  AUC-ROC        : {auc:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Normal  score  : {nm_sc.mean():.4f} ± {nm_sc.std():.4f}")
    print(f"  Anomaly score  : {an_sc.mean():.4f} ± {an_sc.std():.4f}")
    print(f"  Skipped        : {skipped}")
    print()
    print("  Per-category detection (anomaly classes):")
    for cat in sorted(cat_stats):
        if cat == "Normal":
            continue
        s    = cat_stats[cat]
        rate = s["detected"] / s["total"] if s["total"] else 0.0
        bar  = "#" * int(rate * 20)
        print(f"    {cat:<20} {s['detected']:>2}/{s['total']:>2}  {bar}")
    if f1 < 0.70:
        print("\n  [WARN] F1 < 0.70 — model may benefit from fine-tuning.")
    print("="*58 + "\n")

    # ── save outputs ───────────────────────────────────────────────────────────
    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model":              args.model,
        "dataset":            args.dataset,
        "best_threshold":     best_thresh,
        "precision":          float(prec),
        "recall":             float(rec),
        "f1":                 float(f1),
        "accuracy":           float(acc),
        "auc_roc":            float(auc),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_anomaly":          int((labels == 1).sum()),
        "n_normal":           int((labels == 0).sum()),
        "n_total":            len(rows),
        "n_skipped":          skipped,
        "normal_mean_score":  float(nm_sc.mean()) if len(nm_sc) else 0.0,
        "normal_std_score":   float(nm_sc.std())  if len(nm_sc) else 0.0,
        "anomaly_mean_score": float(an_sc.mean()) if len(an_sc) else 0.0,
        "anomaly_std_score":  float(an_sc.std())  if len(an_sc) else 0.0,
    }

    json_path = out_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved → {json_path}")

    csv_path = out_dir / "per_video.csv"
    fieldnames = ["video", "category", "true_label", "anomaly_score",
                  "normal_prob", "pred_class_id", "pred_class"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {csv_path}")

    class_csv = out_dir / "per_class.csv"
    with open(class_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "total", "detected", "detection_rate"])
        for cat in sorted(cat_stats):
            s    = cat_stats[cat]
            rate = s["detected"] / s["total"] if s["total"] else 0.0
            writer.writerow([cat, s["total"], s["detected"], f"{rate:.4f}"])
    print(f"Saved → {class_csv}")

    conf_path = out_dir / "confusion.txt"
    with open(conf_path, "w") as f:
        f.write(f"Threshold : {best_thresh:.4f}\n")
        f.write(f"TP={tp}  FP={fp}\n")
        f.write(f"FN={fn}  TN={tn}\n\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1        : {f1:.4f}\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"AUC-ROC   : {auc:.4f}\n")
    print(f"Saved → {conf_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VideoMAE on UCF-Crime anomaly detection")
    parser.add_argument("--dataset", default=str(DATASET_DIR),
                        help="Path to anomalydetectiondatasetucf directory")
    parser.add_argument("--model",   default=str(MODEL_DIR),
                        help="Path to HF VideoMAE model directory")
    parser.add_argument("--results", default=str(RESULTS_DIR),
                        help="Output directory for results")
    args = parser.parse_args()
    run_eval(args)
