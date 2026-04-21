"""
Evaluate the fine-tuned VideoMAE HF model on UCF-Crime test videos.

This replaces run_videomae_eval.py which incorrectly used the gait autoencoder
with MOG2 instead of the actual VideoMAE model at models/videoMae/best_model/.

Normal class id=7 ("Normal_Videos_Event") per config.json id2label mapping.
anomaly_score = 1 - P(Normal_Videos_Event)

Outputs:
  results/videomae_hf_metrics.json
  results/videomae_hf_eval.csv

Usage:
    python3 scripts/run_videomae_hf_eval.py
    python3 scripts/run_videomae_hf_eval.py --dataset /path/to/ucf --model /path/to/model
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET_DIR     = ROOT / "datasets" / "anomalydetectiondatasetucf"
MODEL_DIR       = ROOT / "models" / "videoMae" / "best_model"
RESULTS_DIR     = ROOT / "results"
NORMAL_CLASS_ID = 7   # "Normal_Videos_Event" — config.json id2label[7]
NUM_FRAMES      = 16  # VideoMAE ViT-B expects exactly 16 frames


def load_video_frames(video_path: Path, num_frames: int = NUM_FRAMES):
    """Sample num_frames uniformly from a video. Returns list of (H,W,3) RGB arrays or None."""
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


def find_test_videos(dataset_dir: Path):
    """
    Returns (anomaly_videos, normal_videos).
    Each element is (Path, int_label) — label 1=anomaly, 0=normal.
    """
    anomaly_videos = []
    test_file = dataset_dir / "Anomaly_Test.txt"
    if test_file.exists():
        with open(test_file) as f:
            for line in f:
                rel = line.strip()
                if not rel:
                    continue
                full = dataset_dir / rel
                if full.exists():
                    anomaly_videos.append((full, 1))

    normal_dir = (dataset_dir / "Normal_Videos_for_Event_Recognition"
                  / "Normal_Videos_for_Event_Recognition")
    normal_videos = []
    if normal_dir.exists():
        for p in sorted(normal_dir.glob("*.mp4")):
            normal_videos.append((p, 0))

    return anomaly_videos, normal_videos


def threshold_sweep(scores: np.ndarray, labels: np.ndarray):
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


def run_eval(args):
    import torch
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading VideoMAE from {args.model} ...")
    processor = VideoMAEImageProcessor.from_pretrained(args.model)
    model     = VideoMAEForVideoClassification.from_pretrained(args.model)
    model.to(device).eval()

    anomaly_vids, normal_vids = find_test_videos(Path(args.dataset))
    print(f"Test set: {len(anomaly_vids)} anomaly + {len(normal_vids)} normal videos")

    rows    = []
    skipped = 0

    for i, (video_path, true_label) in enumerate(anomaly_vids + normal_vids):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(anomaly_vids)+len(normal_vids)}] processed ...")

        frames = load_video_frames(video_path)
        if frames is None:
            skipped += 1
            continue

        try:
            inputs = processor(list(frames), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            probs         = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            normal_prob   = float(probs[NORMAL_CLASS_ID])
            anomaly_score = 1.0 - normal_prob
            pred_class    = int(probs.argmax())
            rows.append({
                "video":         video_path.name,
                "true_label":    true_label,
                "anomaly_score": anomaly_score,
                "normal_prob":   normal_prob,
                "pred_class":    pred_class,
            })
        except Exception as e:
            print(f"  Error on {video_path.name}: {e}")
            skipped += 1

    if not rows:
        print("No videos processed. Check --dataset path.")
        return

    scores = np.array([r["anomaly_score"] for r in rows])
    labels = np.array([r["true_label"]    for r in rows])

    best_thresh = threshold_sweep(scores, labels)
    preds       = (scores >= best_thresh).astype(int)
    tp  = int(((preds == 1) & (labels == 1)).sum())
    fp  = int(((preds == 1) & (labels == 0)).sum())
    fn  = int(((preds == 0) & (labels == 1)).sum())
    tn  = int(((preds == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    acc  = (tp + tn) / len(rows)

    nm_sc = scores[labels == 0]
    an_sc = scores[labels == 1]

    metrics = {
        "best_threshold":     best_thresh,
        "precision":          float(prec),
        "recall":             float(rec),
        "f1":                 float(f1),
        "accuracy":           float(acc),
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

    print("\n=== VideoMAE HF Evaluation ===")
    print(f"Threshold τ*: {best_thresh:.4f}")
    print(f"Precision:    {prec:.4f}   Recall: {rec:.4f}   F1: {f1:.4f}   Acc: {acc:.4f}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Normal  score: {metrics['normal_mean_score']:.4f} ± {metrics['normal_std_score']:.4f}")
    print(f"Anomaly score: {metrics['anomaly_mean_score']:.4f} ± {metrics['anomaly_std_score']:.4f}")
    if f1 < 0.7:
        print("\n[WARNING] F1 < 0.70 — model may need fine-tuning. See notebooks/videomae_eval_fix.ipynb")

    out_dir = Path(args.results)
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / "videomae_hf_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved → {json_path}")

    csv_path = out_dir / "videomae_hf_eval.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video", "true_label", "anomaly_score", "normal_prob", "pred_class"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HF VideoMAE model on UCF-Crime")
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--model",   default=str(MODEL_DIR))
    parser.add_argument("--results", default=str(RESULTS_DIR))
    args = parser.parse_args()
    run_eval(args)
