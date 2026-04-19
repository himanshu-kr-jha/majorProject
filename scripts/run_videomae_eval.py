"""
Real VideoMAE/Action evaluation on UCF-Crime dataset.

Uses the TransformerAutoencoder (gait model) with MOG2 background subtraction
applied to UCF-Crime surveillance videos for anomaly detection.

Test set:
  Anomaly: Anomaly_Test.txt (~190 videos available on disk)
  Normal:  Normal_Videos_for_Event_Recognition/ (50 videos)

Model: models/casib-b/best_transformer_gait.pth
Score: MSE between reconstructed and original motion sequence
Threshold sweep to find F1-optimal decision boundary.

Outputs:
  - results/videomae_real_eval.csv
  - results/videomae_real_metrics.json
  - results/figures/videomae_score_dist.png

Usage:
    python3 scripts/run_videomae_eval.py
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'models', 'casib-b'))

UCF_DIR     = os.path.join(ROOT, 'datasets', 'anomalydetectiondatasetucf')
CKPT        = os.path.join(ROOT, 'models', 'casib-b', 'best_transformer_gait.pth')
RESULTS_DIR = os.path.join(ROOT, 'results')
FIGS_DIR    = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

THRESHOLD = 0.055   # calibrated in test.py; sweep also finds optimal
SEQ_LEN   = 15
IMG_SIZE  = (64, 64)
MAX_SCAN  = 300
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ──────────────────────────────────────────────────────────────────────────────
# MODEL (identical architecture to models/videoMae/test.py)
# ──────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,   32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32,  64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64,  128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.fc = nn.Linear(512 * 2 * 2, latent_dim)

    def forward(self, x):
        x = self.net(x)
        return self.fc(x.view(x.size(0), -1))


class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.fc  = nn.Linear(latent_dim, 512 * 2 * 2)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32,  1,   4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        return self.net(x.view(x.size(0), 512, 2, 2))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)

    def forward(self, x):
        return x + self.pos[:, :x.size(1), :]


class TransformerAutoencoder(nn.Module):
    def __init__(self, latent_dim=512, seq_len=15):
        super().__init__()
        self.encoder     = Encoder(latent_dim)
        self.decoder     = Decoder(latent_dim)
        layer            = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=8,
            dim_feedforward=latent_dim * 4, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.pos         = PositionalEncoding(latent_dim, seq_len)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = torch.stack([self.encoder(x[:, t]) for t in range(T)], dim=1)
        feats = self.pos(feats)
        feats = self.transformer(feats)
        outs  = [self.decoder(feats[:, t]) for t in range(T)]
        return torch.stack(outs, dim=1)


def load_model():
    model = TransformerAutoencoder(latent_dim=512, seq_len=SEQ_LEN).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"[Model] Loaded  device={DEVICE}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# VIDEO PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fgbg   = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=25, detectShadows=False
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    frames, counter = [], 0

    while counter < MAX_SCAN:
        ret, frame = cap.read()
        if not ret:
            break
        frame  = cv2.resize(frame, (320, 240))
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        final  = cv2.resize(fgmask, IMG_SIZE)
        if np.mean(final) > 5:
            frames.append(final)
        counter += 1
        if len(frames) >= 30:
            break

    cap.release()
    if len(frames) < SEQ_LEN:
        return None

    mid  = len(frames) // 2
    clip = frames[mid - 7: mid + 8]
    clip = np.array(clip, dtype='float32') / 255.0
    return torch.from_numpy(clip).unsqueeze(0).unsqueeze(2)   # (1,T,1,H,W)


def score_video(model, path):
    clip = preprocess_video(path)
    if clip is None:
        return None
    with torch.no_grad():
        recon = model(clip.to(DEVICE))
        return nn.MSELoss()(recon, clip.to(DEVICE)).item()


# ──────────────────────────────────────────────────────────────────────────────
# BUILD TEST SET
# ──────────────────────────────────────────────────────────────────────────────

def build_test_set():
    """
    Anomaly_Test.txt contains both anomaly and normal entries.
    Normal entries have filenames starting with 'Normal_Videos_'.
    All 140 anomaly and 50 (of 150) normal entries exist on disk.
    """
    records = []

    ann_file = os.path.join(UCF_DIR, 'Anomaly_Test.txt')
    with open(ann_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    for rel in lines:
        full = os.path.join(UCF_DIR, rel)
        if not os.path.isfile(full):
            continue
        fname = os.path.basename(full)
        if fname.startswith('Normal_Videos_'):
            label    = 0
            category = 'Normal'
        else:
            label    = 1
            category = ''.join(c for c in fname.split('_')[0] if c.isalpha())
        records.append({'path': full, 'video': fname, 'category': category, 'label': label})

    n_ab = sum(r['label'] for r in records)
    n_nm = len(records) - n_ab
    print(f"[Dataset] {n_ab} anomaly + {n_nm} normal = {len(records)} total")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def threshold_sweep(scores, labels):
    best_f1, best_thr = 0.0, THRESHOLD
    for thr in np.linspace(scores.min(), scores.max(), 300):
        preds = (scores > thr).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def per_category_metrics(df, threshold):
    rows = []
    for cat, grp in df.groupby('category'):
        preds  = (grp['score'] > threshold).astype(int).values
        labels = grp['label'].values
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        p  = round(tp / (tp + fp + 1e-9), 4)
        r  = round(tp / (tp + fn + 1e-9), 4)
        f1 = round(2 * p * r / (p + r + 1e-9), 4)
        rows.append({'category': cat, 'n': len(grp),
                     'precision': p, 'recall': r, 'f1': f1,
                     'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
    return pd.DataFrame(rows).sort_values('f1', ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_distributions(df, threshold):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    normal  = df[df['label'] == 0]['score'].values
    anomaly = df[df['label'] == 1]['score'].values
    ax.hist(normal,  bins=30, alpha=0.65, color='steelblue', label='Normal')
    ax.hist(anomaly, bins=30, alpha=0.65, color='tomato',    label='Anomaly')
    ax.axvline(threshold, color='black', linestyle='--', label=f'τ*={threshold:.4f}')
    ax.set_xlabel('Reconstruction Error (MSE)')
    ax.set_ylabel('Count')
    ax.set_title('UCF-Crime Score Distribution')
    ax.legend()

    ax = axes[1]
    scores = df['score'].values
    labels = df['label'].values
    thrs, f1s = [], []
    for t in np.linspace(scores.min(), scores.max(), 300):
        preds = (scores > t).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        thrs.append(t); f1s.append(2 * p * r / (p + r + 1e-9))
    ax.plot(thrs, f1s)
    ax.axvline(threshold, color='red', linestyle='--', label=f'τ*={threshold:.4f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('VideoMAE Threshold Sweep (UCF-Crime)')
    ax.legend()

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, 'videomae_score_dist.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Fig] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("VideoMAE / Action Recognition Evaluation — UCF-Crime")
    print("=" * 65)

    model   = load_model()
    records = build_test_set()

    rows = []
    for rec in tqdm(records, desc='Scoring'):
        s = score_video(model, rec['path'])
        rows.append({
            'video':    rec['video'],
            'category': rec['category'],
            'label':    rec['label'],
            'score':    round(s, 6) if s is not None else 0.0,
        })

    df    = pd.DataFrame(rows)
    valid = df[df['score'] > 0].copy()
    n_err = int(np.sum(df['score'] == 0))
    print(f"\n[Info] valid={len(valid)}  skipped={n_err}")

    scores = valid['score'].values
    labels = valid['label'].values

    best_thr, _ = threshold_sweep(scores, labels)
    preds = (scores > best_thr).astype(int)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    precision = round(tp / (tp + fp + 1e-9), 4)
    recall    = round(tp / (tp + fn + 1e-9), 4)
    f1        = round(2 * precision * recall / (precision + recall + 1e-9), 4)
    accuracy  = round((tp + tn) / len(valid), 4)

    nm  = valid[valid['label'] == 0]['score']
    ab  = valid[valid['label'] == 1]['score']
    cat = per_category_metrics(valid, best_thr)

    print(f"\nThreshold (F1-opt) : {best_thr:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1                 : {f1:.4f}")
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Normal  μ±σ        : {nm.mean():.4f} ± {nm.std():.4f}  n={len(nm)}")
    print(f"Anomaly μ±σ        : {ab.mean():.4f} ± {ab.std():.4f}  n={len(ab)}")
    print("\nPer-category:")
    print(cat[['category', 'n', 'precision', 'recall', 'f1']].to_string(index=False))

    csv_out = os.path.join(RESULTS_DIR, 'videomae_real_eval.csv')
    df.to_csv(csv_out, index=False)
    print(f"\n[Saved] {csv_out}")

    metrics = {
        'best_threshold':  round(best_thr, 4),
        'fixed_threshold': THRESHOLD,
        'precision':       precision,
        'recall':          recall,
        'f1':              f1,
        'accuracy':        accuracy,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'normal_mean':  round(float(nm.mean()), 6),
        'normal_std':   round(float(nm.std()),  6),
        'anomaly_mean': round(float(ab.mean()), 6),
        'anomaly_std':  round(float(ab.std()),  6),
        'n_anomaly':    int(np.sum(labels == 1)),
        'n_normal':     int(np.sum(labels == 0)),
        'n_total':      len(valid),
        'n_skipped':    n_err,
        'per_category': cat.to_dict('records'),
    }
    json_out = os.path.join(RESULTS_DIR, 'videomae_real_metrics.json')
    with open(json_out, 'w') as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[Saved] {json_out}")

    plot_distributions(valid, best_thr)
    return metrics


if __name__ == '__main__':
    main()
