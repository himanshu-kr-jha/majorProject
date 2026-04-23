"""
Real gait evaluation on CASIA-B silhouette dataset.

Label convention (from PDF §3.3):
  nm-* (normal walking)  → label 0 (NORMAL)
  bg-* (bag carrying)    → label 1 (ABNORMAL)
  cl-* (coat wearing)    → label 1 (ABNORMAL)

Anomaly score: 0.3 * MSE + 0.7 * SSIM_loss  (per sequence of 15 frames)

Outputs:
  - results/gait_real_errors.csv
  - results/gait_real_metrics.json
  - results/figures/gait_error_dist.png
  - results/figures/threshold_sweep.png

Usage:
    python3 scripts/run_gait_eval.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd

ROOT     = os.path.join(os.path.dirname(__file__), '..')
GAIT_DIR = os.path.join(ROOT, 'models', 'casib-b')
DATA_DIR = os.path.join(ROOT, 'datasets', 'casia-b')
CKPT     = os.path.join(GAIT_DIR, 'best_transformer_gait.pth')

RESULTS_DIR = os.path.join(ROOT, 'results')
FIGS_DIR    = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
sys.path.insert(0, GAIT_DIR)

SEQ_LEN   = 15
IMG_SIZE  = (64, 64)
STEP      = 1       # use step=1 then subsample below
BATCH     = 32
MAX_SEQS  = 300     # cap per class: 300 nm + 150 bg + 150 cl = 600 total, ~2 min


def load_model():
    import torch
    from train import TransformerAutoencoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = TransformerAutoencoder(latent_dim=128, seq_len=SEQ_LEN).to(device)
    if os.path.exists(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
        print(f"[Gait] Loaded checkpoint: {CKPT}")
    else:
        print(f"[Gait] WARNING: checkpoint not found at {CKPT} — using random weights")
        print("       Results will not be meaningful. Obtain best_transformer_gait.pth.")
    model.eval()
    return model, device


def score_sequences(model, device, seq_list):
    import torch
    from train import GaitSequenceDataset, ssim_loss_sequence
    from torch.utils.data import DataLoader

    ds     = GaitSequenceDataset(seq_list, SEQ_LEN, IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
    scores = []
    mse_fn = torch.nn.MSELoss(reduction='mean')

    with torch.no_grad():
        for clips in loader:
            clips = clips.to(device)
            recon = model(clips)
            for b in range(clips.size(0)):
                c   = clips[b:b+1]
                r   = recon[b:b+1]
                mse  = float(mse_fn(r, c).item())
                ssim = ssim_loss_sequence(r, c)
                scores.append(0.3 * mse + 0.7 * ssim)
    return scores


def apply_temporal_smoothing(scores, window=8):
    out = []
    for i in range(len(scores)):
        lo  = max(0, i - window // 2)
        hi  = min(len(scores), i + window // 2 + 1)
        out.append(float(np.mean(scores[lo:hi])))
    return out


def run_gait_eval():
    from train import build_index_map

    print(f"[Gait] Scanning CASIA-B: {DATA_DIR}")
    rng_sub = np.random.default_rng(99)

    nm_index = build_index_map(DATA_DIR, SEQ_LEN, step=STEP, prefix='nm-')
    bg_index = build_index_map(DATA_DIR, SEQ_LEN, step=STEP, prefix='bg-')
    cl_index = build_index_map(DATA_DIR, SEQ_LEN, step=STEP, prefix='cl-')

    # Subsample to keep eval time manageable on CPU
    ab_max = MAX_SEQS // 2
    if len(nm_index) > MAX_SEQS:
        nm_index = [nm_index[i] for i in rng_sub.choice(len(nm_index), MAX_SEQS, replace=False)]
    if len(bg_index) > ab_max:
        bg_index = [bg_index[i] for i in rng_sub.choice(len(bg_index), ab_max, replace=False)]
    if len(cl_index) > ab_max:
        cl_index = [cl_index[i] for i in rng_sub.choice(len(cl_index), ab_max, replace=False)]

    print(f"  nm (normal)  : {len(nm_index)} sequences (sampled)")
    print(f"  bg (bag)     : {len(bg_index)} sequences (sampled)")
    print(f"  cl (coat)    : {len(cl_index)} sequences (sampled)")

    model, device = load_model()

    print("[Gait] Scoring normal (nm)...")
    nm_scores = apply_temporal_smoothing(score_sequences(model, device, nm_index))

    print("[Gait] Scoring abnormal (bg)...")
    bg_scores = apply_temporal_smoothing(score_sequences(model, device, bg_index))

    print("[Gait] Scoring abnormal (cl)...")
    cl_scores = apply_temporal_smoothing(score_sequences(model, device, cl_index))

    def parse_meta(seq_paths, condition):
        rows = []
        for paths in seq_paths:
            fname = os.path.basename(paths[0])
            parts = fname.split('-')
            subj  = parts[0] if len(parts) > 0 else '?'
            angle = os.path.basename(os.path.dirname(paths[0]))
            rows.append({'subject': subj, 'condition': condition,
                         'angle': angle, 'seq_start': fname})
        return rows

    metas      = (parse_meta(nm_index, 'nm') +
                  parse_meta(bg_index, 'bg') +
                  parse_meta(cl_index, 'cl'))
    all_scores = nm_scores + bg_scores + cl_scores
    all_labels = [0] * len(nm_scores) + [1] * len(bg_scores) + [1] * len(cl_scores)

    rows = []
    for meta, score, label in zip(metas, all_scores, all_labels):
        rows.append({**meta, 'error': round(score, 6), 'label': label})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, 'gait_real_errors.csv')
    df.to_csv(csv_path, index=False)
    print(f"[Gait] Saved errors: {csv_path}  ({len(df)} sequences)")

    errors = np.array(all_scores)
    labels = np.array(all_labels)
    nm_err = errors[labels == 0]
    ab_err = errors[labels == 1]
    nm_mean, nm_std = float(nm_err.mean()), float(nm_err.std())
    print(f"\n  Normal   μ={nm_mean:.4f}  σ={nm_std:.4f}")
    print(f"  Abnormal μ={ab_err.mean():.4f}  σ={ab_err.std():.4f}")

    best_f1  = -1
    # Sweep the actual score range (not the hardcoded PDF range)
    score_min = float(errors.min())
    score_max = float(errors.max())
    best_thr  = score_min + (score_max - score_min) * 0.5
    thr_rows  = []
    for thr in np.linspace(score_min, score_max, 49):
        preds = (errors > thr).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        acc  = (tp + tn) / (len(labels) + 1e-8)
        thr_rows.append({'threshold': round(thr, 4), 'f1': round(f1, 4),
                         'precision': round(prec, 4), 'recall': round(rec, 4),
                         'accuracy':  round(acc,  4)})
        if f1 > best_f1:
            best_f1  = f1
            best_thr = thr

    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(os.path.join(RESULTS_DIR, 'gait_threshold_sweep.csv'), index=False)

    preds_opt = (errors > best_thr).astype(int)
    tp = int(np.sum((preds_opt == 1) & (labels == 1)))
    fp = int(np.sum((preds_opt == 1) & (labels == 0)))
    fn = int(np.sum((preds_opt == 0) & (labels == 1)))
    tn = int(np.sum((preds_opt == 0) & (labels == 0)))
    prec_opt = round(tp / (tp + fp + 1e-8), 4)
    rec_opt  = round(tp / (tp + fn + 1e-8), 4)
    f1_opt   = round(2 * prec_opt * rec_opt / (prec_opt + rec_opt + 1e-8), 4)
    acc_opt  = round((tp + tn) / len(labels), 4)

    metrics = {
        'best_threshold': round(float(best_thr), 4),
        'precision':  prec_opt,
        'recall':     rec_opt,
        'f1':         f1_opt,
        'accuracy':   acc_opt,
        'nm_mean':    round(nm_mean, 4),
        'nm_std':     round(nm_std,  4),
        'n_normal':   int(np.sum(labels == 0)),
        'n_abnormal': int(np.sum(labels == 1)),
    }
    json_path = os.path.join(RESULTS_DIR, 'gait_real_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 55)
    print("GAIT REAL EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Sequences       : {len(df)}  "
          f"(normal={metrics['n_normal']}, abnormal={metrics['n_abnormal']})")
    print(f"  Best threshold  : {metrics['best_threshold']:.4f}  (PDF ref: 0.4521, score scale differs)")
    print(f"  Precision       : {prec_opt:.4f}")
    print(f"  Recall          : {rec_opt:.4f}")
    print(f"  F1              : {f1_opt:.4f}")
    print(f"  Accuracy        : {acc_opt:.4f}")
    print(f"  Normal μ±σ      : {nm_mean:.4f} ± {nm_std:.4f}  (PDF: 0.4305±0.0108)")
    print("=" * 55)

    _plot_error_dist(nm_err, ab_err, best_thr)
    _plot_threshold_sweep(thr_df)
    return metrics


def _plot_error_dist(nm_err, ab_err, threshold):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(
        min(nm_err.min(), ab_err.min()) - 0.005,
        max(nm_err.max(), ab_err.max()) + 0.005,
        50
    )
    ax.hist(nm_err, bins=bins, alpha=0.6, color='steelblue',  label='Normal (nm)')
    ax.hist(ab_err, bins=bins, alpha=0.6, color='darkorange', label='Abnormal (bg+cl)')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold={threshold:.4f}')
    ax.set_xlabel('Anomaly Score (0.3·MSE + 0.7·SSIM loss)')
    ax.set_ylabel('Frequency')
    ax.set_title('Gait Reconstruction Error Distribution — CASIA-B (Real)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(FIGS_DIR, 'gait_error_dist.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Gait] Error dist: {out}")


def _plot_threshold_sweep(thr_df):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thr_df['threshold'], thr_df['f1'],        label='F1',        linewidth=2)
    ax.plot(thr_df['threshold'], thr_df['precision'], label='Precision', linestyle='--')
    ax.plot(thr_df['threshold'], thr_df['recall'],    label='Recall',    linestyle=':')
    best_thr = float(thr_df.loc[thr_df['f1'].idxmax(), 'threshold'])
    ax.axvline(best_thr, color='red', linestyle='--', linewidth=1.5,
               label=f'Best={best_thr:.4f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Gait Threshold Sweep — CASIA-B (Real)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(FIGS_DIR, 'threshold_sweep.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Gait] Threshold sweep: {out}")


if __name__ == '__main__':
    run_gait_eval()
