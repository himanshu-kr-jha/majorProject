"""
Gait threshold optimizer — extends models/casib-b/evaluate.py.
Grid-searches thresholds on CASIA-B NM vs BG/CL splits,
adds temporal smoothing (window=8), saves results CSV + plots.

Usage:
    python src/gait/threshold_optimizer.py --casia_dir /path/to/Casia_b
    python src/gait/threshold_optimizer.py   # simulation mode (no dataset needed)
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

GAIT_MODULE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'casib-b')
sys.path.insert(0, GAIT_MODULE_DIR)

SRC_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, SRC_DIR)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(GAIT_MODULE_DIR, 'best_transformer_gait.pth')
SEQ_LEN = 15
IMAGE_SIZE = (64, 64)
BATCH = 16
SMOOTH_WINDOW = 8  # temporal smoothing from PDF §6.5

THRESHOLDS = [0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.4521, 0.46, 0.47, 0.48, 0.49, 0.50]


def smooth_errors(errors, window=SMOOTH_WINDOW):
    smoothed = []
    for i in range(len(errors)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(errors[start:i + 1]))
    return np.array(smoothed)


def get_errors(model, loader, device):
    errors = []
    mse_fn = nn.MSELoss(reduction='none')
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            recon[recon < 0.1] = 0.0
            mse = mse_fn(recon, batch).mean(dim=[1, 2, 3, 4])
            try:
                from train import ssim_loss_sequence
                ssim_scores = torch.tensor(
                    [ssim_loss_sequence(recon[i:i+1], batch[i:i+1]) for i in range(batch.size(0))]
                ).to(device)
                total = 0.3 * mse + 0.7 * ssim_scores
            except ImportError:
                total = mse
            errors.extend(total.cpu().numpy())
    return np.array(errors)


def run_threshold_sweep(nm_errors, ab_errors, use_smoothing=True):
    from src.evaluation.metrics import threshold_metrics

    if use_smoothing:
        nm_errors = smooth_errors(nm_errors)
        ab_errors = smooth_errors(ab_errors)

    all_errors = np.concatenate([nm_errors, ab_errors])
    all_labels = np.array([0] * len(nm_errors) + [1] * len(ab_errors))

    rows = []
    for t in THRESHOLDS:
        m = threshold_metrics(all_errors, all_labels, t)
        rows.append(m)
        print(f"  t={t:.4f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  Acc={m['accuracy']:.3f}")

    df = pd.DataFrame(rows)
    best = df.loc[df['f1'].idxmax()]
    print(f"\n>>> Best threshold: {best['threshold']:.4f}  "
          f"F1={best['f1']:.4f}  P={best['precision']:.4f}  R={best['recall']:.4f}")
    return df, nm_errors, ab_errors


def main():
    parser = argparse.ArgumentParser(description='Gait anomaly threshold optimizer')
    parser.add_argument('--casia_dir', type=str, default=None,
                        help='Path to CASIA-B dataset root directory')
    parser.add_argument('--no_smoothing', action='store_true',
                        help='Disable temporal smoothing')
    args = parser.parse_args()

    if args.casia_dir is None:
        print("No --casia_dir provided. Running in SIMULATION mode with synthetic data.")
        print("(Distributions match PDF report: Normal µ=0.4305±0.0108)\n")
        rng = np.random.default_rng(42)
        nm_errors = rng.normal(0.4305, 0.0108, 5000)
        ab_errors = np.concatenate([
            rng.normal(0.4227, 0.012, 3000),
            rng.normal(0.465, 0.015, 1000),
        ])
    else:
        try:
            from train import TransformerAutoencoder, build_index_map, GaitSequenceDataset
        except ImportError:
            print("ERROR: Cannot import train.py from models/casib-b/. Ensure it exists.")
            sys.exit(1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        model = TransformerAutoencoder().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded: {MODEL_PATH}")

        nm_index = build_index_map(args.casia_dir, SEQ_LEN, step=15, prefix='nm-')
        ab_index = (build_index_map(args.casia_dir, SEQ_LEN, step=15, prefix='bg-') +
                    build_index_map(args.casia_dir, SEQ_LEN, step=15, prefix='cl-'))

        nm_errors = get_errors(model,
                               DataLoader(__import__('train').GaitSequenceDataset(nm_index, SEQ_LEN, IMAGE_SIZE),
                                          batch_size=BATCH), device)
        ab_errors = get_errors(model,
                               DataLoader(__import__('train').GaitSequenceDataset(ab_index, SEQ_LEN, IMAGE_SIZE),
                                          batch_size=BATCH), device)

    print(f"Normal  errors: mean={np.mean(nm_errors):.4f} ± {np.std(nm_errors):.4f}  n={len(nm_errors)}")
    print(f"Abnormal errors: mean={np.mean(ab_errors):.4f} ± {np.std(ab_errors):.4f}  n={len(ab_errors)}")
    print(f"[PDF reference] Normal µ=0.4305±0.0108, threshold=0.4521\n")

    use_smooth = not args.no_smoothing
    print(f"Temporal smoothing (window={SMOOTH_WINDOW}): {'ON' if use_smooth else 'OFF'}\n")

    df, nm_s, ab_s = run_threshold_sweep(nm_errors, ab_errors, use_smoothing=use_smooth)

    csv_path = os.path.join(RESULTS_DIR, 'threshold_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    from src.evaluation.visualizations import plot_reconstruction_error_dist, plot_threshold_sweep
    best_t = float(df.loc[df['f1'].idxmax(), 'threshold'])
    plot_reconstruction_error_dist(nm_s, ab_s, best_t)
    plot_threshold_sweep(df)
    print("Plots saved to results/figures/")


if __name__ == '__main__':
    main()
