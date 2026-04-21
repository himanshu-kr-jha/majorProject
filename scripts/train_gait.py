"""
Train the TransformerAutoencoder on CASIA-B (normal gait only).

CRITICAL FIX: The existing best_transformer_gait.pth was trained on mixed
normal+abnormal data → near-zero separation (Δμ=0.0002).  This script trains
ONLY on nm-01..nm-04 so the model learns "what normal gait looks like".
Abnormal sequences (bg-*, cl-*) must NEVER appear during training or validation.

Dataset split:
  Train : nm-01..nm-04 (all 124 subjects, all 11 angles) ≈ 5,456 sequences
  Val   : nm-05..nm-06 (all subjects, all angles)        ≈ 2,728 sequences
  Test  : bg-01, bg-02, cl-01, cl-02  (evaluated by scripts/run_gait_eval.py)

Auto-detects GPU vs CPU and adjusts batch/workers accordingly.

Outputs:
  models/casib-b/best_gait_v2.pth    — best val-loss checkpoint
  results/gait_train_log.csv         — epoch, train_loss, val_loss, error_gap

Usage:
    python3 scripts/train_gait.py
    python3 scripts/train_gait.py --epochs 80 --dataset /path/to/casia-b
"""

import os
import sys
import csv
import argparse
import glob as _glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT     = Path(__file__).resolve().parent.parent
GAIT_DIR = ROOT / "models" / "casib-b"
DATA_DIR = ROOT / "datasets" / "casia-b"
RESULTS  = ROOT / "results"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GAIT_DIR))

from train import TransformerAutoencoder, GaitSequenceDataset, _gauss_kernel

# ── Environment-adaptive constants ───────────────────────────────────────────

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32 if DEVICE == "cuda" else 8
NUM_WORKERS = 4  if DEVICE == "cuda" else 0
PIN_MEMORY  = DEVICE == "cuda"

SEQ_LEN    = 15
IMG_SIZE   = (64, 64)
LATENT_DIM = 512

TRAIN_CONDS = {"nm-01", "nm-02", "nm-03", "nm-04"}
VAL_CONDS   = {"nm-05", "nm-06"}
ABN_CONDS   = {"bg-01", "bg-02", "cl-01", "cl-02"}


# ── Differentiable SSIM loss ─────────────────────────────────────────────────

def ssim_loss_tensor(recon: torch.Tensor, original: torch.Tensor,
                     window_size: int = 11) -> torch.Tensor:
    """Returns mean (1-SSIM) as a differentiable tensor."""
    B, T, C, H, W = recon.shape
    kernel  = _gauss_kernel(window_size, sigma=1.5, device=recon.device).expand(C, 1, -1, -1)
    padding = window_size // 2
    C1, C2  = 0.01 ** 2, 0.03 ** 2
    losses  = []
    for t in range(T):
        r, o   = recon[:, t], original[:, t]
        mu_r   = F.conv2d(r,   kernel, padding=padding, groups=C)
        mu_o   = F.conv2d(o,   kernel, padding=padding, groups=C)
        sig_r  = F.conv2d(r*r, kernel, padding=padding, groups=C) - mu_r**2
        sig_o  = F.conv2d(o*o, kernel, padding=padding, groups=C) - mu_o**2
        sig_ro = F.conv2d(r*o, kernel, padding=padding, groups=C) - mu_r*mu_o
        ssim   = ((2*mu_r*mu_o + C1)*(2*sig_ro + C2)) / (
                  (mu_r**2 + mu_o**2 + C1)*(sig_r + sig_o + C2) + 1e-8)
        losses.append(1.0 - ssim.mean())
    return torch.stack(losses).mean()


def combined_loss(recon: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    return 0.5 * F.mse_loss(recon, original) + 0.5 * ssim_loss_tensor(recon, original)


# ── Dataset builder ──────────────────────────────────────────────────────────

def build_split_index(data_dir: Path, conditions: set, seq_len: int, step: int = 1):
    """Build list-of-frame-path-lists for a specific set of condition names."""
    output_dir = data_dir / "output" if (data_dir / "output").is_dir() else data_dir
    index = []
    for subj in sorted(os.listdir(output_dir)):
        subj_path = output_dir / subj
        if not subj_path.is_dir():
            continue
        for cond in sorted(os.listdir(subj_path)):
            if cond not in conditions:
                continue
            cond_path = subj_path / cond
            if not cond_path.is_dir():
                continue
            for angle in sorted(os.listdir(cond_path)):
                angle_path = cond_path / angle
                if not angle_path.is_dir():
                    continue
                frames = sorted(_glob.glob(str(angle_path / "*.png")))
                if len(frames) < seq_len:
                    continue
                for start in range(0, len(frames) - seq_len + 1, step):
                    index.append(frames[start:start + seq_len])
    return index


# ── Mean reconstruction error ────────────────────────────────────────────────

@torch.no_grad()
def mean_recon_error(model, seq_list, max_seqs=400):
    sample = seq_list[:max_seqs]
    ds     = GaitSequenceDataset(sample, SEQ_LEN, IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    errs   = []
    for clips in loader:
        clips = clips.to(DEVICE)
        recon = model(clips)
        for b in range(clips.size(0)):
            errs.append(F.mse_loss(recon[b], clips[b]).item())
    return float(np.mean(errs)) if errs else 0.0


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    print(f"Device:      {DEVICE}")
    print(f"Batch size:  {BATCH_SIZE}   Workers: {NUM_WORKERS}")

    data_dir = Path(args.dataset)
    print("\nBuilding dataset indices ...")
    train_idx = build_split_index(data_dir, TRAIN_CONDS, SEQ_LEN, step=1)
    val_idx   = build_split_index(data_dir, VAL_CONDS,   SEQ_LEN, step=2)
    abn_idx   = build_split_index(data_dir, ABN_CONDS,   SEQ_LEN, step=4)
    print(f"  Train (nm-01..04): {len(train_idx)} sequences")
    print(f"  Val   (nm-05..06): {len(val_idx)} sequences")
    print(f"  Abn monitor:       {min(len(abn_idx), 400)} sequences (not used in training)")

    if not train_idx:
        raise RuntimeError(f"No training sequences in {data_dir}. Check --dataset path.")

    train_dl = DataLoader(
        GaitSequenceDataset(train_idx, SEQ_LEN, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
    )
    val_dl = DataLoader(
        GaitSequenceDataset(val_idx, SEQ_LEN, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    model = TransformerAutoencoder(latent_dim=LATENT_DIM, seq_len=SEQ_LEN).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_val   = float("inf")
    patience_c = 0
    log_rows   = []
    ckpt_path  = GAIT_DIR / "best_gait_v2.pth"

    print(f"\nStarting training — max {args.epochs} epochs, patience={args.patience}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # train
        model.train()
        tr_losses = []
        for clips in train_dl:
            clips = clips.to(DEVICE)
            loss  = combined_loss(model(clips), clips)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())
        tr_loss = float(np.mean(tr_losses))

        # val
        model.eval()
        vl_losses = []
        with torch.no_grad():
            for clips in val_dl:
                clips = clips.to(DEVICE)
                vl_losses.append(combined_loss(model(clips), clips).item())
        vl_loss = float(np.mean(vl_losses))

        sched.step()

        # gap monitoring every 5 epochs
        gap = None
        gap_str = ""
        if epoch % 5 == 0 and abn_idx:
            model.eval()
            nm_err  = mean_recon_error(model, val_idx)
            abn_err = mean_recon_error(model, abn_idx)
            gap     = abn_err - nm_err
            gap_str = f"   gap={gap:+.5f}  (nm={nm_err:.5f}, abn={abn_err:.5f})"
            if gap < 0.005:
                gap_str += "  ← still low"
            else:
                gap_str += "  ✓ separation improving"

        log_rows.append({
            "epoch": epoch, "train_loss": round(tr_loss, 6),
            "val_loss": round(vl_loss, 6), "error_gap": round(gap, 6) if gap else "",
        })

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={tr_loss:.5f}  val={vl_loss:.5f}{gap_str}  [{elapsed:.1f}s]")

        if vl_loss < best_val:
            best_val   = vl_loss
            patience_c = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved {ckpt_path.name}  (best val={vl_loss:.5f})")
        else:
            patience_c += 1
            if patience_c >= args.patience:
                print(f"\nEarly stop at epoch {epoch} (no val improvement for {args.patience} epochs)")
                break

    RESULTS.mkdir(exist_ok=True)
    log_path = RESULTS / "gait_train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "error_gap"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nDone. Best val loss: {best_val:.5f}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Train log  : {log_path}")
    print("\nNext: update CKPT in scripts/run_gait_eval.py to 'best_gait_v2.pth', then run eval.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gait autoencoder (normal-only CASIA-B)")
    parser.add_argument("--dataset",  default=str(DATA_DIR))
    parser.add_argument("--epochs",   type=int,   default=80)
    parser.add_argument("--patience", type=int,   default=15)
    parser.add_argument("--lr",       type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
