"""
CASIA-B gait autoencoder — model definitions.
Provides classes imported by train_gait.py, run_gait_eval.py, threshold_optimizer.py, demo.py:
  - TransformerAutoencoder  (CNN encoder + Transformer + CNN decoder)
  - GaitSequenceDataset
  - build_index_map
  - _gauss_kernel            (kept for eval scripts that use SSIM scoring)
  - ssim_loss_sequence       (kept for eval scripts)

Architecture: latent_dim default reduced 512 -> 128.
Smaller bottleneck forces the model to encode only essential normal gait patterns,
increasing reconstruction error on abnormal sequences (bag/coat) vs normal walking.
"""
import os
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ── Model Architecture ────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,   32, 4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32,  64, 4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256,512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.fc = nn.Linear(512 * 2 * 2, latent_dim)

    def forward(self, x):
        x = self.net(x)
        return self.fc(x.view(x.size(0), -1))


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 2 * 2)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d( 64,  32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d( 32,   1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x).view(x.size(0), 512, 2, 2)
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)

    def forward(self, x):
        return x + self.pos[:, :x.size(1), :]


class TransformerAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, seq_len=15):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=4,
            dim_feedforward=512, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.pos = PositionalEncoding(latent_dim, seq_len)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = self.encoder(x.view(B * T, C, H, W)).view(B, T, -1)
        feats = self.pos(feats)
        feats = self.transformer(feats)
        return self.decoder(feats.reshape(B * T, -1)).view(B, T, C, H, W)


# ── SSIM utilities (kept for eval scripts) ────────────────────────────────────

def _gauss_kernel(size=11, sigma=1.5, device='cpu'):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    k = g[:, None] * g[None, :]
    return k.unsqueeze(0).unsqueeze(0)


def ssim_loss_sequence(recon, original, window_size=11):
    """Mean (1-SSIM) over a (B, T, C, H, W) tensor pair. Returns a Python float."""
    B, T, C, H, W = recon.shape
    device  = recon.device
    kernel  = _gauss_kernel(window_size, sigma=1.5, device=device).expand(C, 1, -1, -1)
    padding = window_size // 2
    C1, C2  = 0.01 ** 2, 0.03 ** 2
    losses  = []
    for t in range(T):
        r = recon[:, t].float()
        o = original[:, t].float()
        mu_r   = F.conv2d(r,   kernel, padding=padding, groups=C)
        mu_o   = F.conv2d(o,   kernel, padding=padding, groups=C)
        sig_r  = F.conv2d(r*r, kernel, padding=padding, groups=C) - mu_r**2
        sig_o  = F.conv2d(o*o, kernel, padding=padding, groups=C) - mu_o**2
        sig_ro = F.conv2d(r*o, kernel, padding=padding, groups=C) - mu_r*mu_o
        ssim_map = ((2*mu_r*mu_o + C1)*(2*sig_ro + C2)) / \
                   ((mu_r**2 + mu_o**2 + C1)*(sig_r + sig_o + C2) + 1e-8)
        losses.append(1.0 - ssim_map.mean())
    return float(torch.stack(losses).mean().item())


# ── CASIA-B Dataset ───────────────────────────────────────────────────────────

def build_index_map(root_dir, seq_len, step=1, prefix=None):
    """
    Scan a CASIA-B directory and return sequence start-point lists.

    Layout: <root_dir>/output/{subj:03d}/{cond}/{angle:03d}/{frames}.png

    Args:
        root_dir : dataset root (contains 'output/' sub-dir, or IS the output dir)
        seq_len  : frames per sequence (15 for this project)
        step     : stride between sequence starts
        prefix   : condition prefix filter e.g. "nm-", "bg-", "cl-"
    Returns:
        List of lists — each inner list is seq_len sorted PNG paths.
    """
    output_dir = os.path.join(root_dir, 'output') \
        if os.path.isdir(os.path.join(root_dir, 'output')) else root_dir

    index = []
    for subj in sorted(os.listdir(output_dir)):
        subj_path = os.path.join(output_dir, subj)
        if not os.path.isdir(subj_path):
            continue
        for cond in sorted(os.listdir(subj_path)):
            if prefix and not cond.startswith(prefix):
                continue
            cond_path = os.path.join(subj_path, cond)
            if not os.path.isdir(cond_path):
                continue
            for angle in sorted(os.listdir(cond_path)):
                angle_path = os.path.join(cond_path, angle)
                if not os.path.isdir(angle_path):
                    continue
                frames = sorted(glob.glob(os.path.join(angle_path, '*.png')))
                if len(frames) < seq_len:
                    continue
                for start in range(0, len(frames) - seq_len + 1, step):
                    index.append(frames[start:start + seq_len])
    return index


class GaitSequenceDataset(Dataset):
    """
    PyTorch Dataset for CASIA-B silhouette sequences.
    Each item: float32 tensor (seq_len, 1, H, W) in [0, 1].
    """

    def __init__(self, index, seq_len, image_size=(64, 64)):
        self.index      = index
        self.seq_len    = seq_len
        self.image_size = image_size

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        frames = []
        for p in self.index[idx]:
            try:
                img = Image.open(p).convert('L')
                img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
                frames.append(np.array(img, dtype=np.float32) / 255.0)
            except Exception:
                frames.append(np.zeros(self.image_size, dtype=np.float32))
        clip = np.stack(frames, axis=0)[:, np.newaxis, :, :]  # (T,1,H,W)
        return torch.from_numpy(clip)
