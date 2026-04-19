"""
MLP-based fusion layer for Gait-YOLO.
Replaces/augments the rule-based hierarchical fusion from PDF §6.5.

Input vector: [yolo_conf, action_prob, gait_error_normalized]  (3-dim)
Output:       4-class alert probabilities (Critical / High / Medium / Low)

Architecture: Linear(3→32) → ReLU → Dropout(0.3) → Linear(32→16) → ReLU → Linear(16→4) → Softmax
"""
import os
import numpy as np
import torch
import torch.nn as nn

ALERT_LEVELS = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']

# Thresholds — YOLO/action from PDF §6.5; gait recalibrated from real CASIA-B eval
_YOLO_CRITICAL_CONF = 0.60
_ACTION_HIGH_PROB   = 0.75
_ACTION_MED_LOW     = 0.40
_GAIT_MEDIUM_THRESH = 0.0491   # real F1-optimal threshold (nm_mean=0.0511, real score scale)
_GAIT_LOW_THRESH    = 0.0520   # nm_mean + 1.5*sigma = 0.0511 + 1.5*0.0006

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'mlp_fusion_weights.pth')


class FusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        """x: (B, 3) — [yolo_conf, action_prob, gait_norm]"""
        return torch.softmax(self.net(x), dim=-1)


def rule_based_label(yolo_conf: float, action_prob: float, gait_error: float) -> int:
    """
    Replicates PDF §6.5 hierarchical decision logic.
    Returns: 0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW
    """
    if yolo_conf >= _YOLO_CRITICAL_CONF:
        return 0
    if action_prob >= _ACTION_HIGH_PROB:
        return 1
    if _ACTION_MED_LOW <= action_prob < _ACTION_HIGH_PROB and gait_error > _GAIT_MEDIUM_THRESH:
        return 2
    if gait_error > _GAIT_LOW_THRESH:
        return 3
    return 3


def generate_bootstrap_dataset(n_samples: int = 10000, seed: int = 42):
    """Synthetic training data via rule-based labels on sampled inputs."""
    rng = np.random.default_rng(seed)

    yolo_conf   = rng.beta(2, 5, n_samples)
    action_prob = rng.beta(2, 3, n_samples)
    # Real CASIA-B scale: nm_mean=0.0511, ab_mean=0.0513, range ~[0.049, 0.057]
    gait_error  = np.clip(rng.normal(0.0511, 0.0008, n_samples), 0.048, 0.058)

    # Inject realistic positives to balance classes
    n_pos = n_samples // 8
    yolo_conf[-n_pos:]            = rng.uniform(0.61, 0.99, n_pos)
    action_prob[-2*n_pos:-n_pos]  = rng.uniform(0.76, 0.99, n_pos)
    gait_error[-3*n_pos:-2*n_pos] = rng.uniform(0.0515, 0.058, n_pos)  # above real threshold

    labels = np.array([
        rule_based_label(float(yolo_conf[i]), float(action_prob[i]), float(gait_error[i]))
        for i in range(n_samples)
    ], dtype=np.int64)

    gait_norm = gait_error / 0.06  # normalize to [0,1] using real empirical max ~0.06
    X = np.stack([yolo_conf, action_prob, gait_norm], axis=1).astype(np.float32)
    return X, labels


def train_fusion_mlp(n_epochs: int = 50, lr: float = 1e-3, verbose: bool = True) -> 'FusionMLP':
    X, y = generate_bootstrap_dataset()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader  = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    model     = FusionMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for xb, yb in loader:
            logits = model.net(xb)
            loss   = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  loss={total_loss/len(loader):.4f}")

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    if verbose:
        print(f"Saved MLP weights → {WEIGHTS_PATH}")
    return model


def load_fusion_mlp() -> 'FusionMLP':
    model = FusionMLP()
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
        model.eval()
    else:
        print("[FusionMLP] No saved weights — training from bootstrap data...")
        model = train_fusion_mlp(verbose=False)
        model.eval()
    return model


class FusionEnsemble:
    """
    Wraps rule-based and MLP fusion.
    Returns ensemble alert with per-modality explanation dict.
    """

    def __init__(self, use_mlp: bool = True):
        self.use_mlp = use_mlp
        self._mlp = load_fusion_mlp() if use_mlp else None

    def predict(self, yolo_conf: float, action_prob: float, gait_error: float) -> dict:
        rule_label = rule_based_label(yolo_conf, action_prob, gait_error)
        result = {
            'rule_alert':  ALERT_LEVELS[rule_label],
            'rule_level':  rule_label,
            'yolo_conf':   round(yolo_conf, 4),
            'action_prob': round(action_prob, 4),
            'gait_error':  round(gait_error, 4),
        }

        if self.use_mlp and self._mlp is not None:
            gait_norm = min(gait_error / 0.06, 1.0)  # real empirical max ~0.06
            x = torch.tensor([[yolo_conf, action_prob, gait_norm]], dtype=torch.float32)
            with torch.no_grad():
                probs = self._mlp(x).squeeze(0).numpy()
            mlp_label = int(np.argmax(probs))
            result.update({
                'mlp_alert': ALERT_LEVELS[mlp_label],
                'mlp_level': mlp_label,
                'mlp_probs': {ALERT_LEVELS[i]: round(float(probs[i]), 4) for i in range(4)},
            })
            # Ensemble: escalate to higher severity (lower index = higher severity)
            final_level = min(rule_label, mlp_label)
            result['final_alert'] = ALERT_LEVELS[final_level]
            result['final_level'] = final_level
        else:
            result['final_alert'] = result['rule_alert']
            result['final_level'] = rule_label

        return result


if __name__ == '__main__':
    print("Training MLP fusion on bootstrap data...")
    train_fusion_mlp(n_epochs=50, verbose=True)

    ensemble = FusionEnsemble(use_mlp=True)
    test_cases = [
        (0.85, 0.30, 0.400, "Weapon detected → CRITICAL"),
        (0.20, 0.82, 0.390, "Violent action → HIGH"),
        (0.15, 0.55, 0.460, "Suspicious + gait anomaly → MEDIUM"),
        (0.10, 0.20, 0.490, "Gait anomaly only → LOW"),
        (0.05, 0.15, 0.390, "All safe → LOW"),
    ]
    print("\n--- Ensemble Predictions ---")
    for yolo_c, act_p, gait_e, desc in test_cases:
        r = ensemble.predict(yolo_c, act_p, gait_e)
        print(f"  {desc}")
        print(f"    Rule={r['rule_alert']}  MLP={r.get('mlp_alert','N/A')}  Final={r['final_alert']}")
