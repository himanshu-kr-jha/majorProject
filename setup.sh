#!/usr/bin/env bash
# =============================================================================
# setup.sh — Gait-YOLO: one-command environment setup + dataset download + training
#
# Usage:
#   bash setup.sh                  # full setup + download + train
#   bash setup.sh --skip-train     # setup + download only (no training)
#   bash setup.sh --skip-download  # skip downloads (datasets already present)
#
# Requires : Python 3.10+, curl, unzip, git
# Optional : nvidia-smi (auto-detected for GPU PyTorch install)
#
# Kaggle auth: reads ~/.kaggle/kaggle.json automatically.
# If missing, you will be prompted for credentials.
# Get your token: https://www.kaggle.com/settings → API → Create New Token
#
# UCF-Crime slug: defaults to odins0n/ucf-crime-dataset
# Override:  UCF_KAGGLE_SLUG=owner/dataset-name bash setup.sh
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

info()  { echo -e "${BLUE}[INFO]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
die()   { echo -e "${RED}[ERR ]${RESET}  $*"; exit 1; }
step()  { echo -e "\n${BOLD}━━━  $*  ━━━${RESET}"; }

# ── Parse flags ────────────────────────────────────────────────────────────────
SKIP_TRAIN=0; SKIP_DOWNLOAD=0
for arg in "$@"; do
  case $arg in
    --skip-train)    SKIP_TRAIN=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    *) warn "Unknown flag: $arg" ;;
  esac
done

mkdir -p logs results
LOG="logs/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║          Gait-YOLO — One-Command Setup               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"
info "Full log: $LOG"

# =============================================================================
# 1. Prerequisites
# =============================================================================
step "1/8  Checking prerequisites"

need_cmd() {
  command -v "$1" &>/dev/null && ok "$1" || warn "$1 not found — $2"
}
need_cmd python3 "install from https://python.org"
need_cmd pip3    "run: python3 -m ensurepip --upgrade"
need_cmd curl    "sudo apt install curl / brew install curl"
need_cmd unzip   "sudo apt install unzip"

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
{ [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; } \
  && die "Python 3.10+ required (found $PY_VER)"
ok "Python $PY_VER"

GPU_AVAILABLE=0; CUDA_VER="0"
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "0")
  ok "GPU: $GPU_NAME  (CUDA $CUDA_VER)"
  GPU_AVAILABLE=1
else
  warn "nvidia-smi not found — CPU-only PyTorch will be installed (training will be slow)"
fi

# =============================================================================
# 2. Virtual environment
# =============================================================================
step "2/8  Python virtual environment"

if [ ! -d "venv" ]; then
  info "Creating venv/ ..."
  python3 -m venv venv
  ok "venv created"
else
  ok "venv/ exists — reusing"
fi

# shellcheck disable=SC1091
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel --quiet
ok "pip/setuptools upgraded"

# =============================================================================
# 3. PyTorch (CUDA-aware)
# =============================================================================
step "3/8  Installing PyTorch"

if python3 -c "import torch" &>/dev/null; then
  ok "PyTorch $(python3 -c 'import torch; print(torch.__version__)') already installed"
else
  CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
  if [ "$GPU_AVAILABLE" -eq 1 ] && [ "$CUDA_MAJOR" -ge 12 ]; then
    info "CUDA 12.x detected → installing torch+cu121 ..."
    pip install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu121 --quiet
  elif [ "$GPU_AVAILABLE" -eq 1 ] && [ "$CUDA_MAJOR" -eq 11 ]; then
    info "CUDA 11.x detected → installing torch+cu118 ..."
    pip install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu118 --quiet
  else
    info "No GPU / unknown CUDA → installing CPU torch ..."
    pip install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cpu --quiet
  fi
  ok "PyTorch installed"
fi

python3 - <<'PYEOF'
import torch
print(f"  version : {torch.__version__}")
print(f"  cuda    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  gpu     : {torch.cuda.get_device_name(0)}")
PYEOF

# =============================================================================
# 4. Project dependencies
# =============================================================================
step "4/8  Installing project dependencies"
pip install -r requirements.txt --quiet
ok "requirements.txt installed"

# =============================================================================
# 5. Kaggle credentials
# =============================================================================
step "5/8  Kaggle API credentials"

KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
if [ -f "$KAGGLE_JSON" ]; then
  KAGGLE_USERNAME=$(python3 -c "import json; print(json.load(open('$KAGGLE_JSON'))['username'])")
  KAGGLE_KEY=$(python3 -c "import json; print(json.load(open('$KAGGLE_JSON'))['key'])")
  ok "Loaded from $KAGGLE_JSON (user: $KAGGLE_USERNAME)"
else
  warn "~/.kaggle/kaggle.json not found"
  echo -e "${YELLOW}Get token at: https://www.kaggle.com/settings → API → Create New Token${RESET}"
  read -rp  "  Kaggle username : " KAGGLE_USERNAME
  read -rsp "  Kaggle API key  : " KAGGLE_KEY; echo ""
  mkdir -p "$HOME/.kaggle"
  printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > "$KAGGLE_JSON"
  chmod 600 "$KAGGLE_JSON"
  ok "Credentials saved to $KAGGLE_JSON"
fi

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
  -u "$KAGGLE_USERNAME:$KAGGLE_KEY" \
  "https://www.kaggle.com/api/v1/datasets/list?search=test&page=1" 2>/dev/null || echo "000")
[ "$HTTP_CODE" = "200" ] && ok "Kaggle credentials valid" \
  || warn "Kaggle API returned HTTP $HTTP_CODE — credentials may be wrong"

# =============================================================================
# 6. Dataset downloads
# =============================================================================
step "6/8  Downloading datasets"

if [ "$SKIP_DOWNLOAD" -eq 0 ]; then
  mkdir -p datasets/tmp

  # Helper: download + extract if target marker file is missing
  kaggle_dl() {
    local NAME="$1" SLUG="$2" DEST="$3" MARKER="$4"
    if [ -e "$MARKER" ]; then
      ok "$NAME already present — skipping"
      return
    fi
    local ZIP="datasets/tmp/${NAME}.zip"
    info "Downloading $NAME (slug: $SLUG) ..."
    if curl -L --progress-bar \
        -u "$KAGGLE_USERNAME:$KAGGLE_KEY" \
        -o "$ZIP" \
        "https://www.kaggle.com/api/v1/datasets/download/$SLUG"; then
      info "Extracting → $DEST ..."
      mkdir -p "$DEST"
      unzip -q "$ZIP" -d "$DEST"
      rm -f "$ZIP"
      ok "$NAME extracted"
    else
      warn "Download failed for $NAME — check slug or credentials. Skipping."
      rm -f "$ZIP"
    fi
  }

  kaggle_dl "guns-knives" \
    "kruthisb999/guns-and-knifes-detection-in-cctv-videos" \
    "datasets/guns-knives" \
    "datasets/guns-knives/combined_gunsnknifes/data.yaml"

  kaggle_dl "casia-b" \
    "trnquanghuyn/casia-b" \
    "datasets/casia-b" \
    "datasets/casia-b/output"

  # UCF-Crime: override slug with UCF_KAGGLE_SLUG env var if needed
  UCF_SLUG="${UCF_KAGGLE_SLUG:-odins0n/ucf-crime-dataset}"
  info "UCF-Crime slug: $UCF_SLUG"
  info "(Override: UCF_KAGGLE_SLUG=owner/slug bash setup.sh)"
  kaggle_dl "ucf-crime" \
    "$UCF_SLUG" \
    "datasets/anomalydetectiondatasetucf" \
    "datasets/anomalydetectiondatasetucf/Anomaly_Test.txt"

  ok "Dataset downloads complete"
else
  info "Skipping downloads (--skip-download flag)"
fi

# =============================================================================
# 7. Training pipeline
# =============================================================================
step "7/8  Training pipeline"

if [ "$SKIP_TRAIN" -eq 0 ]; then
  mkdir -p logs

  run_py() {
    local LABEL="$1"; shift
    local LOGF="logs/${LABEL}.log"
    info "Running: python3 $* → $LOGF"
    if python3 "$@" 2>&1 | tee "$LOGF"; then
      ok "$LABEL done"
    else
      warn "$LABEL had errors — check $LOGF"
    fi
  }

  # Step A: evaluate existing VideoMAE HF model (no training needed)
  run_py "videomae_hf_eval"  scripts/run_videomae_hf_eval.py

  # Step B: retrain gait autoencoder (normal-only — fixes Δμ=0.0002 bug)
  run_py "train_gait"        scripts/train_gait.py

  # Step C: fine-tune YOLO with yolov8s (upgrade from yolov8n)
  run_py "train_yolo"        scripts/train_yolo.py

  # =============================================================================
  # 8. Evaluation pipeline
  # =============================================================================
  step "8/8  Evaluation pipeline"

  # Patch eval scripts to use new checkpoints if they were produced
  if [ -f "models/casib-b/best_gait_v2.pth" ]; then
    sed -i "s|best_transformer_gait\.pth|best_gait_v2.pth|g" scripts/run_gait_eval.py
    ok "run_gait_eval.py → best_gait_v2.pth"
  fi
  if [ -f "models/knifes&pistol/best_v2.pt" ]; then
    sed -i "s|best\.pt\b|best_v2.pt|g" scripts/run_yolo_eval.py
    ok "run_yolo_eval.py → best_v2.pt"
  fi

  run_py "eval_yolo"      scripts/run_yolo_eval.py
  run_py "eval_gait"      scripts/run_gait_eval.py
  run_py "eval_videomae"  scripts/run_videomae_hf_eval.py
  run_py "eval_fusion"    scripts/run_full_real_eval.py
  run_py "ablation"       src/experiments/ablation.py --real

  # Print results summary
  echo ""
  info "Results summary:"
  python3 - <<'PYEOF'
import json, glob, os
for jf in sorted(glob.glob("results/*.json")):
    try:
        d = json.load(open(jf))
        f1    = d.get("f1", d.get("fusion_f1", "—"))
        map50 = d.get("mAP50", "")
        extra = f"  mAP50={map50:.4f}" if map50 else ""
        f1str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
        print(f"  {os.path.basename(jf):<42} F1={f1str}{extra}")
    except Exception:
        pass
PYEOF

else
  info "Skipping training (--skip-train)"
fi

# =============================================================================
echo ""
echo -e "${GREEN}${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║               Setup complete!                        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo "  Log        : $LOG"
echo "  Results    : results/"
echo "  Checkpoints: models/casib-b/best_gait_v2.pth"
echo "               models/knifes&pistol/best_v2.pt"
echo ""
echo "  To activate the environment later:"
echo "    source venv/bin/activate"
echo ""
