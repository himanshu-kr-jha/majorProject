"""
Fine-tune YOLOv8s on the Guns & Knives dataset.

Uses yolov8s (11M params) instead of yolov8n (3.2M) for better accuracy
while remaining real-time capable.  Training on 7,417 images with strong
augmentation — mosaic, mixup, HSV jitter, degrees.

Auto-detects GPU vs CPU and adjusts batch/workers accordingly.
Can resume from an existing checkpoint via --resume.

Outputs:
  models/knifes&pistol/best_v2.pt   — best mAP50 checkpoint (copied after training)
  results/yolo_train_results/       — ultralytics run dir (metrics, curves, weights)

Usage:
    python3 scripts/train_yolo.py
    python3 scripts/train_yolo.py --base yolov8m.pt --epochs 150
    python3 scripts/train_yolo.py --resume results/yolo_train_results/weights/last.pt
"""

import sys
import shutil
import argparse
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "datasets" / "guns-knives" / "combined_gunsnknifes" / "data.yaml"
MODEL_OUT = ROOT / "models" / "knifes&pistol"

sys.path.insert(0, str(ROOT))


def get_env():
    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False
    return {
        "device":  0 if gpu else "cpu",
        "batch":   "auto" if gpu else 4,
        "workers": 8 if gpu else 2,
        "gpu":     gpu,
    }


def train(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed.  Run:  pip install ultralytics")
        sys.exit(1)

    env = get_env()
    print(f"Device:  {'GPU' if env['gpu'] else 'CPU'}")
    print(f"Batch:   {env['batch']}")
    print(f"Workers: {env['workers']}")
    print(f"Data:    {args.data}")
    print(f"Base:    {args.base}\n")

    if not Path(args.data).exists():
        print(f"ERROR: data.yaml not found at {args.data}")
        print("Check --data path or dataset location.")
        sys.exit(1)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        model = YOLO(args.base)
        model.train(
            data          = str(args.data),
            epochs        = args.epochs,
            patience      = args.patience,
            imgsz         = 640,
            batch         = env["batch"],
            device        = env["device"],
            workers       = env["workers"],
            project       = str(ROOT / "results"),
            name          = "yolo_train_results",
            exist_ok      = True,
            optimizer     = "AdamW",
            lr0           = 0.001,
            lrf           = 0.01,
            momentum      = 0.937,
            weight_decay  = 0.0005,
            warmup_epochs = 3.0,
            # augmentation
            mosaic        = 1.0,
            mixup         = 0.15,
            flipud        = 0.1,
            fliplr        = 0.5,
            degrees       = 10.0,
            translate     = 0.1,
            scale         = 0.5,
            hsv_h         = 0.015,
            hsv_s         = 0.7,
            hsv_v         = 0.4,
            copy_paste    = 0.1,
            plots         = True,
            save          = True,
            save_period   = 10,
            val           = True,
            verbose       = True,
        )

    # Copy best.pt to project model directory
    MODEL_OUT.mkdir(exist_ok=True)
    dest = MODEL_OUT / "best_v2.pt"
    for candidate in sorted((ROOT / "results").glob("yolo_train_results*/weights/best.pt")):
        shutil.copy2(candidate, dest)
        print(f"\nBest checkpoint → {dest}")
        break
    else:
        print("\nCould not find best.pt — check results/yolo_train_results/weights/")

    print("Next: update model path in scripts/run_yolo_eval.py to 'best_v2.pt', then run eval.")
    print("Expected improvement: mAP50 ≥ 0.78  (baseline 0.724 with yolov8n)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8s on Guns & Knives dataset")
    parser.add_argument("--data",     default=str(DATA_YAML), help="Path to data.yaml")
    parser.add_argument("--base",     default="yolov8s.pt",   help="Base model (yolov8n/s/m/l)")
    parser.add_argument("--epochs",   type=int, default=100,  help="Max training epochs")
    parser.add_argument("--patience", type=int, default=20,   help="Early-stop patience")
    parser.add_argument("--resume",   default=None,           help="Resume from last.pt path")
    args = parser.parse_args()
    train(args)
