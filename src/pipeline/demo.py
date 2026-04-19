"""
Gait-YOLO integrated inference demo.
Runs all 3 branches matching PDF §6.6 asynchronous schedule:
  - YOLO:            every frame  (~12ms/frame)
  - VideoMAE + Gait: every 4th frame (~85ms/inference)

Usage:
    python src/pipeline/demo.py --video sample.mp4 --output annotated.mp4
    python src/pipeline/demo.py --help
"""
import os
import sys
import json
import time
import argparse
import cv2
import numpy as np
import torch

SRC_DIR  = os.path.join(os.path.dirname(__file__), '..', '..')
GAIT_DIR = os.path.join(SRC_DIR, 'models', 'casib-b')
YOLO_CKPT = os.path.join(SRC_DIR, 'models', 'knifes&pistol', 'best.pt')
MAE_DIR   = os.path.join(SRC_DIR, 'models', 'videoMae', 'best_model')

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, GAIT_DIR)

ALERT_COLORS = {
    'CRITICAL': (0,   0, 255),
    'HIGH':     (0, 100, 255),
    'MEDIUM':   (0, 200, 255),
    'LOW':      (0, 255,   0),
}
GAIT_THRESH   = 0.0491  # recalibrated from real CASIA-B eval (PDF §6.5 used 0.4521)
ACTION_STRIDE = 4       # PDF §6.6


# ── model loaders ─────────────────────────────────────────────────────────────

def load_yolo(ckpt):
    try:
        from ultralytics import YOLO
        m = YOLO(ckpt)
        print(f"[YOLO]     loaded {ckpt}")
        return m
    except Exception as e:
        print(f"[YOLO]     disabled ({e})")
        return None


def load_gait(gait_dir):
    try:
        from train import TransformerAutoencoder
        ckpt   = os.path.join(gait_dir, 'best_transformer_gait.pth')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = TransformerAutoencoder().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        print(f"[Gait]     loaded {ckpt}")
        return model, device
    except Exception as e:
        print(f"[Gait]     disabled ({e})")
        return None, None


def load_mae(model_dir):
    try:
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
        proc  = VideoMAEImageProcessor.from_pretrained(model_dir)
        model = VideoMAEForVideoClassification.from_pretrained(model_dir)
        model.eval()
        print(f"[VideoMAE] loaded {model_dir}")
        return model, proc
    except Exception as e:
        print(f"[VideoMAE] disabled ({e})")
        return None, None


# ── per-branch inference ───────────────────────────────────────────────────────

def infer_yolo(model, frame):
    if model is None:
        return 0.0, []
    res  = model(frame, verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return 0.0, []
    dets = [(boxes.xyxy[i].tolist(), float(boxes.conf[i]),
             res.names[int(boxes.cls[i])]) for i in range(len(boxes))]
    return float(boxes.conf.max()), dets


def infer_gait(model, device, frames_grey64):
    if model is None or len(frames_grey64) < 15:
        return 0.0
    clip = np.stack(frames_grey64[-15:]).astype(np.float32) / 255.0
    clip = torch.from_numpy(clip).unsqueeze(1).unsqueeze(0).to(device)  # (1,15,1,64,64)
    with torch.no_grad():
        recon = model(clip)
        recon[recon < 0.1] = 0.0
        mse = torch.nn.MSELoss()(recon, clip).item()
    return mse


def infer_action(model, proc, frames_rgb224):
    if model is None or len(frames_rgb224) < 16:
        return 0.0, 'Unknown'
    try:
        inp = proc(frames_rgb224[-16:], return_tensors='pt')
        with torch.no_grad():
            probs = torch.softmax(model(**inp).logits, dim=-1).squeeze()
        idx   = int(probs.argmax())
        label = model.config.id2label.get(idx, str(idx))
        return float(probs.max()), label
    except Exception:
        return 0.0, 'Unknown'


# ── overlay drawing ────────────────────────────────────────────────────────────

def draw_overlay(frame, alert, yolo_conf, action_prob, action_label, gait_error, dets):
    color = ALERT_COLORS.get(alert, (255, 255, 255))
    for (x1, y1, x2, y2), conf, cls_name in dets:
        if conf >= 0.60:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f'WEAPON: {cls_name}', (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (w - 235, 5), (w - 5, 110), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f'STATUS: {alert}',                     (w-230, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(frame, f'Obj: {yolo_conf:.2f}',                (w-230, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255,255,255), 1)
    cv2.putText(frame, f'Act: {action_label[:14]} {action_prob:.2f}', (w-230, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
    cv2.putText(frame, f'GaitErr: {gait_error:.3f}',           (w-230, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255,255,255), 1)
    cv2.putText(frame, f'Thr: {GAIT_THRESH}',                  (w-230, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
    return frame


# ── main loop ─────────────────────────────────────────────────────────────────

def run_demo(video_path: str, output_path: str = None):
    from src.fusion.mlp_fusion import FusionEnsemble

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        writer = cv2.VideoWriter(output_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (W, H))

    yolo_m                = load_yolo(YOLO_CKPT)
    gait_m, gait_dev      = load_gait(GAIT_DIR)
    mae_m,  mae_proc      = load_mae(MAE_DIR)
    ensemble              = FusionEnsemble(use_mlp=True)

    gait_buf   = []   # greyscale 64x64
    action_buf = []   # RGB 224x224

    yolo_conf    = 0.0
    action_prob  = 0.0
    action_label = 'Normal'
    gait_error   = 0.0
    dets         = []
    persist      = 0   # 5-frame persistence counter (PDF §6.4)

    alert_log = []
    frame_idx = 0
    t0        = time.time()

    print(f"\nProcessing: {video_path}  ({W}x{H} @ {fps_in:.1f} fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Branch 1 — YOLO every frame
        raw_conf, dets = infer_yolo(yolo_m, frame)
        persist = persist + 1 if raw_conf >= 0.60 else max(0, persist - 1)
        yolo_conf = raw_conf if persist >= 5 else 0.0

        # Branches 2 & 3 — every 4th frame
        if frame_idx % ACTION_STRIDE == 0:
            g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
            r = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),  (224, 224))
            gait_buf.append(g);   action_buf.append(r)
            if len(gait_buf)   > 30: gait_buf.pop(0)
            if len(action_buf) > 30: action_buf.pop(0)

            gait_error             = infer_gait(gait_m, gait_dev, gait_buf)
            action_prob, action_label = infer_action(mae_m, mae_proc, action_buf)

        fusion  = ensemble.predict(yolo_conf, action_prob, gait_error)
        alert   = fusion['final_alert']

        alert_log.append({
            'frame':       frame_idx,
            'timestamp':   round(frame_idx / fps_in, 3),
            'yolo_conf':   round(yolo_conf, 4),
            'action_prob': round(action_prob, 4),
            'gait_error':  round(gait_error, 4),
            'final_alert': alert,
            'rule_alert':  fusion['rule_alert'],
            'mlp_alert':   fusion.get('mlp_alert', 'N/A'),
        })

        annotated = draw_overlay(frame.copy(), alert, yolo_conf,
                                 action_prob, action_label, gait_error, dets)
        if writer:
            writer.write(annotated)

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"  frame {frame_idx:5d}  alert={alert:<8}  "
                  f"fps={frame_idx/elapsed:.1f}")

    cap.release()
    if writer:
        writer.release()
        print(f"\nAnnotated video: {output_path}")

    log_path = os.path.splitext(video_path)[0] + '_alerts_log.json'
    with open(log_path, 'w') as f:
        json.dump(alert_log, f, indent=2)

    elapsed = time.time() - t0
    print(f"Alert log:       {log_path}")
    print(f"Done — {frame_idx} frames in {elapsed:.1f}s  "
          f"({frame_idx/elapsed:.1f} fps effective)")


def main():
    parser = argparse.ArgumentParser(description='Gait-YOLO integrated inference demo')
    parser.add_argument('--video',  required=True, help='Input video path (.mp4 / .avi)')
    parser.add_argument('--output', default=None,  help='Output annotated video path')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: video not found: {args.video}")
        sys.exit(1)

    run_demo(args.video, args.output)


if __name__ == '__main__':
    main()
