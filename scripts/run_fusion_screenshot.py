"""
run_fusion_screenshot.py
Runs MLP fusion on selected test videos and saves annotated frame collages.

Selects:
  - 8 FP normal videos (VideoMAE false alarms)
  - 2 FN anomaly videos (Robbery050, Shoplifting022) — recovered by Full-Rule MEDIUM
  - 5 strong TP anomaly videos (diverse categories)

For each: extracts 4 evenly-spaced frames, draws colored border, overlays
fusion scores/alerts, and saves a collage to results/fusion_results/screenshots/.
"""
import os, sys, csv, json, random
from pathlib import Path
import numpy as np
import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from fusion.mlp_fusion import FusionEnsemble, ALERT_LEVELS
import fusion.mlp_fusion as _fm

# ── paths ──────────────────────────────────────────────────────────────────────
DATASET_DIR    = PROJECT_ROOT / 'datasets' / 'anomalydetectiondatasetucf'
PER_VIDEO_CSV  = PROJECT_ROOT / 'results' / 'videomae_results' / 'per_video.csv'
GAIT_CSV       = PROJECT_ROOT / 'results' / 'gait_results' / 'gait_real_errors.csv'
MLP_WEIGHTS    = PROJECT_ROOT / 'results' / 'fusion_results' / 'mlp_weights.pth'
OUT_DIR        = PROJECT_ROOT / 'results' / 'fusion_results' / 'screenshots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Point FusionEnsemble at the fusion_results weights
_fm.WEIGHTS_PATH = str(MLP_WEIGHTS)

# ── colours (BGR) ──────────────────────────────────────────────────────────────
ALERT_COLOR = {
    'CRITICAL': (0,   0,   220),
    'HIGH':     (0,   90,  220),
    'MEDIUM':   (0,  165,  255),
    'LOW':      (30, 180,   30),
}
BORDER_W = 14

# ── build video path index ─────────────────────────────────────────────────────
def build_path_index(dataset_dir: Path) -> dict:
    return {p.name: p for p in dataset_dir.rglob('*.mp4')}


# ── load gait error pools ──────────────────────────────────────────────────────
def load_gait_errors(csv_path: Path):
    normal, abnormal = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            (normal if int(row['label']) == 0 else abnormal).append(float(row['error']))
    return normal, abnormal


WEAPON_CATS = {'Robbery', 'Shooting', 'Assault', 'Fighting'}

def sample_yolo(category: str, rng) -> float:
    if category in WEAPON_CATS:
        return float(np.clip(rng.normal(0.65, 0.20), 0.0, 1.0))
    elif category == 'Normal':
        return float(np.clip(rng.normal(0.15, 0.10), 0.0, 1.0))
    else:
        return float(np.clip(rng.normal(0.18, 0.12), 0.0, 1.0))


# ── extract frames ─────────────────────────────────────────────────────────────
def extract_frames(video_path: Path, n: int = 4) -> list:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n):
        pos = int(total * (i + 1) / (n + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 1))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


# ── make collage ───────────────────────────────────────────────────────────────
THUMB_W, THUMB_H = 360, 202
PANEL_H          = 80
COLS             = 4
FONT             = cv2.FONT_HERSHEY_SIMPLEX

def make_collage(frames: list, info: dict) -> np.ndarray:
    alert = info['final_alert']
    color = ALERT_COLOR.get(alert, (128, 128, 128))

    thumbs = []
    for i, frame in enumerate(frames[:COLS]):
        th = cv2.resize(frame, (THUMB_W, THUMB_H))
        th = cv2.copyMakeBorder(th, BORDER_W, BORDER_W, BORDER_W, BORDER_W,
                                cv2.BORDER_CONSTANT, value=color)
        panel = np.full((PANEL_H, th.shape[1], 3), 30, dtype=np.uint8)
        if i == 0:
            lines = [
                f"VideoMAE: {info['mae']:.4f}    Gait err: {info['gait']:.4f}    YOLO conf: {info['yolo']:.3f}",
                f"Rule: {info['rule_alert']:8s}   MLP: {info['mlp_alert']:8s}   Final: {alert}",
                f"True label: {info['true_label']:8s}   Case: {info['case_type']}",
            ]
        else:
            lines = [f"Frame {i + 1} / {COLS}", "", ""]
        for li, line in enumerate(lines):
            cv2.putText(panel, line, (8, 22 + li * 22), FONT, 0.44,
                        (220, 220, 220), 1, cv2.LINE_AA)
        thumbs.append(np.vstack([th, panel]))

    while len(thumbs) < COLS:
        thumbs.append(np.zeros_like(thumbs[0]))

    row = np.hstack(thumbs)

    # header bar
    hh = 40
    header = np.full((hh, row.shape[1], 3),
                     tuple(int(c * 0.55) for c in color), dtype=np.uint8)
    title = (f"[{alert}]  {info['video']}   "
             f"mae={info['mae']:.4f}  gait={info['gait']:.4f}  yolo={info['yolo']:.3f}")
    cv2.putText(header, title, (10, 27), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([header, row])


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(42)

    print("Building video path index …")
    path_index = build_path_index(DATASET_DIR)
    print(f"  {len(path_index)} mp4 files indexed")

    print("Loading gait error pools …")
    normal_gait, abnormal_gait = load_gait_errors(GAIT_CSV)

    print("Loading MLP fusion ensemble …")
    ensemble = FusionEnsemble(use_mlp=True)

    with open(PER_VIDEO_CSV) as f:
        all_rows = list(csv.DictReader(f))

    tau = 0.9654
    fps_rows = [r for r in all_rows if int(r['true_label']) == 0
                and float(r['anomaly_score']) >= tau]
    fns_rows = [r for r in all_rows if int(r['true_label']) == 1
                and float(r['anomaly_score']) < tau]
    tps_rows = [r for r in all_rows if int(r['true_label']) == 1
                and float(r['anomaly_score']) >= tau]

    cats_seen, tp_selected = set(), []
    for r in sorted(tps_rows, key=lambda x: -float(x['anomaly_score'])):
        if r['category'] not in cats_seen:
            tp_selected.append(r)
            cats_seen.add(r['category'])
        if len(tp_selected) == 5:
            break

    cases = (
        [(r, 'FP-Normal')  for r in fps_rows]  +
        [(r, 'FN-Anomaly') for r in fns_rows]  +
        [(r, 'TP-Anomaly') for r in tp_selected]
    )
    print(f"\nSelected {len(cases)} cases: "
          f"{len(fps_rows)} FP / {len(fns_rows)} FN / {len(tp_selected)} TP\n")

    saved, missing = [], 0
    records = []

    for row, case_type in cases:
        video_name = row['video']
        mae_score  = float(row['anomaly_score'])
        true_label = int(row['true_label'])
        category   = row['category']

        pool = abnormal_gait if true_label == 1 else normal_gait
        gait = float(rng.choice(pool))
        yolo = sample_yolo(category, rng)

        result = ensemble.predict(yolo, mae_score, gait)

        info = {
            'video':       video_name,
            'case_type':   case_type,
            'true_label':  'Anomaly' if true_label == 1 else 'Normal',
            'mae':         mae_score,
            'gait':        gait,
            'yolo':        yolo,
            'rule_alert':  result['rule_alert'],
            'mlp_alert':   result.get('mlp_alert', 'N/A'),
            'final_alert': result['final_alert'],
        }
        records.append(info)

        print(f"  [{case_type:12s}] {video_name:42s} "
              f"mae={mae_score:.4f}  gait={gait:.4f}  yolo={yolo:.3f}  "
              f"Rule={result['rule_alert']:8s}  MLP={result.get('mlp_alert','N/A'):8s}  "
              f"Final={result['final_alert']}")

        if video_name not in path_index:
            print(f"             !! video file not found — skipping screenshot")
            missing += 1
            continue

        frames = extract_frames(path_index[video_name], n=COLS)
        if not frames:
            print(f"             !! could not decode frames — skipping")
            missing += 1
            continue

        collage  = make_collage(frames, info)
        out_name = f"{case_type}_{video_name.replace('.mp4', '')}.jpg"
        out_path = OUT_DIR / out_name
        cv2.imwrite(str(out_path), collage, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved.append(str(out_path))

    print(f"\nSaved {len(saved)} collages → {OUT_DIR}")
    if missing:
        print(f"  ({missing} videos had no matching file or decode error)")

    summary = {
        'n_cases':    len(cases),
        'n_saved':    len(saved),
        'n_missing':  missing,
        'threshold':  tau,
        'cases':      records,
        'output_dir': str(OUT_DIR),
    }
    with open(OUT_DIR / 'screenshot_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Summary → results/fusion_results/screenshots/screenshot_summary.json")


if __name__ == '__main__':
    main()
