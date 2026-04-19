"""
Real YOLO evaluation on Guns & Knives test set.

Inputs:
  - models/knifes&pistol/best.pt
  - datasets/guns-knives/combined_gunsnknifes/test/images/*.jpg
  - datasets/guns-knives/combined_gunsnknifes/test/labels/*.txt (YOLO format)

Outputs:
  - results/yolo_real_eval.csv
  - results/yolo_real_metrics.json
  - results/figures/yolo_pr_curve.png

Usage:
    python3 scripts/run_yolo_eval.py
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

CKPT      = os.path.join(ROOT, 'models', 'knifes&pistol', 'best.pt')
TEST_IMGS = os.path.join(ROOT, 'datasets', 'guns-knives',
                         'combined_gunsnknifes', 'test', 'images')
TEST_LBLS = os.path.join(ROOT, 'datasets', 'guns-knives',
                         'combined_gunsnknifes', 'test', 'labels')
RESULTS_DIR = os.path.join(ROOT, 'results')
FIGS_DIR    = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

CLASS_NAMES = ['pistol', 'knife']   # from data.yaml nc=2


def parse_gt_labels(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = (float(p) for p in parts[1:5])
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    return inter / (union + 1e-8)


def compute_ap(precisions, recalls):
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_thr = [p for p, r in zip(precisions, recalls) if r >= thr]
        ap += max(prec_at_thr) if prec_at_thr else 0.0
    return ap / 11.0


def run_yolo_eval():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed — run: pip3 install ultralytics")
        sys.exit(1)

    model = YOLO(CKPT)
    print(f"[YOLO] loaded {CKPT}")

    image_paths = sorted(
        glob.glob(os.path.join(TEST_IMGS, '*.jpg')) +
        glob.glob(os.path.join(TEST_IMGS, '*.jpeg')) +
        glob.glob(os.path.join(TEST_IMGS, '*.png'))
    )
    print(f"[YOLO] {len(image_paths)} test images")

    records   = []
    all_dets  = {c: [] for c in range(len(CLASS_NAMES))}
    n_gt      = {c: 0  for c in range(len(CLASS_NAMES))}

    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(TEST_LBLS, stem + '.txt')

        import cv2
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes = parse_gt_labels(lbl_path, w, h)
        for cls_id, *_ in gt_boxes:
            if cls_id < len(CLASS_NAMES):
                n_gt[cls_id] += 1

        results = model(img_path, conf=0.25, iou=0.45, verbose=False)[0]
        boxes   = results.boxes

        gt_matched = [False] * len(gt_boxes)

        preds = []
        if boxes is not None and len(boxes):
            for i in range(len(boxes)):
                pcls  = int(boxes.cls[i])
                pconf = float(boxes.conf[i])
                px1, py1, px2, py2 = boxes.xyxy[i].tolist()
                preds.append((pcls, pconf, px1, py1, px2, py2))

        preds.sort(key=lambda x: -x[1])

        for pcls, pconf, px1, py1, px2, py2 in preds:
            if pcls >= len(CLASS_NAMES):
                continue
            best_iou = 0.0
            best_gi  = -1
            for gi, (gcls, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
                if gcls != pcls or gt_matched[gi]:
                    continue
                v = iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if v > best_iou:
                    best_iou = v
                    best_gi  = gi

            match = int(best_iou >= 0.5 and best_gi >= 0)
            if match:
                gt_matched[best_gi] = True

            all_dets[pcls].append((pconf, match))
            records.append({
                'image':      stem,
                'gt_class':   CLASS_NAMES[pcls] if any(g[0] == pcls for g in gt_boxes) else 'none',
                'pred_class': CLASS_NAMES[pcls],
                'pred_conf':  round(pconf, 4),
                'iou':        round(best_iou, 4),
                'match':      match,
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, 'yolo_real_eval.csv')
    df.to_csv(csv_path, index=False)

    ap_per_class = {}
    for cls_id, dets in all_dets.items():
        if not dets:
            ap_per_class[CLASS_NAMES[cls_id]] = 0.0
            continue
        dets.sort(key=lambda x: -x[0])
        tp_cum = np.cumsum([d[1] for d in dets])
        fp_cum = np.cumsum([1 - d[1] for d in dets])
        prec = tp_cum / (tp_cum + fp_cum + 1e-8)
        rec  = tp_cum / (n_gt[cls_id] + 1e-8)
        ap_per_class[CLASS_NAMES[cls_id]] = round(float(compute_ap(prec, rec)), 4)

    mAP50 = round(float(np.mean(list(ap_per_class.values()))), 4)

    n_pred = len(df)
    tp_tot = int(df['match'].sum()) if len(df) else 0
    fp_tot = n_pred - tp_tot
    fn_tot = sum(n_gt.values()) - tp_tot
    precision = round(tp_tot / (tp_tot + fp_tot + 1e-8), 4)
    recall    = round(tp_tot / (tp_tot + fn_tot + 1e-8), 4)
    f1        = round(2 * precision * recall / (precision + recall + 1e-8), 4)

    metrics = {
        'mAP50':     mAP50,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'pistol_ap': ap_per_class.get('pistol', 0.0),
        'knife_ap':  ap_per_class.get('knife',  0.0),
        'n_images':  len(image_paths),
        'n_gt':      sum(n_gt.values()),
        'n_pred':    n_pred,
        'tp':        tp_tot,
        'fp':        fp_tot,
        'fn':        fn_tot,
    }
    json_path = os.path.join(RESULTS_DIR, 'yolo_real_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 55)
    print("YOLO REAL EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Images evaluated  : {len(image_paths)}")
    print(f"  GT boxes          : {sum(n_gt.values())}  "
          f"(pistol={n_gt[0]}, knife={n_gt[1]})")
    print(f"  Predictions       : {n_pred}")
    print(f"  TP / FP / FN      : {tp_tot} / {fp_tot} / {fn_tot}")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"  F1                : {f1:.4f}")
    print(f"  mAP@0.50          : {mAP50:.4f}")
    for cls, ap in ap_per_class.items():
        print(f"    AP({cls:6s})    : {ap:.4f}")
    print("=" * 55)
    print("[PDF baseline] mAP50=0.819  knife recall=0.865")

    _plot_pr_curve(all_dets, n_gt)
    return metrics


def _plot_pr_curve(all_dets, n_gt):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ['steelblue', 'darkorange']
    for cls_id, dets in all_dets.items():
        if not dets or n_gt[cls_id] == 0:
            continue
        dets_s = sorted(dets, key=lambda x: -x[0])
        tp_cum = np.cumsum([d[1] for d in dets_s])
        fp_cum = np.cumsum([1 - d[1] for d in dets_s])
        prec = tp_cum / (tp_cum + fp_cum + 1e-8)
        rec  = tp_cum / (n_gt[cls_id] + 1e-8)
        ap   = round(float(compute_ap(prec, rec)), 3)
        ax.plot(rec, prec, color=colors[cls_id % len(colors)],
                label=f'{CLASS_NAMES[cls_id]} AP={ap}', linewidth=2)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('YOLO PR Curve — Guns & Knives Test Set')
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(FIGS_DIR, 'yolo_pr_curve.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[YOLO] PR curve: {out}")


if __name__ == '__main__':
    run_yolo_eval()
