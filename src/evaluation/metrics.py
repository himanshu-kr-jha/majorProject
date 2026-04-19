import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)


def compute_classification_report(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    return report


def compute_map_from_csv(results_csv_path):
    df = pd.read_csv(results_csv_path)
    map50_col = [c for c in df.columns if 'mAP50' in c and '95' not in c]
    map95_col = [c for c in df.columns if 'mAP50-95' in c]
    result = {}
    if map50_col:
        result['mAP50'] = float(df[map50_col[0]].iloc[-1])
    if map95_col:
        result['mAP50-95'] = float(df[map95_col[0]].iloc[-1])
    return result


def compute_roc_auc(scores, labels, n_classes=None):
    """
    scores: (N, n_classes) softmax probabilities or (N,) binary
    labels: (N,) integer class indices
    Returns dict of {class_idx: (fpr, tpr, auc_val)}
    """
    scores = np.array(scores)
    labels = np.array(labels)

    if scores.ndim == 1:
        fpr, tpr, _ = roc_curve(labels, scores)
        return {0: (fpr, tpr, auc(fpr, tpr))}

    n_classes = scores.shape[1] if n_classes is None else n_classes
    results = {}
    for i in range(n_classes):
        binary_labels = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, scores[:, i])
        results[i] = (fpr, tpr, auc(fpr, tpr))
    return results


def compute_pr_curves(scores, labels, n_classes=None):
    """
    Returns dict of {class_idx: (precision, recall, ap)}
    """
    scores = np.array(scores)
    labels = np.array(labels)

    if scores.ndim == 1:
        precision, recall, _ = precision_recall_curve(labels, scores)
        return {0: (precision, recall, average_precision_score(labels, scores))}

    n_classes = scores.shape[1] if n_classes is None else n_classes
    results = {}
    for i in range(n_classes):
        binary_labels = (labels == i).astype(int)
        precision, recall, _ = precision_recall_curve(binary_labels, scores[:, i])
        ap = average_precision_score(binary_labels, scores[:, i])
        results[i] = (precision, recall, ap)
    return results


def compute_confusion_matrix(y_true, y_pred, n_classes):
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def threshold_metrics(errors, labels, threshold):
    """
    errors: array of reconstruction errors
    labels: 0=normal, 1=abnormal
    Returns dict with precision, recall, f1, accuracy, tp, tn, fp, fn
    """
    preds = (np.array(errors) > threshold).astype(int)
    labels = np.array(labels)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / len(labels)
    return {
        'threshold': threshold,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'accuracy': round(accuracy, 4),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }
