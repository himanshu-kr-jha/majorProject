import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def _savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', filename='confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(12, 10))
    try:
        import seaborn as sns
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
    except ImportError:
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    plt.tight_layout()
    return _savefig(filename)


def plot_pr_curves(pr_results, class_names, title='Precision-Recall Curves', filename='pr_curves.png'):
    """pr_results: dict {class_idx: (precision, recall, ap)}"""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    for i, name in enumerate(class_names):
        if i not in pr_results:
            continue
        precision, recall, ap = pr_results[i]
        ax.plot(recall, precision, color=colors[i], lw=1.5, label=f'{name} (AP={ap:.2f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _savefig(filename)


def plot_roc_curves(roc_results, class_names, title='ROC Curves', filename='roc_curves.png'):
    """roc_results: dict {class_idx: (fpr, tpr, auc_val)}"""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    for i, name in enumerate(class_names):
        if i not in roc_results:
            continue
        fpr, tpr, auc_val = roc_results[i]
        ax.plot(fpr, tpr, color=colors[i], lw=1.5, label=f'{name} (AUC={auc_val:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _savefig(filename)


def plot_reconstruction_error_dist(nm_errors, ab_errors, threshold,
                                   title='Gait Reconstruction Error Distribution',
                                   filename='gait_error_dist.png'):
    """Extends Fig 4 from PDF report."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(nm_errors, bins=60, alpha=0.6, color='steelblue', label='Normal (NM)', density=True)
    ax.hist(ab_errors, bins=60, alpha=0.6, color='darkorange', label='Abnormal (BG/CL)', density=True)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold = {threshold:.4f}')
    nm_mean = np.mean(nm_errors)
    ax.axvline(nm_mean, color='blue', linestyle=':', linewidth=1,
               label=f'Normal mean = {nm_mean:.4f}')
    ax.set_xlabel('Anomaly Score (0.3·MSE + 0.7·SSIM)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _savefig(filename)


def plot_threshold_sweep(results_df, filename='threshold_sweep.png'):
    """results_df columns: [threshold, precision, recall, f1, accuracy]"""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(results_df['threshold'], results_df['precision'], 'b-o', markersize=4, label='Precision')
    ax.plot(results_df['threshold'], results_df['recall'], 'g-s', markersize=4, label='Recall')
    ax.plot(results_df['threshold'], results_df['f1'], 'r-^', markersize=4, label='F1-Score')
    best_idx = results_df['f1'].idxmax()
    best_t = results_df.loc[best_idx, 'threshold']
    best_f1 = results_df.loc[best_idx, 'f1']
    ax.axvline(best_t, color='red', linestyle='--', linewidth=1,
               label=f'Best threshold = {best_t:.4f} (F1={best_f1:.3f})')
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Gait Threshold Sweep: Precision / Recall / F1', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    return _savefig(filename)


def plot_fp_reduction(fp_df, filename='fp_reduction.png'):
    """fp_df columns: [stage, fp_rate]"""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71']
    bars = ax.bar(fp_df['stage'], fp_df['fp_rate'],
                  color=colors[:len(fp_df)], edgecolor='black', linewidth=0.7)
    for bar, val in zip(bars, fp_df['fp_rate']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('False Positive Rate', fontsize=11)
    ax.set_title('False Positive Rate Reduction Across Fusion Stages', fontsize=13)
    ax.set_ylim([0, max(fp_df['fp_rate']) * 1.3])
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    return _savefig(filename)


def plot_ablation_bar(ablation_df, filename='ablation_f1.png'):
    """ablation_df columns: [configuration, precision, recall, f1]"""
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(ablation_df))
    width = 0.25
    ax.bar(x - width, ablation_df['precision'], width, label='Precision', color='#3498db')
    ax.bar(x,         ablation_df['recall'],    width, label='Recall',    color='#2ecc71')
    ax.bar(x + width, ablation_df['f1'],        width, label='F1-Score',  color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_df['configuration'], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Ablation Study: Per-Configuration Performance', fontsize=13)
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    return _savefig(filename)
