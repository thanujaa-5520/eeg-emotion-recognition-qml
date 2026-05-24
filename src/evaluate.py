"""
Step 6 — Evaluation & plotting.

Metrics : accuracy, precision, recall, F1, ROC-AUC, train time, inference time.
Plots   : confusion matrices, train/validation curves, combined ROC curves.

All plot filenames include the protocol so subject-dependent and
subject-independent outputs never overwrite each other.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                          # headless rendering
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from src.config import RESULTS_DIR

# Consistent colours per model for publication plots
MODEL_COLOURS = {
    "SVM":          "#4C72B0",
    "RandomForest": "#55A868",
    "EEGNet":       "#C44E52",
    "HybridQNN":    "#8172B2",
}


def compute_metrics(y_true, y_pred, y_score,
                    train_time: float, inference_time: float) -> dict:
    """All evaluation metrics for one model on one task/protocol."""
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")                     # only one class present in y_true
    return {
        "accuracy":     round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision":    round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall":       round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "f1":           round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "roc_auc":      round(auc, 4),
        "train_time_s": round(train_time, 2),
        "inference_ms": round(inference_time * 1000, 1),
    }


# ── Confusion matrix ───────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, model_name, task, protocol) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low", "High"]); ax.set_yticklabels(["Low", "High"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{model_name} — {task.capitalize()}\n({protocol.replace('_', '-')})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"cm_{protocol}_{model_name.lower()}_{task}.png")
    plt.savefig(fname, dpi=150)
    plt.close()


# ── Training curves (train vs validation) ──────────────────────────────────────

def save_training_curve(history, model_name, task, protocol) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="train")
    ax1.plot(epochs, history["val_loss"],   "r-s", markersize=3, label="validation")
    ax1.set_title(f"{model_name} — {task} — Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", markersize=3, label="train")
    ax2.plot(epochs, history["val_acc"],   "r-s", markersize=3, label="validation")
    if "best_epoch" in history:
        ax2.axvline(history["best_epoch"], color="green", linestyle="--",
                    label=f"best epoch ({history['best_epoch']})")
    ax2.set_title(f"{model_name} — {task} — Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle(f"Protocol: {protocol.replace('_', '-')}", fontsize=10)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"training_{protocol}_{model_name.lower()}_{task}.png")
    plt.savefig(fname, dpi=150)
    plt.close()


# ── Combined ROC curve (all models on one axes) ────────────────────────────────

def save_combined_roc(roc_data: dict, task: str, protocol: str) -> None:
    """
    roc_data : {model_name: (y_true, y_score)}
    Draws one ROC curve per model on a shared axes for the given task/protocol.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for model_name, (y_true, y_score) in roc_data.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            continue
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})",
                color=MODEL_COLOURS.get(model_name), linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {task.capitalize()} ({protocol.replace('_', '-')})")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"roc_{protocol}_{task}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
