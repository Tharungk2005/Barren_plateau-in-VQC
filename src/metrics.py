"""Evaluation metrics. Positive class = bad credit (+1), per config
label_convention. All metrics are computed from actual model outputs;
none are hand-typed.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)


def predict_labels(outputs, threshold=0.0):
    """outputs in [-1,1] -> predicted label in {-1,+1}."""
    return np.where(np.asarray(outputs) > threshold, 1, -1)


def mse_loss(outputs, labels):
    outputs = np.asarray(outputs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((outputs - labels) ** 2))


def continuous_score(outputs):
    """Map circuit output in [-1,1] to a [0,1] decision score for ROC-AUC
    via an affine transform. This is NOT a calibrated probability; it is
    a monotonic rescaling used only to compute ranking-based AUC.
    """
    return (np.asarray(outputs, dtype=float) + 1.0) / 2.0


def misclassification_cost(y_true, y_pred, cost_matrix):
    """cost_matrix keyed by ('good','bad')->1, ('bad','good')->5, etc.
    y_true, y_pred in {-1 (good), +1 (bad)}.
    """
    label_name = {-1: "good", 1: "bad"}
    total = 0.0
    for yt, yp in zip(y_true, y_pred):
        key = (label_name[int(yt)], label_name[int(yp)])
        total += cost_matrix.get(key, 0)
    return total, total / max(len(y_true), 1)


def compute_classification_metrics(y_true, outputs, threshold=0.0, cost_matrix=None):
    y_true = np.asarray(y_true)
    y_pred = predict_labels(outputs, threshold)
    score = continuous_score(outputs)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "loss_mse": mse_loss(outputs, y_true),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, score))
    except ValueError:
        metrics["roc_auc"] = None  # undefined (e.g. single-class batch)

    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    metrics["confusion_matrix"] = cm.tolist()  # rows=actual[-1,1], cols=pred[-1,1]
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = (
        int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    )

    if cost_matrix is not None:
        total_cost, mean_cost = misclassification_cost(y_true, y_pred, cost_matrix)
        metrics["total_cost"] = total_cost
        metrics["mean_cost"] = mean_cost

    return metrics


def gradient_norm(grad_array):
    g = np.asarray(grad_array, dtype=float).ravel()
    return float(np.linalg.norm(g))


def normalized_gradient_norm(grad_array):
    """Gradient norm divided by sqrt(number of active parameters), so runs
    with different active-parameter counts (e.g. layer-wise stages) remain
    comparable in magnitude-per-parameter terms.
    """
    g = np.asarray(grad_array, dtype=float).ravel()
    n = g.size
    if n == 0:
        return 0.0
    return float(np.linalg.norm(g) / np.sqrt(n))


def plateau_fraction(grad_norm_history, threshold):
    """Fraction of recorded optimizer updates whose gradient norm fell
    below `threshold` (a predeclared plateau indicator).
    """
    g = np.asarray(grad_norm_history, dtype=float)
    if g.size == 0:
        return None
    return float(np.mean(g < threshold))


def first_update_reaching_target(val_metric_history, target, patience=1):
    """Return the (1-indexed) update index at which val_metric_history
    first reaches >= target and stays >= target for `patience` consecutive
    recorded checkpoints, else None (not converged).
    """
    v = np.asarray(val_metric_history, dtype=float)
    for i in range(len(v) - patience + 1):
        if np.all(v[i:i + patience] >= target):
            return i + 1
    return None
