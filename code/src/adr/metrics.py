"""Classification metrics for imbalanced binary tasks + trivial baselines + threshold selection."""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             balanced_accuracy_score, accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix, roc_curve)


def compute_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_true = np.asarray(y_true).astype(int); y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "auroc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "auprc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": float(threshold),
    }
    return m


def best_threshold(y_true, y_prob, criterion="mcc"):
    """Pick decision threshold on the VALIDATION set (never on test)."""
    y_true = np.asarray(y_true).astype(int); y_prob = np.asarray(y_prob)
    if criterion == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        return float(thr[np.argmax(tpr - fpr)])
    grid = np.unique(np.quantile(y_prob, np.linspace(0.02, 0.98, 97)))
    fn = {"mcc": matthews_corrcoef, "f1": lambda a, b: f1_score(a, b, zero_division=0),
          "balanced_acc": balanced_accuracy_score}[criterion]
    scores = [fn(y_true, (y_prob >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def majority_baseline(y_train, y_test) -> dict:
    """Predict the training majority class for every test molecule."""
    p = float(np.mean(y_train))
    y_prob = np.full(len(y_test), p)
    m = compute_metrics(y_test, y_prob, threshold=0.5 if p >= 0.5 else 1.0)
    m["auroc"], m["auprc"] = 0.5, float(np.mean(y_test))
    return m


def random_baseline(y_test, seed=0) -> dict:
    rng = np.random.RandomState(seed)
    return compute_metrics(y_test, rng.rand(len(y_test)))


METRIC_COLS = ["auroc", "auprc", "mcc", "balanced_acc", "accuracy", "f1", "precision", "recall", "specificity"]
