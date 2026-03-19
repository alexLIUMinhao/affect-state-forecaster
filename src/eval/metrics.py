from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def mae(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - targets)))


def rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def pearson_correlation(preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) < 2:
        return float("nan")
    if np.allclose(preds, preds[0]) or np.allclose(targets, targets[0]):
        return float("nan")
    return float(np.corrcoef(preds, targets)[0, 1])


def spearman_correlation(preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) < 2:
        return float("nan")
    pred_ranks = pd.Series(preds).rank(method="average").to_numpy()
    target_ranks = pd.Series(targets).rank(method="average").to_numpy()
    return pearson_correlation(pred_ranks, target_ranks)


def classification_metrics(pred_labels: list[str], true_labels: list[str]) -> dict[str, float]:
    if not pred_labels or not true_labels:
        return {}
    return {
        "majority_sentiment_accuracy": float(accuracy_score(true_labels, pred_labels)),
        "majority_sentiment_macro_f1": float(f1_score(true_labels, pred_labels, average="macro")),
        "majority_sentiment_weighted_f1": float(f1_score(true_labels, pred_labels, average="weighted")),
    }


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    pred_labels: list[str] | None = None,
    true_labels: list[str] | None = None,
) -> dict[str, float]:
    metrics = {
        "mae": mae(preds, targets),
        "rmse": rmse(preds, targets),
        "pearson": pearson_correlation(preds, targets),
        "spearman": spearman_correlation(preds, targets),
    }
    if pred_labels is not None and true_labels is not None and all(label is not None for label in true_labels):
        metrics.update(classification_metrics(pred_labels, true_labels))
    return metrics
