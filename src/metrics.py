import math
from typing import Dict, List, Tuple

import torch


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def confusion_from_preds(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    return tp, tn, fp, fn


def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    tp, tn, fp, fn = confusion_from_preds(y_true, y_pred)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return _safe_div(2 * precision * recall, precision + recall)


def balanced_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    tp, tn, fp, fn = confusion_from_preds(y_true, y_pred)
    tpr = _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    return 0.5 * (tpr + tnr)


def auc_roc(y_true: List[int], y_score: List[float]) -> float:
    """
    Simple ROC-AUC by ranking scores (handles ties).
    y_true: 0/1
    y_score: probability for class 1
    """
    pairs = list(zip(y_score, y_true))
    pairs.sort(key=lambda x: x[0])

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # rank with ties: average ranks
    ranks = []
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks.append((avg_rank, pairs[k][1]))
        i = j

    sum_pos_ranks = sum(r for r, y in ranks if y == 1)
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, float]:
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)
    return {
        "acc": float(acc),
        "bal_acc": float(balanced_accuracy(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(auc_roc(y_true, y_prob)),
    }
