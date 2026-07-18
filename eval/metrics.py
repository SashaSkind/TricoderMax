"""
Minimal ROC / AUC / operating-point metrics (no sklearn dependency).

Scores are "higher = more positive". For the midline measurement the raw mm value
is already higher-for-worse, so it plugs in directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ROC:
    fpr: list[float]
    tpr: list[float]
    thresholds: list[float]
    auc: float


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> ROC:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    P = int((labels == 1).sum())
    N = int((labels == 0).sum())
    if P == 0 or N == 0:
        hi = float(scores.max()) + 1.0 if scores.size else 1.0
        return ROC([0.0, 1.0], [0.0, 1.0], [hi, float(scores.min()) if scores.size else 0.0], 0.5)

    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order]
    tp = fp = 0
    # Finite sentinel above the max score = "classify nothing positive" (0,0 point);
    # keeps the ROC JSON-serializable (no inf).
    fpr, tpr, thr = [0.0], [0.0], [float(s[0]) + 1.0]
    i = 0
    n = len(s)
    while i < n:
        t = s[i]
        while i < n and s[i] == t:  # handle ties: advance all equal scores together
            if y[i] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        tpr.append(tp / P)
        fpr.append(fp / N)
        thr.append(float(t))
    # np.trapezoid (numpy>=2); np.trapz on older numpy.
    _trap = getattr(np, "trapezoid", None) or np.trapz
    auc = float(_trap(tpr, fpr))
    return ROC(fpr, tpr, thr, round(auc, 4))


def precision_at_recall(scores: np.ndarray, labels: np.ndarray, target_recall: float) -> tuple[float, float]:
    """Highest precision achievable at >= target_recall. Returns (precision, threshold).

    Sweeps every score as a threshold, keeps operating points meeting the recall
    floor, and returns the most precise one. This is how we compare 'detector alone'
    vs 'detector + Claude adjustment' at a fixed recall (the verification-value test).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    P = int((labels == 1).sum())
    if P == 0:
        return float("nan"), float("nan")
    best_prec, best_thr = 0.0, float("inf")
    for thr in np.unique(scores):
        pred = scores >= thr
        tp = int((pred & (labels == 1)).sum())
        recall = tp / P
        if recall < target_recall:
            continue
        pp = int(pred.sum())
        prec = tp / pp if pp else 0.0
        if prec > best_prec:
            best_prec, best_thr = prec, float(thr)
    return round(best_prec, 4), best_thr


def sens_spec_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    """Sensitivity and specificity at a decision threshold (positive iff >= threshold)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = scores >= threshold
    P = (labels == 1).sum()
    N = (labels == 0).sum()
    tp = int((pred & (labels == 1)).sum())
    tn = int((~pred & (labels == 0)).sum())
    sens = tp / P if P else float("nan")
    spec = tn / N if N else float("nan")
    return round(float(sens), 4), round(float(spec), 4)
