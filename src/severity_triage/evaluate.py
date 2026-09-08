"""Evaluation: classification quality *and* the business view of triage errors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)

from . import config as C
from .model import decode_scores, severe_probability


def is_severe(scores) -> np.ndarray:
    return np.asarray(scores) >= C.PROGRESS_THRESHOLD_SCORE


def triage_confusion(y_true, progress_pred) -> dict[str, int]:
    """Counts for the binary progress decision."""
    yt = is_severe(y_true)
    yp = np.asarray(progress_pred, dtype=bool)
    return {
        "true_positive": int((yt & yp).sum()),
        "false_negative": int((yt & ~yp).sum()),
        "false_positive": int((~yt & yp).sum()),
        "true_negative": int((~yt & ~yp).sum()),
    }


def triage_metrics(y_true, progress_pred, costs: C.BusinessCosts = C.BusinessCosts()) -> dict[str, float]:
    cm = triage_confusion(y_true, progress_pred)
    tp, fn, fp, tn = cm["true_positive"], cm["false_negative"], cm["false_positive"], cm["true_negative"]
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    n = tp + fn + fp + tn
    return {
        **cm,
        "severe_recall": recall,
        "severe_precision": precision,
        "specificity": specificity,
        "progress_rate": (tp + fp) / n if n else float("nan"),
        "missed_severe_rate": fn / n if n else float("nan"),
        "expected_cost_per_case": (costs.false_negative * fn + costs.false_positive * fp) / n if n else float("nan"),
    }


def score_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "within_one_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mae": mean_absolute_error(y_true, y_pred),
        "quadratic_weighted_kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "under_prediction_rate": float(np.mean(y_pred < y_true)),
        "over_prediction_rate": float(np.mean(y_pred > y_true)),
    }


def probability_metrics(y_true, proba) -> dict[str, float]:
    p_sev = severe_probability(proba)
    return {"severe_auroc": roc_auc_score(is_severe(y_true), p_sev)}


def full_evaluation(y_true, proba, threshold: float, costs: C.BusinessCosts = C.BusinessCosts()) -> dict:
    score, progress, _ = decode_scores(proba, threshold)
    argmax_score = np.asarray(C.SEVERITY_CLASSES)[np.asarray(proba).argmax(axis=1)]
    return {
        "threshold": threshold,
        "score": score_metrics(y_true, score),
        "triage": triage_metrics(y_true, progress, costs),
        "probability": probability_metrics(y_true, proba),
        "argmax_reference": {
            "score": score_metrics(y_true, argmax_score),
            "triage": triage_metrics(y_true, is_severe(argmax_score), costs),
        },
        "confusion_matrix": confusion_matrix(y_true, score, labels=C.SEVERITY_CLASSES).tolist(),
    }


def threshold_curve(y_true, proba, costs: C.BusinessCosts = C.BusinessCosts(), grid: np.ndarray | None = None) -> pd.DataFrame:
    """Triage metrics for a grid of thresholds on P(severe)."""
    grid = np.round(np.arange(0.01, 0.96, 0.01), 2) if grid is None else grid
    rows = []
    for t in grid:
        _, progress, _ = decode_scores(proba, t)
        rows.append({"threshold": float(t), **triage_metrics(y_true, progress, costs)})
    return pd.DataFrame(rows)


def choose_threshold(y_true, proba, costs: C.BusinessCosts = C.BusinessCosts(), min_severe_recall: float = 0.95) -> tuple[float, pd.DataFrame]:
    """Pick the threshold that minimises expected business cost, subject to a recall floor.

    Among thresholds meeting the recall floor we pick the one with the lowest cost; if several
    tie, the highest threshold (fewest unnecessary investigations). If nothing meets the floor,
    fall back to the threshold with the highest recall.
    """
    curve = threshold_curve(y_true, proba, costs)
    eligible = curve[curve["severe_recall"] >= min_severe_recall]
    if eligible.empty:
        eligible = curve[curve["severe_recall"] == curve["severe_recall"].max()]
    best = eligible.sort_values(["expected_cost_per_case", "threshold"], ascending=[True, False]).iloc[0]
    return float(best["threshold"]), curve
