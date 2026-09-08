"""Model definitions and the decision layer that turns probabilities into a severity score."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import build_preprocessor

SEVERE_START_INDEX = C.SEVERITY_CLASSES.index(C.PROGRESS_THRESHOLD_SCORE)  # index of class 4


def build_lgbm_pipeline(params: dict | None = None, *, exclude_internal_scores: bool = False, seed: int = C.RANDOM_SEED) -> Pipeline:
    params = dict(C.TrainingConfig().lgbm_params if params is None else params)
    params.setdefault("random_state", seed)
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(exclude_internal_scores)),
            ("model", LGBMClassifier(**params)),
        ]
    )


def build_baseline_pipeline(*, exclude_internal_scores: bool = False, seed: int = C.RANDOM_SEED) -> Pipeline:
    """Regularised multinomial logistic regression: a transparent reference point."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(exclude_internal_scores)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.3, class_weight="balanced", max_iter=5000, random_state=seed
                ),
            ),
        ]
    )


def severe_probability(proba: np.ndarray) -> np.ndarray:
    """P(SeverityScore >= 4) from the 6-class probability matrix (columns ordered 1..6)."""
    return proba[:, SEVERE_START_INDEX:].sum(axis=1)


def decode_scores(proba: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert class probabilities into (score, progress_flag, p_severe).

    The triage decision is taken first: progress when P(score >= 4) >= threshold. The
    predicted score is then the most likely class *within* the chosen band, so the score in the
    output file is always consistent with the decision it drives. Using a threshold below 0.5
    is how the low tolerance for missed severe cases is encoded.
    """
    proba = np.asarray(proba, dtype=float)
    p_severe = severe_probability(proba)
    # Fail safe: a row whose probabilities are undefined is progressed, never closed automatically.
    progress = ~(p_severe < threshold)
    low_band = proba[:, :SEVERE_START_INDEX].argmax(axis=1) + C.SEVERITY_CLASSES[0]
    high_band = proba[:, SEVERE_START_INDEX:].argmax(axis=1) + C.PROGRESS_THRESHOLD_SCORE
    score = np.where(progress, high_band, low_band)
    return score.astype(int), progress, p_severe


@dataclass
class SeverityModel:
    """Everything needed to score new cases: fitted pipeline + decision threshold + metadata."""

    pipeline: Pipeline
    threshold: float
    classes: list[int] = field(default_factory=lambda: list(C.SEVERITY_CLASSES))
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        proba = self.pipeline.predict_proba(X)
        model_classes = list(self.pipeline.named_steps["model"].classes_)
        if model_classes != self.classes:  # re-order defensively
            proba = proba[:, [model_classes.index(c) for c in self.classes]]
        return proba

    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        proba = self.predict_proba(X)
        score, progress, p_severe = decode_scores(proba, self.threshold)
        out = pd.DataFrame(
            {
                C.ID_COLUMN: X[C.ID_COLUMN].values,
                "PredictedSeverityScore": score,
                "TriageDecision": np.where(progress, "Progress", "Do not progress"),
                "ProbabilitySevere": np.round(p_severe, 4),
            }
        )
        for cls, col in zip(self.classes, proba.T):
            out[f"P_Severity{cls}"] = np.round(col, 4)
        return out

    @property
    def feature_names(self) -> list[str]:
        return list(self.pipeline.named_steps["preprocess"].get_feature_names_out())
