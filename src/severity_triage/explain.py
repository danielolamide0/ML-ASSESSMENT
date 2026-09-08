"""SHAP-based explanations: global feature importance and per-case reasons."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from . import config as C
from .model import SEVERE_START_INDEX, SeverityModel


def transformed_frame(model: SeverityModel, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
    pre = model.pipeline.named_steps["preprocess"]
    return pd.DataFrame(pre.transform(X), columns=model.feature_names, index=X.index)


def shap_values(model: SeverityModel, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:  # noqa: N803
    """Return SHAP values with shape (n_samples, n_features, n_classes) plus the feature frame."""
    Xt = transformed_frame(model, X)
    explainer = shap.TreeExplainer(model.pipeline.named_steps["model"])
    values = explainer.shap_values(Xt)
    if isinstance(values, list):  # older shap API: list of per-class arrays
        values = np.stack(values, axis=-1)
    return values, Xt


def severe_contributions(values: np.ndarray) -> np.ndarray:
    """Collapse per-class SHAP values to a single 'pushes towards severe' contribution.

    Sum of the contributions to classes 4-6 minus classes 1-3 (log-odds space of the
    multiclass model), so positive = raises the chance of being progressed.
    """
    return values[:, :, SEVERE_START_INDEX:].sum(axis=2) - values[:, :, :SEVERE_START_INDEX].sum(axis=2)


def global_importance(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(values).mean(axis=(0, 2))
    severe = np.abs(severe_contributions(values)).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs, "mean_abs_severe_shap": severe})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def top_drivers(model: SeverityModel, X: pd.DataFrame, n: int = 3) -> pd.Series:  # noqa: N803
    """Human-readable top-n reasons per case, e.g. 'EstimatedImpactScore=82 (+1.4)'."""
    values, Xt = shap_values(model, X)
    sev = severe_contributions(values)
    names = np.asarray(model.feature_names)
    out = []
    for i in range(len(Xt)):
        order = np.argsort(-np.abs(sev[i]))[:n]
        parts = []
        for j in order:
            val = Xt.iloc[i, j]
            val_str = f"{val:.0f}" if float(val).is_integer() else f"{val:.2f}"
            parts.append(f"{names[j]}={val_str} ({sev[i, j]:+.2f})")
        out.append("; ".join(parts))
    return pd.Series(out, index=X.index, name="TopDrivers")
