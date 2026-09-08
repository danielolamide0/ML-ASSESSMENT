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

    Sum of the contributions to the raw scores of classes 4-6 minus those of classes 1-3. For a
    softmax model this is a proxy for, not exactly, the log-odds of P(severe); empirically it
    correlates at r ~ 0.99 with logit P(severe). Positive = raises the chance of being progressed.
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


def _format_value(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "missing"
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{val:.0f}" if float(val).is_integer() else f"{val:.2f}"
    return str(val)


def top_drivers(model: SeverityModel, X: pd.DataFrame, n: int = 3) -> pd.Series:  # noqa: N803
    """Human-readable top-n reasons per case, e.g. 'EstimatedImpactScore=82 (+1.4)'.

    Values shown are the *observed* (pre-imputation) values from the engineered frame. A feature
    that was missing and imputed is shown as 'missing' so a reader is never told a number the case
    does not have.
    """
    values, Xt = shap_values(model, X)
    sev = severe_contributions(values)
    names = np.asarray(model.feature_names)
    observed = model.pipeline.named_steps["preprocess"].named_steps["engineer"].transform(X)
    out = []
    for i in range(len(Xt)):
        order = np.argsort(-np.abs(sev[i]))[:n]
        parts = []
        for j in order:
            name = names[j]
            if name in observed.columns:
                val = observed.iloc[i][name]
                val = None if pd.isna(val) else val
            else:  # missing indicator or one-hot column: show the encoded 0/1
                val = float(Xt.iloc[i, j])
            parts.append(f"{name}={_format_value(val)} ({sev[i, j]:+.2f})")
        out.append("; ".join(parts))
    return pd.Series(out, index=X.index, name="TopDrivers")
