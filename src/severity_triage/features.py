"""Feature engineering.

`FeatureEngineer` is a stateless scikit-learn transformer that turns a *cleaned* dataframe into
model-ready columns. It lives inside the saved pipeline so training and inference can never
drift apart. Fitted steps (imputation, one-hot encoding) are in `build_preprocessor`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config as C
from .data import NOMINAL_LEVELS

MISSING_TOKEN = "Missing"

ENGINEERED_NUMERIC = [
    "TotalFinancialLossGBP",
    "LogTotalFinancialLossGBP",
    "LogExpectedFinancialRedressGBP",
    "LogDurationAffectedDays",
    "LogDelayInResolutionDays",
    "HealthImpactCount",
    "MaxImpactLevel",
    "VulnerabilityXImpact",
    "ImpactRiskMean",
    "ImpactRiskGap",
    "RepeatFailureRatio",
    "FailuresPerOrganisation",
    "HasSecondaryFailure",
    "SecondaryMatchesPrimary",
    "LongTermOrLongRecovery",
    "AggravatingFactorCount",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derive features from a cleaned frame. Has no fitted state.

    Parameters
    ----------
    exclude_internal_scores:
        Drop the internal assessment columns (impact/risk scores, remedy band, redress). Used
        for the ablation that shows how much the model relies on them.
    """

    def __init__(self, exclude_internal_scores: bool = False):
        self.exclude_internal_scores = exclude_internal_scores

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        df = X.copy()
        out = pd.DataFrame(index=df.index)

        # Raw numerics (already validated / NaN where invalid)
        for col in C.NUMERIC_COLUMNS:
            out[col] = df[col].astype("float64")

        # Ordered categoricals -> integer codes, NaN preserved
        for col, levels in C.ORDINAL_LEVELS.items():
            codes = pd.Categorical(df[col].astype(object), categories=levels, ordered=True).codes
            out[col] = np.where(codes < 0, np.nan, codes).astype("float64")

        # Booleans -> 0/1 (NaN preserved)
        for col in C.BOOLEAN_COLUMNS:
            out[col] = df[col].astype("boolean").astype("Float64").astype("float64")

        # Nominal categoricals as plain strings; a blank becomes an explicit "Missing" level so the
        # one-hot encoder (which has a fixed category list from the data dictionary) can represent it.
        for col in C.NOMINAL_COLUMNS:
            out[col] = df[col].astype(object).where(df[col].notna(), MISSING_TOKEN).astype(str)

        # --- engineered features -------------------------------------------------------
        money = df[["DirectFinancialLossGBP", "LostIncomeGBP", "AdditionalCostsGBP"]]
        out["TotalFinancialLossGBP"] = money.sum(axis=1, min_count=1)
        out["LogTotalFinancialLossGBP"] = np.log1p(out["TotalFinancialLossGBP"])
        out["LogExpectedFinancialRedressGBP"] = np.log1p(df["ExpectedFinancialRedressGBP"])
        out["LogDurationAffectedDays"] = np.log1p(df["DurationAffectedDays"])
        out["LogDelayInResolutionDays"] = np.log1p(df["DelayInResolutionDays"])

        health = out[C.HEALTH_IMPACT_FLAGS]
        out["HealthImpactCount"] = health.sum(axis=1, min_count=1)

        out["MaxImpactLevel"] = out[["EmotionalImpactLevel", "PhysicalImpactLevel"]].max(axis=1)
        out["VulnerabilityXImpact"] = out["VulnerabilityLevel"] * out["MaxImpactLevel"]

        out["ImpactRiskMean"] = out[["EstimatedImpactScore", "EstimatedRiskScore"]].mean(axis=1)
        out["ImpactRiskGap"] = out["EstimatedImpactScore"] - out["EstimatedRiskScore"]

        failures = out["NumberOfServiceFailures"]
        out["RepeatFailureRatio"] = out["RepeatFailuresCount"] / failures.replace(0, np.nan)
        out["FailuresPerOrganisation"] = failures / out["NumberOfOrganisationsInvolved"].replace(0, np.nan)

        secondary = df["FailureTypeSecondary"].astype(object)
        out["HasSecondaryFailure"] = (secondary.notna() & (secondary != "None")).astype(float)
        out["SecondaryMatchesPrimary"] = (
            secondary.notna() & (secondary == df["FailureTypePrimary"].astype(object))
        ).astype(float)

        out["LongTermOrLongRecovery"] = (
            (out["LongTermImpactFlag"] == 1) | (out["RecoveryTimeMonths"] >= 12)
        ).astype(float)

        aggravating = [
            "SafeguardingConcern",
            "BereavedPerson",
            "LongTermImpactFlag",
            "RepeatedFailurePattern",
            "EscalatedInternally",
            "LegalChallengePotential",
            "MultipleComplaintThemes",
        ]
        out["AggravatingFactorCount"] = out[aggravating].sum(axis=1, min_count=1)

        if self.exclude_internal_scores:
            drop = list(C.INTERNAL_ASSESSMENT_COLUMNS) + [
                "LogExpectedFinancialRedressGBP",
                "ImpactRiskMean",
                "ImpactRiskGap",
            ]
            out = out.drop(columns=[c for c in drop if c in out.columns])

        self.feature_names_out_ = np.asarray(out.columns)
        return out

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "feature_names_out_"):
            return np.asarray(numeric_feature_names(self.exclude_internal_scores) + C.NOMINAL_COLUMNS)
        return self.feature_names_out_


def numeric_feature_names(exclude_internal_scores: bool = False) -> list[str]:
    cols = C.NUMERIC_COLUMNS + list(C.ORDINAL_LEVELS) + C.BOOLEAN_COLUMNS + ENGINEERED_NUMERIC
    if exclude_internal_scores:
        drop = set(C.INTERNAL_ASSESSMENT_COLUMNS) | {
            "LogExpectedFinancialRedressGBP",
            "ImpactRiskMean",
            "ImpactRiskGap",
        }
        cols = [c for c in cols if c not in drop]
    return cols


def build_preprocessor(exclude_internal_scores: bool = False) -> Pipeline:
    """FeatureEngineer -> (median impute | missing indicators | one-hot with dictionary categories).

    One-hot categories are fixed from the data dictionary rather than learned from the data, so the
    output columns are identical for every training fold and at inference, and a value outside the
    dictionary maps to an all-zero row instead of shifting columns.
    """
    numeric = numeric_feature_names(exclude_internal_scores)
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            # One indicator per numeric column, whether or not it was missing in the training data,
            # so the column set is identical across folds and at inference.
            ("missing", MissingIndicator(features="all", sparse=False), numeric),
            (
                "cat",
                OneHotEncoder(
                    categories=[NOMINAL_LEVELS[c] + [MISSING_TOKEN] for c in C.NOMINAL_COLUMNS],
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
                C.NOMINAL_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("engineer", FeatureEngineer(exclude_internal_scores=exclude_internal_scores)),
            ("encode", column_transformer),
        ]
    )
