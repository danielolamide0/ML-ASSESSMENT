import numpy as np
import pandas as pd
import pytest

from severity_triage import config as C
from severity_triage.data import clean
from severity_triage.evaluate import choose_threshold, triage_metrics
from severity_triage.model import build_lgbm_pipeline, decode_scores, severe_probability, SeverityModel


def test_decode_scores_is_consistent_with_decision():
    proba = np.array(
        [
            [0.5, 0.3, 0.1, 0.05, 0.03, 0.02],  # clearly low
            [0.05, 0.05, 0.1, 0.5, 0.2, 0.1],  # clearly severe
            [0.4, 0.3, 0.1, 0.1, 0.05, 0.05],  # low argmax, p_severe = 0.2
        ]
    )
    score, progress, p_sev = decode_scores(proba, threshold=0.5)
    assert score.tolist() == [1, 4, 1]
    assert progress.tolist() == [False, True, False]
    score, progress, _ = decode_scores(proba, threshold=0.15)
    assert score.tolist() == [1, 4, 4]  # lower threshold pushes the borderline case to progress
    assert ((score >= 4) == progress).all()
    np.testing.assert_allclose(p_sev, proba[:, 3:].sum(axis=1))


def test_lower_threshold_never_reduces_recall():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(6), size=500)
    y = rng.integers(1, 7, size=500)
    recalls = [triage_metrics(y, decode_scores(proba, t)[1])["severe_recall"] for t in (0.1, 0.3, 0.5, 0.7)]
    assert recalls == sorted(recalls, reverse=True)


def test_choose_threshold_respects_recall_floor():
    rng = np.random.default_rng(1)
    y = rng.integers(1, 7, size=600)
    proba = rng.dirichlet(np.ones(6), size=600)
    proba[y >= 4, 3:] += 1.0  # make severe cases separable-ish
    proba /= proba.sum(axis=1, keepdims=True)
    t, curve = choose_threshold(y, proba, min_severe_recall=0.95)
    row = curve.loc[curve["threshold"] == t].iloc[0]
    assert row["severe_recall"] >= 0.95


@pytest.fixture(scope="module")
def small_model(raw_train):
    df, _ = clean(raw_train.head(600), is_training=True)
    params = dict(C.TrainingConfig().lgbm_params, n_estimators=50)
    pipe = build_lgbm_pipeline(params).fit(df.drop(columns=[C.TARGET]), df[C.TARGET])
    return SeverityModel(pipeline=pipe, threshold=0.2), df


def test_predict_frame_schema_and_repeatability(small_model, raw_holdback):
    model, _ = small_model
    hb, _ = clean(raw_holdback, is_training=False)
    a = model.predict_frame(hb)
    b = model.predict_frame(hb)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns[:4]) == [C.ID_COLUMN, "PredictedSeverityScore", "TriageDecision", "ProbabilitySevere"]
    assert a["PredictedSeverityScore"].isin(C.SEVERITY_CLASSES).all()
    assert ((a["PredictedSeverityScore"] >= 4) == (a["TriageDecision"] == "Progress")).all()
    np.testing.assert_allclose(a[[f"P_Severity{c}" for c in C.SEVERITY_CLASSES]].sum(axis=1), 1, atol=1e-3)


def test_model_handles_unseen_category_and_missing(small_model, raw_holdback):
    model, _ = small_model
    raw = raw_holdback.head(5).copy()
    raw.loc[0, "ServiceArea"] = "Space Travel"  # not in the data dictionary
    raw.loc[1, "EstimatedImpactScore"] = np.nan
    hb, report = clean(raw, is_training=False)
    assert report.invalid_categories == {"ServiceArea": 1}
    out = model.predict_frame(hb)
    assert len(out) == 5
    p = severe_probability(model.predict_proba(hb))
    assert ((p >= 0) & (p <= 1)).all()
    # A value that somehow bypasses cleaning is ignored by the encoder (all-zero row), not an error
    hb2 = hb.copy()
    hb2["ServiceArea"] = hb2["ServiceArea"].astype(object)
    hb2.loc[2, "ServiceArea"] = "Space Travel"
    Xt = model.pipeline.named_steps["preprocess"].transform(hb2)
    cols = [i for i, n in enumerate(model.feature_names) if n.startswith("ServiceArea_")]
    assert Xt[2, cols].sum() == 0
    assert Xt.shape[1] == len(model.feature_names)


def test_nan_probabilities_fail_safe_to_progress():
    proba = np.full((1, 6), np.nan)
    score, progress, _ = decode_scores(proba, threshold=0.5)
    assert progress.tolist() == [True]
    assert score[0] >= 4


def test_choose_threshold_minimises_cost_and_prefers_higher_threshold_on_ties():
    # Two severe cases with p_severe 0.9 and 0.3; three non-severe with p_severe 0.2, 0.1, 0.05
    p = np.array([0.9, 0.3, 0.2, 0.1, 0.05])
    proba = np.column_stack([1 - p, np.zeros_like(p), np.zeros_like(p), p, np.zeros_like(p), np.zeros_like(p)])
    y = np.array([5, 4, 1, 2, 3])
    t, curve = choose_threshold(y, proba, C.BusinessCosts(false_negative=5, false_positive=1), min_severe_recall=0.0)
    assert 0.2 < t <= 0.3  # catches both severe cases at zero false positives
    best_cost = curve.loc[curve["threshold"] == t, "expected_cost_per_case"].iloc[0]
    assert best_cost == curve["expected_cost_per_case"].min()
    assert t == curve.loc[curve["expected_cost_per_case"] == best_cost, "threshold"].max()


def test_choose_threshold_falls_back_to_max_recall_when_floor_unattainable():
    rng = np.random.default_rng(3)
    proba = rng.dirichlet(np.ones(6), size=200)
    y = rng.integers(1, 7, size=200)
    t, curve = choose_threshold(y, proba, min_severe_recall=1.01)
    assert curve.loc[curve["threshold"] == t, "severe_recall"].iloc[0] == curve["severe_recall"].max()


def test_top_drivers_reports_missing_not_imputed(small_model, raw_holdback):
    from severity_triage.explain import top_drivers

    model, _ = small_model
    raw = raw_holdback.head(3).copy()
    raw["EstimatedImpactScore"] = np.nan
    raw["ExpectedFinancialRedressGBP"] = np.nan
    hb, _ = clean(raw, is_training=False)
    drivers = top_drivers(model, hb, n=8)
    for text in drivers:
        for part in text.split("; "):
            name = part.split("=")[0]
            if name in ("EstimatedImpactScore", "ExpectedFinancialRedressGBP", "LogExpectedFinancialRedressGBP", "ImpactRiskGap"):
                assert part.startswith(f"{name}=missing"), part
