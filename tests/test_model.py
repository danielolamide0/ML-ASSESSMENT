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
    hb, _ = clean(raw_holdback.head(5), is_training=False)
    hb.loc[0, "ServiceArea"] = pd.NA
    hb.loc[1, "EstimatedImpactScore"] = np.nan
    out = model.predict_frame(hb)
    assert len(out) == 5
    p = severe_probability(model.predict_proba(hb))
    assert ((p >= 0) & (p <= 1)).all()
