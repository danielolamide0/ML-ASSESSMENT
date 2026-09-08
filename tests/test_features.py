import numpy as np

from severity_triage import config as C
from severity_triage.data import clean
from severity_triage.features import FeatureEngineer, build_preprocessor, numeric_feature_names


def test_feature_engineer_is_deterministic_and_complete(raw_train):
    df, _ = clean(raw_train, is_training=True)
    fe = FeatureEngineer().fit(df)
    a, b = fe.transform(df), fe.transform(df)
    assert a.equals(b)
    expected = set(numeric_feature_names()) | set(C.NOMINAL_COLUMNS)
    assert set(a.columns) == expected


def test_ordinal_encoding_respects_order(raw_train):
    df, _ = clean(raw_train, is_training=True)
    feats = FeatureEngineer().fit_transform(df)
    codes = feats.loc[df["EmotionalImpactLevel"].notna(), "EmotionalImpactLevel"]
    labels = df.loc[df["EmotionalImpactLevel"].notna(), "EmotionalImpactLevel"]
    assert codes[labels == "Severe"].eq(4).all()
    assert codes[labels == "Minimal"].eq(0).all()


def test_engineered_features_make_sense(raw_train):
    df, _ = clean(raw_train, is_training=True)
    feats = FeatureEngineer().fit_transform(df)
    ok = df[["DirectFinancialLossGBP", "LostIncomeGBP", "AdditionalCostsGBP"]].notna().all(axis=1)
    np.testing.assert_allclose(
        feats.loc[ok, "TotalFinancialLossGBP"],
        df.loc[ok, ["DirectFinancialLossGBP", "LostIncomeGBP", "AdditionalCostsGBP"]].sum(axis=1),
    )
    assert feats["HealthImpactCount"].between(0, len(C.HEALTH_IMPACT_FLAGS)).all()
    assert feats["HasSecondaryFailure"].isin([0, 1]).all()


def test_exclude_internal_scores_removes_them(raw_train):
    df, _ = clean(raw_train, is_training=True)
    feats = FeatureEngineer(exclude_internal_scores=True).fit_transform(df)
    assert not set(C.INTERNAL_ASSESSMENT_COLUMNS) & set(feats.columns)


def test_preprocessor_output_has_no_missing_and_stable_names(raw_train):
    df, _ = clean(raw_train, is_training=True)
    pre = build_preprocessor().fit(df)
    X = pre.transform(df)
    assert not np.isnan(X).any()
    assert X.shape[1] == len(pre.get_feature_names_out())


def test_one_hot_columns_are_fixed_by_dictionary(raw_train):
    """Column set must not depend on which categories happen to be present in a training fold."""
    df, _ = clean(raw_train, is_training=True)
    full = build_preprocessor().fit(df)
    small = build_preprocessor().fit(df.head(50))
    assert list(full.get_feature_names_out()) == list(small.get_feature_names_out())
    assert "Jurisdiction_Missing" in full.get_feature_names_out()
