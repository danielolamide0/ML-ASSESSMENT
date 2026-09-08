import numpy as np
import pandas as pd
import pytest

from severity_triage import config as C
from severity_triage.data import clean


def test_leakage_column_is_dropped(raw_train):
    df, report = clean(raw_train, is_training=True)
    assert "OmbudsmanInvestigationRequired" not in df.columns
    assert report.dropped_columns == ["OmbudsmanInvestigationRequired"]


def test_out_of_range_values_become_missing(raw_train):
    df, report = clean(raw_train, is_training=True)
    for col, (lo, hi) in C.NUMERIC_RANGES.items():
        assert df[col].dropna().between(lo, hi).all(), col
    assert report.out_of_range["EstimatedRiskScore"] == 2  # one below, one above


def test_structural_none_filled(raw_train):
    df, report = clean(raw_train, is_training=True)
    for col in C.STRUCTURAL_NONE_COLUMNS:
        assert df[col].isna().sum() == 0
        assert (df[col] == "None").sum() == report.structural_none_filled[col]


def test_jurisdiction_inferred_from_organisation(raw_train):
    df, report = clean(raw_train, is_training=True)
    assert report.jurisdiction_inferred > 0
    known = df["OrganisationType"].notna()
    expected = df.loc[known, "OrganisationType"].map(C.ORGANISATION_JURISDICTION)
    assert (df.loc[known, "Jurisdiction"] == expected).all()


def test_booleans_are_boolean(raw_train):
    df, _ = clean(raw_train, is_training=True)
    for col in C.BOOLEAN_COLUMNS:
        assert str(df[col].dtype) == "boolean"


def test_invalid_values_are_reported():
    row = {c: np.nan for c in C.ALL_RAW_FEATURES}
    row.update({C.ID_COLUMN: "X-1", "AgeBand": "Ancient", "AnxietyReported": "maybe", "EstimatedRiskScore": 500})
    df, report = clean(pd.DataFrame([row]), is_training=False)
    assert report.invalid_categories["AgeBand"] == 1
    assert report.invalid_booleans["AnxietyReported"] == 1
    assert report.out_of_range["EstimatedRiskScore"] == 1
    assert df["AgeBand"].isna().all()


def test_missing_required_column_raises(raw_holdback):
    with pytest.raises(ValueError, match="missing required columns"):
        clean(raw_holdback.drop(columns=["AgeBand"]), is_training=False)


def test_holdback_cleans_without_target(raw_holdback):
    df, _ = clean(raw_holdback, is_training=False)
    assert len(df) == len(raw_holdback)
    assert C.TARGET not in df.columns
