"""Loading, validation and cleaning of the raw case data.

Cleaning is deterministic and identical for training and inference so that the model always
sees data in the same shape. Every rule is driven by the data dictionary encoded in config.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """What the cleaning step changed, so it can be logged and monitored in production."""

    n_rows: int = 0
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    out_of_range: dict[str, int] = field(default_factory=dict)
    invalid_categories: dict[str, int] = field(default_factory=dict)
    invalid_booleans: dict[str, int] = field(default_factory=dict)
    structural_none_filled: dict[str, int] = field(default_factory=dict)
    jurisdiction_inferred: int = 0
    duplicate_references: int = 0
    missing_after_cleaning: dict[str, int] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load a CSV or the original Excel workbook sheet."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    log.info("Loaded %d rows x %d columns from %s", *df.shape, path)
    return df


def _to_bool(series: pd.Series) -> tuple[pd.Series, int]:
    mapping = {
        "yes": True, "no": False, "true": True, "false": False,
        "1": True, "0": False, "1.0": True, "0.0": False, "y": True, "n": False,
    }
    norm = series.astype("string").str.strip().str.lower()
    out = norm.map(mapping)
    n_invalid = int((out.isna() & series.notna()).sum())
    return out.astype("boolean"), n_invalid


def clean(df: pd.DataFrame, *, is_training: bool = False) -> tuple[pd.DataFrame, DataQualityReport]:
    """Validate and clean a raw dataframe.

    Steps
    -----
    1. Schema check: required columns present; leakage columns dropped.
    2. Types: booleans, numerics, dates coerced; unparseable values become missing.
    3. Range check: numerics outside the data-dictionary range become missing.
    4. Categories: values outside the dictionary become missing; blanks that mean "None"
       are filled with "None".
    5. Jurisdiction inferred from OrganisationType when blank.
    """
    report = DataQualityReport(n_rows=len(df))
    df = df.copy()

    required = set(C.ALL_RAW_FEATURES) | {C.ID_COLUMN}
    if is_training:
        required.add(C.TARGET)
    report.missing_columns = sorted(required - set(df.columns))
    if report.missing_columns:
        raise ValueError(f"Input is missing required columns: {report.missing_columns}")
    known = required | set(C.LEAKAGE_COLUMNS) | {C.TARGET}
    report.unexpected_columns = sorted(set(df.columns) - known)
    if report.unexpected_columns:
        log.warning("Ignoring unexpected columns: %s", report.unexpected_columns)

    to_drop = [c for c in C.LEAKAGE_COLUMNS if c in df.columns]
    if to_drop:
        report.dropped_columns = to_drop
        df = df.drop(columns=to_drop)

    report.duplicate_references = int(df[C.ID_COLUMN].duplicated().sum())
    if report.duplicate_references:
        raise ValueError(
            f"{report.duplicate_references} duplicate {C.ID_COLUMN} values; case references must be unique"
        )
    if df[C.ID_COLUMN].isna().any():
        raise ValueError(f"{C.ID_COLUMN} contains missing values")

    # Booleans
    for col in C.BOOLEAN_COLUMNS:
        df[col], n_bad = _to_bool(df[col])
        if n_bad:
            report.invalid_booleans[col] = n_bad

    # Numerics + range validation
    for col, (lo, hi) in C.NUMERIC_RANGES.items():
        s = pd.to_numeric(df[col], errors="coerce")
        bad = s.notna() & ((s < lo) | (s > hi))
        if bad.any():
            report.out_of_range[col] = int(bad.sum())
            s = s.mask(bad)
        df[col] = s.astype("float64")

    # Dates
    for col in C.DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Categoricals: strip whitespace, validate against dictionary
    for col, levels in C.ORDINAL_LEVELS.items():
        df[col], n_bad = _validate_categories(df[col], levels)
        if n_bad:
            report.invalid_categories[col] = n_bad
    nominal_levels = NOMINAL_LEVELS
    for col in C.NOMINAL_COLUMNS:
        df[col], n_bad = _validate_categories(df[col], nominal_levels[col])
        if n_bad:
            report.invalid_categories[col] = n_bad

    # Blanks that mean "None"
    for col in C.STRUCTURAL_NONE_COLUMNS:
        n = int(df[col].isna().sum())
        if n:
            report.structural_none_filled[col] = n
            df[col] = df[col].fillna("None")

    # Jurisdiction is determined by organisation type
    inferred = df["OrganisationType"].map(C.ORGANISATION_JURISDICTION)
    fill_mask = df["Jurisdiction"].isna() & inferred.notna()
    report.jurisdiction_inferred = int(fill_mask.sum())
    df.loc[fill_mask, "Jurisdiction"] = inferred[fill_mask]

    if is_training:
        df[C.TARGET] = pd.to_numeric(df[C.TARGET], errors="coerce")
        bad_target = ~df[C.TARGET].isin(C.SEVERITY_CLASSES)
        if bad_target.any():
            log.warning("Dropping %d rows with invalid SeverityScore", int(bad_target.sum()))
            df = df.loc[~bad_target]
        df[C.TARGET] = df[C.TARGET].astype(int)

    report.missing_after_cleaning = {
        c: int(n) for c, n in df.isna().sum().items() if n > 0
    }
    log.info("Cleaning report: %s", report.as_dict())
    return df.reset_index(drop=True), report


def _validate_categories(series: pd.Series, levels: list[str]) -> tuple[pd.Series, int]:
    s = series.astype("string").str.strip()
    s = s.where(s != "", other=pd.NA)
    valid = s.isin(levels)
    n_bad = int((~valid & s.notna()).sum())
    return s.where(valid, other=pd.NA), n_bad


def _nominal_levels() -> dict[str, list[str]]:
    failure_types = [
        "Delay",
        "Communication",
        "Policy Application",
        "Record Keeping",
        "Clinical Failure",
        "Administrative Failure",
    ]
    return {
        "ComplaintCategory": [
            "Delay",
            "Communication",
            "Clinical Decision",
            "Administrative Error",
            "Incorrect Advice",
            "Service Quality",
            "Lost Information",
            "Eligibility Decision",
        ],
        "FailureTypePrimary": failure_types,
        "FailureTypeSecondary": ["None"] + failure_types,
        "Jurisdiction": ["Health", "Parliamentary"],
        "OrganisationType": list(C.ORGANISATION_JURISDICTION),
        "PrimaryVulnerability": [
            "None",
            "Physical Health",
            "Mental Health",
            "Learning Disability",
            "Homelessness",
            "Financial Hardship",
            "Caring Responsibilities",
        ],
        "ServiceArea": [
            "Acute Care",
            "Primary Care",
            "Mental Health",
            "Community Care",
            "Social Care",
            "Benefits",
            "Immigration",
            "Taxation",
            "Housing",
            "Justice",
        ],
    }


NOMINAL_LEVELS: dict[str, list[str]] = _nominal_levels()


def load_and_clean(path: str | Path, *, is_training: bool) -> tuple[pd.DataFrame, DataQualityReport]:
    return clean(load_raw(path), is_training=is_training)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[C.TARGET]), df[C.TARGET]
