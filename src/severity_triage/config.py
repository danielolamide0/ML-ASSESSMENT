"""Central configuration: column groups, data-dictionary ranges, business parameters.

Everything that encodes an assumption about the data or the business lives here so that it
is visible in one place and can be reviewed without reading the modelling code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = DATA_DIR / "training_dataset.csv"
HOLDBACK_PATH = DATA_DIR / "holdback_dataset.csv"
MODEL_PATH = MODEL_DIR / "severity_model.joblib"

RANDOM_SEED = 42

ID_COLUMN = "CaseReference"
TARGET = "SeverityScore"
SEVERITY_CLASSES = [1, 2, 3, 4, 5, 6]

# Business rule from the brief: 1-3 not progressed, 4-6 progressed for investigation.
PROGRESS_THRESHOLD_SCORE = 4

# Columns that are only known *after* the severity decision has been made. They must never be
# used as model inputs. OmbudsmanInvestigationRequired is a deterministic restatement of the
# target (Yes <=> SeverityScore >= 4) and is entirely absent from the holdback data.
LEAKAGE_COLUMNS = ["OmbudsmanInvestigationRequired"]

# Internal assessments produced during the initial review. They are strong predictors and are
# present in the holdback data, so they are used by default. `--exclude-internal-scores`
# trains a model without them to show how much performance depends on them.
INTERNAL_ASSESSMENT_COLUMNS = [
    "EstimatedImpactScore",
    "EstimatedRiskScore",
    "ExpectedFinancialRedressGBP",
    "PredictedRemedyBand",
]

DATE_COLUMNS = ["CaseCreatedDate"]

BOOLEAN_COLUMNS = [
    "AdditionalTreatmentRequired",
    "AnxietyReported",
    "BereavedPerson",
    "DepressionReported",
    "DeteriorationInHealth",
    "EscalatedInternally",
    "ImpactOnRelationships",
    "LegalChallengePotential",
    "LongTermImpactFlag",
    "MultipleComplaintThemes",
    "RepeatedFailurePattern",
    "SafeguardingConcern",
    "SleepDisruptionReported",
]

# Ordered categoricals: the order is meaningful and is encoded as an integer.
ORDINAL_LEVELS: dict[str, list[str]] = {
    "AgeBand": ["Under 18", "18-34", "35-49", "50-64", "65-79", "80+"],
    "EmotionalImpactLevel": ["Minimal", "Low", "Moderate", "Significant", "Severe"],
    "PhysicalImpactLevel": ["None", "Minor", "Moderate", "Significant", "Severe"],
    "VulnerabilityLevel": ["None", "Low", "Moderate", "High"],
    "EvidenceStrength": ["Limited", "Moderate", "Strong"],
    "InvestigationRoute": ["Standard", "Fast Track", "Escalated"],
    "PredictedRemedyBand": ["A", "B", "C", "D", "E", "F"],
}

NOMINAL_COLUMNS = [
    "ComplaintCategory",
    "FailureTypePrimary",
    "FailureTypeSecondary",
    "Jurisdiction",
    "OrganisationType",
    "PrimaryVulnerability",
    "ServiceArea",
]

# Categorical columns where a blank means the "None" category rather than an unknown value.
# Evidence: the data dictionary lists "None" as an allowed value, "None" never appears in the
# raw data, and blanks have the lowest mean severity of any level.
STRUCTURAL_NONE_COLUMNS = [
    "PhysicalImpactLevel",
    "VulnerabilityLevel",
    "PrimaryVulnerability",
    "FailureTypeSecondary",
]

# Allowed (min, max) ranges from the data dictionary. Values outside are treated as recording
# errors and set to missing (a negative cost or a 0-100 score of 108 cannot be genuine).
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "AdditionalCostsGBP": (0, 50_000),
    "CaseComplexityScore": (1, 10),
    "DelayInResolutionDays": (0, 2_000),
    "DirectFinancialLossGBP": (0, 100_000),
    "DurationAffectedDays": (1, 3_000),
    "EstimatedImpactScore": (1, 100),
    "EstimatedRiskScore": (1, 100),
    "ExpectedFinancialRedressGBP": (0, 75_000),
    "LostIncomeGBP": (0, 100_000),
    "NumberOfDependentsAffected": (0, 10),
    "NumberOfOrganisationsInvolved": (1, 5),
    "NumberOfServiceFailures": (1, 20),
    "PriorComplaintsRaised": (0, 15),
    "RecoveryTimeMonths": (0, 120),
    "RepeatFailuresCount": (0, 10),
}

NUMERIC_COLUMNS = list(NUMERIC_RANGES)

MONEY_COLUMNS = [
    "AdditionalCostsGBP",
    "DirectFinancialLossGBP",
    "LostIncomeGBP",
    "ExpectedFinancialRedressGBP",
]

# Jurisdiction is a function of the organisation type; used to fill missing Jurisdiction.
ORGANISATION_JURISDICTION = {
    "NHS Trust": "Health",
    "GP Practice": "Health",
    "Integrated Care Board": "Health",
    "Government Department": "Parliamentary",
    "Agency": "Parliamentary",
    "Local Authority": "Parliamentary",
}

HEALTH_IMPACT_FLAGS = [
    "AnxietyReported",
    "DepressionReported",
    "DeteriorationInHealth",
    "SleepDisruptionReported",
    "AdditionalTreatmentRequired",
    "ImpactOnRelationships",
]

ALL_RAW_FEATURES = (
    NUMERIC_COLUMNS + BOOLEAN_COLUMNS + list(ORDINAL_LEVELS) + NOMINAL_COLUMNS + DATE_COLUMNS
)


@dataclass(frozen=True)
class BusinessCosts:
    """Relative cost of triage errors, used to choose the decision threshold.

    The brief says missing a genuinely severe case (a false negative) is far worse than
    progressing a less severe one (a false positive). The ratio, not the absolute numbers,
    drives the threshold. 5:1 is an assumption that should be confirmed with the business.
    """

    false_negative: float = 5.0
    false_positive: float = 1.0


@dataclass
class TrainingConfig:
    seed: int = RANDOM_SEED
    n_folds: int = 5
    exclude_internal_scores: bool = False
    costs: BusinessCosts = field(default_factory=BusinessCosts)
    # Minimum recall on severe cases that the chosen threshold must achieve out-of-fold.
    min_severe_recall: float = 0.95
    lgbm_params: dict = field(
        default_factory=lambda: {
            "objective": "multiclass",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "class_weight": "balanced",
            "n_jobs": 1,
            "verbose": -1,
        }
    )
