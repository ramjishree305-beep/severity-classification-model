"""Feature definitions and preprocessing pipeline.

Feature set is locked in from the Phase 1 leakage audit (docs/phase1_audit.md).
Columns excluded: CaseReference (id), CaseCreatedDate (no signal), and the five
leakage columns (EstimatedImpactScore, EstimatedRiskScore, PredictedRemedyBand,
OmbudsmanInvestigationRequired, ExpectedFinancialRedressGBP).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

TARGET = "SeverityScore"
ID_COL = "CaseReference"

EXCLUDED = [
    "CaseReference",
    "CaseCreatedDate",
    "EstimatedImpactScore",
    "EstimatedRiskScore",
    "PredictedRemedyBand",
    "OmbudsmanInvestigationRequired",
    "ExpectedFinancialRedressGBP",
]

BOOLEAN_COLS = [
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

# Ordinal categoricals with a genuine, evenly-spirited rank order (per data
# dictionary wording). AgeBand is deliberately one-hot encoded instead --
# its bands are unequal-width so imposing a linear ordinal scale is not
# obviously correct (see docs/phase1_audit.md discussion).
ORDINAL_COLS = {
    "EmotionalImpactLevel": ["Minimal", "Low", "Moderate", "Significant", "Severe"],
    "PhysicalImpactLevel": ["None", "Minor", "Moderate", "Significant", "Severe"],
    "VulnerabilityLevel": ["None", "Low", "Moderate", "High"],
    "EvidenceStrength": ["Limited", "Moderate", "Strong"],
}

NOMINAL_COLS = [
    "AgeBand",
    "ComplaintCategory",
    "FailureTypePrimary",
    "FailureTypeSecondary",
    "InvestigationRoute",
    "Jurisdiction",
    "OrganisationType",
    "PrimaryVulnerability",
    "ServiceArea",
]

NUMERIC_COLS = [
    "AdditionalCostsGBP",
    "CaseComplexityScore",
    "DelayInResolutionDays",
    "DirectFinancialLossGBP",
    "DurationAffectedDays",
    "LostIncomeGBP",
    "NumberOfDependentsAffected",
    "NumberOfOrganisationsInvolved",
    "NumberOfServiceFailures",
    "PriorComplaintsRaised",
    "RecoveryTimeMonths",
    "RepeatFailuresCount",
]

NA_VALUES = ["", "NA", "N/A", "NULL", "null", "NaN"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a raw CSV export, preserving 'None' as a real category value."""
    return pd.read_csv(
        path,
        parse_dates=["CaseCreatedDate"],
        keep_default_na=False,
        na_values=NA_VALUES,
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add defensible derived features. Kept separate from raw fields so
    each can be ablated independently during model comparison."""
    df = df.copy()
    df["ActualFinancialImpact"] = (
        df["DirectFinancialLossGBP"].fillna(0)
        + df["LostIncomeGBP"].fillna(0)
        + df["AdditionalCostsGBP"].fillna(0)
    )
    impact_flags = [
        "AnxietyReported",
        "DepressionReported",
        "DeteriorationInHealth",
        "SleepDisruptionReported",
        "BereavedPerson",
    ]
    df["ImpactFlagCount"] = (df[impact_flags] == "Yes").sum(axis=1)
    df["FailureBurden"] = df["NumberOfServiceFailures"] * df["RepeatFailuresCount"].fillna(0)
    return df

ENGINEERED_NUMERIC_COLS = ["ActualFinancialImpact", "ImpactFlagCount", "FailureBurden"]


def build_preprocessor(use_engineered: bool = True) -> ColumnTransformer:
    numeric_cols = NUMERIC_COLS + (ENGINEERED_NUMERIC_COLS if use_engineered else [])

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    boolean_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[["No", "Yes"]] * len(BOOLEAN_COLS))),
    ])
    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=list(ORDINAL_COLS.values()))),
    ])
    nominal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("boolean", boolean_pipe, BOOLEAN_COLS),
        ("ordinal", ordinal_pipe, list(ORDINAL_COLS.keys())),
        ("nominal", nominal_pipe, NOMINAL_COLS),
    ])


def get_feature_columns(use_engineered: bool = True) -> list[str]:
    numeric_cols = NUMERIC_COLS + (ENGINEERED_NUMERIC_COLS if use_engineered else [])
    return numeric_cols + BOOLEAN_COLS + list(ORDINAL_COLS.keys()) + NOMINAL_COLS
