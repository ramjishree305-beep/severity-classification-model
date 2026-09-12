"""Phase 5: score the holdback dataset with the final trained model.

Writes two files, deliberately not both called "predictions" so the
required submission artifact is never ambiguous:

  - outputs/holdback_predictions.csv: the required deliverable, exactly
    CaseReference + PredictedSeverityScore, nothing else.
  - outputs/holdback_operational_analysis.csv: a supporting file showing
    the Phase 3 business-risk decision layer (PredictedSevereProbability,
    RecommendedAction) on top of the same predictions -- not a required
    submission file, provided for transparency into the threshold
    trade-off discussed in docs/phase3_tuning_and_threshold.md.

Run: python src/predict.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from features import ID_COL, engineer_features, get_feature_columns, load_dataset

ROOT = Path(__file__).resolve().parent.parent
SEVERE_THRESHOLD = 0.30  # from docs/phase3_tuning_and_threshold.md


def main():
    pipeline = joblib.load(ROOT / "models" / "final_pipeline.joblib")

    df = load_dataset(ROOT / "data" / "holdback_dataset.csv")
    df = engineer_features(df)
    X = df[get_feature_columns()]

    pred_severity = pipeline.predict(X).ravel().astype(int)
    proba = pipeline.predict_proba(X)
    classes = pipeline.named_steps["clf"].classes_.ravel()
    severe_cols = [i for i, c in enumerate(classes) if c >= 4]
    severe_proba = proba[:, severe_cols].sum(axis=1)

    # Required deliverable: exactly CaseReference + PredictedSeverityScore.
    required = df[[ID_COL]].copy()
    required["PredictedSeverityScore"] = pred_severity

    # Supporting analysis file with the business-risk decision layer --
    # named "operational_analysis", not "predictions", so it's never
    # mistaken for a second copy of the required deliverable.
    analysis = required.copy()
    analysis["PredictedSevereProbability"] = np.round(severe_proba, 4)
    analysis["RecommendedAction"] = np.where(
        severe_proba >= SEVERE_THRESHOLD, "Progress for investigation", "Not progressed"
    )

    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    required_path = outputs_dir / "holdback_predictions.csv"
    analysis_path = outputs_dir / "holdback_operational_analysis.csv"
    required.to_csv(required_path, index=False)
    analysis.to_csv(analysis_path, index=False)

    print(f"Scored {len(required)} holdback cases -> {required_path} (REQUIRED submission file)")
    print(f"Supporting analysis -> {analysis_path} (not required, for transparency)")
    print(required["PredictedSeverityScore"].value_counts().sort_index())
    print(analysis["RecommendedAction"].value_counts())


if __name__ == "__main__":
    main()
