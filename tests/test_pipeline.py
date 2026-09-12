"""Small smoke-test suite. Not exhaustive -- covers the properties that
would silently break the submission if violated: leakage columns creeping
back in, malformed holdback output, or out-of-range predictions.

Run: python -m pytest tests/ -v
(or, without pytest installed: python tests/test_pipeline.py)
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import EXCLUDED, TARGET, get_feature_columns, load_dataset  # noqa: E402


def test_feature_builder_excludes_target_and_leakage_columns():
    cols = get_feature_columns()
    assert TARGET not in cols
    for excluded_col in EXCLUDED:
        assert excluded_col not in cols, f"{excluded_col} should be excluded (leakage/id/no-signal)"


def test_output_has_500_rows():
    out_path = ROOT / "outputs" / "holdback_predictions.csv"
    assert out_path.exists(), "run src/predict.py first"
    df = pd.read_csv(out_path)
    assert len(df) == 500


def test_required_output_columns_exist():
    df = pd.read_csv(ROOT / "outputs" / "holdback_predictions.csv")
    assert list(df.columns) == ["CaseReference", "PredictedSeverityScore"]
    assert df["CaseReference"].is_unique


def test_prediction_scores_are_1_to_6():
    df = pd.read_csv(ROOT / "outputs" / "holdback_predictions.csv")
    assert df["PredictedSeverityScore"].between(1, 6).all()
    assert df["PredictedSeverityScore"].isna().sum() == 0


def test_operational_analysis_recommended_action_values():
    df = pd.read_csv(ROOT / "outputs" / "holdback_operational_analysis.csv")
    assert set(df["RecommendedAction"].unique()) <= {"Progress for investigation", "Not progressed"}
    assert df["PredictedSevereProbability"].between(0, 1).all()


def test_recommended_action_matches_threshold():
    """RecommendedAction must exactly follow the documented 0.30 cut-off on
    PredictedSevereProbability -- catches drift if predict.py's threshold
    constant and its output ever fall out of sync."""
    from predict import SEVERE_THRESHOLD

    df = pd.read_csv(ROOT / "outputs" / "holdback_operational_analysis.csv")
    expected = df["PredictedSevereProbability"].ge(SEVERE_THRESHOLD).map(
        {True: "Progress for investigation", False: "Not progressed"}
    )
    assert (df["RecommendedAction"] == expected).all()


def test_final_pipeline_artifact_exists():
    assert (ROOT / "models" / "final_pipeline.joblib").exists()
    assert (ROOT / "models" / "best_params.json").exists()


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
