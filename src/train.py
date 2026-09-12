"""Phase 5: train the final model on all 2,000 training records, using the
locked feature set (docs/phase1_audit.md) and tuned CatBoost hyperparameters
(docs/phase3_tuning_and_threshold.md).

Run: python src/train.py
Writes models/final_pipeline.joblib
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

from features import TARGET, build_preprocessor, engineer_features, get_feature_columns, load_dataset

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42


def main():
    df = load_dataset(ROOT / "data" / "training_dataset.csv")
    df = engineer_features(df)
    X = df[get_feature_columns()]
    y = df[TARGET]

    best_params = json.loads((ROOT / "models" / "best_params.json").read_text())

    pipeline = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", CatBoostClassifier(
            random_state=RANDOM_STATE, auto_class_weights="Balanced", verbose=False, **best_params
        )),
    ])
    pipeline.fit(X, y)

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(pipeline, models_dir / "final_pipeline.joblib")
    print(f"Trained on {len(X)} rows. Saved models/final_pipeline.joblib")


if __name__ == "__main__":
    main()
