"""Phase 2: compare candidate model families under identical stratified
5-fold CV. Headline business metric is severe-case (SeverityScore 4-6)
recall, not raw accuracy -- see docs/phase1_audit.md and the brief's
explicit low tolerance for missing severe cases.

Run: python src/model_comparison.py
Writes docs/phase2_model_comparison.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import (
    NOMINAL_COLS,
    TARGET,
    build_preprocessor,
    engineer_features,
    get_feature_columns,
    load_dataset,
)

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42


def severe_binary(y):
    return (np.asarray(y) >= 4).astype(int)


def evaluate_fold(y_true, y_pred):
    yb_true, yb_pred = severe_binary(y_true), severe_binary(y_pred)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "severe_recall": recall_score(yb_true, yb_pred, zero_division=0),
        "severe_precision": precision_score(yb_true, yb_pred, zero_division=0),
        "severe_f1": f1_score(yb_true, yb_pred, zero_division=0),
    }


def build_models():
    return {
        "MajorityClassBaseline": Pipeline([
            ("pre", build_preprocessor()),
            ("clf", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
        ]),
        "LogisticRegression": Pipeline([
            ("pre", build_preprocessor()),
            ("scale", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("pre", build_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "LightGBM": Pipeline([
            ("pre", build_preprocessor()),
            ("clf", LGBMClassifier(
                n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, verbosity=-1
            )),
        ]),
        "CatBoost": Pipeline([
            ("pre", build_preprocessor()),
            ("clf", CatBoostClassifier(
                iterations=400, auto_class_weights="Balanced", random_state=RANDOM_STATE, verbose=False
            )),
        ]),
    }


def run_cv(models, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = {name: [] for name in models}
    last_fold_preds = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for name, model in models.items():
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            results[name].append(evaluate_fold(y_val, preds))
            if fold == n_splits - 1:
                last_fold_preds[name] = (y_val, preds)

    return results, last_fold_preds


def summarize(results):
    rows = []
    for name, folds in results.items():
        df = pd.DataFrame(folds)
        row = {"model": name}
        for col in df.columns:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"] = df[col].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def main():
    df = load_dataset(ROOT / "data" / "training_dataset.csv")
    df = engineer_features(df)
    X = df[get_feature_columns()]
    y = df[TARGET]

    models = build_models()
    results, last_fold_preds = run_cv(models, X, y)
    summary = summarize(results)

    lines = ["# Phase 2 Model Comparison\n\n",
             "5-fold stratified CV, same folds across all models. "
             "Class weights balanced throughout given the target imbalance "
             "(class 6 has only 101/2000 examples).\n\n",
             "Headline business metric: **severe_recall** (recall on SeverityScore>=4), "
             "given the brief's stated low tolerance for missing severe cases.\n\n"]

    metric_cols = ["macro_f1_mean", "balanced_accuracy_mean", "severe_recall_mean",
                   "severe_precision_mean", "severe_f1_mean"]
    lines.append(summary[metric_cols].round(3).to_markdown())
    lines.append("\n\n## Full stats (mean +/- std)\n\n")
    lines.append(summary.round(3).to_markdown())

    lines.append("\n\n## Confusion matrix, final CV fold (multiclass)\n\n")
    for name, (y_val, preds) in last_fold_preds.items():
        cm = confusion_matrix(y_val, preds, labels=[1, 2, 3, 4, 5, 6])
        cm_df = pd.DataFrame(cm, index=[f"true_{i}" for i in range(1, 7)],
                              columns=[f"pred_{i}" for i in range(1, 7)])
        lines.append(f"\n**{name}**\n\n")
        lines.append(cm_df.to_markdown())
        lines.append("\n")

    out = ROOT / "docs" / "phase2_model_comparison.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(summary[metric_cols].round(3))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
