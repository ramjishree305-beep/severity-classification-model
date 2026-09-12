"""Phase 3: hyperparameter-tune the CatBoost candidate (best severe-case F1
in Phase 2), then explore the recall/workload trade-off by thresholding the
predicted probability of severe (SeverityScore>=4) rather than using the
model's default argmax decision.

Uses out-of-fold predictions (5-fold stratified CV) throughout, so nothing
here touches the holdback set or overfits to a single split.

Run: python src/tune_and_threshold.py
Writes docs/phase3_tuning_and_threshold.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline

from features import TARGET, build_preprocessor, engineer_features, get_feature_columns, load_dataset

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42
N_SPLITS = 5


def severe_binary(y):
    return (np.asarray(y) >= 4).astype(int)


def make_pipeline(params):
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", CatBoostClassifier(
            random_state=RANDOM_STATE, auto_class_weights="Balanced", verbose=False, **params
        )),
    ])


def cv_severe_f1(params, X, y, skf):
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        pipe = make_pipeline(params)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = pipe.predict(X.iloc[val_idx]).ravel()
        scores.append(f1_score(severe_binary(y.iloc[val_idx]), severe_binary(preds), zero_division=0))
    return float(np.mean(scores))


def tune(X, y):
    grid = ParameterGrid({
        "iterations": [300, 600],
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.1],
    })
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for params in grid:
        score = cv_severe_f1(params, X, y, skf)
        results.append({**params, "severe_f1": score})
    results_df = pd.DataFrame(results).sort_values("severe_f1", ascending=False)
    best_params = {k: v for k, v in results_df.iloc[0].items() if k != "severe_f1"}
    best_params["iterations"] = int(best_params["iterations"])
    best_params["depth"] = int(best_params["depth"])
    return best_params, results_df


def oof_probabilities(params, X, y):
    """Out-of-fold P(SeverityScore in {4,5,6}) for every training row."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(X))
    oof_pred_default = np.zeros(len(X), dtype=int)
    classes_ref = None
    for train_idx, val_idx in skf.split(X, y):
        pipe = make_pipeline(params)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        clf = pipe.named_steps["clf"]
        classes_ref = clf.classes_.ravel()
        proba = pipe.predict_proba(X.iloc[val_idx])
        severe_cols = [i for i, c in enumerate(classes_ref) if c >= 4]
        oof_proba[val_idx] = proba[:, severe_cols].sum(axis=1)
        oof_pred_default[val_idx] = pipe.predict(X.iloc[val_idx]).ravel()
    return oof_proba, oof_pred_default


def threshold_sweep(y_true_binary, proba):
    rows = []
    for t in np.arange(0.1, 0.95, 0.05):
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": round(t, 2),
            "recall": recall_score(y_true_binary, pred, zero_division=0),
            "precision": precision_score(y_true_binary, pred, zero_division=0),
            "f1": f1_score(y_true_binary, pred, zero_division=0),
            "n_flagged": int(pred.sum()),
        })
    return pd.DataFrame(rows)


def main():
    df = load_dataset(ROOT / "data" / "training_dataset.csv")
    df = engineer_features(df)
    X = df[get_feature_columns()]
    y = df[TARGET]
    y_bin = pd.Series(severe_binary(y), index=y.index)

    best_params, grid_results = tune(X, y)
    print("Best params:", best_params)

    oof_proba, oof_pred_default = oof_probabilities(best_params, X, y)
    default_binary = severe_binary(oof_pred_default)
    default_metrics = {
        "recall": recall_score(y_bin, default_binary),
        "precision": precision_score(y_bin, default_binary),
        "f1": f1_score(y_bin, default_binary),
        "n_flagged": int(default_binary.sum()),
    }

    sweep = threshold_sweep(y_bin, oof_proba)
    n_actual_severe = int(y_bin.sum())

    lines = ["# Phase 3: Hyperparameter Tuning and Threshold Trade-off\n\n"]
    lines.append("## CatBoost hyperparameter search (5-fold CV, scored on severe-case F1)\n\n")
    lines.append(grid_results.round(3).to_markdown(index=False))
    lines.append(f"\n\nBest params: `{best_params}`\n\n")

    lines.append("## Default (argmax) multiclass decision, mapped to binary\n\n")
    lines.append(pd.DataFrame([default_metrics]).round(3).to_markdown(index=False))
    lines.append(f"\n\nActual severe cases in training data: {n_actual_severe} / {len(y)}\n\n")

    lines.append("## Threshold sweep on P(SeverityScore >= 4), out-of-fold\n\n")
    lines.append(
        "**Note on methodology:** these probabilities come from 5-fold "
        "out-of-fold predictions, so each row is scored by a model that did "
        "not see it during training -- this is better than reading "
        "thresholds off training-set predictions. However, the same 2,000 "
        "labelled rows were also used to select the CatBoost hyperparameters "
        "above, so this is an **internal training-set operating-point "
        "analysis, not an independent, unbiased performance estimate**. "
        "The holdback set has no labels, so no unbiased estimate of "
        "holdback performance is possible with the data provided. Numbers "
        "below (869 flagged, etc.) describe the 2,000 training records, "
        "*not* the 500-row holdback set.\n\n"
    )
    lines.append(
        "Raising the threshold trades precision for workload; lowering it "
        "trades workload for recall. The brief states a low tolerance for "
        "missing severe cases but does not specify a required recall level, "
        "so the table below is presented as a trade-off for business "
        "sign-off, not as a single \"correct\" answer:\n\n"
    )
    lines.append(sweep.round(3).to_markdown(index=False))

    rec = sweep[sweep["threshold"] == 0.30].iloc[0]
    lines.append(
        f"\n\n**Proposed operating point: threshold {rec['threshold']:.2f}** -> "
        f"recall {rec['recall']:.3f}, precision {rec['precision']:.3f}, "
        f"{int(rec['n_flagged'])} of {len(y)} training cases would be flagged "
        f"for investigation (vs {n_actual_severe} truly severe, "
        f"{int(rec['n_flagged'])-n_actual_severe} additional reviews). "
        "This is proposed, not derived from a stated business requirement "
        "(the brief gives no numeric recall target) -- it should be agreed "
        "with the business against actual investigative capacity before use, "
        "and can be moved along the table above in either direction.\n"
    )

    out = ROOT / "docs" / "phase3_tuning_and_threshold.md"
    out.write_text("".join(lines), encoding="utf-8")

    import json
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "best_params.json").write_text(json.dumps(best_params, indent=2))
    print(f"Wrote {out} and models/best_params.json")


if __name__ == "__main__":
    main()
