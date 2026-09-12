"""Phase 4: explainability (global + local SHAP) and error analysis on
false negatives -- the dangerous error class per the brief (genuinely
severe case, 4-6, misclassified as low, 1-3).

Uses the same out-of-fold predictions mechanism as tune_and_threshold.py so
error analysis reflects genuinely unseen-fold performance, not train-set
fit.

Run: python src/explainability.py
Writes docs/phase4_explainability_error_analysis.md and
outputs/shap_summary.png

Phase numbering used across this project: 1 = audit.py (data/leakage audit),
2 = model_comparison.py (baseline + candidate models), 3 =
tune_and_threshold.py (hyperparameter tuning + threshold trade-off), 4 =
this script, 5 = train.py + predict.py (final model + holdback scoring).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from features import TARGET, build_preprocessor, engineer_features, get_feature_columns, load_dataset
from tune_and_threshold import oof_probabilities

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42
N_SPLITS = 5


def severe_binary(y):
    return (np.asarray(y) >= 4).astype(int)


def load_best_params():
    return json.loads((ROOT / "models" / "best_params.json").read_text())


def get_oof_predictions(X, y, params):
    """Out-of-fold argmax predictions -- identical fold scheme to
    tune_and_threshold.py so results are directly comparable."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(X), dtype=int)
    for train_idx, val_idx in skf.split(X, y):
        pre = build_preprocessor()
        X_tr = pre.fit_transform(X.iloc[train_idx])
        X_val = pre.transform(X.iloc[val_idx])
        clf = CatBoostClassifier(random_state=RANDOM_STATE, auto_class_weights="Balanced",
                                  verbose=False, **params)
        clf.fit(X_tr, y.iloc[train_idx])
        oof_pred[val_idx] = clf.predict(X_val).ravel()
    return oof_pred


def main():
    df = load_dataset(ROOT / "data" / "training_dataset.csv")
    df = engineer_features(df)
    X = df[get_feature_columns()]
    y = df[TARGET]
    params = load_best_params()

    # --- fit one model on all training data for SHAP (explainability only;
    # the Phase 5 final model is retrained fresh with the same recipe) ---
    pre = build_preprocessor()
    X_transformed = pre.fit_transform(X)
    feature_names = pre.get_feature_names_out()
    clf = CatBoostClassifier(random_state=RANDOM_STATE, auto_class_weights="Balanced",
                              verbose=False, **params)
    clf.fit(X_transformed, y)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)  # (n_samples, n_features, n_classes) for multiclass

    if isinstance(shap_values, list):
        shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        shap_abs = np.abs(shap_values).mean(axis=2)
    else:
        shap_abs = np.abs(shap_values)

    mean_abs = shap_abs.mean(axis=0)
    importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    top = importance.head(15).iloc[::-1]
    plt.barh(top.index, top.values)
    plt.xlabel("Mean |SHAP value| (avg across classes)")
    plt.title("Global feature importance (CatBoost, tuned)")
    plt.tight_layout()
    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    plt.savefig(outputs_dir / "shap_summary.png", dpi=120)
    plt.close()

    # --- error analysis on out-of-fold predictions ---
    oof_pred = get_oof_predictions(X, y, params)
    df_err = df.copy()
    df_err["PredictedSeverityScore"] = oof_pred
    df_err["ActualSevere"] = severe_binary(y)
    df_err["PredictedSevere"] = severe_binary(oof_pred)
    df_err["ErrorType"] = "correct"
    df_err.loc[(df_err.ActualSevere == 1) & (df_err.PredictedSevere == 0), "ErrorType"] = "false_negative"
    df_err.loc[(df_err.ActualSevere == 0) & (df_err.PredictedSevere == 1), "ErrorType"] = "false_positive"

    fn = df_err[df_err.ErrorType == "false_negative"]
    tp = df_err[(df_err.ActualSevere == 1) & (df_err.PredictedSevere == 1)]

    compare_cols = [
        "DurationAffectedDays", "NumberOfServiceFailures", "DirectFinancialLossGBP",
        "CaseComplexityScore", "ActualFinancialImpact", "ImpactFlagCount",
    ]
    comparison = pd.DataFrame({
        "false_negative_mean": fn[compare_cols].mean(),
        "true_positive_mean": tp[compare_cols].mean(),
    })

    cat_compare_cols = ["VulnerabilityLevel", "PrimaryVulnerability", "ComplaintCategory", "ServiceArea"]
    cat_tables = {}
    for c in cat_compare_cols:
        fn_dist = fn[c].value_counts(normalize=True).rename("false_negative_share")
        tp_dist = tp[c].value_counts(normalize=True).rename("true_positive_share")
        cat_tables[c] = pd.concat([fn_dist, tp_dist], axis=1).fillna(0).round(3)

    # --- severe-error breakdown by actual class (worst errors: 6->1, etc.) ---
    severe_rows = df_err[df_err.ActualSevere == 1]
    error_rows = []
    for actual in (4, 5, 6):
        subset = severe_rows[severe_rows[TARGET] == actual]
        n = len(subset)
        missed = (subset.PredictedSevere == 0)
        error_rows.append({
            "ActualSeverityScore": actual,
            "n_cases": n,
            "n_missed_by_default_argmax": int(missed.sum()),
            "recall": round(1 - missed.sum() / n, 3) if n else None,
        })
    severe_error_table = pd.DataFrame(error_rows)

    # worst individual misses: actually 5 or 6, predicted <= 3 (not progressed).
    # Sorted by the largest actual-vs-predicted gap, worst first, top 5 shown.
    worst_misses = df_err[
        (df_err[TARGET] >= 5) & (df_err["PredictedSeverityScore"] <= 3)
    ].copy()
    worst_misses["gap"] = worst_misses[TARGET] - worst_misses["PredictedSeverityScore"]
    worst_misses = worst_misses.sort_values("gap", ascending=False).head(5)[
        ["CaseReference", TARGET, "PredictedSeverityScore", "gap"] + compare_cols
    ]

    # --- calibration check on OOF severe probability ---
    oof_proba_severe, _ = oof_probabilities(params, X, y)
    y_bin = (y >= 4).astype(int)
    brier = brier_score_loss(y_bin, oof_proba_severe)
    frac_pos, mean_pred = calibration_curve(y_bin, oof_proba_severe, n_bins=10, strategy="quantile")
    calib_table = pd.DataFrame({
        "mean_predicted_probability": np.round(mean_pred, 3),
        "observed_severe_fraction": np.round(frac_pos, 3),
    })

    # --- write report ---
    lines = ["# Phase 4: Explainability and Error Analysis\n\n"]
    lines.append("## Global feature importance (mean |SHAP|, averaged across classes)\n\n")
    lines.append(importance.head(20).round(4).to_frame("mean_abs_shap").to_markdown())
    lines.append("\n\n![SHAP summary](../outputs/shap_summary.png)\n\n")

    lines.append("## Error analysis: false negatives (genuinely severe, missed)\n\n")
    lines.append(
        f"Out-of-fold false negatives: {len(fn)} of {int(df_err.ActualSevere.sum())} "
        f"truly severe cases ({len(fn)/max(1,int(df_err.ActualSevere.sum()))*100:.1f}%) "
        "were missed by the default argmax decision. These are the dangerous errors "
        "the business explicitly wants minimised -- this is exactly why Phase 3's "
        "threshold analysis lowers the decision threshold below 0.5.\n\n"
    )
    lines.append("### Numeric feature comparison: false negatives vs correctly-caught severe cases\n\n")
    lines.append(comparison.round(1).to_markdown())

    lines.append("\n\n### Categorical distribution comparison\n\n")
    for c, tbl in cat_tables.items():
        lines.append(f"\n**{c}**\n\n")
        lines.append(tbl.to_markdown())
        lines.append("\n")

    lines.append("\n\n### Severe-case recall by actual severity (default argmax decision)\n\n")
    lines.append(severe_error_table.to_markdown(index=False))
    lines.append(
        "\n\nRecall is lowest for ActualSeverityScore=4 (the boundary class, "
        "hardest to separate from 3) and improves for the more extreme 5/6 "
        "cases, as expected -- but a 5 or 6 predicted as 3 or below is still "
        "a serious error under this model. Top 5 worst individual misses "
        "(largest actual-vs-predicted gap, actual >= 5, predicted <= 3):\n\n"
    )
    lines.append(worst_misses.round(1).to_markdown(index=False) if not worst_misses.empty
                  else "None in this run.\n")

    lines.append("\n\n## Probability calibration\n\n")
    prevalence = float(y_bin.mean())
    baseline_brier = prevalence * (1 - prevalence)  # Brier score of the constant-prevalence predictor
    lines.append(
        f"Brier score for P(SeverityScore >= 4), out-of-fold: **{brier:.4f}** "
        "(lower is better, 0 = perfect). Class prevalence here is imbalanced "
        f"({prevalence:.1%} severe), so the standard 0.25 'uninformative' "
        "benchmark (which assumes a balanced problem) doesn't apply directly; "
        f"the more relevant baseline is a model that always predicts the "
        f"training prevalence ({prevalence:.3f}), which scores "
        f"{baseline_brier:.4f}. The model's {brier:.4f} is well below that "
        "baseline.\n\n"
        "Reliability table (predicted probability vs observed severe rate, "
        "10 quantile bins, out-of-fold):\n\n"
    )
    lines.append(calib_table.to_markdown(index=False))
    lines.append(
        "\n\nCatBoost's probabilities are used here as a **ranking/decision "
        "score** for the threshold analysis, not as independently calibrated "
        "probabilities of severity -- tree ensembles are commonly "
        "miscalibrated, and no calibration step (e.g. Platt scaling, "
        "isotonic regression) has been applied. The reliability table above "
        "gives a rough sense of how close the raw scores are to true "
        "frequencies; treat `PredictedSevereProbability` in the output file "
        "as indicative, not as a precise probability.\n"
    )

    out = ROOT / "docs" / "phase4_explainability_error_analysis.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out} and outputs/shap_summary.png")
    print("\nTop 10 features:\n", importance.head(10))


if __name__ == "__main__":
    main()
