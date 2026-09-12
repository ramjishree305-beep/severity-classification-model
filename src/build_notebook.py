"""Builds notebooks/severity_model.ipynb -- a narrative walkthrough of the
whole pipeline that calls into src/ modules rather than duplicating logic,
so there's a single source of truth for every result.

Run: python src/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# Severity Score Classification

AI/ML Engineer Technical Assessment. Full write-up in `README.md`; this
notebook runs the same pipeline (from `src/`) end to end with inline
results, so nothing here duplicates logic that already lives in a tested
module.
""")

code("""import sys
sys.path.insert(0, '../src')
import pandas as pd
pd.set_option('display.max_columns', 50)
""")

md("""## Phase 1 — Data audit and leakage investigation

Full detail in `docs/phase1_audit.md` (regenerate with `python src/audit.py`).
Headline finding: `SeverityScore` distribution is imbalanced (class 6 = 101/2000
rows), and five columns turned out to be near-deterministic functions of the
target -- computed *after* severity is assessed -- so they're excluded from
modelling entirely.""")

code("""from features import load_dataset, engineer_features, TARGET

train_df = load_dataset('../data/training_dataset.csv')
holdback_df = load_dataset('../data/holdback_dataset.csv')
print('train:', train_df.shape, 'holdback:', holdback_df.shape)
train_df[TARGET].value_counts().sort_index()
""")

code("""suspects = ['EstimatedImpactScore', 'EstimatedRiskScore', 'ExpectedFinancialRedressGBP']
train_df[suspects + [TARGET]].corr()[TARGET]
""")

code("""train_df.groupby('PredictedRemedyBand')[TARGET].mean().sort_values()""")

code("""train_df.groupby('OmbudsmanInvestigationRequired')[TARGET].mean()""")

md("""**Verdict:** `EstimatedImpactScore` (r=0.94), `EstimatedRiskScore` (r=0.80),
`PredictedRemedyBand` (monotonic 1.2 -> 6.0), `OmbudsmanInvestigationRequired`
(1.93 vs 4.67), and `ExpectedFinancialRedressGBP` (r=0.56, borderline) are all
excluded as leakage. `CaseReference` is an id, `CaseCreatedDate` has no signal
(month vs severity correlation = -0.004, see `docs/phase1_audit.md`).""")

md("""## Feature engineering and preprocessing pipeline

Shared infrastructure (`src/features.py`), used by every phase below rather
than being its own numbered phase. It is the single source of truth for the locked feature set,
the three engineered features, and the reproducible `sklearn` `ColumnTransformer`
(median/most-frequent imputation, ordinal encoding for genuinely ordered
scales, one-hot for nominal categories). "None" is preserved as a real
category, not treated as missing, for the columns where the data dictionary
defines it as one.""")

code("""from features import build_preprocessor, get_feature_columns

df = engineer_features(train_df)
X = df[get_feature_columns()]
y = df[TARGET]

pre = build_preprocessor()
Xt = pre.fit_transform(X)
print('features in:', X.shape[1], '-> after encoding:', Xt.shape[1])
""")

md("""## Phase 2 — Baseline and candidate model comparison

A majority-class dummy baseline, Logistic Regression, Random Forest,
LightGBM, and CatBoost compared under identical stratified 5-fold CV
(seed 42). Full table and confusion matrices in
`docs/phase2_model_comparison.md` (regenerate with
`python src/model_comparison.py` -- takes a few minutes, not rerun here).
""")

code("""import pandas as pd
comparison_path = '../docs/phase2_model_comparison.md'
print(open(comparison_path).read()[:1400])
""")

md("""**Headline metric is `severe_recall`** (recall on SeverityScore >= 4), not
raw accuracy, per the brief's explicit low tolerance for missing severe
cases. All categorical features are ordinal/one-hot encoded by the shared
`ColumnTransformer` before reaching any model -- CatBoost is not using its
native categorical-feature handling here. It simply performs best on the
encoded categorical-heavy feature set (best severe-case F1 and macro F1),
so it was selected as the candidate to tune further.""")

md("""## Phase 3 — Hyperparameter tuning and the recall/workload trade-off

Grid search over CatBoost depth/iterations/learning_rate, scored on
severe-case F1 under the same CV scheme. Full table in
`docs/phase3_tuning_and_threshold.md` (regenerate with
`python src/tune_and_threshold.py`).""")

code("""print(open('../docs/phase3_tuning_and_threshold.md').read())""")

md("""**Proposed operating threshold: 0.30** on P(SeverityScore >= 4), instead of
the default 0.5 argmax cut-off. This raises severe-case recall from 0.759
to 0.900 at the cost of ~268 additional cases sent for investigation
review (on the 2,000 training rows). The brief gives no numeric recall
target, so 0.30 is a proposed operating point for business sign-off, not
a mathematically optimal answer -- it should move based on actual
investigative capacity. Note also the methodological caveat: this
threshold is chosen using the same out-of-fold predictions used to select
CatBoost's hyperparameters, so it's an internal training-set analysis,
not an independent, unbiased estimate (see `docs/phase3_tuning_and_threshold.md`
for the full discussion).""")

md("""## Phase 4 — Explainability and error analysis

SHAP values (global + the false-negative breakdown) in
`docs/phase4_explainability_error_analysis.md` and `outputs/shap_summary.png`
(regenerate with `python src/explainability.py`).""")

code("""from IPython.display import Image
Image('../outputs/shap_summary.png')""")

code("""print(open('../docs/phase4_explainability_error_analysis.md').read())""")

md("""SHAP shows that the model's strongest predictors (`RecoveryTimeMonths`,
`PhysicalImpactLevel`, `EmotionalImpactLevel`, `DurationAffectedDays`,
`VulnerabilityLevel`) line up with the severity factors named in the
brief -- a useful sanity check on model behaviour, though SHAP shows
influence on the model's predictions, not causal validity against
real-world severity.

False negatives (severe cases missed under the default argmax decision)
skew toward shorter duration and lower financial impact than correctly
caught severe cases -- i.e. borderline cases that look moderate on the
surface. Exploratory analysis of category shares did not reveal an
obvious large disparity across the vulnerability groups examined (this is
not a formal fairness assessment -- no per-group recall, confidence
intervals, or minimum sample-size checks were computed), though
Homelessness and Immigration cases were somewhat over-represented among
misses and are worth monitoring in production. A calibration check
(Brier score 0.107 vs a 0.210 prevalence baseline) suggests useful
probability ranking behaviour, but the scores are not independently
calibrated and should be read as a risk score, not a precise
probability.""")

md("""## Phase 5 — Final model and holdback predictions

Retrain on all 2,000 training rows with the tuned hyperparameters
(`src/train.py`), then score the 500-row holdback set (`src/predict.py`).
""")

code("""!python ../src/train.py""")
code("""!python ../src/predict.py""")

code("""preds = pd.read_csv('../outputs/holdback_predictions.csv')
preds.head(10)""")

code("""preds['PredictedSeverityScore'].value_counts().sort_index()""")

md("""## Summary

- Leakage audit removed 5 columns that would have made the model look
  deceptively strong while solving the wrong problem.
- CatBoost, tuned, selected on severe-case F1 (business-risk-aligned
  metric), not raw accuracy.
- Decision threshold lowered from the default 0.5 to 0.30 to trade
  investigation workload for severe-case recall, per the brief's stated
  risk tolerance.
- SHAP shows the model's strongest predictors align with the severity
  factors named in the brief; exploratory error analysis did not reveal
  an obvious large disparity by vulnerability group among missed severe
  cases, though this was not a formal fairness assessment.
- Final predictions: `outputs/holdback_predictions.csv`.

See `README.md` for the full write-up, repository structure, and
reproduction steps.""")

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

out_path = ROOT / "notebooks" / "severity_model.ipynb"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote {out_path}")
