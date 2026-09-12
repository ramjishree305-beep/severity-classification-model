# Severity Score Classification

AI/ML Engineer Technical Assessment — predicts a 1-6 `SeverityScore` for
public-service complaint cases, used to triage cases into "Not progressed"
(1-3) vs "Progressed for investigation" (4-6).

## Problem framing

The brief is explicit that the organisation has **low tolerance for severe
cases being incorrectly classified as low severity** — a false negative
(missing a genuinely severe case) is far more costly than a false positive
(over-investigating a less severe one). This shaped every modelling
decision below: metric choice, class weighting, and the final decision
threshold are all chosen to protect recall on severe cases, not to
maximise raw accuracy.

The deliverable asks for a predicted *severity score* (1-6), not just the
binary decision, so the model is trained as a 6-class classifier and the
binary triage decision is derived from it afterwards.

## Approach summary

The numbered steps below are a narrative walkthrough, not identical to the
five "Phase" labels used in the `src/` script docstrings and `docs/`
report filenames (1 = `audit.py`, 2 = `model_comparison.py`, 3 =
`tune_and_threshold.py`, 4 = `explainability.py`, 5 = `train.py`/`predict.py`)
— preprocessing and feature engineering (steps 2-3 below) are implemented
in `features.py` and used by every phase rather than being their own script.

1. **Data audit & leakage investigation** (`src/audit.py` → `docs/phase1_audit.md`)
   Five columns were found to be near-deterministic functions of the
   target and **excluded from modelling**:
   - `EstimatedImpactScore` (Pearson r = 0.94 with SeverityScore)
   - `EstimatedRiskScore` (r = 0.80)
   - `PredictedRemedyBand` (mean severity rises monotonically A=1.2 → F=6.0)
     — same reasoning as `OmbudsmanInvestigationRequired`: a remedy
     recommendation is plausibly set after severity is assessed, based on
     the near-perfect ordering and business meaning, not a confirmed
     data-generation timeline
   - `OmbudsmanInvestigationRequired` (mean severity 1.93 vs 4.67 by category)
     — treated as leakage because its near-deterministic relationship with
     severity and its business meaning strongly suggest it represents a
     downstream investigation decision rather than information available
     at initial triage; not confirmed against a data-generation spec
   - `ExpectedFinancialRedressGBP` (r = 0.56; excluded because its business
     meaning suggests it is determined after severity assessment, though
     this is an assumption, not a confirmed fact about the data-generation
     process — see Assumptions below)

   Also excluded: `CaseReference` (identifier) and `CaseCreatedDate` (month
   vs severity correlation = -0.004, no usable signal).

   All remaining fields — including two undocumented columns present in
   the data but missing from the Data Dictionary tab, `AdditionalTreatmentRequired`
   and `ImpactOnRelationships` — were retained as candidate predictors
   after this audit; their use assumes they are genuinely available at
   initial triage, which was not independently confirmed.

2. **Preprocessing** (`src/features.py`)
   Reproducible `sklearn` `ColumnTransformer`: median imputation for
   numerics, most-frequent imputation for categoricals, ordinal encoding
   for genuinely ordered scales (`EmotionalImpactLevel`, `PhysicalImpactLevel`,
   `VulnerabilityLevel`, `EvidenceStrength`), one-hot for nominal categories.
   `AgeBand` is one-hot rather than ordinal, since its bands are unequal
   width and a linear ordinal scale isn't obviously correct.
   "None" is treated as a real category (not missing) for the four columns
   where the data dictionary defines it as one.

3. **Feature engineering** — three derived features, each with a stated
   business rationale, added alongside (not instead of) the raw fields:
   - `ActualFinancialImpact` = direct loss + lost income + additional costs
     (kept separate from `ExpectedFinancialRedressGBP`, which is excluded
     as leakage and conceptually different — a remedy estimate, not
     realised impact)
   - `ImpactFlagCount` = count of true impact flags (anxiety, depression,
     health deterioration, sleep disruption, bereavement)
   - `FailureBurden` = number of service failures × repeat failure count

4. **Model comparison** (`src/model_comparison.py` → `docs/phase2_model_comparison.md`)
   A majority-class dummy baseline anchors the comparison, alongside
   Logistic Regression, Random Forest, LightGBM, and CatBoost, all compared
   under identical stratified 5-fold CV (seed 42):

   | Model | Macro F1 | Severe recall | Severe precision | Severe F1 |
   |---|---|---|---|---|
   | Majority-class baseline | 0.067 | 0.000 | 0.000 | 0.000 |
   | Logistic Regression | 0.410 | 0.767 | 0.675 | 0.718 |
   | Random Forest | 0.442 | 0.624 | 0.778 | 0.690 |
   | LightGBM | 0.429 | 0.685 | 0.722 | 0.702 |
   | **CatBoost** | 0.439 | 0.730 | 0.729 | **0.729** |

   The baseline (always predicts SeverityScore=1) makes the improvement
   from any real model immediately visible. Balanced class weighting was
   applied consistently across all four trained candidates (not just the
   winner) so the comparison wasn't skewed by the target imbalance
   (class 6 = 101/2000 rows) favouring one model over another. **CatBoost**
   won on severe-case F1 (0.729) and macro F1 (0.439). Note: all categorical
   features are ordinal/one-hot encoded by the shared `ColumnTransformer`
   before reaching any model (see `src/features.py`) — CatBoost is not
   using its native categorical-feature handling here, it simply performed
   best on the encoded feature set.

5. **Hyperparameter tuning + threshold trade-off** (`src/tune_and_threshold.py`
   → `docs/phase3_tuning_and_threshold.md`)
   Grid search over depth/iterations/learning_rate, scored on severe-case
   F1 under the same CV scheme (best: depth=6, iterations=600, lr=0.03,
   severe F1 0.748). Then, using out-of-fold predicted probabilities,
   swept the decision threshold on P(SeverityScore ≥ 4) instead of using
   the default argmax cut-off:

   | Threshold | Recall | Precision | Cases flagged (of 2,000 training rows) |
   |---|---|---|---|
   | 0.50 (default argmax) | 0.759 | 0.757 | 602 |
   | 0.40 | 0.824 | 0.688 | 720 |
   | **0.30 (proposed)** | 0.900 | 0.623 | 869 |
   | 0.25 | 0.917 | 0.586 | 941 |

   The brief states a low tolerance for missing severe cases but gives no
   numeric recall target, so **0.30 is proposed as an operating point for
   business sign-off, not derived as a mathematically optimal answer** —
   the full trade-off table (`docs/phase3_tuning_and_threshold.md`) lets
   the threshold move in either direction based on actual investigative
   capacity. At 0.30, on the 2,000 *labelled training rows*, the model
   would flag 869 cases against 601 truly severe ones (268 additional
   reviews) — these figures describe the training data, not the 500-row
   holdback set, which has no labels to check against. The threshold is
   also a capacity lever: e.g. if operational capacity only permits ~700
   investigations per 2,000 cases, a threshold around 0.40 (82.4% recall)
   would be more appropriate than 0.30 — the table lets the business pick
   the point that matches actual capacity rather than the model dictating it.

   **Methodological caveat:** the threshold is chosen using the same
   5-fold out-of-fold predictions used to select CatBoost's
   hyperparameters. This is better than reading a threshold off
   training-set (in-sample) predictions, but it is still an internal
   training-set analysis, not an independent, unbiased estimate — a
   fully rigorous version would use nested cross-validation (inner loop
   for hyperparameters, outer loop for threshold evaluation). That wasn't
   implemented here; the trade-off table should be read as exploratory
   and directionally reliable, not as a certified performance guarantee.

6. **Explainability & error analysis** (`src/explainability.py` →
   `docs/phase4_explainability_error_analysis.md`, `outputs/shap_summary.png`)
   SHAP shows the model's strongest predictors line up with the severity
   factors named in the brief — `RecoveryTimeMonths`, `PhysicalImpactLevel`,
   `EmotionalImpactLevel`, `DurationAffectedDays`, `VulnerabilityLevel` — a
   useful sanity check on model behaviour, though SHAP shows influence on
   the model's predictions, not causal validity against real-world severity.
   Error analysis on out-of-fold false negatives (141 of 601 severe cases
   missed under the default argmax decision) shows they skew toward
   shorter duration and lower financial impact than correctly-caught
   severe cases — i.e. borderline cases that look moderate on the surface
   but are genuinely severe. Recall by actual class is 63% (score 4),
   85% (score 5), 98% (score 6) — weakest at the 3/4 boundary, as expected.
   No obvious large disparity was observed in this exploratory
   false-negative breakdown by vulnerability group — this compares
   category shares, not a proper fairness analysis (no per-group recall,
   confidence intervals, or minimum sample-size checks were computed) —
   though Homelessness and Immigration cases were somewhat over-represented
   among misses and should be monitored. A calibration check (Brier score
   0.107, reliability table in the report) suggests useful probability
   ranking behaviour, but the probabilities are not independently
   calibrated (no Platt scaling or isotonic regression applied) and should
   not be read as precise probabilities. For the proposed operating-point
   analysis, the score is used primarily to rank cases by relative severe
   risk; because it is not independently calibrated, the 0.30 cut-off
   should be treated as an empirical decision threshold rather than as a
   literal 30% probability, and `PredictedSevereProbability` in the output
   file should be read the same way.

7. **Final model** (`src/train.py`, `src/predict.py`)
   Retrained on all 2,000 training rows with the tuned hyperparameters,
   then scored the 500-row holdback set.

## Deliverable: holdback predictions

Two files are produced under `outputs/`. They are deliberately *not* both
named "predictions", so which one is the required submission is never in
question:

```
outputs/
├── holdback_predictions.csv              <- REQUIRED submission file
└── holdback_operational_analysis.csv     <- supporting analysis, not required
```

**Required submission output: `outputs/holdback_predictions.csv`.** This is
the only prediction file required by the assessment. Exactly the requested
columns, nothing else:

| Column | Description |
|---|---|
| `CaseReference` | case id |
| `PredictedSeverityScore` | 1-6, model argmax prediction |

**Supporting analysis output: `outputs/holdback_operational_analysis.csv`.**
Not a required submission file — provided for transparency into the
business-risk decision layer (Phase 3). Same `CaseReference` and
`PredictedSeverityScore` as above, plus:

| Column | Description |
|---|---|
| `PredictedSevereProbability` | P(SeverityScore ≥ 4); indicative, not independently calibrated (see calibration check above) |
| `RecommendedAction` | threshold-based triage decision (0.30 cut-off, see above) |

Note `PredictedSeverityScore` and `RecommendedAction` can occasionally
disagree at the margin (e.g. score 2 but flagged for investigation) — this
is intentional: `RecommendedAction` deliberately uses a lower, recall-biased
threshold than the naive "score ≥ 4" rule, per the business risk decision
in step 5.

**How to read the two outputs:**

```
PredictedSeverityScore (1-6)   <- required model output, the argmax class
        |
        v
P(SeverityScore >= 4)          <- derived from the same model's probabilities
        |
        v
   threshold = 0.30            <- proposed business operating point, tunable
        |
        v
RecommendedAction              <- separate risk-sensitive decision layer
```

`PredictedSeverityScore` is the literal deliverable. `RecommendedAction` is
an additional, independently-thresholded triage recommendation built on
top of it — the two are related but not the same decision rule, by design.

## Repository structure

```
data/               training_dataset.csv, holdback_dataset.csv (exported from the source .xlsx)
src/
  features.py       locked feature set + preprocessing pipeline + engineered features
  audit.py          Phase 1 data/leakage audit
  model_comparison.py  Phase 2 baseline + candidate model comparison (5-fold CV)
  tune_and_threshold.py  Phase 3 hyperparameter search + threshold trade-off
  explainability.py Phase 4 SHAP + error analysis
  train.py          fits and saves the final model
  predict.py         scores the holdback set
docs/               generated audit/analysis reports (markdown, regenerated by the scripts above)
models/             best_params.json, final_pipeline.joblib (generated)
outputs/            holdback_predictions.csv, shap_summary.png (generated)
notebooks/          severity_model.ipynb — narrative walkthrough of the above
```

## Reproduction

```bash
pip install -r requirements.txt
python src/audit.py
python src/model_comparison.py
python src/tune_and_threshold.py
python src/explainability.py
python src/train.py
python src/predict.py
```

All randomness is controlled by a single fixed seed (42), used consistently
for every train/validation split and every model, so results are
repeatable end to end. The notebook (`notebooks/severity_model.ipynb`) runs
the same pipeline with inline commentary and plots.

## Assumptions

- The five excluded columns are genuinely unavailable at triage time. This
  is an assumption based on their statistical relationship to the target
  and their business meaning (remedy/investigation decisions are outcomes
  of the severity assessment, not inputs to it) — not confirmed against a
  data-generation spec, since none was provided.
- "None" is treated as a valid category (not missing) for
  `VulnerabilityLevel`, `PhysicalImpactLevel`, `PrimaryVulnerability`, and
  `FailureTypeSecondary`, per the data dictionary's stated allowed values.
- The 0.30 decision threshold was selected using the training-set
  out-of-fold recall/precision trade-off (not independently calibrated —
  see the Methodological caveat above); it should be revisited if the true
  operating class balance (or the organisation's investigative capacity)
  differs materially from what's seen here.
- Two columns (`AdditionalTreatmentRequired`, `ImpactOnRelationships`)
  appear in both datasets but are absent from the Data Dictionary tab.
  They were treated as ordinary Boolean predictors on inspection, same
  format as the documented Boolean fields.

## Limitations

- Only 2,000 labelled training cases are available, and class 6 (the
  rarest severity level) has only 101 of them — CV estimates for that
  class carry more uncertainty than for the larger classes.
- The five columns excluded as leakage were identified by their
  statistical relationship to the target and their business meaning, not
  confirmed against a data-generation specification — none was provided.
- The 0.30 decision threshold is an internal training-set analysis (see
  the methodological caveat above), selected using the training-set
  out-of-fold recall/precision trade-off, and should be revisited if
  real-world case mix or investigative capacity differs materially.
- The holdback set has no labels, so no unbiased estimate of holdback
  performance is possible with the data provided — reported metrics
  throughout are training-set cross-validation estimates.
- Model probabilities are not independently calibrated (see the
  calibration check in `docs/phase4_explainability_error_analysis.md`);
  they are used as a ranking/decision score, not a precise probability.
- This model is intended to support triage decisions, not replace human
  review — particularly for cases near the decision boundary.
# severity-classification-model
