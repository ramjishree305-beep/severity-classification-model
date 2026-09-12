# Phase 3: Hyperparameter Tuning and Threshold Trade-off

## CatBoost hyperparameter search (5-fold CV, scored on severe-case F1)

|   depth |   iterations |   learning_rate |   severe_f1 |
|--------:|-------------:|----------------:|------------:|
|       6 |          600 |            0.03 |       0.748 |
|       6 |          300 |            0.03 |       0.747 |
|       4 |          300 |            0.03 |       0.746 |
|       4 |          600 |            0.03 |       0.746 |
|       8 |          300 |            0.03 |       0.744 |
|       6 |          300 |            0.1  |       0.739 |
|       8 |          600 |            0.03 |       0.738 |
|       4 |          600 |            0.1  |       0.734 |
|       4 |          300 |            0.1  |       0.733 |
|       6 |          600 |            0.1  |       0.73  |
|       8 |          300 |            0.1  |       0.717 |
|       8 |          600 |            0.1  |       0.713 |

Best params: `{'depth': 6, 'iterations': 600, 'learning_rate': 0.03}`

## Default (argmax) multiclass decision, mapped to binary

|   recall |   precision |    f1 |   n_flagged |
|---------:|------------:|------:|------------:|
|    0.765 |        0.73 | 0.747 |         630 |

Actual severe cases in training data: 601 / 2000

## Threshold sweep on P(SeverityScore >= 4), out-of-fold

**Note on methodology:** these probabilities come from 5-fold out-of-fold predictions, so each row is scored by a model that did not see it during training -- this is better than reading thresholds off training-set predictions. However, the same 2,000 labelled rows were also used to select the CatBoost hyperparameters above, so this is an **internal training-set operating-point analysis, not an independent, unbiased performance estimate**. The holdback set has no labels, so no unbiased estimate of holdback performance is possible with the data provided. Numbers below (869 flagged, etc.) describe the 2,000 training records, *not* the 500-row holdback set.

Raising the threshold trades precision for workload; lowering it trades workload for recall. The brief states a low tolerance for missing severe cases but does not specify a required recall level, so the table below is presented as a trade-off for business sign-off, not as a single "correct" answer:

|   threshold |   recall |   precision |    f1 |   n_flagged |
|------------:|---------:|------------:|------:|------------:|
|        0.1  |    0.992 |       0.465 | 0.633 |        1282 |
|        0.15 |    0.97  |       0.511 | 0.669 |        1142 |
|        0.2  |    0.938 |       0.545 | 0.69  |        1034 |
|        0.25 |    0.917 |       0.586 | 0.715 |         941 |
|        0.3  |    0.9   |       0.623 | 0.736 |         869 |
|        0.35 |    0.87  |       0.658 | 0.749 |         795 |
|        0.4  |    0.824 |       0.688 | 0.749 |         720 |
|        0.45 |    0.794 |       0.721 | 0.755 |         662 |
|        0.5  |    0.759 |       0.757 | 0.758 |         602 |
|        0.55 |    0.722 |       0.771 | 0.746 |         563 |
|        0.6  |    0.686 |       0.795 | 0.736 |         518 |
|        0.65 |    0.639 |       0.807 | 0.713 |         476 |
|        0.7  |    0.586 |       0.83  | 0.687 |         424 |
|        0.75 |    0.519 |       0.855 | 0.646 |         365 |
|        0.8  |    0.433 |       0.884 | 0.581 |         294 |
|        0.85 |    0.354 |       0.903 | 0.509 |         236 |
|        0.9  |    0.235 |       0.922 | 0.374 |         153 |

**Proposed operating point: threshold 0.30** -> recall 0.900, precision 0.623, 869 of 2000 training cases would be flagged for investigation (vs 601 truly severe, 268 additional reviews). This is proposed, not derived from a stated business requirement (the brief gives no numeric recall target) -- it should be agreed with the business against actual investigative capacity before use, and can be moved along the table above in either direction.
