# Phase 2 Model Comparison

5-fold stratified CV, same folds across all models. Class weights balanced throughout given the target imbalance (class 6 has only 101/2000 examples).

Headline business metric: **severe_recall** (recall on SeverityScore>=4), given the brief's stated low tolerance for missing severe cases.

| model                 |   macro_f1_mean |   balanced_accuracy_mean |   severe_recall_mean |   severe_precision_mean |   severe_f1_mean |
|:----------------------|----------------:|-------------------------:|---------------------:|------------------------:|-----------------:|
| MajorityClassBaseline |           0.067 |                    0.167 |                0     |                   0     |            0     |
| LogisticRegression    |           0.41  |                    0.429 |                0.767 |                   0.675 |            0.718 |
| RandomForest          |           0.442 |                    0.43  |                0.624 |                   0.778 |            0.69  |
| LightGBM              |           0.429 |                    0.426 |                0.685 |                   0.722 |            0.702 |
| CatBoost              |           0.439 |                    0.439 |                0.73  |                   0.729 |            0.729 |

## Full stats (mean +/- std)

| model                 |   macro_f1_mean |   macro_f1_std |   balanced_accuracy_mean |   balanced_accuracy_std |   severe_recall_mean |   severe_recall_std |   severe_precision_mean |   severe_precision_std |   severe_f1_mean |   severe_f1_std |
|:----------------------|----------------:|---------------:|-------------------------:|------------------------:|---------------------:|--------------------:|------------------------:|-----------------------:|-----------------:|----------------:|
| MajorityClassBaseline |           0.067 |          0     |                    0.167 |                   0     |                0     |               0     |                   0     |                  0     |            0     |           0     |
| LogisticRegression    |           0.41  |          0.013 |                    0.429 |                   0.022 |                0.767 |               0.066 |                   0.675 |                  0.025 |            0.718 |           0.04  |
| RandomForest          |           0.442 |          0.033 |                    0.43  |                   0.03  |                0.624 |               0.053 |                   0.778 |                  0.062 |            0.69  |           0.031 |
| LightGBM              |           0.429 |          0.021 |                    0.426 |                   0.02  |                0.685 |               0.028 |                   0.722 |                  0.039 |            0.702 |           0.018 |
| CatBoost              |           0.439 |          0.031 |                    0.439 |                   0.03  |                0.73  |               0.037 |                   0.729 |                  0.04  |            0.729 |           0.028 |

## Confusion matrix, final CV fold (multiclass)


**MajorityClassBaseline**

|        |   pred_1 |   pred_2 |   pred_3 |   pred_4 |   pred_5 |   pred_6 |
|:-------|---------:|---------:|---------:|---------:|---------:|---------:|
| true_1 |      101 |        0 |        0 |        0 |        0 |        0 |
| true_2 |       98 |        0 |        0 |        0 |        0 |        0 |
| true_3 |       80 |        0 |        0 |        0 |        0 |        0 |
| true_4 |       59 |        0 |        0 |        0 |        0 |        0 |
| true_5 |       41 |        0 |        0 |        0 |        0 |        0 |
| true_6 |       21 |        0 |        0 |        0 |        0 |        0 |

**LogisticRegression**

|        |   pred_1 |   pred_2 |   pred_3 |   pred_4 |   pred_5 |   pred_6 |
|:-------|---------:|---------:|---------:|---------:|---------:|---------:|
| true_1 |       69 |       25 |        6 |        0 |        0 |        1 |
| true_2 |       23 |       35 |       22 |       10 |        4 |        4 |
| true_3 |        5 |       26 |       22 |       15 |        9 |        3 |
| true_4 |        1 |        6 |        8 |       23 |       16 |        5 |
| true_5 |        0 |        1 |        3 |        8 |       18 |       11 |
| true_6 |        0 |        1 |        0 |        3 |        7 |       10 |

**RandomForest**

|        |   pred_1 |   pred_2 |   pred_3 |   pred_4 |   pred_5 |   pred_6 |
|:-------|---------:|---------:|---------:|---------:|---------:|---------:|
| true_1 |       74 |       19 |        6 |        2 |        0 |        0 |
| true_2 |       31 |       42 |       12 |       12 |        1 |        0 |
| true_3 |        9 |       35 |       13 |       17 |        6 |        0 |
| true_4 |        1 |       11 |       11 |       27 |        7 |        2 |
| true_5 |        0 |        3 |       10 |       11 |       16 |        1 |
| true_6 |        0 |        0 |        1 |        4 |       10 |        6 |

**LightGBM**

|        |   pred_1 |   pred_2 |   pred_3 |   pred_4 |   pred_5 |   pred_6 |
|:-------|---------:|---------:|---------:|---------:|---------:|---------:|
| true_1 |       72 |       22 |        4 |        3 |        0 |        0 |
| true_2 |       24 |       42 |       17 |       13 |        2 |        0 |
| true_3 |        6 |       32 |       15 |       19 |        6 |        2 |
| true_4 |        1 |        7 |       13 |       22 |       13 |        3 |
| true_5 |        0 |        2 |        9 |       10 |       16 |        4 |
| true_6 |        0 |        1 |        2 |        5 |        7 |        6 |

**CatBoost**

|        |   pred_1 |   pred_2 |   pred_3 |   pred_4 |   pred_5 |   pred_6 |
|:-------|---------:|---------:|---------:|---------:|---------:|---------:|
| true_1 |       71 |       22 |        5 |        3 |        0 |        0 |
| true_2 |       20 |       45 |       21 |       10 |        1 |        1 |
| true_3 |        8 |       25 |       18 |       21 |        8 |        0 |
| true_4 |        0 |        4 |       15 |       25 |       13 |        2 |
| true_5 |        0 |        1 |        8 |        7 |       17 |        8 |
| true_6 |        0 |        0 |        1 |        4 |        9 |        7 |
