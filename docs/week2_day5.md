# Week 2 Day 5 — Random Forest Fraud Detection

## Objective

Train a Random Forest classifier, evaluate its performance, analyze
feature importance, compare its results with Logistic Regression, and
save reproducible model artifacts.

## Dataset

Input directory: `data/processed/week2_day3/`

The existing prepared dataset was reused without changing the train/test split.

- Training records: 800
- Test records: 200
- Training fraud cases: 10
- Test fraud cases: 3
- Features: 20
- Random state: 42

The training and test feature and target CSV files were verified to have
matching SHA-256 checksums against the existing baseline branch.

## Implementation

Script: `src/train_random_forest.py`

Random Forest configuration:

- Estimators: 300
- Class weighting: `balanced`
- Random state: 42
- Parallel processing: enabled

The Logistic Regression comparison uses the existing baseline settings:
`StandardScaler`, `C=1.0`, `max_iter=1000`, `lbfgs` solver, and balanced
class weights.

Both models are evaluated on the same held-out test set. Evaluation
includes accuracy, fraud precision, fraud recall, fraud F1-score,
macro F1, ROC-AUC, average precision, confusion matrices, and error rates.

## Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.9900 | 1.0000 |
| Fraud precision | 0.6000 | 1.0000 |
| Fraud recall | 1.0000 | 1.0000 |
| Fraud F1-score | 0.7500 | 1.0000 |
| Macro F1 | 0.8724 | 1.0000 |
| ROC-AUC | 1.0000 | 1.0000 |
| Average precision (PR-AUC) | 1.0000 | 1.0000 |
| True negatives | 195 | 197 |
| False positives | 2 | 0 |
| False negatives | 0 | 0 |
| True positives | 3 | 3 |

On this test split, Random Forest classified all 200 records correctly,
including the three fraud cases. Logistic Regression also detected all
three fraud cases but generated two false positives.

## Feature Importance

The Random Forest impurity-based feature importance ranking identified
these ten highest-ranked features:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `numeric__new_balance_orig` | 0.212258 |
| 2 | `numeric__dest_balance_change` | 0.155947 |
| 3 | `numeric__orig_balance_change` | 0.132007 |
| 4 | `numeric__amount` | 0.125711 |
| 5 | `numeric__log_amount` | 0.121279 |
| 6 | `numeric__amount_to_orig_balance_ratio` | 0.084077 |
| 7 | `numeric__new_balance_dest` | 0.064445 |
| 8 | `numeric__old_balance_orig` | 0.045459 |
| 9 | `numeric__old_balance_dest` | 0.026732 |
| 10 | `numeric__dest_balance_error` | 0.008831 |

These scores describe the fitted model's impurity-based importance. They
do not establish causation or prove that a feature independently detects fraud.

## Generated Artifacts

- Model: `models/random_forest_model.joblib`
- Feature ranking: `results/feature_importance.csv`
- Comparison table: `results/model_comparison.csv`
- Comparison metrics: `results/model_comparison.json`
- Random Forest metrics: `results/random_forest_evaluation.json`
- Random Forest report: `results/random_forest_classification_report.txt`
- Logistic Regression comparison metrics:
  `results/logistic_regression_comparison.json`
- Confusion matrix: `docs/figures/random_forest_confusion_matrix.png`
- Feature importance chart: `docs/figures/random_forest_feature_importance.png`
- ROC comparison: `docs/figures/model_comparison_roc_curve.png`
- Precision-recall comparison:
  `docs/figures/model_comparison_precision_recall.png`

## Automated Tests

Test file: `tests/test_train_random_forest.py`

Six tests passed, covering dataset loading, Random Forest training and
evaluation, Logistic Regression training, feature importance ranking,
invalid estimator counts, and missing dataset handling.

Run the tests from the repository root:

`PYTHONPATH=src python -m unittest discover -s tests -p 'test_train_random_forest.py' -v`

## Limitations

The test set contains only three fraudulent transactions, and the training
set contains only ten. Fraud recall therefore changes in large increments
when even one case is missed. The apparently perfect Random Forest result
is preliminary and should not be treated as evidence of production readiness.

The small sample does not support reliable estimates of real-world fraud
detection. Further validation requires a substantially larger representative
dataset and preferably time-based evaluation on later transactions.
Threshold selection, false-positive costs, and monitoring should also be
considered before deployment.

## Reproduce

Install dependencies:

`python -m pip install -r requirements.txt`

Train and evaluate both classifiers:

`python src/train_random_forest.py`
