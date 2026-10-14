# Week 2 Day 3 — ML-Ready Dataset Preparation

## Objective
Prepare the feature-engineered PaySim transaction data for fraud
classification by validating the target, separating features and labels,
creating a reproducible train-test split, and addressing class imbalance.

## Input
- `data/processed/paysim_feature_engineered.csv`
- Source: Week 1/2 Day 2 feature-engineering work
- Records: 1,000
- Target column: `is_fraud`

## Implementation
- Script: `src/prepare_ml_dataset.py`
- Target: `y` (`is_fraud`)
- Features: numeric transaction/balance features and encoded `payment_type`
- Excludes transaction/customer identifiers, raw timestamp, target leakage
  candidates and constant/uninformative fields.
- Uses an 80/20 stratified train-test split with random state 42.
- Fits missing-value handling and categorical encoding on training data only.
- Saves the fitted preprocessing pipeline for reuse.
- Calculates balanced class weights using training labels only.
- Does not resample the test set.

## Output Files
Generated in `data/processed/week2_day3/`:
- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`
- `preprocessor.joblib`
- `split_summary.json`

## Validation Results
- Input records: 1,000
- Original class counts: 987 non-fraud, 13 fraud
- Training rows: 800 (790 non-fraud, 10 fraud)
- Test rows: 200 (197 non-fraud, 3 fraud)
- Encoded feature count: 20
- Missing values in prepared feature matrices: 0
- Constant feature removed: `transaction_month`
- Automated preparation tests: 3 passed

## Class Imbalance and Limitations
No oversampling was applied. Balanced class weights were calculated from
the training set for use by a compatible classifier during model training.
Only 13 fraud records are present, so evaluation metrics—especially recall
and fraud detection rates—will be highly uncertain. This dataset is a
small development sample, not production-level validation.

## Reproduce
From the project root, install dependencies and run:

`python src/prepare_ml_dataset.py`

The script uses the default input and output paths above.

## Scope
This task prepares datasets for model training. Model training and model
performance evaluation are separate tasks.
