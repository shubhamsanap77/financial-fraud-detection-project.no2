# Week 3 Day 3 — Data Validation and Prediction Preparation

## Objective

Validate transaction input, handle missing and invalid values, and prepare
transactions for prediction using the trained Random Forest model and the
existing training-fitted preprocessing pipeline.

## Input and Model Compatibility

- Input: `data/processed/paysim_feature_engineered.csv`
- Input rows: 1,000
- Model input features: 16 raw feature columns
- Transformed features: 20 encoded columns
- Preprocessor: `data/processed/week2_day3/preprocessor.joblib`
- Model: `models/random_forest_model.joblib`

The existing preprocessor is reused without refitting. This preserves the
feature encoding and imputation statistics learned during model preparation.

## Implementation

- Script: `src/validate_prediction_data.py`
- Unit tests: `tests/test_validate_prediction_data.py`

Validation checks include:

- Required model input columns.
- Missing or invalid transaction amounts.
- Malformed numeric values and infinite values.
- Negative account balances.
- Transaction hour and day-of-week ranges.
- Missing or unsupported payment types.
- Duplicate transaction IDs when an ID column is present.

Eligible missing numeric feature values are passed to the saved preprocessor
for imputation. Missing amounts and unsupported payment types are rejected.
Rejected records are kept separately with reasons rather than silently scored.

The script transforms accepted transactions into model-ready features,
obtains fraud probabilities, and applies a configurable classification
threshold (default 0.5). Existing fraud labels are not used as prediction
features and are omitted from the prediction output.

## Validation Results

The clean sample was processed successfully:

- Input rows: 1,000
- Accepted rows: 1,000
- Rejected rows: 0
- Prediction-ready features: 20 per transaction
- Rows predicted: 1,000
- Missing values after preprocessing: 0
- Non-finite values after preprocessing: 0
- Predicted fraud cases at threshold 0.5: 13

All 9 targeted validation tests passed. The full project suite passed all
20 tests, including the existing Cassandra integration tests, dataset
preparation tests, Random Forest tests, and the new validation tests.

The count of 13 predicted fraud cases is a model output, not an accuracy
measurement or proof that 13 transactions are actually fraudulent.

## Generated Outputs

Directory: `data/processed/week3_day3/`

- `prediction_ready_features.csv` — transformed model input features
- `predictions.csv` — accepted input records with fraud probability and predicted label
- `rejected_transactions.csv` — rejected input records and validation reasons
- `data_quality_report.json` — audit counts, rejection reasons, and output paths

## Reproduce

Install dependencies from the repository root:

`python -m pip install -r requirements.txt`

Run validation, preprocessing, and prediction:

`python src/validate_prediction_data.py`

To select a different probability threshold:

`python src/validate_prediction_data.py --threshold 0.3`

Run the targeted tests:

`PYTHONPATH=src python -m unittest discover -s tests -p 'test_validate_prediction_data.py' -v`

Run the full project suite:

`PYTHONPATH=src python -m unittest discover -s tests -v`

## Limitations

The input sample contains only 13 known fraud examples. The output from
running predictions across this file is not an independent evaluation of
the model. Reliable performance claims require a larger representative
dataset and a separate evaluation procedure.
