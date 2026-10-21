# financial-fraud-detection-project.no2
Real-Time Financial Fraud Detection Pipeline using Kafka, Cassandra, Python and Docker
# Financial Fraud Detection

Real-Time Financial Fraud Detection Pipeline using Python, Kafka, Cassandra, and Docker.

## Project Overview

This project focuses on building a real-time financial fraud detection pipeline for identifying potentially fraudulent transactions.

## Technologies

- Python
- Apache Kafka
- Cassandra
- Docker
- Machine Learning
- GitHub

## Team

- Shubham Sanap
- Himanshu
- Rohan
- Rajesh
- Omkar


## Week 2 Day 3 — ML-Ready Fraud Dataset

Prepared the feature-engineered PaySim dataset for fraud classification.

- **Script:** `src/prepare_ml_dataset.py`
- **Target:** `is_fraud`
- **Split:** 80% training / 20% testing, stratified with random state 42
- **Preprocessing:** Numeric missing-value handling, categorical encoding, and saved preprocessing pipeline
- **Class imbalance:** Training-set class weights calculated; test set left unchanged
- **Tests:** Three automated preparation tests

### Run

Install the project dependencies, then execute from the repository root:

`python src/prepare_ml_dataset.py`

### Outputs

Files are written to `data/processed/week2_day3/`:
- `X_train.csv` and `X_test.csv`
- `y_train.csv` and `y_test.csv`
- `preprocessor.joblib`
- `split_summary.json`

**Data limitation:** The current dataset contains 1,000 transactions but only 13 fraud examples. Evaluation results from this sample will be highly uncertain and should not be treated as production-level evidence.

## Week 2 Day 5 — Random Forest Training and Evaluation

Trained a class-balanced Random Forest classifier and compared it with
Logistic Regression using the same prepared training and test datasets.

- **Training script:** `src/train_random_forest.py`
- **Random Forest:** 300 estimators, balanced class weights, random state 42
- **Evaluation:** Accuracy, fraud precision, recall, F1-score, ROC-AUC, average precision, and confusion matrix
- **Feature importance:** Ranked feature CSV and top-10 importance chart
- **Comparison:** CSV/JSON metrics and ROC and precision-recall plots
- **Tests:** Six model-specific tests; all eleven project tests passed

### Run

Install dependencies and execute from the repository root:

`python -m pip install -r requirements.txt`

`python src/train_random_forest.py`

### Outputs

- Model: `models/random_forest_model.joblib`
- Feature ranking: `results/feature_importance.csv`
- Model comparison: `results/model_comparison.csv`
- Evaluation metrics: `results/random_forest_evaluation.json`
- Figures: `docs/figures/`

**Limitation:** The current training dataset contains only 10 fraud cases,
and the test set contains only 3. The promising evaluation results are
preliminary and are not evidence of production readiness.

## Week 3 Day 3 — Data Validation and Prediction Preparation

Implemented input validation and prediction preparation using the existing
training-fitted preprocessing pipeline and Random Forest model.

- **Script:** `src/validate_prediction_data.py`
- **Validation:** Missing/invalid amounts, malformed numeric values, negative
  balances, transaction-time ranges, payment types, and duplicate transaction IDs
- **Missing data:** Eligible missing numeric features are imputed by the saved
  preprocessor; invalid required values are rejected with reasons
- **Prediction preparation:** 16 input features transformed into 20 model features
- **Outputs:** Prepared feature CSV, fraud predictions, rejected records, and JSON audit report
- **Tests:** Nine targeted validation tests passed; all 20 project tests passed

### Run

`python -m pip install -r requirements.txt`

`python src/validate_prediction_data.py`

### Outputs

Generated in `data/processed/week3_day3/`:
- `prediction_ready_features.csv`
- `predictions.csv`
- `rejected_transactions.csv`
- `data_quality_report.json`

**Limitation:** The sample contains only 13 known fraud examples. The 13 positive
predictions in this run are model outputs and must not be presented as measured
accuracy or independent evidence of real-world performance.
