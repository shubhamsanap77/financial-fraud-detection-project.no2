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
