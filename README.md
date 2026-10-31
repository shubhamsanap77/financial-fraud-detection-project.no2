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
## Docker Environment Setup

### Prerequisites
- Docker Desktop
- WSL 2
- Windows virtualization enabled

### Verify Docker Installation

```bash
docker --version
docker compose version
```

### Build Docker Image

From the project root directory:

```bash
docker build -t fraud-detection-base .
```

### Verify Docker Image

```bash
docker images
```

The `fraud-detection-base:latest` image should be visible.

### Docker Environment

- Base Image: `python:3.11-slim`
- Working Directory: `/app`
- Docker Engine: Docker Desktop with WSL 2


Day 5 – PaySim Kafka Producer

The Kafka producer reads cleaned PaySim transaction events from a JSON file and supports dry-run validation or publishing to the Kafka topic "fraud-transactions".

Prerequisites

- Python 3
- Apache Kafka broker running for live streaming

Install Dependencies

python -m pip install -r requirements.txt

Dry-Run Test

python producer/kafka_producer.py --input data/processed/paysim_stream_ready.json --dry-run --limit 5

Publish Transactions to Kafka

Ensure Kafka is running and the "fraud-transactions" topic exists, then run:

python producer/kafka_producer.py --input data/processed/paysim_stream_ready.json --limit 5

Default Kafka broker: "localhost:9092"

Default topic: "fraud-transactions"

Day 7 – Integration Testing Results

Kafka Integration Test

- Kafka broker started successfully using Docker Compose.
- Kafka container status verified as healthy.
- Created and verified the "fraud-transactions" topic.
- Published 5 PaySim transaction events using "producer/kafka_producer.py".
- Consumed 5 messages from Kafka using the Kafka console consumer.
- Result: Producer-to-Kafka-to-consumer flow passed for 5 messages.

Test Scope and Limitations

- The integration test used the Kafka setup from Shubham's Day 2 project ZIP.
- Cassandra integration has not yet been verified.
- End-to-end fraud detection has not yet been verified.

Day 7 – Integration Testing and Verification

Tests Performed

- Verified that the Kafka Docker container was running and healthy.
- Verified that the Cassandra Docker container was running.
- Confirmed that the "fraud-transactions" Kafka topic exists.
- Published 5 PaySim transaction events using the Python Kafka Producer.
- Consumed 5 transaction messages from Kafka successfully.
- Verified that the Cassandra keyspace and "transactions" table exist.
- Ran Cassandra integration tests successfully: 2 tests passed.

Test Results

Component| Result
Kafka topic verification| Passed
Kafka producer test| Passed
Kafka message consumption| Passed – 5 messages
Cassandra integration tests| Passed – 2 tests
GitHub push| Successful

Note: Kafka message delivery and Cassandra integration tests were verified separately. End-to-end Kafka-to-Cassandra processing has not yet been verified.


Week 2 – Day 2: Feature Engineering

- Loaded the processed PaySim transaction dataset containing 1,000 records.
- Converted transaction amount and balance columns to numeric data types.
- Created nine additional features related to balance changes, balance errors, transaction amount, and transaction time.
- Checked missing values across the dataset; the validation output reported zero missing values.
- Checked duplicate transaction IDs; zero duplicates were found.
- Saved the feature-engineered dataset to "data/processed/paysim_feature_engineered.csv".
- Verified the output dataset shape: 1,000 rows and 25 columns.
- Fraud distribution: 13 fraud transactions and 987 non-fraud transactions.
- Identified class imbalance for consideration during model training.

Output: "data/processed/paysim_feature_engineered.csv"

Week 2 – Day 6: XGBoost Fraud Detection Model

Objective

Train and evaluate an XGBoost classifier to detect fraudulent financial transactions using the feature-engineered PaySim dataset.

Implementation

- Used the feature-engineered dataset: "data/processed/paysim_feature_engineered.csv".
- Applied stratified train-test splitting with an 80:20 ratio.
- Excluded identifiers and target-leakage-related columns from model features.
- Used class weighting through "scale_pos_weight" to address class imbalance.
- Evaluated the model using fraud precision, recall, F1-score, ROC-AUC, average precision, and a confusion matrix.
- Generated feature importance rankings.

Evaluation Results

Metric| Result
Fraud Recall| 1.0000
Fraud F1-score| 1.0000
ROC-AUC| 1.0000
Average Precision| 1.0000

Confusion Matrix

[[197, 0],
 [0, 3]]

The test set contained 197 non-fraud transactions and 3 fraud transactions. All 200 test records were classified correctly in this run.

Limitation: The dataset contains only 13 fraud transactions overall. These results are preliminary and should not be interpreted as evidence of guaranteed real-world performance. Further validation on a larger, representative dataset is required.

Generated Outputs

- "outputs/week2_day6/xgboost_metrics.json"
- "outputs/week2_day6/xgboost_feature_importance.csv"

Training Script

- "src/train_xgboost.py"

Dependencies

- "pandas"
- "scikit-learn"
- "xgboost"

Week 2 - Day 6: XGBoost Model and Model Comparison
Trained an XGBoost fraud detection model.
Compared Logistic Regression, Random Forest, and XGBoost.
Evaluated accuracy, fraud precision, fraud recall, fraud F1-score, ROC-AUC, and average precision.
Saved XGBoost metrics, feature importance, and model comparison outputs under outputs/week2_day6/.
Preliminary Results
Model	Accuracy	Fraud Precision	Fraud Recall	Fraud F1
Logistic Regression	0.99	0.60	1.00	0.75
Random Forest	1.00	1.00	1.00	1.00
XGBoost	1.00	1.00	1.00	1.00
Note: These results are preliminary because the dataset contains only 13 fraud cases, with 3 fraud cases in the test set. Further validation on a larger dataset is required.
