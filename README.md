# financial-fraud-detection-project.no2
Real-Time Financial Fraud Detection Pipeline using Kafka, Cassandra, Python and Docker

# Financial Fraud Detection

Real-Time Financial Fraud Detection Pipeline using Python, Kafka, Cassandra, Docker, and Machine Learning.

## Project Overview

This project focuses on building an enterprise-grade real-time financial fraud detection pipeline for identifying potentially fraudulent transactions. The system streams transactions via Apache Kafka, stores them in Apache Cassandra, engineers domain-specific fraud indicators on the PaySim benchmark dataset, trains regularized ML classification models with class-imbalance weighting, and performs real-time/batch model inference with risk categorization and automated security alerting.

## Technologies

- **Python** (Data preprocessing, feature engineering, ML models, Kafka clients, Cassandra driver)
- **Apache Kafka** (Real-time message streaming)
- **Apache Cassandra** (NoSQL high-throughput database storage)
- **Docker & Docker Compose** (Containerized infrastructure)
- **Scikit-Learn & Machine Learning** (Logistic Regression, Random Forest, class imbalance weighting, metrics evaluation)
- **GitHub** (Collaborative branch-based workflow)

## Team

- Shubham Sanap
- Himanshu
- Rohan
- Rajarsh Rajguru
- Omkar

---

## Table of Contents
1. [Week 3 Day 4 — ML Model Integration, Fraud Prediction & Results Generation](#week-3-day-4--ml-model-integration-fraud-prediction--results-generation)
2. [Week 2 Day 4 — Logistic Regression Model Training & Evaluation](#week-2-day-4--logistic-regression-model-training--evaluation)
3. [Week 1 Day 4 — PaySim Dataset Preparation & Transaction-Data Analysis](#week-1-day-4--paysim-dataset-preparation--transaction-data-analysis)
4. [Project File Structure (Organized by Weeks)](#project-file-structure-organized-by-weeks)
5. [How to Execute](#how-to-execute)
6. [Automated Test Verification](#automated-test-verification)

---

## Week 3 Day 4 — ML Model Integration, Fraud Prediction & Results Generation

**Assigned to:** Rajarsh  
**Branch:** `rajarsh`  
**Deliverables:** ML Model Predictor Service, Scored Prediction Results, Fraud Alerts, and Diagnostic Visualizations  
**Detailed Report:** [`docs/week3_day4.md`](docs/week3_day4.md) / [`docs/week3/week3_day4.md`](docs/week3/week3_day4.md)

### Objectives & Work Completed
1. **Model Integration Pipeline**:
   - Developed an inference and prediction service ([`src/integrate_ml_model.py`](src/integrate_ml_model.py)) wrapping the trained ML classifier (`models/logistic_regression_model.joblib`) and preprocessor (`data/processed/week2_day3/preprocessor.joblib`).
   - Supports both high-throughput batch scoring and real-time single-event scoring for streaming Kafka consumers.

2. **Transaction Prediction & Risk Scoring**:
   - Processed 1,000 transactions and generated continuous fraud probabilities $P(\text{fraud})$.
   - Classified transactions into **`LEGITIMATE`** ($0$) and **`FRAUD`** ($1$) based on decision threshold ($\tau = 0.50$).
   - Categorized transactions into standardized risk tiers:
     - **CRITICAL** ($P \ge 0.85$): 13 transactions (1.30%)
     - **HIGH** ($0.50 \le P < 0.85$): 5 transactions (0.50%)
     - **MEDIUM** ($0.20 \le P < 0.50$): 7 transactions (0.70%)
     - **LOW** ($P < 0.20$): 975 transactions (97.50%)

3. **Prediction Results & Security Alerts**:
   - **Total Scored**: 1,000 transactions
   - **Predicted Legitimate**: 982 transactions (**98.20%**)
   - **Predicted Fraudulent**: 18 transactions (**1.80%**)
   - **Fraud Alerts Generated**: **18** high-risk security alerts exported to [`data/processed/week3_day4/fraud_alerts.json`](data/processed/week3_day4/fraud_alerts.json).
   - Exported enriched dataset to [`data/processed/week3_day4/prediction_results.csv`](data/processed/week3_day4/prediction_results.csv).
   - Exported executive summary to [`data/processed/week3_day4/prediction_summary.json`](data/processed/week3_day4/prediction_summary.json).

4. **Diagnostic Visualizations Generated**:
   - `docs/figures/week3/prediction_probability_distribution.png`: Histogram displaying probability density separation.
   - `docs/figures/week3/predictions_by_payment_type.png`: Stacked breakdown of predictions across transaction types.
   - `docs/figures/week3/risk_level_breakdown.png`: Distribution across `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` risk tiers.

5. **Automated Unit Testing**:
   - Built [`tests/test_integrate_ml_model.py`](tests/test_integrate_ml_model.py) verifying model loading, batch predictions, real-time single scoring, risk tiers, and threshold adjustments.
   - **All 26/26 automated unit tests pass successfully.**

---

## Week 2 Day 4 — Logistic Regression Model Training & Evaluation

**Assigned to:** Rajarsh  
**Deliverable:** Trained Model Pipeline, Performance Evaluation, and Confusion Matrix  
**Detailed Report:** [`docs/week2_day4.md`](docs/week2_day4.md) / [`docs/week2/week2_day4.md`](docs/week2/week2_day4.md)

1. **Model Architecture & Imbalance Handling**:
   - Built [`src/train_logistic_regression.py`](src/train_logistic_regression.py) using `StandardScaler` and `LogisticRegression(C=1.0, solver='lbfgs')`.
   - Prevented data leakage by fitting scaling parameters on training data only.
   - Addressed severe class imbalance with `class_weight='balanced'` ($w_1 = 40.0, w_0 = 0.506$).
2. **Evaluation Metrics (Holdout Test Split: 200 txns)**:
   - **Fraud Recall**: `1.0000` (100.0% — **zero missed frauds**)
   - **Fraud Precision**: `0.6000` (60.0%)
   - **Fraud F1-Score**: `0.7500` (75.0%)
   - **Accuracy**: `0.9900` (99.0%)
   - **ROC-AUC**: `1.0000`
   - **Confusion Matrix**: TN=195, FP=2, FN=0, TP=3
3. **Saved Artifacts**:
   - Serialized model: [`models/logistic_regression_model.joblib`](models/logistic_regression_model.joblib).
   - Metrics: [`results/evaluation_results.json`](results/evaluation_results.json) & [`results/classification_report.txt`](results/classification_report.txt).

---

## Week 1 Day 4 — PaySim Dataset Preparation & Transaction-Data Analysis

**Assigned to:** Rajarsh  
**Deliverable:** Clean / Usable PaySim Data & Transaction Analysis  
**Detailed Report:** [`docs/week1_day4.md`](docs/week1_day4.md) / [`docs/week1/week1_day4.md`](docs/week1/week1_day4.md)

1. **PaySim Ingestion & Cleaning**: Preprocessing script [`src/data_preprocessing.py`](src/data_preprocessing.py) ensuring 100% data retention.
2. **Domain-Specific Feature Engineering**: Calculated balance discrepancies (`errorBalanceOrig`, `errorBalanceDest`) and ISO-8601 timestamps.
3. **Kafka & Cassandra Alignment**: Mapped schema to Kafka topic `fraud-transactions` and Cassandra table `fraud_detection.transactions`.
4. **Key EDA Discovery**: 100% of fraud incidents in PaySim occur exclusively in `TRANSFER` and `CASH_OUT` and drain 100% of the origin balance.
5. **Streaming Adapter**: [`src/stream_paysim_producer.py`](src/stream_paysim_producer.py) for streaming transactions into Kafka.

---

## Project File Structure (Organized by Weeks)

The project is structured with dedicated weekly folders for clean modularity:

```text
financial-fraud-detection-project.no2/
├── .gitignore
├── README.md                              # Complete project documentation & guide
├── requirements.txt                       # Project dependencies
│
├── data/
│   ├── raw/
│   │   └── paysim_sample.csv              # [Week 1 Day 4] Baseline raw PaySim dataset
│   └── processed/
│       ├── week1/                         # [WEEK 1 FOLDER]
│       │   ├── paysim_cleaned.csv         # Cleaned PaySim dataset
│       │   └── paysim_stream_ready.json   # Stream-ready JSON events for Kafka
│       ├── week2/                         # [WEEK 2 FOLDER]
│       │   ├── X_train.csv                # Training feature matrix
│       │   ├── X_test.csv                 # Testing feature matrix
│       │   ├── y_train.csv                # Training labels
│       │   ├── y_test.csv                 # Testing labels
│       │   ├── preprocessor.joblib        # Fitted ColumnTransformer
│       │   └── split_summary.json         # Split metadata & class weights
│       ├── week2_day3/                    # (Week 2 Day 3 source split)
│       ├── week3/                         # [WEEK 3 FOLDER]
│       │   ├── prediction_results.csv     # Enriched prediction records
│       │   ├── fraud_alerts.json          # Security fraud alerts
│       │   └── prediction_summary.json    # Prediction audit report
│       ├── week3_day4/                    # (Week 3 Day 4 output folder)
│       └── paysim_feature_engineered.csv  # 1,000 feature-engineered transactions
│
├── models/
│   ├── logistic_regression_model.joblib   # [Week 2 Day 4] Serialized trained model
│   └── evaluation_results.json            # Model evaluation metrics copy
│
├── results/
│   ├── week2/                             # [WEEK 2 RESULTS]
│   │   ├── evaluation_results.json        # Evaluation metrics JSON
│   │   └── classification_report.txt      # Text classification report
│   └── week3/                             # [WEEK 3 RESULTS]
│       ├── prediction_summary.json        # Executive prediction audit
│       ├── fraud_alerts.json              # Flagged fraud alerts
│       └── prediction_results.csv         # Prediction outputs
│
├── docs/
│   ├── week1/                             # [WEEK 1 DOCS]
│   │   ├── week1_day4.md                  # PaySim prep & EDA report
│   │   └── transaction_analysis_report.md # Statistical transaction report
│   ├── week2/                             # [WEEK 2 DOCS]
│   │   ├── week2_day4.md                  # Logistic Regression evaluation report
│   │   └── week2_day3.md                  # ML dataset preparation report
│   ├── week3/                             # [WEEK 3 DOCS]
│   │   └── week3_day4.md                  # Model integration & prediction report
│   ├── week1_day4.md                      # (Root doc copy for direct links)
│   ├── week2_day4.md                      # (Root doc copy for direct links)
│   ├── week3_day4.md                      # (Root doc copy for direct links)
│   └── figures/                           # Diagnostic Visualization Figures
│       ├── week1/                         # [Week 1 Charts]
│       │   ├── transaction_type_distribution.png
│       │   ├── fraud_by_transaction_type.png
│       │   └── amount_distribution_by_class.png
│       ├── week2/                         # [Week 2 Charts]
│       │   ├── confusion_matrix.png
│       │   ├── roc_curve.png
│       │   ├── precision_recall_curve.png
│       │   └── logistic_regression_coefficients.png
│       └── week3/                         # [Week 3 Charts]
│           ├── prediction_probability_distribution.png
│           ├── predictions_by_payment_type.png
│           └── risk_level_breakdown.png
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py              # [Week 1 Day 4] Cleaning, feature engineering & export
│   ├── transaction_analysis.py            # [Week 1 Day 4] EDA, statistics & visualization engine
│   ├── generate_sample_data.py            # [Week 1 Day 4] PaySim synthetic distribution generator
│   ├── stream_paysim_producer.py          # [Week 1 Day 4] Kafka streaming producer adapter
│   ├── prepare_ml_dataset.py              # [Week 2 Day 3] ML train/test splitting and encoding
│   ├── train_logistic_regression.py       # [Week 2 Day 4] Model training & performance evaluation
│   └── integrate_ml_model.py              # [Week 3 Day 4] Model inference & prediction service
│
└── tests/
    ├── __init__.py
    ├── test_data_preprocessing.py         # [Week 1 Day 4] Unit tests for preprocessing
    ├── test_transaction_analysis.py       # [Week 1 Day 4] Unit tests for analysis & metrics
    ├── test_prepare_ml_dataset.py         # [Week 2 Day 3] Unit tests for ML dataset preparation
    ├── test_train_logistic_regression.py  # [Week 2 Day 4] Unit tests for Logistic Regression
    └── test_integrate_ml_model.py         # [Week 3 Day 4] Unit tests for model integration
```

---

## How to Execute

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Week 3 Day 4 (ML Model Integration & Prediction)
Executes the inference engine on transactions, classifies fraudulent and legitimate records, generates security alerts, and exports diagnostic charts:
```bash
python src/integrate_ml_model.py
```

### 3. Run Week 2 Day 4 (Train Logistic Regression Model)
```bash
python src/train_logistic_regression.py
```

### 4. Run Week 1 Day 4 (PaySim Data Preparation & EDA)
```bash
python src/data_preprocessing.py
python src/transaction_analysis.py
```

### 5. Run Full Automated Unit Test Suite
```bash
python -m unittest discover tests -v
```

---

## Automated Test Verification

```text
Ran 26 tests in 3.396s
OK
```
All **26 unit tests** covering data preparation, transaction analysis, model training, and model prediction pass with 100% success.

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
