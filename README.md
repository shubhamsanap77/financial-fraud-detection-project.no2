# financial-fraud-detection-project.no2
Real-Time Financial Fraud Detection Pipeline using Kafka, Cassandra, Python and Docker

# Financial Fraud Detection

Real-Time Financial Fraud Detection Pipeline using Python, Kafka, Cassandra, and Docker.

## Project Overview

This project focuses on building a real-time financial fraud detection pipeline for identifying potentially fraudulent transactions. The pipeline streams transactional events through Apache Kafka, persists records into Apache Cassandra, and processes/analyzes transaction distributions using the PaySim benchmark dataset with machine learning models.

## Technologies

- **Python** (Data preprocessing, feature engineering, ML models, Kafka clients, Cassandra driver)
- **Apache Kafka** (Real-time message streaming)
- **Apache Cassandra** (NoSQL high-throughput database storage)
- **Docker & Docker Compose** (Containerized infrastructure)
- **Scikit-Learn & Machine Learning** (Logistic Regression, class imbalance weighting, metrics evaluation)
- **GitHub** (Collaborative branch-based workflow)

## Team

- Shubham Sanap
- Himanshu
- Rohan
- Rajarsh Rajguru
- Omkar

---

## Table of Contents
1. [Week 1 Day 4 — PaySim Dataset Preparation & Transaction-Data Analysis](#week-1-day-4--paysim-dataset-preparation--transaction-data-analysis)
2. [Week 2 Day 4 — Logistic Regression Model Training & Evaluation](#week-2-day-4--logistic-regression-model-training--evaluation)
3. [Project File Structure](#project-file-structure)
4. [How to Execute](#how-to-execute)
5. [Automated Test Verification](#automated-test-verification)

---

## Week 1 Day 4 — PaySim Dataset Preparation & Transaction-Data Analysis

**Assigned to:** Rajarsh  
**Branch:** `rajarsh`  
**Deliverable:** Clean / Usable PaySim Data & Transaction Analysis  
**Detailed Report:** [`docs/week1_day4.md`](docs/week1_day4.md) & [`docs/transaction_analysis_report.md`](docs/transaction_analysis_report.md)

### Objectives & Work Completed
1. **PaySim Dataset Ingestion & Cleaning**:
   - Built an automated data cleaning and validation pipeline ([`src/data_preprocessing.py`](src/data_preprocessing.py)).
   - Enforced strict schema constraints on all 11 PaySim columns (`step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`).
   - Removed corrupted/negative values and duplicate transaction records with 100% data retention.

2. **Domain-Specific Feature Engineering**:
   - **Sender Balance Discrepancy**: $\text{errorBalanceOrig} = (\text{oldbalanceOrg} - \text{amount}) - \text{newbalanceOrig}$
   - **Recipient Balance Discrepancy**: $\text{errorBalanceDest} = (\text{oldbalanceDest} + \text{amount}) - \text{newbalanceDest}$
   - **Temporal Mapping**: Converted hourly simulation `step` (1–744) into ISO-8601 timestamps (`timestamp`).

3. **Pipeline & Schema Alignment (Kafka + Cassandra)**:
   - Generated standard UUID `transaction_id` matching Cassandra's primary key (`transaction_id text PRIMARY KEY`) and Kafka message key.
   - Mapped fields: `nameOrig` $\rightarrow$ `user_id`, `type` $\rightarrow$ `payment_type`, added `currency = 'USD'`.
   - Exported clean, stream-ready dataset payloads in both CSV ([`data/processed/paysim_cleaned.csv`](data/processed/paysim_cleaned.csv)) and JSON format ([`data/processed/paysim_stream_ready.json`](data/processed/paysim_stream_ready.json)).

4. **Exploratory Data Analysis (EDA) & Fraud Behavior Discovery**:
   - **Fraud Exclusivity**: 100% of fraud incidents are isolated strictly to **`TRANSFER`** and **`CASH_OUT`** transactions (`PAYMENT`, `CASH_IN`, and `DEBIT` contain 0 fraud).
   - **Account Draining**: In **100%** of fraud events, the originating account balance was emptied completely (`newbalanceOrig == 0.0`).
   - **Static Heuristic Failure**: Evaluated `isFlaggedFraud` threshold rule; it missed >85% of real fraud, justifying the need for this real-time ML streaming pipeline.
   - Generated high-resolution diagnostic figures:
     - `docs/figures/transaction_type_distribution.png`
     - `docs/figures/fraud_by_transaction_type.png`
     - `docs/figures/amount_distribution_by_class.png`

5. **Kafka Streaming Adapter**:
   - Developed [`src/stream_paysim_producer.py`](src/stream_paysim_producer.py) to stream clean PaySim transaction payloads into the `fraud-transactions` Kafka topic.

---

## Week 2 Day 4 — Logistic Regression Model Training & Evaluation

**Assigned to:** Rajarsh  
**Branch:** `rajarsh`  
**Deliverable:** Trained Model, Performance Evaluation & Saved Artifacts  
**Detailed Report:** [`docs/week2_day4.md`](docs/week2_day4.md)

### Objectives & Work Completed
1. **Model Architecture & Training Pipeline**:
   - Built an end-to-end training pipeline ([`src/train_logistic_regression.py`](src/train_logistic_regression.py)) utilizing `StandardScaler` and `LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)`.
   - Prevented data leakage by fitting scaling parameters strictly on training features.
   - Addressed extreme fraud class imbalance using `class_weight='balanced'` ($w_1 = 40.0$, $w_0 = 0.506$).

2. **Model Performance Evaluation (Test Set: 200 transactions)**:
   - **Fraud Recall**: `1.0000` (100.0% — **zero missed fraud events**)
   - **Fraud Precision**: `0.6000` (60.0%)
   - **Fraud F1-Score**: `0.7500` (75.0%)
   - **Overall Accuracy**: `0.9900` (99.0%)
   - **ROC-AUC Score**: `1.0000`
   - **Average Precision (PR-AUC)**: `1.0000`

   **Confusion Matrix**:
   | Predicted \ True | True Non-Fraud (0) | True Fraud (1) |
   | :--- | :--- | :--- |
   | **Pred Non-Fraud (0)** | **195** (TN) | **0** (FN — zero missed) |
   | **Pred Fraud (1)** | **2** (FP) | **3** (TP) |

3. **Saved Model Artifacts & Deliverables**:
   - Trained model pipeline serialized to [`models/logistic_regression_model.joblib`](models/logistic_regression_model.joblib).
   - Evaluation metrics exported to [`results/evaluation_results.json`](results/evaluation_results.json).
   - Text classification report saved to [`results/classification_report.txt`](results/classification_report.txt).
   - Generated visual diagnostic plots in `docs/figures/`:
     - `docs/figures/confusion_matrix.png`
     - `docs/figures/roc_curve.png`
     - `docs/figures/precision_recall_curve.png`
     - `docs/figures/logistic_regression_coefficients.png`

---

## Project File Structure

All files created across both Week 1 Day 4 and Week 2 Day 4 are committed and present:

```text
financial-fraud-detection-project.no2/
├── .gitignore
├── README.md                              # Complete project & execution guide
├── requirements.txt                       # Project dependencies
├── data/
│   ├── raw/
│   │   └── paysim_sample.csv              # [Week 1 Day 4] Baseline raw PaySim dataset (5,000 txns)
│   └── processed/
│       ├── paysim_cleaned.csv             # [Week 1 Day 4] Cleaned dataset with engineered features
│       ├── paysim_stream_ready.json       # [Week 1 Day 4] Stream-ready JSON events for Kafka
│       └── week2_day3/                    # [Week 2 Day 3] ML-ready train/test splits
│           ├── X_train.csv
│           ├── X_test.csv
│           ├── y_train.csv
│           ├── y_test.csv
│           └── split_summary.json
├── models/
│   ├── logistic_regression_model.joblib   # [Week 2 Day 4] Trained and serialized ML model
│   └── evaluation_results.json            # [Week 2 Day 4] Model evaluation metrics copy
├── results/
│   ├── evaluation_results.json            # [Week 2 Day 4] Structured evaluation metrics
│   └── classification_report.txt          # [Week 2 Day 4] Plain-text classification report
├── docs/
│   ├── week1_day4.md                      # [Week 1 Day 4] PaySim prep & EDA documentation
│   ├── transaction_analysis_report.md     # [Week 1 Day 4] Statistical transaction report
│   ├── week2_day3.md                      # [Week 2 Day 3] ML dataset prep documentation
│   ├── week2_day4.md                      # [Week 2 Day 4] Logistic Regression evaluation report
│   └── figures/                           # High-resolution diagnostic figures
│       ├── confusion_matrix.png           # [Week 2 Day 4]
│       ├── roc_curve.png                  # [Week 2 Day 4]
│       ├── precision_recall_curve.png     # [Week 2 Day 4]
│       ├── logistic_regression_coefficients.png # [Week 2 Day 4]
│       ├── amount_distribution_by_class.png     # [Week 1 Day 4]
│       ├── fraud_by_transaction_type.png        # [Week 1 Day 4]
│       └── transaction_type_distribution.png    # [Week 1 Day 4]
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py              # [Week 1 Day 4] Cleaning, feature engineering & export
│   ├── transaction_analysis.py            # [Week 1 Day 4] EDA, statistics & visualization engine
│   ├── generate_sample_data.py            # [Week 1 Day 4] PaySim synthetic distribution generator
│   ├── stream_paysim_producer.py          # [Week 1 Day 4] Kafka streaming producer adapter
│   ├── prepare_ml_dataset.py              # [Week 2 Day 3] ML train/test splitting and encoding
│   └── train_logistic_regression.py       # [Week 2 Day 4] Model training, evaluation & metrics export
└── tests/
    ├── __init__.py
    ├── test_data_preprocessing.py         # [Week 1 Day 4] Unit tests for preprocessing
    ├── test_transaction_analysis.py       # [Week 1 Day 4] Unit tests for analysis & metrics
    ├── test_prepare_ml_dataset.py         # [Week 2 Day 3] Unit tests for ML dataset preparation
    └── test_train_logistic_regression.py  # [Week 2 Day 4] Unit tests for Logistic Regression
```

---

## How to Execute

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Week 1 Day 4 Work (Data Preparation & EDA Analysis)
```bash
# Prepare and clean raw PaySim data:
python src/data_preprocessing.py \
  --input data/raw/paysim_sample.csv \
  --output-csv data/processed/paysim_cleaned.csv \
  --output-json data/processed/paysim_stream_ready.json

# Run EDA analysis and generate charts:
python src/transaction_analysis.py \
  --input data/processed/paysim_cleaned.csv \
  --report docs/transaction_analysis_report.md \
  --figures-dir docs/figures

# Dry-run Kafka streaming adapter:
python src/stream_paysim_producer.py --dry-run --limit 5
```

### 3. Run Week 2 Day 4 Work (Train & Evaluate Logistic Regression)
```bash
python src/train_logistic_regression.py
```

### 4. Run Automated Unit Tests (19/19 Passed)
```bash
python -m unittest discover tests -v
```

---

## Automated Test Verification
```text
Ran 19 tests in 2.456s
OK
```
All unit tests covering Week 1 Day 4 and Week 2 Day 4 pass with 100% success.
