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

## Week 2 Day 4 — Logistic Regression Model Training & Evaluation

**Assigned to:** Rajarsh  
**Branch:** `rajarsh`  
**Deliverable:** Trained Model, Performance Evaluation & Saved Artifacts  

### Objectives & Work Completed
1. **Model Architecture & Training Pipeline**:
   - Built an end-to-end training pipeline ([`src/train_logistic_regression.py`](src/train_logistic_regression.py)) utilizing `StandardScaler` and `LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)`.
   - Prevented data leakage by fitting scaling parameters strictly on training features.
   - Addressed extreme fraud class imbalance using `class_weight='balanced'` ($w_1 = 40.0$, $w_0 = 0.506$).

2. **Model Performance Evaluation**:
   - Evaluated predictions on the holdout test set (`X_test.csv`, `y_test.csv`):
     - **Fraud Precision**: `0.6000` (60.0%)
     - **Fraud Recall**: `1.0000` (100.0% — **zero missed fraud events**)
     - **Fraud F1-Score**: `0.7500` (75.0%)
     - **Overall Accuracy**: `0.9900` (99.0%)
     - **ROC-AUC Score**: `1.0000`
     - **Average Precision (PR-AUC)**: `1.0000`
   - **Confusion Matrix**:
     - True Negatives (TN): **195**
     - False Positives (FP): **2**
     - False Negatives (FN): **0** (no missed frauds)
     - True Positives (TP): **3**

3. **Saved Model Artifacts & Deliverables**:
   - Trained model pipeline serialized to [`models/logistic_regression_model.joblib`](models/logistic_regression_model.joblib).
   - Evaluation metrics exported to [`results/evaluation_results.json`](results/evaluation_results.json).
   - Text classification report saved to [`results/classification_report.txt`](results/classification_report.txt).
   - Generated visual diagnostic plots in `docs/figures/`:
     - `docs/figures/confusion_matrix.png`
     - `docs/figures/roc_curve.png`
     - `docs/figures/precision_recall_curve.png`
     - `docs/figures/logistic_regression_coefficients.png`
   - Complete documentation written in [`docs/week2_day4.md`](docs/week2_day4.md).

4. **Automated Unit Tests**:
   - Added automated tests in [`tests/test_train_logistic_regression.py`](tests/test_train_logistic_regression.py).
   - All **19/19** tests across the entire repository pass (**OK**).

---

## Week 1 Day 4 — PaySim Dataset Preparation & Transaction-Data Analysis

**Assigned to:** Rajarsh  
**Deliverable:** Clean / Usable PaySim Data & Transaction Analysis  

1. **PaySim Cleaning & Validation**: Automated validation in [`src/data_preprocessing.py`](src/data_preprocessing.py) ensuring 100% data retention.
2. **Domain-Specific Feature Engineering**: Calculated balance discrepancies (`errorBalanceOrig`, `errorBalanceDest`) and ISO-8601 timestamps.
3. **Kafka & Cassandra Alignment**: Mapped schema to Kafka topic `fraud-transactions` and Cassandra table `fraud_detection.transactions`.
4. **Key EDA Discovery**: 100% of fraud transactions in PaySim occur exclusively in `TRANSFER` and `CASH_OUT` and drain 100% of the origin balance. Documentation in [`docs/week1_day4.md`](docs/week1_day4.md).

---

## Project Structure

```text
financial-fraud-detection-project.no2/
├── .gitignore
├── README.md                              # Project & execution documentation
├── requirements.txt                       # Project dependencies
├── data/
│   ├── raw/
│   │   └── paysim_sample.csv              # Baseline raw PaySim dataset (5,000 txns)
│   └── processed/
│       ├── paysim_cleaned.csv             # Cleaned dataset with engineered features
│       ├── paysim_stream_ready.json       # Stream-ready JSON events for Kafka
│       └── week2_day3/                    # ML-ready train/test splits (Day 3)
│           ├── X_train.csv
│           ├── X_test.csv
│           ├── y_train.csv
│           ├── y_test.csv
│           └── split_summary.json
├── models/
│   ├── logistic_regression_model.joblib   # Trained and serialized ML model
│   └── evaluation_results.json            # Model evaluation metrics copy
├── results/
│   ├── evaluation_results.json            # Structured evaluation metrics
│   └── classification_report.txt          # Plain-text classification report
├── docs/
│   ├── week2_day4.md                      # Week 2 Day 4 documentation & evaluation
│   ├── week2_day3.md                      # Week 2 Day 3 ML dataset prep documentation
│   ├── week1_day4.md                      # Week 1 Day 4 PaySim prep & EDA documentation
│   ├── transaction_analysis_report.md     # Statistical transaction report
│   └── figures/                           # Generated visualization charts
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── precision_recall_curve.png
│       ├── logistic_regression_coefficients.png
│       ├── amount_distribution_by_class.png
│       ├── fraud_by_transaction_type.png
│       └── transaction_type_distribution.png
├── src/
│   ├── __init__.py
│   ├── train_logistic_regression.py       # Model training, evaluation & metrics export
│   ├── prepare_ml_dataset.py              # ML train/test splitting and encoding
│   ├── data_preprocessing.py              # Cleaning, feature engineering & export
│   ├── transaction_analysis.py            # EDA, statistics & visualization engine
│   ├── generate_sample_data.py            # PaySim synthetic distribution generator
│   └── stream_paysim_producer.py          # Kafka streaming producer adapter
└── tests/
    ├── __init__.py
    ├── test_train_logistic_regression.py  # Unit tests for Logistic Regression
    ├── test_prepare_ml_dataset.py         # Unit tests for ML dataset preparation
    ├── test_data_preprocessing.py         # Unit tests for preprocessing
    └── test_transaction_analysis.py       # Unit tests for analysis & metrics
```

---

## How to Execute

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and Evaluate Logistic Regression Model (Week 2 Day 4)
Trains the model on `data/processed/week2_day3/`, generates confusion matrix and curves in `docs/figures/`, and saves artifacts in `models/` and `results/`:
```bash
python src/train_logistic_regression.py
```

### 3. Run Automated Unit Tests
```bash
python -m unittest discover tests -v
```

### 4. (Optional) Run Data Preprocessing Pipeline (Week 1 Day 4)
```bash
python src/data_preprocessing.py \
  --input data/raw/paysim_sample.csv \
  --output-csv data/processed/paysim_cleaned.csv \
  --output-json data/processed/paysim_stream_ready.json
```

### 5. (Optional) Run Transaction EDA Analysis
```bash
python src/transaction_analysis.py \
  --input data/processed/paysim_cleaned.csv \
  --report docs/transaction_analysis_report.md \
  --figures-dir docs/figures
```

### 6. (Optional) Test Kafka Streaming Producer (Live or Dry-Run)
```bash
python src/stream_paysim_producer.py --dry-run --limit 5
```

---

## Verification & Status

- **Automated Tests**: `Ran 19 tests in 2.456s — OK` (19/19 passed)
- **Model Evaluation**: Precision: 0.6000, Recall: 1.0000, F1: 0.7500, ROC-AUC: 1.0000
- **Artifacts Saved**: `models/logistic_regression_model.joblib`, `results/evaluation_results.json`, `results/classification_report.txt`
