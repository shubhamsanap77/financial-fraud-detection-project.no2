# financial-fraud-detection-project.no2
Real-Time Financial Fraud Detection Pipeline using Kafka, Cassandra, Python and Docker

# Financial Fraud Detection

Real-Time Financial Fraud Detection Pipeline using Python, Kafka, Cassandra, and Docker.

## Project Overview

This project focuses on building a real-time financial fraud detection pipeline for identifying potentially fraudulent transactions. The pipeline streams transactional events through Apache Kafka, persists records into Apache Cassandra, and processes/analyzes transaction distributions using the PaySim benchmark dataset.

## Technologies

- **Python** (Data preprocessing, EDA analysis, Kafka clients, Cassandra driver)
- **Apache Kafka** (Real-time message streaming)
- **Apache Cassandra** (NoSQL high-throughput database storage)
- **Docker & Docker Compose** (Containerized infrastructure)
- **Machine Learning & Analytics** (Feature engineering, imbalance handling, EDA)
- **GitHub** (Collaborative branch-based workflow)

## Team

- Shubham Sanap
- Himanshu
- Rohan
- Rajarsh Rajguru
- Omkar

---

## Week 1 Day 4 — PaySim Dataset Preparation & Transaction-Data Analysis

**Assigned to:** Rajarsh  
**Branch:** `rajarsh`  
**Deliverable:** Clean / Usable PaySim Data & Transaction Analysis  

### Objectives & Work Completed
1. **PaySim Dataset Ingestion & Cleaning**:
   - Built an automated data cleaning and validation pipeline ([`src/data_preprocessing.py`](src/data_preprocessing.py)).
   - Enforced schema constraints on all 11 PaySim columns (`step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`).
   - Removed corrupted/negative values and duplicate transaction records with 100% data retention.

2. **Domain-Specific Feature Engineering**:
   - **Sender Balance Discrepancy**: $\text{errorBalanceOrig} = (\text{oldbalanceOrg} - \text{amount}) - \text{newbalanceOrig}$
   - **Recipient Balance Discrepancy**: $\text{errorBalanceDest} = (\text{oldbalanceDest} + \text{amount}) - \text{newbalanceDest}$
   - **Temporal Mapping**: Converted hourly simulation `step` (1–744) into ISO-8601 timestamps.

3. **Pipeline & Schema Alignment (Kafka + Cassandra)**:
   - Generated standard UUID `transaction_id` matching Cassandra's primary key (`transaction_id text PRIMARY KEY`) and Kafka message key.
   - Mapped fields: `nameOrig` $\rightarrow$ `user_id`, `type` $\rightarrow$ `payment_type`, added `currency = 'USD'`.
   - Exported clean, stream-ready dataset payloads in both CSV and JSON formats.

4. **Exploratory Data Analysis (EDA) & Fraud Behavior Discovery**:
   - **Fraud Exclusivity**: 100% of fraud incidents are isolated strictly to **`TRANSFER`** and **`CASH_OUT`** transactions (`PAYMENT`, `CASH_IN`, and `DEBIT` contain 0 fraud).
   - **Account Draining**: In **100%** of fraud events, the originating account balance was emptied completely (`newbalanceOrig == 0.0`).
   - **Static Heuristic Failure**: Evaluated `isFlaggedFraud` threshold rule; it missed >85% of real fraud, justifying the need for this real-time ML streaming pipeline.
   - Exported summary report ([`docs/transaction_analysis_report.md`](docs/transaction_analysis_report.md)) and high-resolution figures to `docs/figures/`.

5. **Automated Testing Suite**:
   - Created 10 automated unit tests covering data validation, feature engineering, and analytics. Ran 10/10 tests successfully (**OK**).

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
│       └── paysim_stream_ready.json       # Stream-ready JSON events for Kafka
├── docs/
│   ├── week1_day4.md                      # Official Day 4 verification & report
│   ├── transaction_analysis_report.md     # Detailed statistical analysis report
│   └── figures/                           # Generated EDA visualization charts
│       ├── amount_distribution_by_class.png
│       ├── fraud_by_transaction_type.png
│       └── transaction_type_distribution.png
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py              # Cleaning, feature engineering & export
│   ├── transaction_analysis.py            # EDA, statistics & visualization engine
│   ├── generate_sample_data.py            # PaySim synthetic distribution generator
│   └── stream_paysim_producer.py          # Kafka streaming producer adapter
└── tests/
    ├── __init__.py
    ├── test_data_preprocessing.py         # Unit tests for preprocessing
    └── test_transaction_analysis.py       # Unit tests for analysis & metrics
```

---

## How to Execute

### 1. Clone and Checkout the Branch
```bash
git clone https://github.com/shubhamsanap77/financial-fraud-detection-project.no2.git
cd financial-fraud-detection-project.no2
git checkout rajarsh
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Generate a New PaySim Sample Dataset
```bash
python src/generate_sample_data.py --rows 5000 --output data/raw/paysim_sample.csv
```

### 4. Run the Data Preprocessing Pipeline
Cleans the raw data, engineers balance discrepancy features, aligns fields with the Kafka/Cassandra schema, and exports both CSV and stream-ready JSON files:
```bash
python src/data_preprocessing.py \
  --input data/raw/paysim_sample.csv \
  --output-csv data/processed/paysim_cleaned.csv \
  --output-json data/processed/paysim_stream_ready.json \
  --stream-limit 1000
```

### 5. Run Transaction Data Analysis & Generate Visualizations
Computes class distributions, fraud type patterns, and exports charts to `docs/figures/` and report to `docs/transaction_analysis_report.md`:
```bash
python src/transaction_analysis.py \
  --input data/processed/paysim_cleaned.csv \
  --report docs/transaction_analysis_report.md \
  --figures-dir docs/figures
```

### 6. Test Kafka Streaming Producer (Live or Dry-Run)
Stream clean PaySim events into the `fraud-transactions` Kafka topic:
```bash
# Dry-run validation (no running Kafka broker required):
python src/stream_paysim_producer.py --dry-run --limit 5

# Live Kafka streaming (when Docker Kafka is running on localhost:9092):
python src/stream_paysim_producer.py --servers localhost:9092 --topic fraud-transactions --limit 50
```

### 7. Run Automated Unit Tests
```bash
python -m unittest discover tests -v
```

---

## Verification & Status

- **Automated Tests**: `Ran 10 tests in 1.910s — OK` (10/10 passed)
- **Data Quality Audit**: 5,000 / 5,000 valid records (100% clean retention rate)
- **Branch Status**: Pushed and up-to-date with `origin/rajarsh`
