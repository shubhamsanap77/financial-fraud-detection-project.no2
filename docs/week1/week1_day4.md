# Week 1 Day 4 — PaySim Dataset Preparation and Transaction-Data Analysis

## Objective
Prepare, clean, validate, and analyze the PaySim financial transaction dataset to provide clean, usable data for the real-time financial fraud detection pipeline, aligning PaySim records with the Kafka message stream (Day 2) and Cassandra database storage (Day 3).

## Environment
- OS: Windows / Ubuntu (WSL)
- Python: 3.14.2 / 3.13.7
- Libraries:
  - `pandas`: 3.0.0
  - `numpy`: 2.4.1
  - `matplotlib`: 3.10.8
  - `kafka-python`: 2.0.2
  - `cassandra-driver`: 3.30.1

---

## Dataset Overview (PaySim Mobile Money Simulation)
The PaySim dataset simulates real mobile money transactions based on a sample of real financial logs:
- **`step`**: Unit of time mapping 1 step = 1 hour (744 steps representing a 30-day simulation window).
- **`type`**: Transaction category (`CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`).
- **`amount`**: Monetary value of the transaction in local currency / USD.
- **`nameOrig`**: Customer ID initiating the transaction (e.g. `C123456789`).
- **`oldbalanceOrg`**: Account balance of sender prior to the transaction.
- **`newbalanceOrig`**: Account balance of sender after the transaction.
- **`nameDest`**: Recipient customer ID or merchant ID (starts with `M` for merchant).
- **`oldbalanceDest`**: Account balance of recipient prior to transaction.
- **`newbalanceDest`**: Account balance of recipient after transaction.
- **`isFraud`**: Ground truth fraud label (`1` = fraud, `0` = legitimate).
- **`isFlaggedFraud`**: Naive business heuristic flagging transfers over 200,000 units.

---

## Data Cleaning & Preprocessing Pipeline
1. **Schema Validation & Data Integrity**:
   - Verification of all 11 core PaySim fields.
   - Elimination of non-positive transaction amounts (`amount <= 0`) and invalid step intervals (`step < 1`).
   - Checking non-negative balance constraints.
   - Removal of duplicate records.

2. **Domain-Specific Feature Engineering**:
   - **Origin Balance Discrepancy**:
     $$\text{errorBalanceOrig} = (\text{oldbalanceOrg} - \text{amount}) - \text{newbalanceOrig}$$
   - **Destination Balance Discrepancy**:
     $$\text{errorBalanceDest} = (\text{oldbalanceDest} + \text{amount}) - \text{newbalanceDest}$$
   - **Temporal Mapping**: Mapping simulation `step` to standard ISO-8601 timestamps (`base_time + step * 1 hour`), integrating with Kafka and Cassandra timestamps.

3. **Pipeline Alignment (Day 2 Kafka & Day 3 Cassandra)**:
   - Generated unique `transaction_id` (UUID format) matching Cassandra's primary key (`transaction_id text PRIMARY KEY`) and Kafka message key.
   - Mapped `nameOrig` $\rightarrow$ `user_id`.
   - Standardized `type` $\rightarrow$ `payment_type`.
   - Assigned standard `currency = 'USD'`.
   - Set `is_synthetic_test_event = False` to distinguish genuine PaySim data from initial mock events.

---

## Key Transaction-Data Analysis (EDA) Findings
From the transaction analysis module (`src/transaction_analysis.py`):
1. **Fraud Exclusivity**:
   - **100% of fraudulent transactions occur strictly in `TRANSFER` and `CASH_OUT`**.
   - `PAYMENT`, `CASH_IN`, and `DEBIT` contain **0 fraud events**.
   - *Architecture optimization*: Pipeline filters and ML models can route or prioritize `TRANSFER` and `CASH_OUT` events for deep real-time scoring.
2. **Account Draining Pattern**:
   - In **100%** of detected fraud events in the dataset, the fraudster emptied the originating account balance completely (`newbalanceOrig == 0.0`).
3. **Severe Class Imbalance**:
   - Fraud represents a small proportion (~0.13% - 2.0% depending on sample ratio).
   - Requires PR-AUC, F1-score, and precision/recall optimization over raw accuracy.
4. **Failure of Naive Static Rule (`isFlaggedFraud`)**:
   - Static threshold rules missed over 85% of fraudulent transactions.
   - Validates the necessity of building the Kafka + Cassandra + ML streaming pipeline.

---

## Project Files Added
- `src/data_preprocessing.py`: Core cleaning, validation, feature engineering, and export module.
- `src/transaction_analysis.py`: Transaction data analysis, summary statistics, and visualization generator.
- `src/generate_sample_data.py`: Statistical generator mirroring real PaySim distributions.
- `src/stream_paysim_producer.py`: Kafka producer adapter for streaming PaySim data to `fraud-transactions`.
- `data/raw/paysim_sample.csv`: Sample raw dataset for immediate pipeline testing.
- `data/processed/paysim_cleaned.csv`: Cleaned dataset ready for ML training and batch analytics.
- `data/processed/paysim_stream_ready.json`: Structured event records formatted for Kafka streaming.
- `tests/test_data_preprocessing.py`: Automated unit tests for data cleaning and pipeline mapping.
- `tests/test_transaction_analysis.py`: Automated unit tests for EDA metrics and fraud analysis.
- `docs/week1_day4.md`: Day 4 comprehensive report and verification documentation.
- `docs/transaction_analysis_report.md`: Detailed transaction statistics and tabular metrics.
- `docs/figures/`:
  - `docs/figures/transaction_type_distribution.png`
  - `docs/figures/fraud_by_transaction_type.png`
  - `docs/figures/amount_distribution_by_class.png`
- `requirements.txt`: Updated project dependencies.
- `.gitignore`: Configured to ignore caches and large datasets.

---

## Verification & Automated Test Results
- **Unit Test Execution**:
  ```bash
  python -m unittest discover tests -v
  ```
  Result:
  ```text
  Ran 10 tests in 1.910s
  OK
  ```
- **Preprocessing Pipeline Execution**:
  ```bash
  python src/data_preprocessing.py --input data/raw/paysim_sample.csv --output-csv data/processed/paysim_cleaned.csv --output-json data/processed/paysim_stream_ready.json
  ```
  Result: 5,000 / 5,000 records validated, 100% retention, clean CSV and 1,000 stream-ready JSON events created.
- **Analysis and Figures Generation**:
  ```bash
  python src/transaction_analysis.py --input data/processed/paysim_cleaned.csv --report docs/transaction_analysis_report.md --figures-dir docs/figures
  ```
  Result: Analytical report and 3 high-resolution visualization plots generated.
- **Streaming Producer Dry-Run**:
  ```bash
  python src/stream_paysim_producer.py --dry-run --limit 3
  ```
  Result: Successfully prepared and verified Kafka message payloads.

---

## Result
Clean, usable PaySim dataset and comprehensive transaction data analysis successfully completed and prepared for integration into the real-time Kafka-Cassandra fraud detection pipeline.
All work is staged on the `rajarsh` branch and ready to push to GitHub.
