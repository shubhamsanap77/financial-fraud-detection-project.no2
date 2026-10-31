# PaySim Transaction-Data Analysis Report

## 1. Executive Summary
- **Total Transactions Analyzed**: 5,000
- **Legitimate Transactions**: 4,900 (98.00%)
- **Fraudulent Transactions**: 2.0% (100 transactions)
- **Class Imbalance Ratio**: 1:49
- **Total Monetary Volume**: $253,744,943.13

## 2. Transaction Type Distribution and Fraud Exposure
The analysis confirms a foundational domain characteristic of PaySim: **Fraudulent activity occurs strictly within `TRANSFER` and `CASH_OUT` transaction types**. `PAYMENT`, `CASH_IN`, and `DEBIT` show zero fraud events.

| Transaction Type | Total Count | Proportion (%) | Fraud Count | Type Fraud Rate (%) | Share of All Frauds (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CASH_IN** | 954 | 19.08% | 0 | 0.0% | 0.0% |
| **CASH_OUT** | 1,758 | 35.16% | 41 | 2.3322% | 41.0% |
| **DEBIT** | 95 | 1.9% | 0 | 0.0% | 0.0% |
| **PAYMENT** | 1,741 | 34.82% | 0 | 0.0% | 0.0% |
| **TRANSFER** | 452 | 9.04% | 59 | 13.0531% | 59.0% |

## 3. Transaction Amount Characteristics
- **Overall Mean Amount**: $50,748.99
- **Overall Median Amount**: $20,330.83
- **Legitimate Mean Amount**: $41,468.10
- **Fraudulent Mean Amount**: $505,512.42
- **Fraudulent Median Amount**: $491,697.62
- **Maximum Fraud Amount**: $969,973.79

*Key Observation*: Fraudulent transactions demonstrate substantially higher average amounts ($505,512.42) compared to normal retail legitimate transactions ($41,468.10).

## 4. Fraud Behavioral Patterns
- **Account Draining Phenomenon**: In **100.0%** of fraudulent transactions (100 instances), the originating account balance was emptied completely (`newbalanceOrig == 0.0`).
- **Fraud Type Exclusivity**: Frauds exclusively manifest as **`CASH_OUT, TRANSFER`**.

## 5. Evaluation of Static Rule-Based Detection (`isFlaggedFraud`)
The legacy system flags transactions as `isFlaggedFraud` using a naive threshold (transfers over $200,000).
- **True Positives**: 10
- **False Negatives (Missed Frauds)**: 90
- **False Positives**: 3
- **Detection Sensitivity**: 10.0%

*Implication*: Static rule-based systems fail to detect over 80-90% of actual fraud events. This demonstrates the critical necessity for real-time machine learning pipelines utilizing Kafka and Cassandra.

## 6. Generated Visualizations
- `docs/figures/transaction_type_distribution.png`: Overall distribution of transaction volumes.
- `docs/figures/fraud_by_transaction_type.png`: Isolation of fraud occurrences by transaction type.
- `docs/figures/amount_distribution_by_class.png`: Boxplot comparison of transaction amounts.
