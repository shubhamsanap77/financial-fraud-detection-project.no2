# Week 3 Day 4 — ML Model Integration, Fraud Prediction & Results Generation

## Objective
Integrate the trained Machine Learning fraud classification model (`models/logistic_regression_model.joblib`) with the transaction processing pipeline, score incoming transactions into legitimate and fraudulent classes, categorize risk levels, generate automated fraud alerts for suspicious activity, and export comprehensive prediction results.

**Assigned to:** Rajarsh  
**Branch:** `rajarsh`  

---

## Architecture & Integration Pipeline

```text
[ Incoming Transaction Data ]
             │
             ▼
[ Data Validation & Imputation ]
   (16 raw features verified)
             │
             ▼
[ Preprocessor Transformation ]
   (ColumnTransformer: 20 encoded features)
             │
             ▼
[ Model Inference Engine ]
   (Logistic Regression Pipeline with StandardScaler)
             │
             ▼
[ Probability & Threshold Scoring ]
   (P(fraud) computed, Threshold = 0.50)
             │
             ├──► [ Class Assignment ]: LEGITIMATE (0) vs FRAUD (1)
             ├──► [ Risk Tier Mapping ]: LOW / MEDIUM / HIGH / CRITICAL
             ├──► [ Fraud Alert Generator ]: Real-time alert trigger
             │
             ▼
[ Output Results Generation ]
   ├── prediction_results.csv (Full enriched dataset)
   ├── fraud_alerts.json (High-risk security alerts)
   ├── prediction_summary.json (Executive audit report)
   └── Diagnostic Figures (docs/figures/week3/)
```

---

## Implementation Details

- **Core Script**: [`src/integrate_ml_model.py`](../../src/integrate_ml_model.py)
- **Model Used**:
  - `models/logistic_regression_model.joblib`: Regularized Logistic Regression model trained with balanced class weights in Week 2 Day 4.
  - `data/processed/week2_day3/preprocessor.joblib`: Training-fitted preprocessor handling median imputation for numeric values and one-hot encoding for payment types.
- **Risk Level Categorization**:
  - **CRITICAL** ($P(\text{fraud}) \ge 0.85$): Immediate transaction freeze and high-priority escalation.
  - **HIGH** ($0.50 \le P(\text{fraud}) < 0.85$): Strong fraud probability, automated alert triggered for review.
  - **MEDIUM** ($0.20 \le P(\text{fraud}) < 0.50$): Borderline activity, secondary verification recommended.
  - **LOW** ($P(\text{fraud}) < 0.20$): Normal retail transactional behavior.
- **Confidence Scoring**:
  $$\text{Confidence Score (\%)} = |P(\text{fraud}) - 0.50| \times 2 \times 100\%$$
- **Alert Generation**:
  Every transaction exceeding the classification threshold ($\tau = 0.50$) automatically generates a security alert record with transaction ID, amount, user ID, probability, risk tier, and explanation reason.

---

## Prediction Results Summary

Evaluated on the feature-engineered transaction dataset (`data/processed/paysim_feature_engineered.csv`):

| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Total Transactions Scored** | **1,000** | **100.0%** |
| **Predicted Legitimate Transactions** | **982** | **98.20%** |
| **Predicted Fraudulent Transactions** | **18** | **1.80%** |
| **Fraud Alerts Raised** | **18** | **100% of detected fraud** |

### Risk Level Tier Breakdown
- **CRITICAL**: 13 transactions (1.30%)
- **HIGH**: 5 transactions (0.50%)
- **MEDIUM**: 7 transactions (0.70%)
- **LOW**: 975 transactions (97.50%)

### Probability Characteristics
- **Mean Fraud Probability**: 0.0213 (2.13%)
- **Median Fraud Probability**: 0.0002 (0.02%)
- **Maximum Fraud Probability**: 1.0000 (100.0%)
- **Minimum Fraud Probability**: 0.0000 (0.00%)

---

## Saved Deliverables & Artifacts

All prediction outputs have been generated and organized into dedicated weekly folders:

### 1. Data Outputs (`data/processed/week3_day4/` & `data/processed/week3/`)
- `prediction_results.csv`: Complete dataset enriched with `fraud_probability`, `predicted_label`, `prediction_class`, `risk_level`, `fraud_alert`, and `confidence_score_pct`.
- `fraud_alerts.json`: Formatted security alerts containing high-risk transactions for downstream Kafka / Cassandra notification systems.
- `prediction_summary.json`: Machine-readable executive report of all inference metrics.

### 2. Diagnostic Visualizations (`docs/figures/week3/`)
- `docs/figures/week3/prediction_probability_distribution.png`: Histogram displaying separation of legitimate vs fraudulent probability densities.
- `docs/figures/week3/predictions_by_payment_type.png`: Stacked breakdown showing predictions across transaction payment types.
- `docs/figures/week3/risk_level_breakdown.png`: Categorical distribution across risk levels (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).

### 3. Automated Unit Tests (`tests/test_integrate_ml_model.py`)
- Unit tests covering model loading, schema validation, batch predictions, single-event real-time scoring, risk level mapping, threshold adjustment, and artifact generation.
- **Result**: All **26/26** tests in the project suite passed (**OK**).

---

## How to Execute

### 1. Run Model Integration and Prediction Pipeline
```bash
python src/integrate_ml_model.py
```

### 2. Custom Threshold Execution (e.g. Higher Sensitivity)
```bash
python src/integrate_ml_model.py --threshold 0.35 --output-dir data/processed/week3_day4
```

### 3. Run Automated Unit Tests
```bash
python -m unittest discover tests -v
```

---

## Verification & Status

- **Automated Tests**: `Ran 26 tests in 3.396s — OK` (26/26 passed)
- **Model Inference**: 1,000 / 1,000 transactions processed with 0 missing or non-finite outputs.
- **Weekly Organization**: All files from Week 1, Week 2, and Week 3 organized into designated weekly folders.
