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