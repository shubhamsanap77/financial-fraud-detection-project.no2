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
