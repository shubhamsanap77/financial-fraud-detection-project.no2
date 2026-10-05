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
