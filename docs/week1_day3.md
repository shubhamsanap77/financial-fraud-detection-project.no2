# Week 1 Day 3 — Cassandra Setup and Database Configuration

## Objective
Set up Apache Cassandra for the real-time financial fraud detection pipeline.

## Environment
- Ubuntu 26.04 LTS on WSL
- Apache Cassandra 5.0.9
- OpenJDK 17
- Python 3.13.7
- cassandra-driver 3.30.1

## Cassandra Keyspace
CREATE KEYSPACE IF NOT EXISTS fraud_detection
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

## Transactions Table
CREATE TABLE IF NOT EXISTS fraud_detection.transactions (
    transaction_id text PRIMARY KEY,
    timestamp timestamp,
    user_id text,
    amount double,
    currency text,
    payment_type text,
    is_synthetic_test_event boolean
);

## Project Files
- cassandra/schema.cql
- src/cassandra_client.py
- tests/test_cassandra.py
- requirements.txt

## Verification
Cassandra node status: UN 127.0.0.1
Cassandra version: 5.0.9
Python connection test: Cassandra connection successful. Version: 5.0.9
Automated tests: Ran 2 tests — OK

## Result
Cassandra is installed, running, connected to Python, and configured with the required transactions table.
