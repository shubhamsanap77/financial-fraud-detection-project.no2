CREATE KEYSPACE IF NOT EXISTS fraud_detection
WITH replication = {
    'class': 'SimpleStrategy',
    'replication_factor': 1
};

CREATE TABLE IF NOT EXISTS fraud_detection.transactions (
    transaction_id text PRIMARY KEY,
    timestamp timestamp,
    user_id text,
    amount double,
    currency text,
    payment_type text,
    is_synthetic_test_event boolean
);
