"""PaySim Streaming Producer Adapter.

Streams cleaned PaySim transactions into the Kafka 'fraud-transactions' topic,
feeding the real-time Cassandra ingestion and fraud detection pipeline.
Supports both live Kafka publishing and offline dry-run validation.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Any, List

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "fraud-transactions")


def load_stream_events(json_path: str) -> List[Dict[str, Any]]:
    """Load stream-ready PaySim events from JSON."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Stream events file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        events = json.load(f)
    return events


def stream_events(
    events: List[Dict[str, Any]],
    bootstrap_servers: str = BOOTSTRAP_SERVERS,
    topic: str = TOPIC,
    delay_seconds: float = 0.5,
    max_events: int = 10,
    dry_run: bool = False,
) -> int:
    """Stream PaySim events to Kafka topic or perform dry-run."""
    print(f"Preparing to stream up to {max_events} events (dry_run={dry_run}).")
    sent_count = 0

    if dry_run:
        for event in events[:max_events]:
            sent_count += 1
            print(
                f"[DRY-RUN] Event {sent_count}/{max_events}: txn_id={event['transaction_id'][:8]}... "
                f"type={event['payment_type']} amount=${event['amount']:,.2f} "
                f"user={event['user_id']} isFraud={event['is_fraud']}"
            )
            time.sleep(delay_seconds)
        return sent_count

    try:
        from kafka import KafkaProducer

        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers.split(","),
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            key_serializer=lambda key: key.encode("utf-8"),
            acks="all",
            retries=3,
        )
    except Exception as exc:
        print(f"Could not connect to Kafka at {bootstrap_servers}: {exc}")
        print("Falling back to dry-run validation mode.")
        return stream_events(events, delay_seconds=delay_seconds, max_events=max_events, dry_run=True)

    try:
        for event in events[:max_events]:
            future = producer.send(topic, key=event["transaction_id"], value=event)
            metadata = future.get(timeout=10)
            sent_count += 1
            print(
                f"Sent transaction_id={event['transaction_id']} "
                f"topic={metadata.topic} partition={metadata.partition} offset={metadata.offset} "
                f"isFraud={event['is_fraud']}"
            )
            time.sleep(delay_seconds)
    finally:
        producer.flush()
        producer.close()

    return sent_count


def main():
    parser = argparse.ArgumentParser(description="Stream PaySim transactions to Kafka topic")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/paysim_stream_ready.json",
        help="Path to stream-ready JSON events",
    )
    parser.add_argument("--servers", type=str, default=BOOTSTRAP_SERVERS, help="Kafka bootstrap servers")
    parser.add_argument("--topic", type=str, default=TOPIC, help="Kafka target topic")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between transactions in seconds")
    parser.add_argument("--limit", type=int, default=10, help="Max transactions to stream")
    parser.add_argument("--dry-run", action="store_true", help="Simulate streaming without Kafka connection")
    args = parser.parse_args()

    events = load_stream_events(args.input)
    sent = stream_events(
        events=events,
        bootstrap_servers=args.servers,
        topic=args.topic,
        delay_seconds=args.delay,
        max_events=args.limit,
        dry_run=args.dry_run,
    )
    print(f"Successfully processed {sent} PaySim transaction events.")


if __name__ == "__main__":
    main()
