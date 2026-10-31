"""Cassandra connection utilities for the fraud detection pipeline."""

import os

from cassandra.cluster import Cluster, Session


CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "127.0.0.1")
CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
CASSANDRA_KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "fraud_detection")


def get_session() -> Session:
    """Create a Cassandra session connected to the project keyspace."""
    cluster = Cluster([CASSANDRA_HOST], port=CASSANDRA_PORT)
    session = cluster.connect(CASSANDRA_KEYSPACE)
    return session


def health_check() -> str:
    """Return the Cassandra release version to verify connectivity."""
    session = get_session()
    try:
        row = session.execute(
            "SELECT release_version FROM system.local"
        ).one()
        return row.release_version
    finally:
        session.shutdown()
        session.cluster.shutdown()


if __name__ == "__main__":
    version = health_check()
    print(f"Cassandra connection successful. Version: {version}")
