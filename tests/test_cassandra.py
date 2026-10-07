"""Tests for the Week 1 Day 3 Cassandra setup."""

import unittest
import uuid
from datetime import datetime, timezone

from cassandra_client import CASSANDRA_KEYSPACE, get_session


class CassandraIntegrationTest(unittest.TestCase):
    """Verify Cassandra connectivity and transaction CRUD behavior."""

    def setUp(self):
        self.session = get_session()

    def tearDown(self):
        self.session.shutdown()
        self.session.cluster.shutdown()

    def test_keyspace_and_table_exist(self):
        row = self.session.execute(
            """
            SELECT keyspace_name, table_name
            FROM system_schema.tables
            WHERE keyspace_name = %s
              AND table_name = %s
            """,
            (CASSANDRA_KEYSPACE, "transactions"),
        ).one()

        self.assertIsNotNone(row)
        self.assertEqual(row.keyspace_name, CASSANDRA_KEYSPACE)
        self.assertEqual(row.table_name, "transactions")

    def test_transaction_insert_and_read(self):
        transaction_id = f"test-{uuid.uuid4()}"
        timestamp = datetime.now(timezone.utc)

        self.session.execute(
            """
            INSERT INTO transactions (
                transaction_id,
                timestamp,
                user_id,
                amount,
                currency,
                payment_type,
                is_synthetic_test_event
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                transaction_id,
                timestamp,
                "user_test",
                99.99,
                "USD",
                "TRANSFER",
                True,
            ),
        )

        row = self.session.execute(
            """
            SELECT transaction_id, user_id, amount, currency,
                   payment_type, is_synthetic_test_event
            FROM transactions
            WHERE transaction_id = %s
            """,
            (transaction_id,),
        ).one()

        self.assertIsNotNone(row)
        self.assertEqual(row.transaction_id, transaction_id)
        self.assertEqual(row.user_id, "user_test")
        self.assertAlmostEqual(row.amount, 99.99, places=2)
        self.assertEqual(row.currency, "USD")
        self.assertEqual(row.payment_type, "TRANSFER")
        self.assertTrue(row.is_synthetic_test_event)

        self.session.execute(
            "DELETE FROM transactions WHERE transaction_id = %s",
            (transaction_id,),
        )


if __name__ == "__main__":
    unittest.main()
