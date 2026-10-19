import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType, StructField, StringType,
    DoubleType, LongType
)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "fraud-transactions")

schema = StructType([
    StructField("step", LongType(), True),
    StructField("type", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("nameOrig", StringType(), True),
    StructField("oldbalanceOrg", DoubleType(), True),
    StructField("newbalanceOrig", DoubleType(), True),
    StructField("nameDest", StringType(), True),
    StructField("oldbalanceDest", DoubleType(), True),
    StructField("newbalanceDest", DoubleType(), True),
    StructField("isFraud", LongType(), True),
])

def main():
    spark = (
        SparkSession.builder
        .appName("RealTimeFraudDetection")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"Connecting to Kafka topic: {KAFKA_TOPIC}")

    stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    transactions = (
        stream.selectExpr("CAST(value AS STRING) AS json_value")
        .select(from_json(col("json_value"), schema).alias("tx"))
        .select("tx.*")
    )

    # Baseline rule: flag transactions marked as fraud in the event.
    # ML model scoring will be integrated in the next step.
    results = transactions.withColumn(
        "fraud_alert",
        col("isFraud") == 1
    )

    query = (
        results.writeStream
        .outputMode("append")
        .format("console")
        .option("truncate", "false")
        .option("checkpointLocation", "/tmp/fraud-spark-checkpoint")
        .start()
    )

    print("Fraud detection stream started. Press Ctrl+C to stop.")
    query.awaitTermination()

if __name__ == "__main__":
    main()