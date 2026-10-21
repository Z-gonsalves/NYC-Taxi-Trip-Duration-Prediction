
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    count,
    max as spark_max,
    min as spark_min,
    window,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    LongType,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "streaming" / "incoming"
OUTPUT_DIR = PROJECT_DIR / "streaming" / "output"
CHECKPOINT_DIR = PROJECT_DIR / "streaming" / "checkpoint"


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("NYCTaxiStructuredStreaming")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    schema = StructType([
        StructField("id", StringType(), True),
        StructField("vendor_id", IntegerType(), True),
        StructField("pickup_datetime", StringType(), True),
        StructField("dropoff_datetime", StringType(), True),
        StructField("passenger_count", IntegerType(), True),
        StructField("pickup_longitude", DoubleType(), True),
        StructField("pickup_latitude", DoubleType(), True),
        StructField("dropoff_longitude", DoubleType(), True),
        StructField("dropoff_latitude", DoubleType(), True),
        StructField("store_and_fwd_flag", StringType(), True),
        StructField("trip_duration", LongType(), True),
        StructField("pickup_hour", IntegerType(), True),
        StructField("pickup_dayofweek", IntegerType(), True),
        StructField("pickup_month", IntegerType(), True),
        StructField("is_weekend", IntegerType(), True),
        StructField("distance_km", DoubleType(), True),
    ])

    print("Starting Spark Structured Streaming...")
    print("Source: simulated replay of historical NYC taxi records.")
    print(f"Watching: {INPUT_DIR}")

    incoming = (
        spark.readStream
        .schema(schema)
        .option("maxFilesPerTrigger", 1)
        .json(str(INPUT_DIR))
    )

    valid_trips = incoming.filter(
        incoming.trip_duration.isNotNull()
        & (incoming.trip_duration > 0)
        & incoming.distance_km.isNotNull()
        & (incoming.distance_km >= 0)
    )

    # Process each incoming micro-batch and report its statistics.
    def report_batch(batch_df, batch_id):
        stats = (
            batch_df.agg(
                count("*").alias("records_processed"),
                avg("trip_duration").alias("avg_duration_seconds"),
                avg("distance_km").alias("avg_distance_km"),
                spark_min("trip_duration").alias("min_duration_seconds"),
                spark_max("trip_duration").alias("max_duration_seconds"),
            )
            .collect()[0]
        )

        print(f"\n--- Streaming micro-batch {batch_id} ---")
        print(f"Records processed: {stats['records_processed']}")
        print(f"Average trip duration: {stats['avg_duration_seconds']:.2f} seconds"
              if stats["avg_duration_seconds"] is not None else "Average trip duration: N/A")
        print(f"Average distance: {stats['avg_distance_km']:.3f} km"
              if stats["avg_distance_km"] is not None else "Average distance: N/A")
        print(f"Minimum duration: {stats['min_duration_seconds']} seconds")
        print(f"Maximum duration: {stats['max_duration_seconds']} seconds")

    query = (
        valid_trips.writeStream
        .foreachBatch(report_batch)
        .option("checkpointLocation", str(CHECKPOINT_DIR))
        .trigger(processingTime="2 seconds")
        .start()
    )

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\nStopping Spark streaming consumer...")
        query.stop()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
