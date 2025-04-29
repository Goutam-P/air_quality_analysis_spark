from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_session():
    return SparkSession.builder \
        .appName("AirQualityIngestion") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false") \
        .getOrCreate()


def handle_outliers(df):
    logger.info("Handling outliers by capping extreme values")
    return df.withColumn(
        "pm2_5", when(col("pm2_5") < 0, 0.0).when(
            col("pm2_5") > 500, 500.0).otherwise(col("pm2_5"))
    ).withColumn(
        "temperature", when(col("temperature") < -40, -40.0).when(
            col("temperature") > 50, 50.0).otherwise(col("temperature"))
    ).withColumn(
        "humidity", when(col("humidity") < 0, 0.0).when(
            col("humidity") > 100, 100.0).otherwise(col("humidity"))
    ).withColumn("is_valid", lit(True))


def ingest_data():
    logger.info("Running ingestion for pipeline...")
    spark = create_spark_session()

    # Read preprocessed CSV
    df = spark.read.option("header", "true").option(
        "inferSchema", "true").csv("ingestion/task2_feature_enhanced.csv")

    # Select only necessary columns
    df = df.select(
        col("location_id").cast("integer"),
        col("timestamp"),
        col("location"),
        col("lat").cast("double"),
        col("lon").cast("double"),
        col("pm2_5").cast("double"),
        col("temperature").cast("double"),
        col("humidity").cast("double")
    )

    # Optional watermark for streaming-style consistency
    df = df.withWatermark("timestamp", "10 minutes")

    # Handle outliers
    df = handle_outliers(df)

    # Return as Pandas for downstream use in ML/Dashboard
    return df
