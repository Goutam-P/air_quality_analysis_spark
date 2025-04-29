from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType


def classify_aqi(pm2_5_value):
    if pm2_5_value is None:
        return "Unknown"
    elif pm2_5_value <= 12:
        return "Good"
    elif pm2_5_value <= 35.4:
        return "Moderate"
    else:
        return "Unhealthy"


def run_sql_queries(df):
    # Register as a temporary SQL view
    df.createOrReplaceTempView("air_quality")

    # Optional: Run a SQL example (e.g., average PM2.5 per location)
    top_locations = df.sql_ctx.sql("""
        SELECT location, ROUND(AVG(pm2_5), 2) AS avg_pm2_5
        FROM air_quality
        GROUP BY location
        ORDER BY avg_pm2_5 DESC
        LIMIT 5
    """)
    top_locations.show()

    # Add AQI category using UDF
    aqi_udf = udf(classify_aqi, StringType())
    aqi_df = df.withColumn("AQI_Category", aqi_udf(col("pm2_5")))

    # Save classified AQI results
    aqi_df.select("timestamp", "location", "pm2_5", "AQI_Category") \
        .write.mode("overwrite").option("header", "true") \
        .csv("outputs/section3/aqi_classification.csv")
