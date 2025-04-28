# section3_sql_analysis.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, lead, when, row_number, udf
from pyspark.sql.window import Window
from pyspark.sql.types import StringType

# Step 1: Start Spark Session
spark = SparkSession.builder \
    .appName("Air Quality SQL Analysis - Section 3") \
    .getOrCreate()

# Step 2: Load feature-enhanced data
df = spark.read.option("header", "true").option("inferSchema", "true").csv("task2_feature_enhanced.csv")

# Step 3: Create Temp View
df.createOrReplaceTempView("air_quality")

# Step 4.1: Top Locations with Highest Avg PM2.5 (No Limit)
latest_date = df.agg({"date": "max"}).collect()[0][0]
print(f"Latest Date in Data: {latest_date}")

top_locations_query = f"""
WITH avg_pm25_by_location AS (
    SELECT location, ROUND(AVG(pm2_5),2) AS avg_pm25
    FROM air_quality
    WHERE date = '{latest_date}'
    GROUP BY location
)
SELECT location, avg_pm25
FROM avg_pm25_by_location
WHERE avg_pm25 = (SELECT MAX(avg_pm25) FROM avg_pm25_by_location)
"""

top_locations = spark.sql(top_locations_query)

# Save
top_locations.write.mode("overwrite").option("header", "true").csv("../outputs/section3/top_locations_pm25.csv")

# Step 4.2: Peak Pollution Times (No Limit)
peak_pollution_query = """
SELECT timestamp, location, pm2_5
FROM air_quality
WHERE pm2_5 IS NOT NULL
ORDER BY pm2_5 DESC
"""
peak_pollution = spark.sql(peak_pollution_query)

# Save
peak_pollution.write.mode("overwrite").option("header", "true").csv("../outputs/section3/peak_pollution_times.csv")

# Step 4.3: Trend Analysis Using Window Functions (ROW_NUMBER, LAG, LEAD)
window_spec = Window.partitionBy("location").orderBy("timestamp")

trend_df = df.withColumn("row_num", row_number().over(window_spec)) \
             .withColumn("prev_pm2_5", lag(col("pm2_5")).over(window_spec)) \
             .withColumn("next_pm2_5", lead(col("pm2_5")).over(window_spec)) \
             .withColumn("pm2_5_change_prev", col("pm2_5") - col("prev_pm2_5")) \
             .withColumn("pm2_5_change_next", col("next_pm2_5") - col("pm2_5")) \
             .withColumn("trend", when(col("pm2_5_change_next") > 0, "Increasing")
                                  .when(col("pm2_5_change_next") < 0, "Decreasing")
                                  .otherwise("Stable")) \
             .select("timestamp", "location", "pm2_5", "prev_pm2_5", "next_pm2_5", "pm2_5_change_prev", "pm2_5_change_next", "trend")

# Save
trend_df.write.mode("overwrite").option("header", "true").csv("../outputs/section3/trend_analysis_pm25.csv")

# Step 4.4: AQI Classification UDF (All Data)
def classify_aqi(pm2_5_value):
    if pm2_5_value is None:
        return "Unknown"
    elif pm2_5_value <= 12:
        return "Good"
    elif pm2_5_value <= 35.4:
        return "Moderate"
    else:
        return "Unhealthy"

aqi_udf = udf(classify_aqi, StringType())

aqi_classified_df = df.withColumn("AQI_Category", aqi_udf(col("pm2_5")))

# Save
aqi_classified_df.select("timestamp", "location", "pm2_5", "AQI_Category") \
    .write.mode("overwrite").option("header", "true").csv("../outputs/section3/aqi_classification.csv")

print("✅ Section 3 completed successfully! All outputs saved to /outputs/section3/")
