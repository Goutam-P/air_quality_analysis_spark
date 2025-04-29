
# Air Quality Analysis Using Spark

## 📌 Project Overview

This project builds a modular, near-real-time air quality analysis pipeline using PySpark. It ingests sensor data via a TCP server, merges pollution and weather metrics, applies data cleaning and feature engineering, performs SQL-based trend analysis, trains predictive models with Spark MLlib, and visualizes results on an interactive dashboard. Outputs from each stage are stored independently (CSV/Parquet/PostgreSQL) to support parallel development and reproducibility.

---

## 🧩 Section 1: Data Ingestion and Initial Pre-Processing

### ✅ Objectives

- Simulate live data streaming from a TCP server.
- Parse datetime and detect schema correctness.
- Merge PM2.5, temperature, and humidity data by timestamp and region.
- Enrich with external weather data (temperature and humidity).
- Validate the final dataset quality.

---

## 🛠️ Project Structure

```
ingestion/
│
├── ingestion_task1.py                # Spark job to stream and clean data
├── merge_and_sort.py                 # Spark job to merge sensor metrics into unified records
├── tcp_log_file_streaming_server.py  # Simulated TCP server sending log data
├── test_reading_client.py            # Testing client for TCP connection
├── locations_metadata.csv            # Optional metadata for location mapping
├── download_from_s3.py               # Script to fetch files from S3
```

---

## 🚀 Getting Started

### 1. Requirements

- Python 3.8+
- Apache Spark 3.x (Structured Streaming)
- PySpark
- Docker (optional, for TCP server testing)
- Git

### 2. Installation

```bash
git clone https://github.com/Goutam-P/air_quality_analysis_spark.git
cd air_quality_analysis_spark
pip install -r requirements.txt
```

---

## ⚙️ Execution Steps

### Step 1: Download Files from S3

```bash
python ingestion/download_from_s3.py
```

### Step 2: Run the Simulated TCP Server

```bash
python ingestion/tcp_log_file_streaming_server.py
```

### Step 3: Ingest and Preprocess Streamed Data

```bash
spark-submit ingestion/ingestion_task1.py
```

### Step 4: Merge and Sort Metrics

```bash
spark-submit ingestion/merge_and_sort.py
```

---

## 📊 Output

The output of this stage is a **cleaned and enriched DataFrame** written to:
- Console (for debugging), and/or
- Local Parquet/CSV directory (e.g., `/ingestion/data/pending/final_task1`)

---

## 📊 Section 3: Spark SQL Exploration & Correlation Analysis

### ✅ Objectives

- Register cleaned feature-enhanced air quality data as a **temporary SQL view**.
- Develop **complex analytical queries** to identify regions with the highest PM2.5 levels.
- Perform **trend analysis** using **SQL window functions** (`ROW_NUMBER()`, `LAG()`, `LEAD()`).
- Implement a **UDF-based Air Quality Index (AQI) classification** to assess pollution risk levels.
- Save all outputs into organized CSV files.

---

## 🧩 Queries and Operations

### 1. Top Locations by Highest Average PM2.5

Using a CTE and MAX aggregation to find regions with the highest average:

```python
WITH avg_pm25_by_location AS (
    SELECT location, ROUND(AVG(pm2_5),2) AS avg_pm25
    FROM air_quality
    WHERE date = '{latest_date}'
    GROUP BY location
)
SELECT location, avg_pm25
FROM avg_pm25_by_location
WHERE avg_pm25 = (SELECT MAX(avg_pm25) FROM avg_pm25_by_location)
```

Saved Output: `/outputs/section3/top_locations_pm25.csv`

---

### 2. Peak Pollution Time Intervals

Ordering PM2.5 readings in descending order:

```python
SELECT timestamp, location, pm2_5
FROM air_quality
WHERE pm2_5 IS NOT NULL
ORDER BY pm2_5 DESC
```

Saved Output: `/outputs/section3/peak_pollution_times.csv`

---

### 3. Trend Analysis Using Window Functions

Calculating trends using LAG, LEAD, and ROW_NUMBER:

```python
window_spec = Window.partitionBy("location").orderBy("timestamp")

trend_df = df.withColumn("row_num", row_number().over(window_spec))              .withColumn("prev_pm2_5", lag(col("pm2_5")).over(window_spec))              .withColumn("next_pm2_5", lead(col("pm2_5")).over(window_spec))              .withColumn("pm2_5_change_prev", col("pm2_5") - col("prev_pm2_5"))              .withColumn("pm2_5_change_next", col("next_pm2_5") - col("pm2_5"))              .withColumn("trend", when(col("pm2_5_change_next") > 0, "Increasing")
                                  .when(col("pm2_5_change_next") < 0, "Decreasing")
                                  .otherwise("Stable"))
```

Saved Output: `/outputs/section3/trend_analysis_pm25.csv`

---

### 4. Air Quality Index (AQI) Classification

Custom UDF for classifying air quality:

```python
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
```

Saved Output: `/outputs/section3/aqi_classification.csv`

---

## Section 4:

Section 4 focuses on building, training, and evaluating a predictive model using Spark MLlib to forecast Air Quality Index (AQI) categories based on sensor readings (temperature, humidity, and PM2.5 trends).

## Steps Performed;

1. Load Feature-Enhanced Dataset:
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Air Quality ML Modeling").getOrCreate()

# Load the dataset generated in Task 2
df = spark.read.option("header", "true").option("inferSchema", "true").csv("task2_feature_enhanced.csv")
```

2. Create AQI Category Label

```python
from pyspark.sql.functions import when

# Define AQI categories based on PM2.5 values
df = df.withColumn("AQI_Category",
    when(df.pm2_5 <= 12, "Good")
    .when(df.pm2_5 <= 35.4, "Moderate")
    .otherwise("Unhealthy")
)
```

3. Feature Selection and Label Preparation
```python
from pyspark.ml.feature import StringIndexer, VectorAssembler

# Index AQI categories into numeric labels
indexer = StringIndexer(inputCol="AQI_Category", outputCol="label")
df = indexer.fit(df).transform(df)

# Assemble features
assembler = VectorAssembler(
    inputCols=["temperature", "humidity", "pm2_5_lag_1", "pm2_5_rate_of_change"],
    outputCol="features",
    handleInvalid="skip"
)

final_df = assembler.transform(df)
```

4. Train-Test Split:
# Split data
```python
train_data, test_data = final_df.randomSplit([0.7, 0.3], seed=42)
```
5. Train Random Forest Classifier
```python
from pyspark.ml.classification import RandomForestClassifier
```

# Initialize and train the model
```python
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50, maxDepth=5)
model = rf.fit(train_data)
```

6. Evaluate Model Performance
```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Predictions
predictions = model.transform(test_data)

# Evaluators
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

# Results
accuracy = evaluator_acc.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

print(f"\u2705 Model Evaluation Results:")
print(f" - Accuracy: {accuracy:.4f}")
print(f" - F1 Score: {f1_score:.4f}")
```

Result Achieved:

Accuracy: 96.26%

F1 Score: 96.12%


## Final Output Saved
# Save important fields (timestamp, location, true label, predicted label)
```python
predictions.select("timestamp", "location", "label", "prediction") \
    .write.mode("overwrite").option("header", "true") \
    .csv("../outputs/section4/final_predictions")
```


# Section 5:
## Objectives
Integrate the full pipeline from raw ingestion to model predictions.

Generate interactive charts that communicate trends, spikes, and air quality levels effectively.

Store final outputs and reports in CSV format for future analysis or external use.

## Pipeline Integration Summary
The complete end-to-end workflow (section5_pipeline.py) combines all modular components:

Ingestion: Reads, cleans, and merges sensor data.

Transformation: Handles outliers and performs feature enrichment.

SQL Analysis: Identifies trends, hotspots, and classifications using Spark SQL.

ML Modeling: Predicts AQI categories using Random Forest classifier.

Output: Stores results in outputs/final_output.csv.

```
python section5_pipeline.py

```

The final predictions stored in outputs/final_output.csv are used to visualize air quality trends using Plotly in Google Colab.

 # 🧩 Visualizations

 ## 1. Time-Series Line Chart – PM2.5 Concentration by Location
 ```
 import plotly.express as px

fig = px.line(df, x='timestamp', y='pm2_5', color='location',
              title='Time-Series of PM2.5 Concentration by Location')
fig.show()

```
 ## 2. AQI Category Pie Chart – Risk Distribution

 ```
fig = px.pie(df, names='AQI_Category', title='AQI Category Distribution')
fig.show()


```
 ## 3. PM2.5 Spike Events – Above Safe Threshold (150 µg/m³)
 ```
 spikes = df[df['pm2_5'] > 150]

fig = px.scatter(spikes, x='timestamp', y='pm2_5', color='location',
                 title='PM2.5 Spike Events (Above 150 µg/m³)')
fig.show()


```

 ## 4. Correlation Heatmap – PM2.5, Temperature, Humidity
 ```
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df[['pm2_5', 'temperature', 'humidity']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap: PM2.5, Temp, Humidity')
plt.show()

```

# 🎯 Outcome of Section 5

A complete, reproducible pipeline with:

Ingested and enriched data

Analytical and ML-driven insights

Visualizations for stakeholder reporting

All outputs saved for monitoring and future processing