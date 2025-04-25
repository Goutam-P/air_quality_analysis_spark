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
├── tcp_log_file_streaming_server.py # Simulated TCP server sending log data
├── test_reading_client.py           # Testing client for TCP connection
├── locations_metadata.csv           # Optional metadata for location mapping
├── download_from_s3.py              # Script to fetch files from S3

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

This script pulls required input data such as logs or weather data from an S3 bucket to your local environment.

### Step 2: Run the Simulated TCP Server

```bash
python ingestion/tcp_log_file_streaming_server.py
```

This simulates a real-time stream of air quality sensor logs (PM2.5, etc.).

### Step 3: Ingest and Preprocess Streamed Data

```bash
spark-submit ingestion/ingestion_task1.py
```

This script reads streamed data, applies watermarking, parses datetime fields, drops irrelevant columns, and validates schema.

### Step 4: Merge and Sort Metrics

```bash
spark-submit ingestion/merge_and_sort.py
```

This script merges multiple datasets from different batch files into a single, unified DataFrame, combining all metrics (PM2.5, temperature, humidity) by timestamp and region to produce a consolidated dataset for downstream processing.


---

## 📊 Output

The output of this stage is a **cleaned and enriched DataFrame** written to:

- Console (for debugging), and/or
- Local Parquet/CSV directory (e.g., `/ingestion/data/pending/final_task1`)
