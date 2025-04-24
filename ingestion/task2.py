import pandas as pd
import numpy as np
import glob
import os

# STEP 1: Load all CSV files from the folder
file_paths = glob.glob("/workspaces/air_quality_analysis_spark/ingestion/data/pending/final_task1/part-00000-3b283761-98e2-4230-ba48-3c6205ead6a6-c000.csv")

if not file_paths:
    raise FileNotFoundError("❌ No CSV files found in the specified path.")

# Read and concatenate all CSVs
df = pd.concat([pd.read_csv(f, parse_dates=["timestamp"]) for f in file_paths], ignore_index=True)

# STEP 2: Handle Outliers
df = df[df["pm2_5"] < 1000]
df["temperature"] = np.where(df["temperature"] > 60, np.nan, df["temperature"])
df["humidity"] = np.where((df["humidity"] > 100) | (df["humidity"] < 0), np.nan, df["humidity"])

# STEP 3: Impute Missing Values (Median)
df["pm2_5"].fillna(df["pm2_5"].median(), inplace=True)
df["temperature"].fillna(df["temperature"].median(), inplace=True)
df["humidity"].fillna(df["humidity"].median(), inplace=True)

# STEP 4: Normalize Features (Z-score)
for col in ["pm2_5", "temperature", "humidity"]:
    mean = df[col].mean()
    std = df[col].std()
    df[f"{col}_zscore"] = (df[col] - mean) / std

# STEP 5: Aggregations
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour

daily_avg = df.groupby(["date", "location"]).agg({
    "pm2_5": "mean",
    "temperature": "mean",
    "humidity": "mean"
}).reset_index()
daily_avg.to_csv("daily_aggregates.csv", index=False)

hourly_avg = df.groupby(["date", "hour", "location"]).agg({
    "pm2_5": "mean",
    "temperature": "mean",
    "humidity": "mean"
}).reset_index()
hourly_avg.to_csv("hourly_aggregates.csv", index=False)

# STEP 6: Rolling Average, Lag & Rate-of-Change
df.sort_values(by=["location", "timestamp"], inplace=True)
df["pm2_5_rolling_avg_3"] = df.groupby("location")["pm2_5"].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["pm2_5_lag_1"] = df.groupby("location")["pm2_5"].shift(1)
df["pm2_5_rate_of_change"] = df["pm2_5"] - df["pm2_5_lag_1"]

# Save final output
df.to_csv("task2_feature_enhanced.csv", index=False)
print("✅ Task 2 completed. Outputs:")
print(" - task2_feature_enhanced.csv")
print(" - daily_aggregates.csv")
print(" - hourly_aggregates.csv")