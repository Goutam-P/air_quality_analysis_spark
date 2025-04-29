from ingestion.ingestion_task1 import ingest_data
from ingestion.section3_sql_analysis import run_sql_queries
from ingestion.section4_ml_modeling import preprocess_data, load_trained_model, predict_aqi
import os


def main():
    print("[1] Ingesting data...")
    df_spark = ingest_data()  # ✅ This is a Spark DataFrame

    print("[2] Skipping merge step — using preprocessed data directly")
    df_clean = df_spark  # already merged + cleaned

    print("[3] Running SQL analysis...")
    run_sql_queries(df_clean)

    print("[4] Loading ML model & predicting AQI...")
    df_preprocessed = preprocess_data(df_clean)

    model = load_trained_model(df_preprocessed)
    df_predictions = predict_aqi(model, df_preprocessed)

    print("[5] Saving final predictions to outputs/final_output.csv...")
    os.makedirs("outputs", exist_ok=True)
    df_predictions.select("timestamp", "location", "pm2_5", "AQI_Category", "prediction") \
        .toPandas().to_csv("outputs/final_output.csv", index=False)

    print("✅ Pipeline completed successfully. Output saved at: outputs/final_output.csv")


if __name__ == "__main__":
    main()
