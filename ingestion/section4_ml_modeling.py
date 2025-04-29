# section4_ml_modeling.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def start_spark_session():
    return SparkSession.builder \
        .appName("Air Quality ML Modeling - Section 4") \
        .getOrCreate()


def load_data(input_path):
    spark = start_spark_session()
    df = spark.read.option("header", "true").option(
        "inferSchema", "true").csv(input_path)
    return df, spark


def preprocess_data(df):
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
    df = df.withColumn("AQI_Category", aqi_udf(col("pm2_5")))

    selected_features = ["temperature", "humidity", "pm2_5"]
    df = df.dropna(subset=selected_features)

    indexer = StringIndexer(inputCol="AQI_Category", outputCol="label")
    df_indexed = indexer.fit(df).transform(df)

    assembler = VectorAssembler(
        inputCols=selected_features, outputCol="features")
    df_final = assembler.transform(df_indexed)

    return df_final


def load_trained_model(train_data):
    rf = RandomForestClassifier(
        featuresCol="features", labelCol="label", numTrees=50, maxDepth=5)
    model = rf.fit(train_data)
    return model


def predict_aqi(model, test_data):
    return model.transform(test_data)


def save_predictions(predictions):
    predictions.select("timestamp", "location", "label", "prediction") \
        .write.mode("overwrite").option("header", "true") \
        .csv("../outputs/section4/final_predictions")


def main():
    input_path = "task2_feature_enhanced.csv"
    df, spark = load_data(input_path)
    df_final = preprocess_data(df)

    train_data, test_data = df_final.randomSplit([0.7, 0.3], seed=42)
    model = load_trained_model(train_data)

    predictions = predict_aqi(model, test_data)

    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1")

    accuracy = evaluator_accuracy.evaluate(predictions)
    f1_score = evaluator_f1.evaluate(predictions)

    print(f" Model Evaluation Results:")
    print(f" - Accuracy: {accuracy:.4f}")
    print(f" - F1 Score: {f1_score:.4f}")

    save_predictions(predictions)
    print("Predictions saved successfully at /outputs/section4/final_predictions/")

    spark.stop()


if __name__ == "__main__":
    main()
