# section4_ml_modeling.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import StringType

# Step 1: Start Spark Session
spark = SparkSession.builder \
    .appName("Air Quality ML Modeling - Section 4") \
    .getOrCreate()

# Step 2: Load the feature-enhanced dataset
input_path = "task2_feature_enhanced.csv"  # Adjust path if needed
df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)

# Step 3: Create AQI_Category Column (if not present)
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

# Step 4: Show Sample Data
df.select("timestamp", "location", "pm2_5", "AQI_Category", "temperature", "humidity").show(5)

print("Dataset loaded and AQI_Category prepared!")

from pyspark.ml.feature import StringIndexer, VectorAssembler

# Step 5: Feature Selection
selected_features = [
    "temperature",
    "humidity",
    "pm2_5_lag_1",
    "pm2_5_rate_of_change"
]

# Step 6: Drop rows with NULLs in selected features
df = df.dropna(subset=selected_features)

# Step 7: Handle Label - Convert AQI_Category into Numeric
indexer = StringIndexer(inputCol="AQI_Category", outputCol="label")
df_indexed = indexer.fit(df).transform(df)

# Step 8: Assemble Features into a Single Vector
assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
df_final = assembler.transform(df_indexed)

# Step 9: Show Final Prepared Data
df_final.select("features", "label").show(5, truncate=False)

print(" Features assembled and labels indexed! Ready for model training.")

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Step 10: Split data into train and test sets
train_data, test_data = df_final.randomSplit([0.7, 0.3], seed=42)

# Step 11: Train Random Forest Classifier
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50, maxDepth=5)
model = rf.fit(train_data)

# Step 12: Make predictions
predictions = model.transform(test_data)

# Step 13: Evaluate model
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

accuracy = evaluator_accuracy.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

print(f" Model Evaluation Results:")
print(f" - Accuracy: {accuracy:.4f}")
print(f" - F1 Score: {f1_score:.4f}")

# Step 14: Save predictions for Section 5 usage (without features column)
predictions.select("timestamp", "location", "label", "prediction") \
    .write.mode("overwrite").option("header", "true") \
    .csv("../outputs/section4/final_predictions")

print("Predictions saved successfully at /outputs/section4/final_predictions/")


