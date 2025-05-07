# evaluate_model.py

import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Fix Python path so we can import from utils/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import handle_missing_values

# Initialize Spark session
spark = SparkSession.builder.appName("StrokeModelEvaluation").getOrCreate()

# Load data
df = spark.read.csv("data/healthcare-dataset-stroke-data.csv", header=True, inferSchema=True)

# Apply same preprocessing as training
df = handle_missing_values(df)

# Load the trained model pipeline
model = PipelineModel.load("models/stroke_gbt_model")

# Make predictions
predictions = model.transform(df)

# Preview predictions
predictions.select("label", "prediction", "probability").show(5)

# Evaluation metrics
binary_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
multiclass_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

roc_auc = binary_eval.evaluate(predictions)
accuracy = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: "accuracy"})
f1_score = multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: "f1"})

# Print results
print(f"\n🔍 Evaluation Metrics")
print(f"🔹 ROC AUC     : {roc_auc:.4f}")
print(f"🔹 Accuracy    : {accuracy:.4f}")
print(f"🔹 F1 Score    : {f1_score:.4f}")
