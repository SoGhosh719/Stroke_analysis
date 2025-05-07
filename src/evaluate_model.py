# evaluate_model.py

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from utils.preprocessing import handle_missing_values

# Initialize Spark session
spark = SparkSession.builder.appName("StrokeModelEvaluation").getOrCreate()

# Load data
df = spark.read.csv("data/healthcare-dataset-stroke-data.csv", header=True, inferSchema=True)

# Apply same preprocessing
df = handle_missing_values(df)

# Load model
model = PipelineModel.load("models/stroke_gbt_model")
predictions = model.transform(df)

# Show prediction preview
predictions.select("label", "prediction", "probability").show(5)

# Evaluate
binary_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
multiclass_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

print(f"🔹 ROC AUC: {binary_eval.evaluate(predictions):.4f}")
print(f"🔹 Accuracy: {multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: 'accuracy'}):.4f}")
print(f"🔹 F1 Score: {multiclass_eval.evaluate(predictions, {multiclass_eval.metricName: 'f1'}):.4f}")
