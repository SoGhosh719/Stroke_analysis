
# evaluate_model.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("StrokeModelEvaluation").getOrCreate()

# Load data
df = spark.read.csv("data/healthcare-dataset-stroke-data.csv", header=True, inferSchema=True)

# Handle missing BMI
mean_bmi = df.select(mean("bmi")).first()[0]
df = df.withColumn("bmi", when(col("bmi").isNull(), mean_bmi).otherwise(col("bmi")))

# Fill nulls in categorical columns
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
for colname in categorical_cols:
    df = df.fillna({colname: "Unknown"})

# Load trained model
model = PipelineModel.load("models/stroke_gbt_model")

# Generate predictions
predictions = model.transform(df)

# Show predictions preview
predictions.select("label", "prediction", "probability").show(5, truncate=False)

# Evaluate with BinaryClassificationEvaluator (ROC AUC)
binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
roc_auc = binary_evaluator.evaluate(predictions)
print(f"🔹 ROC AUC: {roc_auc:.4f}")

# Evaluate with MulticlassClassificationEvaluator
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

accuracy = accuracy_evaluator.evaluate(predictions)
f1 = f1_evaluator.evaluate(predictions)

print(f"🔹 Accuracy: {accuracy:.4f}")
print(f"🔹 F1 Score: {f1:.4f}")

print("✅ Evaluation complete.")
