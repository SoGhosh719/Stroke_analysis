# train_model.py

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from utils.preprocessing import handle_missing_values, get_feature_pipeline
import os

# Initialize Spark session
spark = SparkSession.builder.appName("StrokeModelTraining").getOrCreate()

# Load data
df = spark.read.csv("data/healthcare-dataset-stroke-data.csv", header=True, inferSchema=True)

# Apply preprocessing
df = handle_missing_values(df)

# Get pipeline stages
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
numeric_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
stages = get_feature_pipeline(categorical_cols, numeric_cols)

# Add classifier
classifier = GBTClassifier(labelCol="label", featuresCol="scaled_features", maxIter=20)
stages.append(classifier)

# Build and train pipeline
pipeline = Pipeline(stages=stages)
model = pipeline.fit(df)

# Save model
os.makedirs("models", exist_ok=True)
model.write().overwrite().save("models/stroke_gbt_model")

print("✅ Model trained and saved to models/stroke_gbt_model")
