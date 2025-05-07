# train_model.py

import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier

# Add root path for utils import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import handle_missing_values, get_feature_pipeline

# Initialize Spark session
spark = SparkSession.builder.appName("StrokeModelTraining").getOrCreate()

# Load data
df = spark.read.csv("data/healthcare-dataset-stroke-data.csv", header=True, inferSchema=True)

# Apply preprocessing
df = handle_missing_values(df)

# Define columns
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
numeric_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]

# Get preprocessing pipeline stages
stages = get_feature_pipeline(categorical_cols, numeric_cols)

# Add classifier
classifier = GBTClassifier(labelCol="label", featuresCol="scaled_features", maxIter=20)
stages.append(classifier)

# Build full pipeline
pipeline = Pipeline(stages=stages)

# Train model
model = pipeline.fit(df)

# Save model
os.makedirs("models", exist_ok=True)
model.write().overwrite().save("models/stroke_gbt_model")

print("✅ Model trained and saved to models/stroke_gbt_model")
