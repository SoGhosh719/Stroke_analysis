# train_model.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, when, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
import os

# Initialize Spark session
spark = SparkSession.builder.appName("StrokeModelTraining").getOrCreate()

# Load data
df = spark.read.csv("data/healthcare-dataset-stroke-data.csv", header=True, inferSchema=True)

# Handle missing values
mean_bmi = df.select(mean("bmi")).first()[0]
df = df.withColumn("bmi", when(col("bmi").isNull(), mean_bmi).otherwise(col("bmi")))

# Fill nulls in categorical columns
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
for colname in categorical_cols:
    df = df.fillna({colname: "Unknown"})

# Index label
label_indexer = StringIndexer(inputCol="stroke", outputCol="label")

# Index categorical features
indexers = [StringIndexer(inputCol=colname, outputCol=colname + "_indexed") for colname in categorical_cols]

# Final feature columns
feature_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"] + [c + "_indexed" for c in categorical_cols]

# Vectorize features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Classifier: Gradient-Boosted Trees
gbt = GBTClassifier(labelCol="label", featuresCol="scaled_features", maxIter=20)

# Build pipeline
pipeline = Pipeline(stages=indexers + [label_indexer, assembler, scaler, gbt])

# Train model
model = pipeline.fit(df)

# Save model
os.makedirs("models", exist_ok=True)
model.write().overwrite().save("models/stroke_gbt_model")

print("✅ Model trained and saved to models/stroke_gbt_model")

