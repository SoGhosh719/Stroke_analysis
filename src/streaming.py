# streaming.py

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel

# Add project root to path for utils import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import handle_missing_values

# Initialize Spark session
spark = SparkSession.builder.appName("StrokeStreaming").getOrCreate()

# Define schema
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("age", DoubleType(), True),
    StructField("hypertension", IntegerType(), True),
    StructField("heart_disease", IntegerType(), True),
    StructField("ever_married", StringType(), True),
    StructField("work_type", StringType(), True),
    StructField("Residence_type", StringType(), True),
    StructField("avg_glucose_level", DoubleType(), True),
    StructField("bmi", DoubleType(), True),
    StructField("smoking_status", StringType(), True),
    StructField("stroke", IntegerType(), True)
])

# Read streaming data
df_stream = spark.readStream \
    .schema(schema) \
    .option("maxFilesPerTrigger", 1) \
    .csv("stream_input/")

# Preprocess
df_stream = handle_missing_values(df_stream)

# Load trained model
model = PipelineModel.load("models/stroke_gbt_model")

# Apply model
predictions = model.transform(df_stream)

# Select output columns
output = predictions.select("id", "age", "avg_glucose_level", "bmi", "prediction", "probability")

# Write stream to console
query = output.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()
