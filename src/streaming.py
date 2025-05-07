# streaming.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when
from pyspark.ml import PipelineModel
import time

# Initialize Spark session with streaming support
spark = SparkSession.builder \
    .appName("StrokeStreamingInference") \
    .getOrCreate()

# Define schema for input data (matches original dataset)
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType

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
    StructField("stroke", IntegerType(), True)  # Optional for inference
])

# Load the trained pipeline model
model = PipelineModel.load("models/stroke_gbt_model")

# Monitor a directory for new data (simulate streaming)
input_path = "stream_input/"

# Read the streaming data
streaming_df = spark.readStream \
    .schema(schema) \
    .option("maxFilesPerTrigger", 1) \
    .csv(input_path)

# Fill missing bmi values and categories
mean_bmi = 28.893236911794673  # use same as training
streaming_df = streaming_df.withColumn("bmi", when(col("bmi").isNull(), mean_bmi).otherwise(col("bmi")))

categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
for colname in categorical_cols:
    streaming_df = streaming_df.fillna({colname: "Unknown"})

# Apply trained pipeline to streaming data
predictions = model.transform(streaming_df)

# Select and display relevant outputs
results = predictions.select("id", "age", "avg_glucose_level", "bmi", "prediction", "probability")

query = results.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()

