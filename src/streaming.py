# streaming.py

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from utils.preprocessing import handle_missing_values

spark = SparkSession.builder.appName("StrokeStreaming").getOrCreate()

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

df_stream = spark.readStream.schema(schema).option("maxFilesPerTrigger", 1).csv("stream_input/")
df_stream = handle_missing_values(df_stream)

model = PipelineModel.load("models/stroke_gbt_model")
predictions = model.transform(df_stream)

output = predictions.select("id", "age", "avg_glucose_level", "bmi", "prediction", "probability")

query = output.writeStream.outputMode("append").format("console").option("truncate", False).start()
query.awaitTermination()
