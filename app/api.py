# api.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml import PipelineModel
import uvicorn

# Initialize Spark session
spark = SparkSession.builder.appName("StrokeAPI").getOrCreate()

# Load the trained model
model = PipelineModel.load("models/stroke_gbt_model")

# Create FastAPI app
app = FastAPI(title="Stroke Risk Prediction API")

# Define request schema
class StrokeInput(BaseModel):
    gender: Literal["Male", "Female", "Other"]
    age: float
    hypertension: int
    heart_disease: int
    ever_married: Literal["Yes", "No"]
    work_type: str
    Residence_type: Literal["Urban", "Rural"]
    avg_glucose_level: float
    bmi: float
    smoking_status: str

@app.get("/")
def welcome():
    return {"message": "Welcome to the Stroke Prediction API"}

@app.post("/predict")
def predict_stroke(data: StrokeInput):
    # Convert incoming data to Spark DataFrame
    row = Row(**data.dict())
    df = spark.createDataFrame([row])

    # Default imputation if needed (use same defaults as training)
    df = df.fillna({
        "bmi": 28.9,
        "smoking_status": "Unknown",
        "gender": "Other",
        "ever_married": "No",
        "work_type": "Private",
        "Residence_type": "Urban"
    })

    # Predict using pipeline model
    result = model.transform(df)
    prediction = result.select("prediction", "probability").first()

    return {
        "prediction": int(prediction["prediction"]),
        "probability_no_stroke": round(prediction["probability"][0], 3),
        "probability_stroke": round(prediction["probability"][1], 3)
    }

# Run with: uvicorn app.api:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
