import streamlit as st
from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel

# --- Cached Spark + Model Load ---
@st.cache_resource
def load_spark():
    return SparkSession.builder.appName("StrokeDashboard").getOrCreate()

@st.cache_resource
def load_model():
    return PipelineModel.load("models/stroke_gbt_model")

spark = load_spark()
model = load_model()

# --- Streamlit UI ---
st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")
st.title("🧠 Stroke Risk Predictor")
st.caption("Use the sidebar to enter patient details and predict stroke risk.")

# --- Sidebar Inputs ---
st.sidebar.header("Patient Info")
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
age = st.sidebar.slider("Age", 0, 100, 50)
hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 28.0)
smoking_status = st.sidebar.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# --- Prediction ---
if st.sidebar.button("Predict Stroke Risk"):
    input_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }

    row = Row(**input_data)
    df = spark.createDataFrame([row])

    # Handle missing fields just in case
    df = df.fillna({
        "bmi": 28.9,
        "smoking_status": "Unknown",
        "gender": "Other",
        "ever_married": "No",
        "work_type": "Private",
        "Residence_type": "Urban"
    })

    # Predict
    result = model.transform(df)
    pred = result.select("prediction", "probability").first()

    stroke_prob = round(pred["probability"][1], 2)
    stroke_prediction = int(pred["prediction"])

    # --- Display Results ---
    st.markdown("---")
    st.subheader("🩺 Prediction Results")
    if stroke_prediction == 1:
        st.error(f"⚠️ High risk of stroke detected!\n\n**Probability: {stroke_prob:.2f}**")
    else:
        st.success(f"✅ Low risk of stroke.\n\n**Probability: {stroke_prob:.2f}**")

    st.progress(stroke_prob)

# --- Footer ---
st.markdown("---")
st.caption("🔧 Developed using PySpark + Streamlit")
