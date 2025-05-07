from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when, count
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Initialize Spark session
spark = SparkSession.builder.appName("StrokeEDA").getOrCreate()

# Load data
df = spark.read.csv("data/healthcare-dataset-stroke-data.csv", header=True, inferSchema=True)

# Basic schema and sample
df.printSchema()
df.show(5)

# Null check
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# Fill nulls in BMI with mean
mean_bmi = df.select(mean("bmi")).first()[0]
df = df.withColumn("bmi", when(col("bmi").isNull(), mean_bmi).otherwise(col("bmi")))
df = df.withColumn("bmi", df["bmi"].cast("double"))  # Ensure bmi is numeric

# Convert to pandas for visualizations
pdf = df.toPandas()
pdf["bmi"] = pd.to_numeric(pdf["bmi"], errors="coerce")
pdf = pdf.dropna(subset=["bmi", "age", "avg_glucose_level", "stroke"])  # Drop rows with non-numeric data

# Create output folder
os.makedirs("outputs", exist_ok=True)

# ----------------- BASIC PLOTS -----------------

# Plot: Stroke Class Distribution Pie Chart
plt.figure(figsize=(6, 6))
labels = ['No Stroke', 'Stroke']
explode = [0, 0.1]
colors = ['#66b3ff', '#ff9999']
pdf['stroke'].value_counts().plot.pie(labels=labels, autopct='%1.1f%%', startangle=90,
                                      explode=explode, colors=colors, shadow=True)
plt.title('Stroke Class Distribution')
plt.ylabel('')
plt.tight_layout()
plt.savefig("outputs/stroke_class_distribution.png")
plt.clf()

# Plot: Age Distribution by Stroke Status
plt.figure(figsize=(8, 5))
sns.histplot(data=pdf, x="age", hue="stroke", kde=True, palette="Set2", bins=30)
plt.title("Age Distribution by Stroke")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("outputs/age_distribution_by_stroke.png")
plt.clf()

# Plot: Average Glucose Level Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=pdf, x="avg_glucose_level", hue="stroke", kde=True, bins=30, palette="coolwarm")
plt.title("Glucose Level Distribution by Stroke")
plt.savefig("outputs/glucose_distribution_by_stroke.png")
plt.clf()

# Plot: BMI vs Age Scatter
plt.figure(figsize=(8, 5))
sns.scatterplot(data=pdf, x="age", y="bmi", hue="stroke", alpha=0.6)
plt.title("BMI vs Age by Stroke")
plt.savefig("outputs/bmi_vs_age.png")
plt.clf()

# Plot: Countplot for Categorical Variables
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
for colname in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=pdf, x=colname, hue="stroke", palette="pastel")
    plt.title(f"{colname} distribution by stroke")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"outputs/{colname}_distribution_by_stroke.png")
    plt.clf()

# ----------------- COMBINED SUBPLOTS -----------------

numeric_columns = ["age", "avg_glucose_level", "bmi"]
fig, axes = plt.subplots(len(numeric_columns), 3, figsize=(15, 12))
plt.subplots_adjust(hspace=0.4)

for i, colname in enumerate(numeric_columns):
    sns.kdeplot(data=pdf, x=colname, hue="stroke", fill=True, ax=axes[i, 0])
    axes[i, 0].set_title(f"KDE: {colname}")

    sns.boxplot(data=pdf, x="stroke", y=colname, ax=axes[i, 1])
    axes[i, 1].set_title(f"Boxplot: {colname}")

    sns.scatterplot(data=pdf, x=colname, y="stroke", alpha=0.5, ax=axes[i, 2])
    axes[i, 2].set_title(f"Scatter: {colname} vs Stroke")

plt.suptitle("Numeric Feature Analysis by Stroke", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig("outputs/numeric_feature_subplots.png")
plt.clf()

print("✅ EDA complete. All plots saved to 'outputs/' folder.")
