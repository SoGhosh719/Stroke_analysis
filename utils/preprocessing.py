
# preprocessing.py

from pyspark.sql import DataFrame
from pyspark.sql.functions import when, mean, col
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

def handle_missing_values(df: DataFrame) -> DataFrame:
    """Fill missing BMI with mean and categorical nulls with 'Unknown'."""
    # Impute BMI
    mean_bmi = df.select(mean("bmi")).first()[0]
    df = df.withColumn("bmi", when(col("bmi").isNull(), mean_bmi).otherwise(col("bmi")))

    # Fill nulls in categorical columns
    categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    for colname in categorical_cols:
        df = df.fillna({colname: "Unknown"})

    return df

def get_feature_pipeline(categorical_cols, numeric_cols, label_col="stroke"):
    """Return list of preprocessing stages for a pipeline."""
    stages = []

    # Label indexing
    label_indexer = StringIndexer(inputCol=label_col, outputCol="label")
    stages.append(label_indexer)

    # Categorical column indexers
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_indexed") for c in categorical_cols]
    stages.extend(indexers)

    # Assemble features
    indexed_cat = [f"{c}_indexed" for c in categorical_cols]
    feature_cols = numeric_cols + indexed_cat
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

    stages.extend([assembler, scaler])
    return stages
