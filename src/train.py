# src/train.py

import pandas as pd
import joblib

from data_processing import feature_pipeline

df = pd.read_csv("../data/data.csv")

X = df.drop("FraudResult", axis=1)
y = df["FraudResult"]

X_processed = feature_pipeline.fit_transform(X, y)

joblib.dump(
    feature_pipeline,
    "../models/feature_pipeline.pkl"
)

print("Pipeline saved successfully")