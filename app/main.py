# app/main.py

"""
main.py

FastAPI app for predicting penguin species from input features using a trained XGBoost model.

Features:
- Prediction endpoint using Pydantic input validation
- Consistent one-hot encoding with training
- Model and metadata loading at app startup
- Structured logging and error handling
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import logging
import json
import os
import pandas as pd
from xgboost import XGBClassifier
from typing import List

# Enums and Pydantic model as per assignment
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    Male = "male"
    Female = "female"

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI()

# Globals to hold model and metadata
model: XGBClassifier = None
feature_cols: List[str] = []
label_classes: List[str] = []

# Load model and metadata at startup
@app.on_event("startup")
def load_model_and_metadata() -> None:
    """
    Loads the trained XGBoost model and metadata from the app/data directory.
    """
    global model, feature_cols, label_classes

    model_path = "app/data/model.json"
    features_path = "app/data/features.json"
    labels_path = "app/data/labels.json"

    try:
        if not (os.path.exists(model_path) and os.path.exists(features_path) and os.path.exists(labels_path)):
            logging.error("Model or metadata files missing in app/data.")
            raise RuntimeError("Model or metadata files missing.")

        model = XGBClassifier()
        model.load_model(model_path)
        logging.info("Model loaded from {}".format(model_path))

        with open(features_path, "r") as f:
            feature_cols = json.load(f)
        logging.info("Feature columns loaded.")

        with open(labels_path, "r") as f:
            label_classes = json.load(f)
        logging.info("Label classes loaded.")

    except Exception as e:
        logging.error("Failed to load model or metadata: {}".format(e))
        raise

# Root endpoint for health check
@app.get("/")
def root() -> dict:
    """
    Health check endpoint to verify the API is running.
    """
    return {"message": "Penguin species prediction API is running."}

# Prediction endpoint
@app.post("/predict")
def predict(features: PenguinFeatures) -> dict:
    """
    Accepts input features and returns the predicted penguin species.
    """
    try:
        # Convert input to dict and then to DataFrame
        input_dict = features.dict()
        df_input = pd.DataFrame([input_dict])

        # Prepare one-hot encoded columns for sex and island exactly as training
        one_hot_cols = [col for col in feature_cols if col.startswith("sex_") or col.startswith("island_")]

        # Initialize one-hot columns to 0
        for col in one_hot_cols:
            df_input[col] = 0

        # Set the appropriate one-hot columns to 1
        sex_col = "sex_" + input_dict["sex"]
        island_col = "island_" + input_dict["island"]

        if sex_col in df_input.columns:
            df_input[sex_col] = 1
        if island_col in df_input.columns:
            df_input[island_col] = 1

        # Drop the original sex and island columns
        df_input = df_input.drop(columns=["sex", "island"])

        # Reorder columns to match training feature order exactly
        df_input = df_input.reindex(columns=feature_cols, fill_value=0)

        # Predict using the model
        pred_int = model.predict(df_input)[0]
        pred_label = label_classes[pred_int]

        logging.info("Prediction successful: {}".format(pred_label))

        return {"predicted_species": pred_label}

    except Exception as e:
        logging.error("Prediction failed: {}".format(e))
        raise HTTPException(status_code=400, detail="Prediction failed: " + str(e))