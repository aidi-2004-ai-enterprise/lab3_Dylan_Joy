# app/main.py

"""
main.py

FastAPI app for predicting penguin species from input features using a trained XGBoost model.

Features:
- Prediction endpoint using Pydantic input validation
- Consistent one-hot encoding with training
- Model and metadata loading at app startup
- Structured logging and error handling
- Loads model and metadata from Google Cloud Storage using service account
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
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from google.cloud import storage
import tempfile

# Load environment variables from .env file
load_dotenv()

# Read GCP config from environment variables
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
bucket_name = os.getenv("GCS_BUCKET_NAME")
model_blob_name = os.getenv("MODEL_BLOB_NAME")
features_blob_name = os.getenv("FEATURES_BLOB_NAME")
labels_blob_name = os.getenv("LABELS_BLOB_NAME")

print("Credentials Path:", credentials_path)
print("Bucket Name:", bucket_name)
print("Model Blob Name:", model_blob_name)
print("Features Blob Name:", features_blob_name)
print("Labels Blob Name:", labels_blob_name)

# Set Google credentials environment variable for GCP client libraries
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Enums and Pydantic model
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

# Globals to hold model and metadata
model: XGBClassifier = None
feature_cols: List[str] = []
label_classes: List[str] = []

def download_blob(bucket_name: str, blob_name: str, destination_file_name: str):
    """
    Downloads a blob from the GCP bucket to a local file.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    logging.info(f"Downloading blob {blob_name} from bucket {bucket_name} to {destination_file_name}")
    blob.download_to_filename(destination_file_name)
    logging.info(f"Blob {blob_name} downloaded successfully.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_cols, label_classes

    try:
        # Use a temporary directory to avoid file lock issues on Windows
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.json")
            features_path = os.path.join(tmpdir, "features.json")
            labels_path = os.path.join(tmpdir, "labels.json")

            # Download files from GCS bucket
            download_blob(bucket_name, model_blob_name, model_path)
            download_blob(bucket_name, features_blob_name, features_path)
            download_blob(bucket_name, labels_blob_name, labels_path)

            # Load the model
            model = XGBClassifier()
            model.load_model(model_path)
            logging.info(f"Model loaded from {model_path}")

            # Load feature columns
            with open(features_path, "r") as f:
                feature_cols = json.load(f)
            logging.info("Feature columns loaded.")

            # Load label classes
            with open(labels_path, "r") as f:
                label_classes = json.load(f)
            logging.info("Label classes loaded.")

            yield  # app runs here

    except Exception as e:
        logging.error(f"Failed to load model or metadata: {e}")
        raise

# Initialize FastAPI app with lifespan handler
app = FastAPI(lifespan=lifespan)

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

        logging.info(f"Prediction successful: {pred_label}")

        return {"predicted_species": pred_label}

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail="Prediction failed: " + str(e))
