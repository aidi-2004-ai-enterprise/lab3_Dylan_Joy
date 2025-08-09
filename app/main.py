# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from typing import List, Any
from contextlib import asynccontextmanager
from google.cloud import storage
import tempfile
import os
import json
import pandas as pd
from xgboost import XGBClassifier
import logging
from dotenv import load_dotenv
import uvicorn

# Load .env locally
load_dotenv()

# Config from env vars
bucket_name = os.getenv("GCS_BUCKET_NAME")
model_blob_name = os.getenv("MODEL_BLOB_NAME")
features_blob_name = os.getenv("FEATURES_BLOB_NAME")
labels_blob_name = os.getenv("LABELS_BLOB_NAME")
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if credentials_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Enums & Pydantic input model
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

# Globals for model and metadata
model: XGBClassifier | None = None
feature_cols: List[str] = []
label_classes: List[str] = []

def download_blob_from_gcs(bucket_name: str, blob_name: str, destination_file_name: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    logger.info(f"Downloading gs://{bucket_name}/{blob_name} to {destination_file_name}")
    blob.download_to_filename(destination_file_name)
    logger.info("Download complete.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_cols, label_classes
    try:
        if bucket_name and model_blob_name and features_blob_name and labels_blob_name:
            # Download model & metadata from GCS
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.json")
                features_path = os.path.join(tmpdir, "features.json")
                labels_path = os.path.join(tmpdir, "labels.json")

                download_blob_from_gcs(bucket_name, model_blob_name, model_path)
                download_blob_from_gcs(bucket_name, features_blob_name, features_path)
                download_blob_from_gcs(bucket_name, labels_blob_name, labels_path)

                model = XGBClassifier()
                model.load_model(model_path)
                logger.info("Model loaded from GCS.")

                with open(features_path) as f:
                    feature_cols = json.load(f)
                with open(labels_path) as f:
                    label_classes = json.load(f)

                logger.info(f"Loaded {len(feature_cols)} features and {len(label_classes)} labels from GCS.")
                yield

        else:
            # Local fallback (for dev)
            logger.info("Env vars missing, loading model from local app/data/")
            model_path = os.path.join("app", "data", "model.json")
            features_path = os.path.join("app", "data", "features.json")
            labels_path = os.path.join("app", "data", "labels.json")

            if not (os.path.exists(model_path) and os.path.exists(features_path) and os.path.exists(labels_path)):
                raise RuntimeError("Local model/metadata files missing in app/data")

            model = XGBClassifier()
            model.load_model(model_path)

            with open(features_path) as f:
                feature_cols = json.load(f)
            with open(labels_path) as f:
                label_classes = json.load(f)

            logger.info(f"Loaded local model, {len(feature_cols)} features, {len(label_classes)} labels.")
            yield
    except Exception as e:
        logger.exception("Failed to load model or metadata: %s", e)
        raise

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Penguin species prediction API is running."}

@app.post("/predict")
def predict(features: PenguinFeatures):
    try:
        if model is None or not feature_cols or not label_classes:
            raise HTTPException(status_code=500, detail="Model or metadata not loaded.")

        input_dict = features.dict()
        # Ensure enum values are strings
        input_dict["sex"] = input_dict["sex"].value if hasattr(input_dict["sex"], "value") else input_dict["sex"]
        input_dict["island"] = input_dict["island"].value if hasattr(input_dict["island"], "value") else input_dict["island"]

        df = pd.DataFrame([input_dict])

        # One-hot encode sex and island columns (as per training)
        one_hot_cols = [c for c in feature_cols if c.startswith("sex_") or c.startswith("island_")]
        for col in one_hot_cols:
            df[col] = 0

        sex_col = "sex_" + input_dict["sex"]
        island_col = "island_" + input_dict["island"]

        if sex_col in df.columns:
            df[sex_col] = 1
        if island_col in df.columns:
            df[island_col] = 1

        df = df.drop(columns=["sex", "island"])
        df = df.reindex(columns=feature_cols, fill_value=0)

        pred_int = int(model.predict(df)[0])
        pred_label = label_classes[pred_int]

        logger.info(f"Predicted species: {pred_label}")
        return {"predicted_species": pred_label}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
