import os
import pickle
import logging
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

import csv
from pathlib import Path

MONITORING_PATH = "data/monitoring/predictions.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature', 'duration_ms', 'explicit', 'year'
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
label_encoder = None


def load_model_and_encoder():
    global model, label_encoder
    client = MlflowClient()
    versions = client.search_model_versions("name='spotify-popularity-classifier'")
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    run_id = latest.run_id
    logger.info(f"Loading model version {latest.version} from run {run_id}")
    model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
    local_path = client.download_artifacts(run_id, "label_encoder.pkl", "/tmp")
    with open(local_path, "rb") as f:
        label_encoder = pickle.load(f)
    logger.info("Model and label encoder loaded successfully")


@app.on_event("startup")
def startup_event():
    load_model_and_encoder()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


class ManualFeaturesRequest(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    time_signature: int
    duration_ms: int
    explicit: int
    year: int


@app.post("/predict/manual")
def predict_manual(request: ManualFeaturesRequest):
    features = request.model_dump()
    df = pd.DataFrame([features])[FEATURES]
    prediction_encoded = model.predict(df)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    # Log input features to CSV for Evidently monitoring
    log_path = Path(MONITORING_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: features[k] for k in FEATURES})

    logger.info(f"PREDICTION_INPUT: {features}")
    logger.info(f"PREDICTION_OUTPUT: {prediction_label}")
    return {"prediction": prediction_label, "features": features}