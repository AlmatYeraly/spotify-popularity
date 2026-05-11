import pandas as pd
import numpy as np
import os
import logging
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
import pickle

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train.log")
    ]
)
logger = logging.getLogger(__name__)

PROCESSED_PATH = "data/processed/tracks_processed.csv"
MLFLOW_EXPERIMENT = "spotify-popularity"

FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature', 'duration_ms', 'explicit', 'year'
]

PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "eval_metric": "mlogloss",
    "random_state": 42
}

def load_data():
    logger.info(f"Loading processed data from {PROCESSED_PATH}")
    df = pd.read_csv(PROCESSED_PATH)
    logger.info(f"Loaded {len(df):,} rows")

    X = df[FEATURES]
    y = df['popularity_bucket']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    ), le


def train():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        logger.info("Starting training run...")

        (X_train, X_test, y_train, y_test), le = load_data()

        # Log parameters
        mlflow.log_params(PARAMS)

        # Train
        model = XGBClassifier(**PARAMS)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, target_names=le.classes_)

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Macro: {f1_macro:.4f}")
        logger.info(f"\n{report}")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)

        # Log model
        mlflow.xgboost.log_model(
            model,
            name="model",
            registered_model_name="spotify-popularity-classifier"
        )

        # Save label encoder as artifact
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)
        mlflow.log_artifact("label_encoder.pkl")
        os.remove("label_encoder.pkl")

        logger.info("Run complete.")


if __name__ == "__main__":
    train()