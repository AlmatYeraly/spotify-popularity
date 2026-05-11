import pandas as pd
import numpy as np
import os
import logging

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/preprocess.log")
    ]
)

logger = logging.getLogger(__name__)

RAW_PATH = "data/raw/tracks.csv"
PROCESSED_PATH = "data/processed/tracks_processed.csv"

FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature', 'duration_ms', 'explicit', 'year'
]

POPULARITY_THRESHOLDS = {'low': 25, 'high': 50}

def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data...")
    initial_rows = len(df)

    df = df.dropna(subset=['name'])
    df = df[df['popularity'] > 0]

    logger.info(f"Removed {initial_rows - len(df):,} rows (missing names + zero popularity)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering features...")

    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    median_year = df['year'].median()
    df['year'] = df['year'].fillna(median_year)

    return df


def bucket(score: int) -> str:
    low_thresh = POPULARITY_THRESHOLDS['low']
    high_thresh = POPULARITY_THRESHOLDS['high']
    if score <= low_thresh:
        return 'low'
    elif score <= high_thresh:
        return 'medium'
    else:
        return 'high'


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Assigning popularity buckets...")
    df['popularity_bucket'] = df['popularity'].apply(bucket)
    logger.info(f"Label distribution:\n{df['popularity_bucket'].value_counts()}")
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols_to_save = FEATURES + ['popularity_bucket']
    df[cols_to_save].to_csv(path, index=False)
    logger.info(f"Saved {len(df):,} rows to {path}")


def run():
    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    df = assign_labels(df)
    save_data(df, PROCESSED_PATH)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    run()