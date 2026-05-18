FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature', 'duration_ms', 'explicit', 'year'
]

POPULARITY_THRESHOLDS = {'low': 25, 'high': 50}

MLFLOW_EXPERIMENT = "spotify-popularity"
MLFLOW_MODEL_NAME = "spotify-popularity-classifier"
