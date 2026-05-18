# Spotify Popularity MLOps Pipeline

> Built with the help of Claude. The main goal of this project was to learn MLOps tools and how they fit together in a real pipeline.

Predicts whether a Spotify track is low, medium, or high popularity using audio features (danceability, energy, tempo, etc.). The model is an XGBoost classifier trained on a static Kaggle dataset of ~600k tracks.

**Note on data:** Spotify's API no longer provides audio features for new tracks, so fresh data ingestion is not possible. The pipeline retrains on the same static dataset.

## Stack

| Layer | Tool |
|---|---|
| Experiment tracking + model registry | MLflow |
| Orchestration | Airflow |
| Serving | FastAPI |
| Drift monitoring | Evidently |
| Infrastructure | Docker Compose |

## Pipeline

```
download (placeholder) -> preprocess -> train -> promote -> monitor drift
```

The `promote` step compares the newly trained model's accuracy against the current production model in the MLflow registry. It only promotes if the new model is better. The serving API loads exclusively from the `@production` alias.

## Running locally

```bash
docker-compose up
```

| Service | URL |
|---|---|
| Airflow UI | http://localhost:8080 (admin / admin) |
| MLflow UI | http://localhost:5000 |
| Prediction API | http://localhost:8000 |

Trigger the `spotify_retraining` DAG once to train and promote the first model, then the serving container will be ready.

To generate a drift report manually:

```bash
docker-compose --profile monitoring run evidently
```

## API

`POST /predict/manual` accepts audio features and returns a popularity bucket:

```json
{
  "danceability": 0.8,
  "energy": 0.6,
  "key": 5,
  "loudness": -5.0,
  "mode": 1,
  "speechiness": 0.05,
  "acousticness": 0.1,
  "instrumentalness": 0.0,
  "liveness": 0.1,
  "valence": 0.7,
  "tempo": 120.0,
  "time_signature": 4,
  "duration_ms": 200000,
  "explicit": 0,
  "year": 2020
}
```

## Next steps

Deploy to a cloud host. Airflow will likely be dropped from the hosted version due to resource cost, with the retraining pipeline shown via screenshots and documentation instead.
