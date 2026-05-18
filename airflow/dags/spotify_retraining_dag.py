from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="spotify_retraining",
    default_args=default_args,
    description="Retrain Spotify popularity model on fresh data",
    schedule_interval="@weekly",
    start_date=datetime(2026, 5, 12),
    catchup=False,         # don't backfill missed runs
    tags=["spotify", "mlops"],
) as dag:

    download = BashOperator(
        task_id="download_data",
        bash_command="echo 'Placeholder: would fetch fresh data from Spotify API here'",
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="cd /app && python src/data/preprocess.py",
    )

    train = BashOperator(
        task_id="train",
        bash_command="cd /app && python src/training/train.py",
    )

    promote = BashOperator(
        task_id="promote_model",
        bash_command="cd /app && python src/training/promote.py",
    )

    monitor = BashOperator(
        task_id="monitor_drift",
        bash_command="cd /app && python src/monitoring/monitor.py",
    )

    # dependency chain
    download >> preprocess >> train >> promote >> monitor