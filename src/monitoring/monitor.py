import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetSummaryMetric
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REFERENCE_PATH = "data/processed/tracks_processed.csv"
CURRENT_PATH = "data/monitoring/predictions.csv"
REPORT_PATH = "data/monitoring/drift_report.html"

FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature', 'duration_ms', 'explicit', 'year'
]

def run():
    logger.info("Loading reference data...")
    reference = pd.read_csv(REFERENCE_PATH)[FEATURES]

    if not os.path.exists(CURRENT_PATH):
        logger.error(f"No prediction logs found at {CURRENT_PATH}. Make some predictions first.")
        return

    logger.info("Loading current (live) data...")
    current = pd.read_csv(CURRENT_PATH)[FEATURES]
    logger.info(f"Reference: {len(reference):,} rows | Current: {len(current):,} rows")

    # Sample reference down to a manageable size
    reference_sample = reference.sample(n=min(5000, len(reference)), random_state=42)

    report = Report(metrics=[
        DatasetSummaryMetric(),
        DataDriftPreset(),
    ])

    report.run(reference_data=reference_sample, current_data=current)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report.save_html(REPORT_PATH)
    logger.info(f"Drift report saved to {REPORT_PATH}")

if __name__ == "__main__":
    run()