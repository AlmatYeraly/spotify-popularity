import os
import logging
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from src.config import MLFLOW_MODEL_NAME

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def promote():
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
    if not versions:
        logger.error("No model versions found. Run training first.")
        return

    # Latest registered version is the candidate (just trained by the previous DAG task)
    candidate = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    candidate_accuracy = client.get_run(candidate.run_id).data.metrics.get("accuracy", 0.0)
    logger.info(f"Candidate: version {candidate.version}, accuracy={candidate_accuracy:.4f}")

    try:
        prod = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "production")
        current_accuracy = client.get_run(prod.run_id).data.metrics.get("accuracy", 0.0)
        logger.info(f"Current production: version {prod.version}, accuracy={current_accuracy:.4f}")

        if candidate_accuracy > current_accuracy:
            client.set_registered_model_alias(MLFLOW_MODEL_NAME, "production", candidate.version)
            logger.info(
                f"Promoted v{candidate.version} ({candidate_accuracy:.4f}) "
                f"over v{prod.version} ({current_accuracy:.4f})"
            )
        else:
            logger.info(
                f"Candidate v{candidate.version} ({candidate_accuracy:.4f}) did not beat "
                f"current production v{prod.version} ({current_accuracy:.4f}). Keeping current."
            )

    except MlflowException:
        # No production alias exists yet — promote unconditionally
        client.set_registered_model_alias(MLFLOW_MODEL_NAME, "production", candidate.version)
        logger.info(f"No production model found. Promoted v{candidate.version} as first production model.")


if __name__ == "__main__":
    promote()
