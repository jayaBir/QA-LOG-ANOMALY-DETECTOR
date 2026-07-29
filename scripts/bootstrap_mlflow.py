"""Verify that an explicitly approved serving model is available in MLflow."""
import os

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "qa-log-anomaly-detector")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "production")


def latest_version(client: MlflowClient):
    versions = list(client.search_model_versions(f"name='{MODEL_NAME}'"))
    return max(versions, key=lambda version: int(version.version), default=None)


def main():
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        current = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        print(f"{MODEL_NAME}@{MODEL_ALIAS} already points to version {current.version}; nothing to do.")
        return
    except MlflowException:
        pass

    version = latest_version(client)
    if version is None:
        raise RuntimeError(
            f"No approved production model exists: {MODEL_NAME} has no registered versions. "
            "Run the separate training and validation pipeline, then explicitly promote a version."
        )

    raise RuntimeError(
        f"{MODEL_NAME} has registered candidate version {version.version}, but no '{MODEL_ALIAS}' alias. "
        "Validate it and explicitly assign the production alias before deployment."
    )


if __name__ == "__main__":
    main()
