
"""Verify that an approved serving model is available in MLflow.

If the registry is empty, bootstrap the very first model automatically.
Subsequent deployments still require an explicit production alias.
"""

import os
import sys
import time
from pathlib import Path

# Add project root (/app) to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

    # Production alias already exists
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        print(
            f"{MODEL_NAME}@{MODEL_ALIAS} already points to version "
            f"{current.version}; nothing to do."
        )
        return
    except MlflowException:
        pass

    version = latest_version(client)

    # First deployment: registry is empty
    if version is None:
        print("No registered model found.")
        print("Running pipeline to create the initial production model...")

        from run_pipeline import run_pipeline

        run_pipeline()

        # Wait for MLflow to register the model version
        version = None
        for _ in range(15):
            version = latest_version(client)
            if version is not None:
                break
            time.sleep(1)

        if version is None:
            raise RuntimeError(
                "Pipeline completed, but no model version was registered."
            )

        client.set_registered_model_alias(
            MODEL_NAME,
            MODEL_ALIAS,
            version.version,
        )

        print(
            f"Assigned '{MODEL_ALIAS}' alias to model version "
            f"{version.version}."
        )
        return

    # Existing model versions but no production alias
    raise RuntimeError(
        f"{MODEL_NAME} has registered candidate version {version.version}, "
        f"but no '{MODEL_ALIAS}' alias. "
        "Validate it and explicitly assign the production alias before deployment."
    )


if __name__ == "__main__":
    main()