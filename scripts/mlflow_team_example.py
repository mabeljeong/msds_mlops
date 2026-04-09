"""
Step 10 — log a run to the shared MLflow server from your laptop.

Usage (from repo root, venv activated):

  export MLFLOW_TRACKING_URI="http://EXTERNAL_IP:5000"   # or rely on default below
  python scripts/mlflow_team_example.py

Then open the MLflow UI and open experiment "team-project" to see the run.
"""

import os

import mlflow

# Shared server
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://8.229.86.3:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "team-project")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_metric("accuracy", 0.92)

    print(f"Logged run to {TRACKING_URI} under experiment '{EXPERIMENT_NAME}'.")


if __name__ == "__main__":
    main()
