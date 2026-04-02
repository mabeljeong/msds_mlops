"""
Log the same placeholder sklearn model used by the API to a local MLflow store.

Run from the repo root (after pip install -r requirements.txt):

  python scripts/register_mlflow_placeholder.py

Then set the printed environment variables and start uvicorn. /health should show
model_source \"mlflow\" instead of \"placeholder\".
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

from app.model_loader import FEATURE_COLUMNS, build_placeholder_model  # noqa: E402


def main() -> None:
    mlruns = ROOT / "mlruns"
    mlruns.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns.as_uri())
    mlflow.set_experiment("RentIQ")

    model = build_placeholder_model()
    sample = {
        "bedrooms": 2.0,
        "bathrooms": 1.0,
        "sqft": 900.0,
        "walk_score": 85.0,
        "transit_score": 70.0,
        "median_zip_rent": 3200.0,
    }
    import pandas as pd

    X = pd.DataFrame([sample])[FEATURE_COLUMNS]
    signature = infer_signature(X, model.predict(X))

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X,
        )
        run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/model"
    tracking = mlruns.as_uri()

    print()
    print("Local MLflow run created. Use these in your shell:")
    print()
    print(f'  export MLFLOW_TRACKING_URI="{tracking}"')
    print(f'  export MLFLOW_MODEL_URI="{model_uri}"')
    print(f'  export MLFLOW_MODEL_VERSION="{run_id[:8]}"')
    print()
    print("Then:")
    print('  uvicorn app.main:app --host 0.0.0.0 --port 8000')
    print()
    print("GET /health should report model_source: mlflow")
    print()


if __name__ == "__main__":
    main()
