# RentIQ (msds_mlops)

RentIQ is an end-to-end ML pipeline for rental market intelligence. This repository includes a **FastAPI inference service** for monthly rent prediction.

## GitHub repository

Replace with your public repository URL after you push, for example: `https://github.com/<your-username>/msds_mlops`

## Where the API endpoints are defined

All HTTP routes are implemented in [`app/main.py`](app/main.py):

- `GET /` — welcome payload
- `GET /health` — confirms the process is up and reports whether the model came from MLflow or the built-in placeholder
- `POST /predict` — validated JSON body → rent prediction

## Run the API locally (without Docker)

Requirements: Python 3.9+ (3.11+ recommended; the Docker image uses 3.11).

```bash
cd msds_mlops
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### MLflow Model Registry (optional)

By default the service loads a **placeholder** `sklearn.linear_model.LinearRegression` trained on synthetic data with the same feature names as the API.

To load from MLflow instead, set:

| Variable | Purpose |
|----------|---------|
| `MLFLOW_TRACKING_URI` | Tracking server or local store (e.g. `http://127.0.0.1:5000` or `./mlruns`) |
| `MLFLOW_MODEL_URI` | Model URI, e.g. `models:/RentIQ-RentModel/Production` or `runs:/<run_id>/model` |
| `MLFLOW_MODEL_VERSION` | Optional string echoed in `/predict` responses |

If `MLFLOW_MODEL_URI` is set but loading fails, the service falls back to the placeholder and logs a warning.

Registered MLflow models should accept a **single-row** pandas DataFrame with the same column names as the request fields (`bedrooms`, `bathrooms`, `sqft`, `walk_score`, `transit_score`, `median_zip_rent`), as produced by this API.

## Build and run with Docker

Replace `YOUR_DOCKERHUB_USER` with your Docker Hub (or registry) username. **Use the same image name in your submission screenshot and in this README.**

```bash
docker build -t YOUR_DOCKERHUB_USER/rentiq-api:latest .
docker run --rm -p 8000:8000 YOUR_DOCKERHUB_USER/rentiq-api:latest
```

With MLflow:

```bash
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_MODEL_URI=models:/RentIQ-RentModel/Production \
  YOUR_DOCKERHUB_USER/rentiq-api:latest
```

Publish (after `docker login`):

```bash
docker push YOUR_DOCKERHUB_USER/rentiq-api:latest
```

## Example `/predict` request

**Input (JSON)** — all fields are validated by Pydantic before inference:

| Field | Type | Notes |
|-------|------|--------|
| `bedrooms` | number | 0–20 |
| `bathrooms` | number | 0–20 |
| `sqft` | number | must be positive |
| `walk_score` | number | 0–100 |
| `transit_score` | number | optional, default `50`, 0–100 |
| `median_zip_rent` | number | benchmark rent for the ZIP (USD/month) |

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 2,
    "bathrooms": 1,
    "sqft": 900,
    "walk_score": 85,
    "transit_score": 70,
    "median_zip_rent": 3200
  }'
```

**Example response** (values depend on model; placeholder model shown):

```json
{
  "predicted_rent_usd": 2958.21,
  "model_source": "placeholder",
  "model_version": null
}
```

When serving an MLflow model, `model_source` is `"mlflow"` and `model_version` may be set via `MLFLOW_MODEL_VERSION`.

## Async `/predict` and threading

`POST /predict` is declared as `async def`. Blocking sklearn / MLflow inference runs in a worker thread via `asyncio.to_thread` so the event loop can still accept connections while a prediction runs.

For **CPU-bound** sklearn inference on a single process, throughput is still limited by the GIL and core count; async shines when the handler performs **I/O** (HTTP to other services, databases, blob reads). A workload with heavy concurrent I/O or many slow clients benefits more from async patterns than a pure `predict(X)` on a small matrix.

## Project layout

- `app/main.py` — FastAPI app and routes
- `app/schemas.py` — Pydantic request/response models
- `app/model_loader.py` — MLflow vs placeholder loading
- `Dockerfile` — production image
- `requirements.txt` — Python dependencies

## Submission checklist (course)

- [ ] GitHub repo link in your PDF
- [ ] Direct link to [`app/main.py`](app/main.py) on GitHub (blob URL)
- [ ] Screenshot of published Docker image; image name matches this README
- [ ] Run instructions (local + Docker) and `/predict` example (this README)
