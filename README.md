# RentIQ (msds_mlops)

RentIQ is an end-to-end ML pipeline for rental market intelligence. This repository includes a **FastAPI inference service** for monthly rent prediction.

## GitHub repository (for your PDF)

**Repository:** [https://github.com/mabeljeong/msds_mlops](https://github.com/mabeljeong/msds_mlops)

**Endpoints are defined in:** [app/main.py on `main`](https://github.com/mabeljeong/msds_mlops/blob/main/app/main.py)  
(Copy that URL into your submission PDF as the direct link to the route definitions.)

## Run the API locally (without Docker)

Requirements: Python 3.9+ (3.11+ recommended; the Docker image uses 3.11).

```bash
git clone https://github.com/mabeljeong/msds_mlops.git
cd msds_mlops
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Smoke test** (in another terminal):

```bash
curl -s http://127.0.0.1:8000/
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"bedrooms":2,"bathrooms":1,"sqft":900,"walk_score":85,"transit_score":70,"median_zip_rent":3200}'
```

### (Optional) Connect a real MLflow model

By default the API uses an in-process **placeholder** `LinearRegression` (no MLflow).

To load a model from MLflow so `/health` shows `model_source: mlflow`:

1. **Local file store (no server required)** — register the same placeholder into `./mlruns`:

   ```bash
   python scripts/register_mlflow_placeholder.py
   ```

   Copy the printed `export MLFLOW_TRACKING_URI=...` and `export MLFLOW_MODEL_URI=...` lines, run them in your shell, then start `uvicorn` again.

2. **Remote tracking server** — set `MLFLOW_TRACKING_URI` to your server (e.g. `http://127.0.0.1:5000`) and `MLFLOW_MODEL_URI` to a registry URI such as `models:/YourModelName/Production`.

| Variable | Purpose |
|----------|---------|
| `MLFLOW_TRACKING_URI` | Tracking store URI (e.g. `file:///absolute/path/to/mlruns` or `http://host:5000`) |
| `MLFLOW_MODEL_URI` | Model URI, e.g. `runs:/<run_id>/model` or `models:/RentIQ-RentModel/Production` |
| `MLFLOW_MODEL_VERSION` | Optional string echoed in `/predict` responses |

If `MLFLOW_MODEL_URI` is set but loading fails, the service falls back to the placeholder and logs a warning.

Registered MLflow models should accept a **single-row** pandas DataFrame with columns: `bedrooms`, `bathrooms`, `sqft`, `walk_score`, `transit_score`, `median_zip_rent`.

## Build and run with Docker

Use an image name that matches what you push to Docker Hub (below we use **`mabeljeong/rentiq-api:latest`** — change the namespace if your Docker Hub username is different).

```bash
docker build -t mabeljeong/rentiq-api:latest .
docker run --rm -p 8000:8000 mabeljeong/rentiq-api:latest
```

With MLflow (tracking server on the host):

```bash
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_MODEL_URI=models:/YourModelName/Production \
  mabeljeong/rentiq-api:latest
```

**Publish to Docker Hub** (run on your machine after `docker login`):

```bash
docker push mabeljeong/rentiq-api:latest
```

Take a **screenshot** of the image on Docker Hub showing **repository name + tag** (e.g. `mabeljeong/rentiq-api` and `latest`). That name must match the README and your PDF.

## Example `/predict` request

**Input (JSON)** — validated by Pydantic before inference:

| Field | Type | Notes |
|-------|------|--------|
| `bedrooms` | number | 0–20 |
| `bathrooms` | number | 0–20 |
| `sqft` | number | must be positive |
| `walk_score` | number | 0–100 |
| `transit_score` | number | optional, default `50`, 0–100 |
| `median_zip_rent` | number | benchmark rent for the ZIP (USD/month) |

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

**Example output** (numbers vary slightly with placeholder vs MLflow; shape is the same):

```json
{
  "predicted_rent_usd": 2958.21,
  "model_source": "placeholder",
  "model_version": null
}
```

With MLflow loaded, `model_source` is `"mlflow"` and `model_version` may be set.

## Async `/predict`

`POST /predict` uses `async def` and runs blocking inference in a worker thread via `asyncio.to_thread`. For CPU-only sklearn work, async helps less than for I/O-heavy handlers; see comments in `app/main.py`.

## Project layout

- [`app/main.py`](app/main.py) — FastAPI app and routes
- [`app/schemas.py`](app/schemas.py) — Pydantic request/response models
- [`app/model_loader.py`](app/model_loader.py) — MLflow vs placeholder loading
- [`scripts/register_mlflow_placeholder.py`](scripts/register_mlflow_placeholder.py) — optional local MLflow registration helper
- [`Dockerfile`](Dockerfile) — container image
- [`requirements.txt`](requirements.txt) — dependencies

---

## Assignment checklist (what to do locally)

| Step | Action |
|------|--------|
| 1 | *(Optional)* Run `python scripts/register_mlflow_placeholder.py`, export printed vars, confirm `GET /health` shows `"model_source":"mlflow"`. |
| 2 | Run `uvicorn app.main:app --port 8000` and verify `GET /`, `GET /health`, `POST /predict`. |
| 3 | `docker build -t mabeljeong/rentiq-api:latest .` — must finish with no errors. |
| 4 | `docker run --rm -p 8000:8000 mabeljeong/rentiq-api:latest` and repeat the curl checks. |
| 5 | `docker login` then `docker push mabeljeong/rentiq-api:latest`. |
| 6 | Screenshot Docker Hub showing image name + tag. |
| 7 | `git add`, `git commit`, `git push` so GitHub has `app/`, `Dockerfile`, `README`. |
| 8 | PDF: paste **repo URL** and **blob link** to `app/main.py` (links at top of this README). |
| 9 | PDF: confirm README documents local run, Docker run, and `/predict` example (this file). |
| 10 | Export submission PDF with repo link, endpoint file link, and Docker screenshot. |

**Note:** Steps 3–6 require Docker Desktop (or Docker Engine) running on your machine. This README uses GitHub user `mabeljeong`; use your Docker Hub username in the image name if it differs.
