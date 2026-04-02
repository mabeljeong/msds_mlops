# RentIQ API

FastAPI service for rent prediction (`/`, `/health`, `/predict`). Routes live in [`app/main.py`](app/main.py).

**Repo:** https://github.com/mabeljeong/msds_mlops  
**Endpoints file:** https://github.com/mabeljeong/msds_mlops/blob/main/app/main.py

---

## Run locally (no Docker)

Python 3.9+ recommended.

```bash
cd msds_mlops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Docker

Build and run (change `maniatwal` to your Docker Hub username if you push elsewhere):

```bash
docker build -t maniatwal/rentiq-api:latest .
docker run --rm -p 8000:8000 maniatwal/rentiq-api:latest
```

Then open http://127.0.0.1:8000 or use the curl below.

---

## `POST /predict`

**Request body (JSON):**

```json
{
  "bedrooms": 2,
  "bathrooms": 1,
  "sqft": 900,
  "walk_score": 85,
  "transit_score": 70,
  "median_zip_rent": 3200
}
```

`transit_score` is optional (defaults to 50).

**Example curl:**

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"bedrooms":2,"bathrooms":1,"sqft":900,"walk_score":85,"transit_score":70,"median_zip_rent":3200}'
```

**Example response** (placeholder model; numbers may differ slightly):

```json
{
  "predicted_rent_usd": 2958.21,
  "model_source": "placeholder",
  "model_version": null
}
```

---

## MLflow

Tracking UI: http://8.229.86.3:5000  

Point the API at that server and at a registered model URI (from the Models tab), then start the app:

```bash
export MLFLOW_TRACKING_URI="http://8.229.86.3:5000"
export MLFLOW_MODEL_URI="models:/YourRegisteredModelName/Production"   # example — use your real name and stage
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker:

```bash
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://8.229.86.3:5000 \
  -e MLFLOW_MODEL_URI=models:/YourRegisteredModelName/Production \
  maniatwal/rentiq-api:latest
```

If loading fails, the app falls back to the placeholder model.

**Log a run from your laptop** (shows up in the shared UI under experiment `team-project`):

```bash
export MLFLOW_TRACKING_URI="http://8.229.86.3:5000"
python scripts/mlflow_team_example.py
```

Local-only test (no remote server): `python scripts/register_mlflow_placeholder.py` and use the printed `export` lines.
