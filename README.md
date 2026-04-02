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

Optional: load a model from MLflow by setting `MLFLOW_TRACKING_URI` and `MLFLOW_MODEL_URI` before starting the app. To log a small local model for testing, run `python scripts/register_mlflow_placeholder.py` and use the `export` lines it prints.
