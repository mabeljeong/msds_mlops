# RentIQ API

FastAPI service for rent prediction and a single-page web UI. Endpoints: `/`, `/health`, `/predict`, `/flag_overpriced`, `/listings`, `/rank`. Routes live in [`app/main.py`](app/main.py); the SPA lives in [`web/`](web/).

**Repo:** https://github.com/mabeljeong/msds_mlops  
**Endpoints file:** https://github.com/mabeljeong/msds_mlops/blob/main/app/main.py

The app loads environment variables from a **`.env` file in the repo root** (via `python-dotenv`). If `MLFLOW_MODEL_URI` is missing or loading fails, it uses a **placeholder** predictor (fine for demos; not the trained XGBoost bundle).

## Demo

Screen recording of the RentIQ UI and API: **[open on Google Drive](https://drive.google.com/file/d/1VZnCJNUvVCCqqr_wUjOg1ab2VWwmtMp5/view?usp=sharing)**.

---

## Run locally (no Docker)

Python 3.10+ recommended (see `requirements.txt`).

```bash
cd msds_mlops
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### With the registered MLflow model (recommended)

Point the app at your **tracking server** and a **model URI** from the Model Registry (version number, stage, or `runs:/...`).

Example registry version **1** for model **`RentIQRentPredictor`**:

```bash
export MLFLOW_TRACKING_URI="http://8.229.86.3:5000"
export MLFLOW_MODEL_URI="models:/RentIQRentPredictor/1"
export MLFLOW_MODEL_VERSION="1"   # optional; echoed on /predict only, not inferred from MLflow

uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Or add to **`.env`** (same keys as above). The app calls `load_dotenv` on startup, so you can omit the `export` lines if `.env` is set.

**UI:** open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) — the SPA is served from the API root and calls the API on the same origin.

**Sanity check:**

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Expect `"model_source": "mlflow"` when the bundle loaded. If you see `"placeholder"`, check `detail` and your `MLFLOW_*` variables.

**Optional:** regenerate demo listings for `/rank` if needed:

```bash
python scripts/build_rank_listings.py
```

### Without MLflow (placeholder only)

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

---

## Docker

Build and run (change `maniatwal` to your Docker Hub username if you push elsewhere):

```bash
docker build -t maniatwal/rentiq-api:latest .
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://8.229.86.3:5000 \
  -e MLFLOW_MODEL_URI=models:/RentIQRentPredictor/1 \
  -e MLFLOW_MODEL_VERSION=1 \
  maniatwal/rentiq-api:latest
```

Then open http://127.0.0.1:8000 or use the curl examples below.

---

## `POST /predict`

**Request body (JSON):**

```json
{
  "zip_code": "94103",
  "bedrooms": 2,
  "bathrooms": 1,
  "walk_score": 95,
  "transit_score": 92,
  "census_median_income": 130000,
  "census_renter_ratio": 0.62,
  "census_vacancy_rate": 0.04,
  "crime_total_month_zip_log1p_latest": 5.2,
  "zori_baseline": 3450,
  "zhvi_level": 1250000,
  "zhvi_12mo_delta": 45000,
  "redfin_mom_pct": 0.01,
  "redfin_yoy_pct": 0.06
}
```

All fields except `zip_code` and `bedrooms` are optional; the trained pipeline imputes
missing numeric features and can use precomputed `bedrooms_x_census_income` /
`walk_score_x_transit_score` if supplied.

**Example response** (MLflow tri-model bundle — point + quantile band):

```json
{
  "predicted_rent_usd": 3601.08,
  "fair_rent_p25": 2732.92,
  "fair_rent_p75": 4914.10,
  "model_source": "mlflow",
  "model_version": "1"
}
```

## `POST /flag_overpriced`

Pass the same feature fields plus `actual_rent_usd`. With the real model, the fair-rent band is **p25–p75**; the listing is flagged overpriced when actual rent is **above p75** (with fallbacks documented in code when quantiles are missing):

```json
{
  "predicted_rent_usd": 3601.08,
  "fair_rent_p25": 2732.92,
  "fair_rent_p75": 4914.10,
  "delta_usd": 598.92,
  "delta_pct": 0.1663,
  "flag_overpriced": true,
  "flag_reason": "actual_rent_above_p75",
  "top_shap_contributors": [
    {"feature": "numeric__bedrooms_x_census_income", "shap_contribution_log_rent": -0.063},
    {"feature": "numeric__zori_baseline", "shap_contribution_log_rent": -0.035}
  ]
}
```

When quantiles are unavailable, the API falls back to delta thresholds and `flag_reason` may be `delta_pct_exceeds_threshold` instead.

## Web UI (`/`) + `POST /rank`

A single-page app (vanilla HTML/JS, no build step) ships at the API root. It loads demo SF listings from `GET /listings`, then calls `POST /rank` when weights, budget, beds, or ZIP filters change. `/rank` runs `flag_overpriced` on each listing, blends scores with user weights, and returns a sorted list. Fair-rent **p25–p75** bands on the cards come from the MLflow model when the API is configured as above.

If you open `web/index.html` via `file://`, the client uses `http://localhost:8000` as the API base; otherwise it uses the same origin as the page.

---

## Training

Prerequisites:

- `data/processed/listings_clean.csv` (and other inputs expected by [`scripts/build_listings_features.py`](scripts/build_listings_features.py))
- Then `data/features/listings_features.csv` from the feature build

```bash
python scripts/build_listings_features.py
export MLFLOW_TRACKING_URI="http://8.229.86.3:5000"
export MLFLOW_EXPERIMENT_NAME="RentIQ"
export MLFLOW_REGISTERED_MODEL_NAME="RentIQRentPredictor"
python scripts/train_rent_predictor.py
```

The training script logs a parent run with nested children (variants × hyperparameters + ZORI baselines), selects the best config, fits the **point + q25 + q75** bundle, and registers **`MLFLOW_REGISTERED_MODEL_NAME`** (default `RentIQRentPredictor`) in the Model Registry.

**Remote training from your laptop:** the experiment **`RentIQ`** on the shared server must use an **`mlflow-artifacts:/...` artifact root** so the client uploads artifacts over HTTP. If the experiment instead uses a bare server path (e.g. `/home/.../artifacts`), logging from another machine will fail; create a new experiment with `artifact_location='mlflow-artifacts:/...'` or fix the tracking server configuration (`mlflow server` with proxied artifacts).

After training, the script prints `export` lines including `runs:/<run_id>/model`; you can still load by run URI or by `models:/RentIQRentPredictor/<version>`.

### Model card and demo pack

- [`docs/model_card.md`](docs/model_card.md) — training data, CV performance, interval calibration, known limitations, V2 roadmap.
- [`demo/example_listings.json`](demo/example_listings.json) — hand-picked `/flag_overpriced` examples. Regenerate with `python scripts/build_demo_examples.py` (requires `MLFLOW_MODEL_URI` and tracking URI).
- [`demo/conformal_calibration.json`](demo/conformal_calibration.json) — CQR offsets. Regenerate with `python scripts/conformal_calibrate.py`. Not applied at serve time by default (see model card).

---

## MLflow

**Tracking UI:** http://8.229.86.3:5000  

**Registered model name:** `RentIQRentPredictor` (versions increment on each successful registration from training).

**Example API env (registry version 1):**

```bash
export MLFLOW_TRACKING_URI="http://8.229.86.3:5000"
export MLFLOW_MODEL_URI="models:/RentIQRentPredictor/1"
export MLFLOW_MODEL_VERSION="1"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

If you assign a **stage** (e.g. `Production`) in the MLflow UI, you can use `models:/RentIQRentPredictor/Production` instead.

If loading fails, the app falls back to the placeholder model (`/health` will show `model_source: placeholder`).

**Log a toy run** (experiment `team-project` by default):

```bash
export MLFLOW_TRACKING_URI="http://8.229.86.3:5000"
python scripts/mlflow_team_example.py
```

**Local-only MLflow smoke test** (no remote server): `python scripts/register_mlflow_placeholder.py` and use the printed `export` lines (placeholder sklearn model, not the production bundle).
