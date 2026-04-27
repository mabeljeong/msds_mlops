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
missing numeric features and computes the `bedrooms_x_census_income` /
`walk_score_x_transit_score` interactions if they're not supplied.

**Example response** (mlflow tri-model bundle):

```json
{
  "predicted_rent_usd": 3601.08,
  "fair_rent_p10": 2732.92,
  "fair_rent_p90": 4914.10,
  "model_source": "mlflow",
  "model_version": "76c95ef8"
}
```

## `POST /flag_overpriced`

Pass the same feature fields plus `actual_rent_usd`. The response now includes a
fair-rent interval and the flag fires when the listing is above the model's 90th
percentile prediction (falls back to the +10% threshold when quantile models
aren't available):

```json
{
  "predicted_rent_usd": 3601.08,
  "fair_rent_p10": 2732.92,
  "fair_rent_p90": 4914.10,
  "delta_usd": 598.92,
  "delta_pct": 0.1663,
  "flag_overpriced": false,
  "flag_reason": "actual_rent_above_p90",
  "top_shap_contributors": [
    {"feature": "numeric__bedrooms_x_census_income", "shap_contribution_log_rent": -0.063},
    {"feature": "numeric__zori_baseline", "shap_contribution_log_rent": -0.035}
  ]
}
```

## Training

Train the rent predictor end-to-end (parent run with nested children for each
target variant + hyperparameter config + ZORI baseline, then a final tri-model
bundle for the API):

```bash
python scripts/build_listings_features.py
python scripts/train_rent_predictor.py
```

The training script logs to MLflow:

- per-fold MAE/MAPE/R² (un-logged scale) for every (variant, hp) child run,
- 95% CI half-width on CV MAE,
- ZORI-only baseline and `mae_lift_vs_zori`,
- `reports/leakage_audit.json`, `reports/feature_importances.csv`, `reports/cv_fold_metrics.json`,
- `reports/interval_coverage_*.json` for the final q10/q90 models,
- `reports/model_selection_summary.csv` ranked by CV MAE then std.

### Model card and demo pack

- [`docs/model_card.md`](docs/model_card.md) — training data, CV performance, interval calibration, known limitations, V2 roadmap.
- [`demo/example_listings.json`](demo/example_listings.json) — 5 hand-picked `/flag_overpriced` examples (overpriced, underpriced, in-band ×2, edge case). Regenerate with `python scripts/build_demo_examples.py` (requires `MLFLOW_MODEL_URI`).
- [`demo/conformal_calibration.json`](demo/conformal_calibration.json) — CQR offsets that lift coverage from ~73% to ~80%. Regenerate with `python scripts/conformal_calibrate.py`. Not yet applied at serve time (V2).

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
