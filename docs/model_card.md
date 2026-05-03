# Model Card — RentIQ Rent Predictor (v2)

| | |
|---|---|
| Model name | `RentIQRentPredictor` (MLflow registered model name; version depends on registry) |
| Owners | RentIQ MLOps team |
| Last training run | Parent run name `rentiq_v2_selection`; final bundle nested run `rentiq_v2_final_bundle` (IDs are per tracking store — see MLflow UI or training script stdout) |
| Artifact | Tri-model `pyfunc` bundle: point regressor + **q25** + **q75** quantile heads (same features, `RentPredictor` in `app/rent_predictor.py`) |
| Serving entry points | FastAPI in `app/main.py`: `POST /predict`, `POST /flag_overpriced`, `POST /rank`, plus `GET /health`, `GET /listings`, `GET /` |
| Intended use | Estimating fair-market monthly rent and surfacing potentially overpriced listings for San Francisco apartments |
| Out-of-scope | Single-family homes, sublets, short-term rentals, markets outside San Francisco, listings outside the $2K–$8K band |

---

## 1. Training data

- **Source feature table:** `data/features/listings_features.csv` (688 listing rows + header; 16 model features), produced by `scripts/build_listings_features.py`.
- **Listing population:** Curated core augmented with additional scraped SF rentals; rows include `zip_code`, `listing_month` (and related date fields in the CSV), bed/bath, walk/transit, and joined market context. Only the 16 `FEATURE_COLUMNS` in `app/rent_predictor.py` are fed to the model.
- **External signals (point-in-time joined to `listing_month`):**
  - `crime_total_month_zip_log1p_latest` from SFPD incident data, aggregated to ZIP/month.
  - Zillow ZORI (`zori_baseline`) and ZHVI (`zhvi_level`, `zhvi_12mo_delta`).
  - Redfin metro momentum (`redfin_mom_pct`, `redfin_yoy_pct`).
  - U.S. Census ACS demographics (`census_median_income`, `census_renter_ratio`, `census_vacancy_rate`).
  - Walk Score / Transit Score (`walk_score`, `transit_score`).
- **Engineered interactions:** `bedrooms_x_census_income`, `walk_score_x_transit_score`.
- **Target scope used by shipped model:** `rent_usd ∈ [2_000, 8_000]` (variant `scope_2000_8000`). 600 of the 688 rows fall in this band; rows outside are excluded from training. The model is not trustworthy for ultra-low rent rooms or luxury rentals at the tails.
- **Target transformation:** `log1p(rent_usd)`; predictions are exponentiated and clipped to ≥ 0 at serve time.

### Feature schema (17 columns: 16 features + target)

| Column | Type | Notes |
|---|---|---|
| `zip_code` | string | One-hot encoded |
| `bedrooms`, `bathrooms` | float | Median-imputed |
| `walk_score`, `transit_score` | float | Median-imputed |
| `census_median_income`, `census_renter_ratio`, `census_vacancy_rate` | float | Latest ACS year per ZIP |
| `crime_total_month_zip_log1p_latest` | float | SFPD ZIP-month panel, point-in-time join |
| `zori_baseline`, `zhvi_level`, `zhvi_12mo_delta` | float | Zillow ZIP-month panel, PIT join |
| `redfin_mom_pct`, `redfin_yoy_pct` | float | Redfin SF metro panel, PIT join |
| `bedrooms_x_census_income`, `walk_score_x_transit_score` | float | Engineered interactions |
| `rent_usd` | float | Target |

---

## 2. Architecture and training procedure

- **Estimator:** `xgboost.XGBRegressor` inside a scikit-learn `Pipeline` with a `ColumnTransformer` (`OneHotEncoder` for `zip_code`, median imputation for numerics). Shared defaults include `learning_rate=0.05`, `subsample=0.85`, `colsample_bytree=0.8`, `reg_lambda=2.0` (`build_training_pipeline` in `app/rent_predictor.py`).
- **Three heads, identical features:**
  - Point: `objective=reg:squarederror`
  - q25: `objective=reg:quantileerror`, `quantile_alpha=0.25`
  - q75: `objective=reg:quantileerror`, `quantile_alpha=0.75`
- **Selected hyperparameters (winner from a 5-config sweep × 3 target variants):** `max_depth=3`, `min_child_weight=3`, `n_estimators=220`, plus the defaults above.
- **Cross-validation:** 5-fold `KFold(shuffle=True, random_state=42)` on the `scope_2000_8000` filtered dataset; metrics are reported on the un-logged dollar scale.
- **Model selection rule:** Lowest `cv_mae_mean`, with `cv_mae_std` as tiebreaker; ZORI baseline run (`scope_2000_8000`, `baseline_zori`) supplies `mae_lift_vs_zori`. Runs with lift below 10% get MLflow tag `value_add_lt_10pct=true` (current selection: `false`).
- **Leakage / degeneracy audit (`app/audit.py`):** Before model selection, nested runs log Pearson/Spearman target correlations, near-deterministic feature checks, top-feature-importance share (warns if any feature exceeds 50%), and fold-level importance stability.
- **Training entry point:** `python scripts/train_rent_predictor.py`. MLflow experiment name defaults to **`RentIQ`** (override with `MLFLOW_EXPERIMENT_NAME`); registered model name defaults to `RentIQRentPredictor` (`MLFLOW_REGISTERED_MODEL_NAME`).

---

## 3. Cross-validated performance (winning config, scope $2K–$8K, n=600)

Figures below match a full run of `scripts/train_rent_predictor.py` on the current `listings_features.csv` (May 2026); retraining can move them slightly.

| Metric | Value | Notes |
|---|---|---|
| `cv_mae_mean` | **$758.54 ± $44.90** (95% CI half-width) | std across folds ≈ $45.81 |
| `cv_mape_mean` | **18.0%** | per-fold mean of MAPE on dollar scale |
| `cv_r2_mean` | **0.494** | un-logged residuals |
| `mae_lift_vs_zori` | **+33.7%** | vs `scope_2000_8000` ZORI-only baseline (~$1,143.75 MAE) |
| `interval_coverage_mean` | **45.8%** | mean fraction of validation rows with `p25 ≤ actual ≤ p75` (nominal central mass for q25/q75 is 50%) |
| `interval_width_mean_usd` | **~$1,189** | mean of `(p75 − p25)` on the dollar scale across folds |

Variant comparison (best HP per variant):

| Variant | n | CV MAE | CV MAPE | CV R² |
|---|---|---|---|---|
| `scope_2000_8000` | 600 | $758.54 | 18.0% | 0.494 |
| `winsorized_1_99` | 688 | $970.52 | 28.0% | 0.496 |
| `raw_positive` | 688 | $1,025.97 | 29.3% | 0.429 |
| `baseline_zori` (scope) | 600 | $1,143.75 | 25.0% | −0.167 |

The scoped variant is the one shipped.

---

## 4. Feature importance (winning point model, full-data refit)

Top 10 features by gain (full table in MLflow: nested run `rentiq_v2_final_bundle` → `artifacts/reports/feature_importances_final.csv`). Names below omit the `numeric__` / `categorical__zip_code_` prefixes from the pipeline.

| Rank | Feature | Gain |
|---|---|---|
| 1 | **`bedrooms_x_census_income`** | 0.170 |
| 2 | `bedrooms` | 0.075 |
| 3 | `zori_baseline` | 0.070 |
| 4 | `census_renter_ratio` | 0.066 |
| 5 | `zhvi_level` | 0.047 |
| 6 | `zip_code=94129` | 0.046 |
| 7 | `census_median_income` | 0.042 |
| 8 | `zip_code=94015` | 0.040 |
| 9 | `crime_total_month_zip_log1p_latest` | 0.035 |
| 10 | `zip_code=94108` | 0.033 |

- `walk_score_x_transit_score` is mid-pack (≈ rank 19 among all one-hot and numeric columns) with gain ≈ 0.019; `walk_score` and `transit_score` alone are lower (≈ ranks 27 and 31).
- `redfin_mom_pct` and `redfin_yoy_pct` show zero gain in this refit once `zhvi_*` and ZORI are present; pruning remains a candidate for future work.

---

## 5. Interval calibration

- **Raw q25/q75 heads (CV, scope $2K–$8K):** Empirical coverage ≈ **45.8%** vs a **50%** nominal central interval; mean width ≈ **$1,189** (see §3).
- **Conformal calibration (CQR, Romano et al. 2019).** `scripts/conformal_calibrate.py` holds out **15%** of in-scope rows (`random_state=13`), fits point + q25 + q75 on the rest, and chooses a **symmetric dollar offset** `q̂` so the holdout achieves nominal coverage **`NOMINAL_COVERAGE` (default 0.50)**. Example output after re-running on the current table: raw holdout coverage ≈ **41%**, offset **q̂ ≈ $86.32**, calibrated coverage ≈ **51%**, mean width ≈ **$1,276**. Values are written to `demo/conformal_calibration.json` with usage note: **`p25' = max(p25 − offset_usd, 0)`**, **`p75' = p75 + offset_usd`**. The shipped `RentPredictorPyfunc` does **not** apply this offset by default.
- **Overpriced flag:** When q25/q75 models are present, `flag_overpriced` uses **`actual_rent_usd > p75`** (`flag_reason`: `actual_rent_above_p75`). If intervals are missing, it falls back to delta thresholds (`threshold_pct`, `threshold_usd`). Conformal widening would change flag rates if offsets were applied at serve time.

---

## 6. Known limitations

- **No square-footage feature.** Bedroom/bathroom counts are a coarse proxy; unusual layouts will misprice.
- **No listing-quality features.** Amenity, condition, parking, laundry, etc., are not modeled.
- **Crime granularity is ZIP-month.** Block-level differences and crime type mix are not captured.
- **Small training set (600 in-scope rows).** Fold-to-fold MAE std is tens of dollars; small MAE deltas between runs are rarely definitive.
- **Augmented data provenance.** Mix of curated and scraped listings can bias toward what was online at scrape time.
- **Scoped to $2K–$8K SF only.** The API does not reject out-of-band inputs, but extrapolation is unreliable.
- **Static joins.** External features use point-in-time `listing_month`; missing month falls back to latest panel values (see feature pipeline docs — a bad `listing_month` can weaken PIT correctness).
- **Quantile intervals are imperfectly calibrated** on raw heads (coverage slightly below 50% in CV); conformal offsets are available but not enabled in the default bundle.

---

## 7. V2 roadmap

1. **Apply conformal offsets at serve time.** Load `demo/conformal_calibration.json` (or a baked-in artifact) and adjust **p25/p75** inside `RentPredictor` / `RentPredictorPyfunc`. Re-run CQR whenever the model is retrained.
2. **Square-footage imputation** from public assessor or listing enrichment.
3. **Landlord / building risk side-features** if reliable labels exist.
4. **Listing-quality signals** (structured amenities or text/vision) if scope allows.
5. **Time-aware validation** once enough contiguous months exist; random KFold can understate temporal drift.
6. **Threshold tuning for the flag** against human labels or policy targets.
7. **Drift / data-quality monitoring** on inputs and predictions.
8. **Champion–challenger** evaluation with a frozen holdout before promoting registry versions.

---

## 8. How to reproduce

```bash
python scripts/build_listings_features.py
python scripts/train_rent_predictor.py
python scripts/conformal_calibrate.py        # optional; refreshes demo/conformal_calibration.json
python scripts/build_demo_examples.py         # requires MLFLOW_MODEL_URI (see training script stdout)
```

Inspect the parent run named **`rentiq_v2_selection`** in the MLflow experiment (default name **`RentIQ`**) for nested runs, `reports/model_selection_summary.csv`, audit artifacts, and the **`rentiq_v2_final_bundle`** artifacts (interval summaries, final importances, registered model).
