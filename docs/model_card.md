# Model Card — RentIQ Rent Predictor (v2)

| | |
|---|---|
| Model name | `RentIQRentPredictor` (registered MLflow model, version 5) |
| Owners | RentIQ MLOps team |
| Last training run | parent `rentiq_v2_selection`, final bundle `rentiq_v2_final_bundle` (run id `76c95ef…d6ec4e`) |
| Artifact | tri-model `pyfunc` bundle: point regressor + q10 + q90 quantile heads |
| Serving entry points | `POST /predict`, `POST /flag_overpriced` (FastAPI in `app/main.py`) |
| Intended use | Estimating fair-market monthly rent and surfacing potentially overpriced listings for San Francisco apartments |
| Out-of-scope | Single-family homes, sublets, short-term rentals, markets outside San Francisco, listings outside the $2K–$8K band |

---

## 1. Training data

- **Source feature table:** `data/features/listings_features.csv` (688 rows, 16 model features), produced by `scripts/build_listings_features.py`.
- **Listing population:** ~300-listing curated core augmented to 688 via additional scraped SF rentals; each row carries `zip_code`, `listing_month`, bed/bath, walk/transit, and joined market context.
- **External signals (point-in-time joined to `listing_month`):**
  - `crime_total_month_zip_log1p_latest` from SFPD incident data, aggregated to ZIP/month.
  - Zillow ZORI (`zori_baseline`) and ZHVI (`zhvi_level`, `zhvi_12mo_delta`).
  - Redfin metro momentum (`redfin_mom_pct`, `redfin_yoy_pct`).
  - U.S. Census ACS demographics (`census_median_income`, `census_renter_ratio`, `census_vacancy_rate`).
  - Walk Score / Transit Score (`walk_score`, `transit_score`).
- **Engineered interactions:** `bedrooms_x_census_income`, `walk_score_x_transit_score`.
- **Target scope used by shipped model:** `rent_usd ∈ [2_000, 8_000]` (the `scope_2000_8000` variant). 600 of the 688 rows fall in this band; rows outside are excluded from training. Model is therefore not trustworthy for ultra-low rent rooms or luxury rentals at the tails.
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

- **Estimator:** `XGBoost.XGBRegressor` wrapped in a scikit-learn `Pipeline` with a `ColumnTransformer` (`OneHotEncoder` for `zip_code`, median imputation for numerics).
- **Three heads, identical features:**
  - Point: `objective=reg:squarederror`
  - q10: `objective=reg:quantileerror, quantile_alpha=0.1`
  - q90: `objective=reg:quantileerror, quantile_alpha=0.9`
- **Selected hyperparameters (winner from a 5-config sweep × 3 target variants):** `max_depth=3, min_child_weight=3, n_estimators=220, learning_rate=0.05, subsample=0.85, colsample_bytree=0.8, reg_lambda=2.0`.
- **Cross-validation:** 5-fold `KFold(shuffle=True, random_state=42)` on the `scope_2000_8000` filtered dataset; metrics are reported on the un-logged dollar scale.
- **Model selection rule:** lowest `cv_mae_mean`, with `cv_mae_std` as the tiebreaker; ZORI baseline run (`scope_2000_8000`, `baseline_zori`) used to compute `mae_lift_vs_zori`. Runs with lift below 10% would be tagged `value_add_lt_10pct=true` (current bundle: tag is `false`).
- **Leakage / degeneracy audit (`app/audit.py`):** before model selection we log Pearson/Spearman target correlations, near-deterministic feature checks, top-feature-importance share (warns if any feature exceeds 50%), and fold-level importance stability. The shipped run logged no audit warnings.
- **Training entry point:** `python scripts/train_rent_predictor.py` (parent MLflow run with one nested run per `(variant, hp_config)` and per baseline; final bundle logged and registered as `RentIQRentPredictor`).

---

## 3. Cross-validated performance (winning config, scope $2K–$8K, n=600)

| Metric | Value | Notes |
|---|---|---|
| `cv_mae_mean` | **$758.54 ± $44.90** (95% CI half-width) | std across folds = $45.81 |
| `cv_mape_mean` | **17.997%** | per-fold mean of `MAPE` on dollar scale |
| `cv_r2_mean` | **0.494** | un-logged residuals |
| `mae_lift_vs_zori` | **+33.7%** | vs `scope_2000_8000` ZORI-only baseline ($1,143.75 MAE) |
| `interval_coverage_mean` | **0.73** (target 0.80, std 0.026) | fraction of fold rows with `p10 ≤ actual ≤ p90` |
| `interval_width_mean_usd` | **$2,266.65** | mean of `p90 - p10` across folds |

Variant comparison (best HP per variant):

| Variant | n | CV MAE | CV MAPE | CV R² |
|---|---|---|---|---|
| `scope_2000_8000` | 600 | $758.54 | 18.0% | 0.494 |
| `winsorized_1_99` | 688 | $970.52 | 28.0% | 0.496 |
| `raw_positive` | 688 | $1,025.97 | 29.3% | 0.429 |
| `baseline_zori` (scope) | 600 | $1,143.75 | 25.0% | −0.167 |

The scoped variant won on every metric and is the only one shipped.

---

## 4. Feature importance (winning point model, full-data refit)

Top 10 features by gain (full table at `mlruns/.../reports/feature_importances_final.csv`):

| Rank | Feature | Gain |
|---|---|---|
| 1 | **`bedrooms_x_census_income`** (engineered interaction) | 0.170 |
| 2 | `bedrooms` | 0.075 |
| 3 | `zori_baseline` | 0.070 |
| 4 | `census_renter_ratio` | 0.066 |
| 5 | `zhvi_level` | 0.047 |
| 6 | `zip_code=94129` | 0.046 |
| 7 | `census_median_income` | 0.042 |
| 8 | `zip_code=94015` | 0.040 |
| 9 | `crime_total_month_zip_log1p_latest` | 0.035 |
| 10 | `zip_code=94108` | 0.033 |

- `bedrooms_x_census_income` is the **#1 feature** by gain — the engineered interaction does pull weight, validating the time spent on it.
- `walk_score_x_transit_score` lands at rank 19/42 with gain 0.019 (≈1.9%) — middle of the pack with measurable but minor lift; `walk_score` alone (rank 26) and `transit_score` alone (rank 30) each contribute less than the interaction term, so the interaction is the right way to encode the signal even though its share is modest.
- `redfin_mom_pct` and `redfin_yoy_pct` register zero gain, suggesting metro-level momentum is fully captured by `zhvi_*` once those are included; pruning is a candidate for v2.

---

## 5. Interval calibration

- **Raw quantile heads (CV, scope $2K–$8K):** 73% empirical coverage at the nominal 80% level, mean width ≈ $2.3K.
- **Conformal calibration (CQR, Romano et al. 2019).** A 15% holdout (90 listings, `random_state=13`) gave raw coverage **75.6%** and a symmetric dollar offset of `q̂ = $102.41` was sufficient to lift calibrated coverage to **81.1%** at the cost of widening intervals to ≈ $2.4K (≈ +9%). Offsets are saved in `demo/conformal_calibration.json` and can be applied at serve time as `p10' = max(p10 − offset, 0)`, `p90' = p90 + offset`. The shipped pyfunc does **not** apply the offset by default — flag in the V2 roadmap.
- **Implication for the overpriced flag:** the flag fires when `actual_rent_usd > p90`. With raw intervals, a non-trivial share of in-band listings sit above the empirical p90, so the flag is biased towards over-flagging. CQR offsets close most of that gap.

---

## 6. Known limitations

- **No square-footage feature.** A high-leverage size signal is missing; bedroom/bathroom counts are a coarse proxy. Listings with unusual size for their bed count (e.g., huge studios, micro 2-beds) will misprice.
- **No listing-quality features.** No amenity tags, year built, condition, view, parking, in-unit laundry, etc. Rent variation driven by these will look like noise.
- **Crime granularity is ZIP-month.** Block-level safety differences inside a ZIP are invisible. Crime feature is a single log-transformed total — type of crime (violent vs property) and recency weighting are not modeled.
- **Small training set (600 in-scope rows).** Folds carry std≈$46; a single retrain on resampled data shifts MAE by tens of dollars. The 95% CI half-width on MAE is roughly the same magnitude as the lift over baseline, so claims like "$30 MAE improvement" are not statistically meaningful at this N.
- **Augmented data provenance.** Of the 688 rows in the feature table, ~300 come from a curated core and the remainder from scraped sources. Selection bias in the scrape pool can pull the predictor toward listings that happened to be online at scrape time.
- **Scoped to $2K–$8K SF only.** The shipped model rejects no inputs but will extrapolate dangerously outside this range and outside SF ZIPs.
- **Static joins.** External features are point-in-time joined to `listing_month`, but if `listing_month` is missing we fall back to the per-ZIP latest panel value. That fallback is a leakage risk if a listing's `listing_month` is set to a value that didn't exist at the time the listing was actually live; we mitigate by checking `listing_month` is in the past relative to the panel max.
- **Quantile head is mildly under-covering.** 73% empirical at nominal 80% (see §5).

---

## 7. V2 roadmap

1. **Apply conformal offsets at serve time.** Read `demo/conformal_calibration.json` (or a baked-in artifact) and adjust `p10/p90` inside `RentPredictorPyfunc.predict`. Re-run CQR every retrain.
2. **Square-footage imputation.** Train a separate sqft imputer keyed on `zip_code, bedrooms, bathrooms, building age` from public assessor data; feed imputed sqft + uncertainty into the rent model.
3. **Landlord scoring side-feature.** Aggregate complaint / eviction / response-time signals per landlord and join as a categorical-strength feature; expose its SHAP contribution in `/flag_overpriced`.
4. **Listing-quality embeddings.** Pull listing photos / description text and fine-tune a small embedding model; include the top components as numeric features.
5. **Time-aware split.** Replace random KFold with a forward-chaining time-series split once we have ≥ 18 months of consistent data; current random split underestimates real-world drift.
6. **Threshold tuning for the flag.** Sweep flag thresholds against a labeled "actually overpriced" review set, and expose per-tenant risk tolerance as a serving parameter.
7. **Drift / data-quality monitoring.** Track input feature distributions, missingness, and prediction distribution per-ZIP per-week; alert on shift.
8. **A/B harness.** Stand up a shadow model + champion-challenger evaluation against a frozen test set so we can ship retraining with a defensible win bar (improvement > 1.96 × CV-MAE std).

---

## 8. How to reproduce

```bash
python scripts/build_listings_features.py
python scripts/train_rent_predictor.py
python scripts/conformal_calibrate.py        # optional, offsets only
python scripts/build_demo_examples.py         # requires MLFLOW_MODEL_URI
```

Inspect the parent run in MLflow UI under experiment `rentiq_v2_selection` for nested runs, audit artifacts, and the model selection summary CSV.
