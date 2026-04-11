# RentIQ — Cloud Budget Estimate & Justification

Project scope: ~1,000 San Francisco listings, GCP stack, targeting **2,000 monthly active users (MAU)**.

User assumption: SF has ~580,000 renters; active apartment searchers at any time are roughly 3–5%.
2,000 MAU represents a realistic early-stage user base. At ~25 API calls per user per month,
this equates to ~50,000 requests/month to the prediction API.

---

## Monthly Cost Breakdown


| Service                                      | Usage                                                            | Est. Monthly Cost          |
| -------------------------------------------- | ---------------------------------------------------------------- | -------------------------- |
| Google Cloud Storage                         | ~5 GB (listings, Zillow CSVs, model artifacts, logs)             | ~$0.10                     |
| Cloud Run (FastAPI container)                | ~50,000 requests/mo, 512 MB RAM per instance                     | ~$3–8                      |
| Cloud SQL (user preferences / session store) | PostgreSQL db-f1-micro, 10 GB storage                            | ~$10–15                    |
| Artifact Registry (Docker images)            | 2–3 images ~1 GB total                                           | ~$0.10                     |
| MLflow (e2-micro VM)                         | Persistent experiment tracking + model registry                  | ~$6–8                      |
| Compute — model retraining                   | Weekly retrain, XGBoost/LightGBM, e2-standard-2 spot, ~1 hr/week | ~$1–2                      |
| WalkScore API                                | ~1,000 calls/mo (new listings only; cached otherwise)            | Free (5,000/day free tier) |
| US Census ACS API                            | Monthly refresh                                                  | Free                       |


**Total estimated: ~$20–45 / month**

---

## Justification by Component

### Google Cloud Storage (~$0.10)

Raw and processed CSVs (Apartments.com listings, Zillow ZORI/ZHVI, Census data),
model artifacts, and application logs scale to roughly 5 GB at 2,000 MAU.
GCS standard storage at $0.02/GB/month keeps this near-free.

### Cloud Run — FastAPI (~$3–8)

At 50,000 requests/month the service stays well within Cloud Run's free tier
(2 million requests/month free). However, with real users we allocate 512 MB RAM
and allow autoscaling to 3 instances during peak hours, which adds modest CPU cost.
Estimated: ~$3–8/month under normal load.

### Cloud SQL — User Preferences (~$10–15)

Real users require persistent storage for saved searches, preference profiles, and
recommendation history. A db-f1-micro PostgreSQL instance with 10 GB SSD handles
2,000 MAU comfortably.

### Artifact Registry (~$0.10)

Stores Docker images for the FastAPI service and any batch jobs.
At 2–3 images (~1 GB total), storage cost is negligible.

### MLflow Tracking Server (~$6–8)

An always-on e2-micro VM (~$6–8/mo) hosts the MLflow tracking server and acts as
the model registry. At 2,000 MAU, model versions need to be reliably accessible
for the prediction API, so a persistent server (rather than local-only) is justified.

### Model Retraining (~$1–2)

With real users, the model should retrain weekly on fresh listings data.
A spot e2-standard-2 instance (~$0.08/hr) running ~1 hr/week costs ~$1.30/month.
XGBoost/LightGBM on ~1,000 rows completes well within that window.

### WalkScore API (Free)

Only called when new listings are ingested (not per user request).
With ~1,000 SF listings refreshed periodically, usage stays within the 5,000 calls/day free tier.
Responses are cached in GCS to avoid redundant calls.

### Census ACS API (Free)

Monthly data refresh for zip/tract-level median rent and income features.
No quota concerns at this scale.

---

## Cost Scenarios


| Scenario                                         | Monthly Cost |
| ------------------------------------------------ | ------------ |
| Minimal (GCS + Cloud Run only, no DB)            | ~$5–10       |
| Standard production (full stack, 2,000 MAU)      | ~$20–45      |
| Growth stage (10,000 MAU, larger DB + Cloud Run) | ~$80–150     |


---

