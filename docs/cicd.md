# CI/CD

## Workflows

| File | Triggers | Purpose |
| --- | --- | --- |
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | push to `main`, PRs to `main`, tag `v*` | `ruff check`, full `pytest` with coverage, Docker build smoke (skipped on tags) |
| [`.github/workflows/release.yml`](../.github/workflows/release.yml) | tag `v*` | Build + push image to Artifact Registry, deploy to Cloud Run, refresh data on GCS |

A normal push or PR runs only `ci.yml`. A version tag (`git tag v1.0.0 && git push origin v1.0.0`) runs both `ci.yml` (the build smoke job is skipped for tags) and `release.yml`.

## Local equivalents

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .
pytest tests/ -v --cov=app --cov=scripts --cov-report=term
docker build -t rentiq-api:local .
```

## Required GitHub secrets

These are all consumed by [`.github/workflows/release.yml`](../.github/workflows/release.yml). Set them at **Settings → Secrets and variables → Actions → New repository secret**.

### GCP authentication (Workload Identity Federation)

| Secret | Example | Notes |
| --- | --- | --- |
| `GCP_PROJECT_ID` | `rentiq-prod-123456` | The Cloud project that owns Artifact Registry, Cloud Run, and GCS |
| `GCP_REGION` | `us-central1` | Used for both Artifact Registry (`<region>-docker.pkg.dev`) and Cloud Run |
| `GCP_AR_REPO` | `rentiq` | Artifact Registry **repo** (not project). Created with `gcloud artifacts repositories create rentiq --repository-format=docker --location=$GCP_REGION` |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | `projects/123456789/locations/global/workloadIdentityPools/github-pool/providers/github-provider` | Full resource name of the WIF provider |
| `GCP_SERVICE_ACCOUNT` | `github-actions@rentiq-prod-123456.iam.gserviceaccount.com` | Service account the WIF provider impersonates |

The service account needs these IAM roles on the project:

- `roles/artifactregistry.writer` (push images)
- `roles/run.admin` (deploy Cloud Run revisions)
- `roles/iam.serviceAccountUser` (set the Cloud Run runtime service account)
- `roles/storage.objectAdmin` on `GCS_DATA_BUCKET` (upload data)

### Cloud Run runtime configuration

| Secret | Example | Notes |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | `http://8.229.86.3:5000` | Tracking server the API connects to at startup |
| `MLFLOW_MODEL_URI` | `models:/RentIQRentPredictor/1` | Registered model URI loaded by [`app/model_loader.py`](../app/model_loader.py) |
| `MLFLOW_MODEL_VERSION` | `1` | Echoed in `/predict` responses |

### Data refresh

| Secret | Example | Notes |
| --- | --- | --- |
| `GCS_DATA_BUCKET` | `rentiq-data` | Target bucket for [`scripts/upload_to_gcs.py`](../scripts/upload_to_gcs.py) |

## Workload Identity Federation setup (one-time)

```bash
PROJECT_ID="rentiq-prod-123456"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
SA="github-actions@${PROJECT_ID}.iam.gserviceaccount.com"
REPO="mabeljeong/msds_mlops"  # GitHub owner/repo

gcloud iam service-accounts create github-actions \
  --project="$PROJECT_ID" --display-name="GitHub Actions"

gcloud iam workload-identity-pools create github-pool \
  --project="$PROJECT_ID" --location=global --display-name="GitHub Actions pool"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --project="$PROJECT_ID" --location=global \
  --workload-identity-pool=github-pool \
  --display-name="GitHub OIDC" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository == '${REPO}'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

gcloud iam service-accounts add-iam-policy-binding "$SA" \
  --project="$PROJECT_ID" --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/${REPO}"
```

`GCP_WORKLOAD_IDENTITY_PROVIDER` is then:

```
projects/<PROJECT_NUMBER>/locations/global/workloadIdentityPools/github-pool/providers/github-provider
```

## Triggering a release

```bash
git tag v0.1.0
git push origin v0.1.0
```

The Actions tab will show `release.yml` running three jobs in sequence + parallel:

```
build-and-push -> deploy-cloud-run
              \-> data-refresh
```

After deploy, hit the Cloud Run URL printed by the deploy job and confirm `/health` returns `model_source: "mlflow"`.

## Note on the model artifact

The Docker image **does not** contain the model. On startup, [`app/model_loader.py`](../app/model_loader.py) calls `mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)` with the env-var configuration above. As long as Cloud Run can reach `MLFLOW_TRACKING_URI` over the network, no manual artifact push to GCP is required. If the tracking server is unreachable, the API silently falls back to a synthetic-data placeholder predictor and `/health` will report `model_source: "placeholder"`.
