# Deploying RentRadar to Google Cloud Run

This walks you through getting the React + FastAPI app live on a public URL
your team can share.

## What gets deployed

A single Docker image built from [`../Dockerfile`](../Dockerfile):

1. **Stage 1** (`node:20-alpine`) builds the React app from `frontend/`
   into `frontend/dist/`.
2. **Stage 2** (`python:3.11-slim`) installs FastAPI deps, copies `app/`,
   `demo/`, and the React build, and runs uvicorn.

Cloud Run starts the container, exposes it at a public HTTPS URL, scales
to zero when idle, and only bills you per request.

## One-time setup

```bash
# Pick the right project (you already did this; included for the team)
gcloud config set project YOUR-PROJECT-ID

# Enable the services Cloud Run + Artifact Registry need (idempotent)
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

# Create an Artifact Registry repo to hold the image (us-central1 picked
# arbitrarily; pick whatever region the rest of the team uses)
gcloud artifacts repositories create rentradar \
  --repository-format=docker \
  --location=us-central1 \
  --description="RentRadar container images"
```

## Deploying

The simplest path: have Cloud Build build the image from source, push to
Artifact Registry, and deploy to Cloud Run, all in one command.

```bash
# From the repo root
gcloud run deploy rentradar \
  --source=. \
  --region=us-central1 \
  --allow-unauthenticated \
  --port=8000 \
  --memory=1Gi \
  --cpu=1 \
  --set-env-vars="MLFLOW_TRACKING_URI=http://8.229.86.3:5000,MLFLOW_MODEL_URI=models:/RentIQRentPredictor/1,MLFLOW_MODEL_VERSION=1"
```

What this does:

- `--source=.` uploads the repo to Cloud Build, which uses `Dockerfile`
  to build the image (the multi-stage build runs `npm install` +
  `npm run build` for you — no Node required on your laptop).
- `--allow-unauthenticated` makes the URL public (anyone with the link
  can hit it). Drop this flag for a private deploy.
- `--port=8000` matches the `EXPOSE` in our Dockerfile.
- `--set-env-vars=...` wires up MLflow so `/health` reports
  `model_source: mlflow` instead of `placeholder`. Drop these to deploy
  with the placeholder predictor.

When it finishes, gcloud prints something like:

```
Service [rentradar] revision [rentradar-00001-abc] has been deployed
and is serving 100 percent of traffic.
Service URL: https://rentradar-xxxxxxxx-uc.a.run.app
```

That URL is what you share with the team.

## Updating after code changes

Re-run the same `gcloud run deploy` command. Cloud Run zero-downtime
swaps the new revision in once it passes its startup probe.

## Watching logs

```bash
gcloud run services logs tail rentradar --region=us-central1
```

## Cost note

Cloud Run's free tier is generous: 2M requests / 360k vCPU-seconds /
180k GiB-seconds per month. A demo app that's mostly idle and bursts
during demos costs **$0** in practice. The container scales to zero
when there's no traffic, so you don't pay for an idle server.

## Common gotchas

- **First request is slow** (~3–5s). Cold start while the container
  boots from zero. Subsequent requests within a few minutes are warm.
- **`/health` shows `placeholder`** — your `MLFLOW_*` env vars aren't
  reaching the container, or the MLflow tracking server isn't reachable
  from Cloud Run's network. Check `gcloud run services describe rentradar`
  to confirm env vars; check logs for the load failure reason.
- **Build fails with "node: command not found"** — your local Docker
  daemon ran out of disk space mid-build. Run `docker system prune` and
  retry. (Cloud Build doesn't have this problem.)
- **`port=8000` mismatch** — Cloud Run defaults to `$PORT=8080`. We
  override to 8000 in the deploy command and the Dockerfile honors
  `${PORT}` via `CMD ["sh", "-c", "uvicorn ... --port ${PORT}"]`, so
  either value works as long as the deploy and EXPOSE stay aligned.
