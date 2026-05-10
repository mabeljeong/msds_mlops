# --- Stage 1: build the React frontend ---------------------------------- #
# We pin Node to a current LTS so local `npm run build` and the container
# build behave the same way. The output (frontend/dist/) is copied into the
# Python image below; Node itself is discarded.
FROM node:20-alpine AS frontend

WORKDIR /frontend

# Copy package manifests first so npm install is cached when only source
# files (not dependencies) change.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend/ ./
RUN npm run build


# --- Stage 2: the FastAPI image ----------------------------------------- #
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY demo ./demo

# Drop the React build into the location app/main.py expects: the
# REACT_DIST_DIR resolution looks at <repo_root>/frontend/dist, which maps
# to /app/frontend/dist inside the container.
COPY --from=frontend /frontend/dist ./frontend/dist

# Cloud Run injects $PORT at runtime; default to 8000 for plain `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
