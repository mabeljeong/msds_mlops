"""RentIQ FastAPI inference service."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from app.model_loader import LoadedModel, load_model
from app.schemas import HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)

_loaded: Optional[LoadedModel] = None


def get_model() -> LoadedModel:
    assert _loaded is not None
    return _loaded


def _predict_sync(model: LoadedModel, features: dict[str, float]) -> float:
    return model.predict_row(features)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loaded
    _loaded = load_model()
    yield
    _loaded = None


app = FastAPI(
    title="RentIQ API",
    description="Intelligent apartment recommendations — rent prediction service",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "RentIQ",
        "message": "Welcome to the RentIQ ML inference API.",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    m = get_model()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_source=m.source,
        detail=m.load_message,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest) -> PredictResponse:
    # Pydantic validates body before inference
    features = body.model_dump()
    m = get_model()
    predicted = await asyncio.to_thread(_predict_sync, m, features)
    return PredictResponse(
        predicted_rent_usd=round(predicted, 2),
        model_source=m.source,
        model_version=m.version,
    )
