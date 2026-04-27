"""RentIQ FastAPI inference service."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from app.model_loader import LoadedModel, load_model
from app.schemas import (
    FlagOverpricedRequest,
    FlagOverpricedResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO)

_loaded: Optional[LoadedModel] = None


def get_model() -> LoadedModel:
    assert _loaded is not None
    return _loaded


def _predict_sync(model: LoadedModel, features: dict) -> dict:
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
    features = body.model_dump()
    m = get_model()
    result = await asyncio.to_thread(_predict_sync, m, features)
    return PredictResponse(
        predicted_rent_usd=round(float(result["predicted_rent_usd"]), 2),
        fair_rent_p10=(
            round(float(result["fair_rent_p10"]), 2) if result.get("fair_rent_p10") is not None else None
        ),
        fair_rent_p90=(
            round(float(result["fair_rent_p90"]), 2) if result.get("fair_rent_p90") is not None else None
        ),
        model_source=m.source,
        model_version=m.version,
    )


@app.post("/flag_overpriced", response_model=FlagOverpricedResponse)
async def flag_overpriced(body: FlagOverpricedRequest) -> FlagOverpricedResponse:
    listing = body.model_dump()
    m = get_model()
    result = await asyncio.to_thread(m.flag_overpriced, listing)
    return FlagOverpricedResponse(**result)
