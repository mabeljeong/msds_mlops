"""RentIQ FastAPI inference service."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.model_loader import LoadedModel, load_model
from app.ranker import COMPONENT_KEYS, composite, normalize_weights, score_listings
from app.schemas import (
    FlagOverpricedRequest,
    FlagOverpricedResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RankedListing,
    RankRequest,
    RankResponse,
)

logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
LISTINGS_FIXTURE = ROOT / "demo" / "listings_for_rank.json"

_loaded: Optional[LoadedModel] = None
_listings_cache: list[dict[str, Any]] = []


def get_model() -> LoadedModel:
    assert _loaded is not None
    return _loaded


def _predict_sync(model: LoadedModel, features: dict) -> dict:
    return model.predict_row(features)


def _flag_sync(model: LoadedModel, listing: dict) -> dict:
    return model.flag_overpriced(listing)


def _load_listings_fixture() -> list[dict[str, Any]]:
    if not LISTINGS_FIXTURE.exists():
        return []
    return json.loads(LISTINGS_FIXTURE.read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loaded, _listings_cache
    _loaded = load_model()
    _listings_cache = _load_listings_fixture()
    logging.info("Loaded %d demo listings from %s", len(_listings_cache), LISTINGS_FIXTURE)
    yield
    _loaded = None
    _listings_cache = []


app = FastAPI(
    title="RentIQ API",
    description="Intelligent apartment recommendations — rent prediction service",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Static frontend ---------------------------------------------------- #
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


@app.get("/")
def root() -> Any:
    """Serve the SPA when present; otherwise return the JSON service banner."""
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {
        "service": "RentIQ",
        "message": "Welcome to the RentIQ ML inference API.",
    }


# ----- API endpoints ------------------------------------------------------ #
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    m = get_model()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_source=m.source,
        detail=m.load_message,
    )


@app.get("/listings")
def listings() -> dict[str, Any]:
    """Return the demo listings the frontend ranks against."""
    return {"count": len(_listings_cache), "listings": _listings_cache}


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
    result = await asyncio.to_thread(_flag_sync, m, listing)
    return FlagOverpricedResponse(**result)


@app.post("/rank", response_model=RankResponse)
async def rank(body: RankRequest) -> RankResponse:
    """Score every listing with `flag_overpriced`, blend with user weights, and sort."""
    if not body.listings:
        raise HTTPException(status_code=400, detail="listings must be non-empty")

    m = get_model()
    raw_listings = [l.model_dump() for l in body.listings]

    # Score each listing through the model. asyncio.gather over to_thread keeps the
    # event loop responsive for ~30 listings without blocking.
    flag_results = await asyncio.gather(
        *[asyncio.to_thread(_flag_sync, m, listing) for listing in raw_listings]
    )

    scored = score_listings(raw_listings, flag_results, budget_usd=body.budget_usd)
    weights = normalize_weights(body.weights.model_dump())

    enriched: list[tuple[float, RankedListing]] = []
    for src, sc in zip(raw_listings, scored):
        comp = composite(sc, weights)
        flag = sc.flag_result
        enriched.append((
            comp,
            RankedListing(
                listing_id=src.get("listing_id"),
                title=src.get("title"),
                address=src.get("address"),
                url=src.get("url"),
                zip_code=str(src.get("zip_code", "")).zfill(5),
                bedrooms=float(src.get("bedrooms", 0.0)),
                bathrooms=src.get("bathrooms"),
                lat=src.get("lat"),
                lng=src.get("lng"),
                actual_rent_usd=float(src.get("actual_rent_usd", 0.0)),
                predicted_rent_usd=float(flag.get("predicted_rent_usd", 0.0)),
                fair_rent_p10=flag.get("fair_rent_p10"),
                fair_rent_p90=flag.get("fair_rent_p90"),
                delta_usd=flag.get("delta_usd"),
                delta_pct=flag.get("delta_pct"),
                flag_overpriced=bool(flag.get("flag_overpriced", False)),
                flag_reason=flag.get("flag_reason"),
                component_scores={k: round(sc.component_scores[k], 4) for k in COMPONENT_KEYS},
                composite_score=round(comp, 4),
                rank=0,  # filled below
            ),
        ))

    enriched.sort(key=lambda item: item[0], reverse=True)
    results = []
    for i, (_, item) in enumerate(enriched, start=1):
        item.rank = i
        results.append(item)

    if body.top_n is not None:
        results = results[: body.top_n]

    return RankResponse(
        weights_normalized={k: round(weights[k], 4) for k in COMPONENT_KEYS},
        n_input=len(raw_listings),
        n_returned=len(results),
        results=results,
    )
