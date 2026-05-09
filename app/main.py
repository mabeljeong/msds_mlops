"""RentIQ FastAPI inference service."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
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
load_dotenv(ROOT / ".env")

# Frontend resolution order:
#   1) React production build at `frontend/dist/` (built by Vite, used in
#      Docker / Cloud Run via the multi-stage Dockerfile).
#   2) Legacy vanilla SPA at `web/` (kept as a fallback during the React
#      migration so contributors without a Node toolchain can still serve
#      *something* via uvicorn).
REACT_DIST_DIR = ROOT / "frontend" / "dist"
LEGACY_WEB_DIR = ROOT / "web"
WEB_DIR = REACT_DIST_DIR if REACT_DIST_DIR.exists() else LEGACY_WEB_DIR

LISTINGS_FIXTURE = ROOT / "demo" / "listings_for_rank.json"


def get_model(request: Request) -> LoadedModel:
    model = getattr(request.app.state, "loaded_model", None)
    assert model is not None, "Model not loaded; lifespan did not run"
    return model


def get_listings_cache(request: Request) -> list[dict[str, Any]]:
    return getattr(request.app.state, "listings_cache", [])


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
    app.state.loaded_model = load_model()
    app.state.listings_cache = _load_listings_fixture()
    logging.info(
        "Loaded %d demo listings from %s", len(app.state.listings_cache), LISTINGS_FIXTURE
    )
    yield
    app.state.loaded_model = None
    app.state.listings_cache = []


app = FastAPI(
    title="RentRadar API",
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
# Legacy /web mount is kept so contributors who haven't built the React app
# yet can still load the original vanilla SPA via /web/index.html (handy for
# local debugging during the React migration). The new React build, when
# present, takes over the root path below.
if LEGACY_WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(LEGACY_WEB_DIR), html=True), name="legacy-web")


@app.get("/")
def root() -> Any:
    """Serve the SPA when present; otherwise return the JSON service banner."""
    index = WEB_DIR / "index.html"
    if index.exists():
        # Ensure the SPA shell is always revalidated so updated skin/assets
        # are picked up immediately after deploy. The hashed asset bundles
        # under /assets/... are still aggressively cacheable (handled by the
        # StaticFiles mount below).
        return FileResponse(index, headers={"Cache-Control": "no-store"})
    return {
        "service": "RentRadar",
        "message": "Welcome to the RentRadar ML inference API.",
    }


# ----- API endpoints ------------------------------------------------------ #
@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    m = get_model(request)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_source=m.source,
        detail=m.load_message,
        rank_component_keys=list(COMPONENT_KEYS),
    )


@app.get("/listings")
def listings(request: Request) -> dict[str, Any]:
    """Return the demo listings the frontend ranks against."""
    cache = get_listings_cache(request)
    return {"count": len(cache), "listings": cache}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    features = body.model_dump()
    m = get_model(request)
    result = await asyncio.to_thread(_predict_sync, m, features)
    return PredictResponse(
        predicted_rent_usd=round(float(result["predicted_rent_usd"]), 2),
        fair_rent_p25=(
            round(float(result["fair_rent_p25"]), 2) if result.get("fair_rent_p25") is not None else None
        ),
        fair_rent_p75=(
            round(float(result["fair_rent_p75"]), 2) if result.get("fair_rent_p75") is not None else None
        ),
        model_source=m.source,
        model_version=m.version,
    )


@app.post("/flag_overpriced", response_model=FlagOverpricedResponse)
async def flag_overpriced(request: Request, body: FlagOverpricedRequest) -> FlagOverpricedResponse:
    listing = body.model_dump()
    m = get_model(request)
    result = await asyncio.to_thread(_flag_sync, m, listing)
    return FlagOverpricedResponse(**result)


@app.post("/rank", response_model=RankResponse)
async def rank(request: Request, body: RankRequest) -> RankResponse:
    """Score every listing with `flag_overpriced`, blend with user weights, and sort."""
    if not body.listings:
        raise HTTPException(status_code=400, detail="listings must be non-empty")

    m = get_model(request)
    raw_listings = [l.model_dump() for l in body.listings]

    # Score each listing through the model. asyncio.gather over to_thread keeps the
    # event loop responsive for ~30 listings without blocking.
    flag_results = await asyncio.gather(
        *[asyncio.to_thread(_flag_sync, m, listing) for listing in raw_listings]
    )

    scored = score_listings(raw_listings, flag_results)
    weights = normalize_weights(body.weights.model_dump())

    enriched: list[tuple[float, dict[str, Any], object]] = [
        (composite(sc, weights), sc.flag_result, sc) for sc in scored
    ]

    order = sorted(
        range(len(enriched)), key=lambda i: enriched[i][0], reverse=True
    )

    results: list[RankedListing] = []
    for rank_idx, i in enumerate(order, start=1):
        comp, flag, sc = enriched[i]
        ordered_components = {k: sc.component_scores[k] for k in COMPONENT_KEYS}
        results.append(
            RankedListing.from_rank_inputs(
                src=raw_listings[i],
                flag=flag,
                component_scores=ordered_components,
                composite_score=comp,
                rank=rank_idx,
            )
        )

    if body.top_n is not None:
        results = results[: body.top_n]

    return RankResponse(
        weights_normalized={k: round(weights[k], 4) for k in COMPONENT_KEYS},
        n_input=len(raw_listings),
        n_returned=len(results),
        results=results,
    )


# ----- Static asset catch-all -------------------------------------------- #
# Mount the React build at root so Vite's hashed bundle paths
# (e.g. /assets/index-abc123.js) resolve. This must come AFTER every
# explicit @app.get/@app.post route above; Starlette matches routes in
# registration order, so the explicit API endpoints win against this
# catch-all mount. When no React build exists yet, this no-ops and the
# legacy /web mount above still serves the vanilla SPA.
if REACT_DIST_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(REACT_DIST_DIR), html=True),
        name="spa",
    )
