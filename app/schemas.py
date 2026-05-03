from typing import Any, Mapping, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Input features aligned to listings_features training table v2."""

    zip_code: str = Field(..., min_length=5, max_length=5, description="5-digit listing ZIP code")
    bedrooms: float = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: Optional[float] = Field(None, ge=0, le=20, description="Number of bathrooms")
    walk_score: Optional[float] = Field(None, ge=0, le=100, description="Walk Score (0-100)")
    transit_score: Optional[float] = Field(None, ge=0, le=100, description="Transit Score (0-100)")
    census_median_income: Optional[float] = Field(None, ge=0, description="ZIP-level census median income")
    census_renter_ratio: Optional[float] = Field(None, ge=0, le=1, description="ZIP-level renter household share")
    census_vacancy_rate: Optional[float] = Field(None, ge=0, le=1, description="ZIP-level vacancy rate")
    crime_total_month_zip_log1p_latest: Optional[float] = Field(
        None, ge=0, description="Log1p monthly crime volume (point-in-time)"
    )
    zori_baseline: Optional[float] = Field(None, ge=0, description="ZIP-level Zillow observed rent index")
    zhvi_level: Optional[float] = Field(None, ge=0, description="ZIP-level Zillow home value index")
    zhvi_12mo_delta: Optional[float] = Field(None, description="ZIP-level 12-month home value delta")
    redfin_mom_pct: Optional[float] = Field(None, description="Metro rent momentum month-over-month")
    redfin_yoy_pct: Optional[float] = Field(None, description="Metro rent momentum year-over-year")
    bedrooms_x_census_income: Optional[float] = Field(
        None, description="Optional precomputed bedrooms x census_median_income"
    )
    walk_score_x_transit_score: Optional[float] = Field(
        None, description="Optional precomputed walk_score x transit_score"
    )


class FlagOverpricedRequest(PredictRequest):
    actual_rent_usd: float = Field(
        ...,
        gt=0,
        le=100_000,
        description="Observed listing rent in USD to compare against prediction",
    )


class PredictResponse(BaseModel):
    predicted_rent_usd: float = Field(..., description="Predicted monthly rent in USD")
    fair_rent_p25: Optional[float] = Field(None, description="25th percentile rent prediction (USD)")
    fair_rent_p75: Optional[float] = Field(None, description="75th percentile rent prediction (USD)")
    model_source: str = Field(..., description="'mlflow' or 'placeholder'")
    model_version: Optional[str] = Field(None, description="MLflow model version if applicable")


class FlagOverpricedResponse(BaseModel):
    predicted_rent_usd: float
    fair_rent_p25: Optional[float] = None
    fair_rent_p75: Optional[float] = None
    delta_usd: Optional[float] = None
    delta_pct: Optional[float] = None
    flag_overpriced: bool
    flag_reason: Optional[str] = None
    top_shap_contributors: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str
    detail: Optional[str] = None
    rank_component_keys: list[str] = Field(
        default_factory=list,
        description="Server-canonical ordering of /rank component scores. Clients use this to render component labels.",
    )


class UserWeights(BaseModel):
    """User preferences for the /rank composite score (any non-negative scale, auto-normalized)."""

    safety: float = Field(1.0, ge=0, le=100)
    walk: float = Field(1.0, ge=0, le=100)
    transit: float = Field(1.0, ge=0, le=100)


class RankRequestListing(FlagOverpricedRequest):
    """A listing payload for ranking. Adds optional id/title/lat-lng for UI."""

    listing_id: Optional[str] = Field(None, description="Stable id for the listing card")
    title: Optional[str] = None
    address: Optional[str] = None
    url: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None


class RankRequest(BaseModel):
    listings: list[RankRequestListing] = Field(..., min_length=1, max_length=1000)
    weights: UserWeights = Field(default_factory=UserWeights)
    top_n: Optional[int] = Field(None, ge=1, le=1000)


class RankedListing(BaseModel):
    listing_id: Optional[str] = None
    title: Optional[str] = None
    address: Optional[str] = None
    url: Optional[str] = None
    zip_code: str
    bedrooms: float
    bathrooms: Optional[float] = None
    lat: Optional[float] = None
    lng: Optional[float] = None

    actual_rent_usd: float
    predicted_rent_usd: float
    fair_rent_p25: Optional[float] = None
    fair_rent_p75: Optional[float] = None
    delta_usd: Optional[float] = None
    delta_pct: Optional[float] = None
    flag_overpriced: bool
    flag_reason: Optional[str] = None

    component_scores: dict[str, float] = Field(
        ..., description="safety, walk, transit — all in [0, 1]"
    )
    composite_score: float = Field(..., description="Weighted sum of component_scores, in [0, 1]")
    rank: int

    @classmethod
    def from_rank_inputs(
        cls,
        src: Mapping[str, Any],
        flag: Mapping[str, Any],
        component_scores: Mapping[str, float],
        composite_score: float,
        rank: int,
    ) -> "RankedListing":
        """Assemble a RankedListing from a source listing and its flag-overpriced result."""
        return cls(
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
            fair_rent_p25=flag.get("fair_rent_p25"),
            fair_rent_p75=flag.get("fair_rent_p75"),
            delta_usd=flag.get("delta_usd"),
            delta_pct=flag.get("delta_pct"),
            flag_overpriced=bool(flag.get("flag_overpriced", False)),
            flag_reason=flag.get("flag_reason"),
            component_scores={k: round(float(v), 4) for k, v in component_scores.items()},
            composite_score=round(float(composite_score), 4),
            rank=rank,
        )


class RankResponse(BaseModel):
    weights_normalized: dict[str, float]
    n_input: int
    n_returned: int
    results: list[RankedListing]
