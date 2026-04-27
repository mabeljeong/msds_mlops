from typing import Any, Optional

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
    fair_rent_p10: Optional[float] = Field(None, description="10th percentile rent prediction (USD)")
    fair_rent_p90: Optional[float] = Field(None, description="90th percentile rent prediction (USD)")
    model_source: str = Field(..., description="'mlflow' or 'placeholder'")
    model_version: Optional[str] = Field(None, description="MLflow model version if applicable")


class FlagOverpricedResponse(BaseModel):
    predicted_rent_usd: float
    fair_rent_p10: Optional[float] = None
    fair_rent_p90: Optional[float] = None
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


class UserWeights(BaseModel):
    """User preferences for the /rank composite score (any non-negative scale, auto-normalized)."""

    price_fairness: float = Field(1.0, ge=0, le=100)
    safety: float = Field(1.0, ge=0, le=100)
    walk: float = Field(1.0, ge=0, le=100)
    transit: float = Field(1.0, ge=0, le=100)
    affordability: float = Field(1.0, ge=0, le=100)


class RankRequestListing(FlagOverpricedRequest):
    """A listing payload for ranking. Adds optional id/title/lat-lng for UI."""

    listing_id: Optional[str] = Field(None, description="Stable id for the listing card")
    title: Optional[str] = None
    address: Optional[str] = None
    url: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None


class RankRequest(BaseModel):
    listings: list[RankRequestListing] = Field(..., min_length=1, max_length=200)
    weights: UserWeights = Field(default_factory=UserWeights)
    budget_usd: Optional[float] = Field(
        None, gt=0, le=100_000, description="Optional renter budget; drives the affordability score"
    )
    top_n: Optional[int] = Field(None, ge=1, le=200)


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
    fair_rent_p10: Optional[float] = None
    fair_rent_p90: Optional[float] = None
    delta_usd: Optional[float] = None
    delta_pct: Optional[float] = None
    flag_overpriced: bool
    flag_reason: Optional[str] = None

    component_scores: dict[str, float] = Field(
        ..., description="price_fairness, safety, walk, transit, affordability — all in [0, 1]"
    )
    composite_score: float = Field(..., description="Weighted sum of component_scores, in [0, 1]")
    rank: int


class RankResponse(BaseModel):
    weights_normalized: dict[str, float]
    n_input: int
    n_returned: int
    results: list[RankedListing]
