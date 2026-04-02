from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Input features for rent prediction (aligned with RentIQ feature engineering)."""

    bedrooms: float = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=20, description="Number of bathrooms")
    sqft: float = Field(..., gt=0, le=50_000, description="Interior square feet")
    walk_score: float = Field(..., ge=0, le=100, description="Walk Score (0–100)")
    transit_score: float = Field(50.0, ge=0, le=100, description="Transit Score (0–100)")
    median_zip_rent: float = Field(
        ...,
        gt=0,
        le=100_000,
        description="Median rent in ZIP from external benchmark (e.g. ZORI), USD/month",
    )


class PredictResponse(BaseModel):
    predicted_rent_usd: float = Field(..., description="Predicted monthly rent in USD")
    model_source: str = Field(..., description="'mlflow' or 'placeholder'")
    model_version: Optional[str] = Field(None, description="MLflow model version if applicable")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str
    detail: Optional[str] = None
