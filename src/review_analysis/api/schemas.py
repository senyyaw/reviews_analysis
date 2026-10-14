"""Request/response models. They drive validation and the OpenAPI docs at /docs."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from review_analysis.metrics import Metrics, Proportion, Representativeness
from review_analysis.models import AppInfo
from review_analysis.sources.lookup import validate_storefront


class Problem(BaseModel):
    """RFC 9457 problem details."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None


class CollectRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "strategy": "random",
                    "count": 100,
                    "window_days": 365,
                    "storefronts": ["us", "gb"],
                    "seed": 42,
                },
            ]
        },
    )

    strategy: Literal["random", "recent"] = Field(
        "random",
        description="random: seeded simple random sample of the window's pool; "
        "recent: the newest reviews",
    )
    count: int = Field(100, ge=1, le=1000, description="reviews in the sample")
    window_days: int = Field(365, ge=1, le=1095, description="collect reviews from the last N days")
    storefronts: list[str] = Field(
        default_factory=lambda: ["us", "gb", "ca", "au", "in", "ie", "nz"],
        min_length=1,
        max_length=20,
        description="ISO country codes of App Store storefronts",
    )
    seed: int | None = Field(42, description="random strategy only; makes the sample reproducible")

    @field_validator("storefronts")
    @classmethod
    def _valid_storefronts(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(validate_storefront(s) for s in v))


class CollectionSummary(BaseModel):
    collection_id: str
    snapshot: str
    sample: str = Field(description='"pool" = every review in the window, or a sample name')
    app: AppInfo
    strategy: Literal["pool", "random", "recent"]
    n: int
    pool_size: int
    seed: int | None
    window_start: datetime
    window_end: datetime
    collected_at: datetime
    storefronts: list[str]
    analysed: bool = Field(description="sentiment/insights available")
    warnings: list[str]
    links: dict[str, str]


class SentimentSummary(BaseModel):
    n: int
    distribution: dict[str, Proportion]  # negative / neutral / positive (text, model)
    complaint: Proportion
    model: str = "fine-tuned RoBERTa (en) / XLM-R (other languages), int8 ONNX"


class MetricsResponse(BaseModel):
    collection_id: str
    metrics: Metrics
    listing: dict[str, Any] = Field(
        description="App Store listing rating (all ratings incl. "
        "star-only) vs the written reviews analysed here"
    )
    representativeness: Representativeness | None = Field(
        None, description="random samples only: sample vs the pool it was drawn from"
    )
    stars_vs_text: dict[str, Any] | None = None


class InsightsResponse(BaseModel):
    collection_id: str
    sentiment: SentimentSummary
    most_mentioned: dict[str, Any]
    negative_keywords: dict[str, Any] | None = Field(
        None, description="most common aspect phrases of negative reviews (counts, any size)"
    )
    distinctive_terms: dict[str, Any] = Field(
        description="phrases that set negative reviews apart (Fightin' Words, tested)"
    )
    drivers: dict[str, Any]
    themes: dict[str, Any] | None = Field(
        None,
        description="theme prevalence, rating impact and "
        "quotes (available once themes are assigned)",
    )
    improvement_areas: list[dict[str, Any]] = Field(
        default_factory=list,
        description="biggest problem areas, ranked, with evidence and verbatim quotes",
    )
    notes: list[str] = Field(default_factory=list)
