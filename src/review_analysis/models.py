"""Core data models shared by the collector, the analysis and the API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ReviewSource = Literal["apple_web", "apple_rss"]


class AppInfo(BaseModel):
    """App metadata from the iTunes Lookup API."""

    app_id: int
    name: str
    seller: str
    storefront: str
    listing_avg_rating: float | None = None
    listing_rating_count: int | None = None
    version: str | None = None
    current_version_release_date: datetime | None = None
    genre: str | None = None


class Review(BaseModel):
    """One written App Store review, normalised across sources."""

    model_config = ConfigDict(frozen=True)

    review_id: str
    app_id: int
    storefront: str
    date: datetime
    rating: int = Field(ge=1, le=5)
    title: str
    body: str
    # A pseudonymous hash, never the raw user name. See sources.apple_web.author_hash.
    author_hash: str
    is_edited: bool = False
    developer_response: str | None = None
    source: ReviewSource
    fetched_at: datetime

    @property
    def text(self) -> str:
        """Title and body joined. Titles often carry the verdict ("Heavy", "Unusable on iPad")."""
        if self.title and self.body:
            return f"{self.title}. {self.body}"
        return self.title or self.body


class StorefrontStats(BaseModel):
    """What happened while walking one storefront. Kept for the manifest and the report."""

    storefront: str
    pages_fetched: int = 0
    reviews_seen: int = 0
    reviews_in_window: int = 0
    invalid_skipped: int = 0
    oldest_date_seen: datetime | None = None
    stop_reason: (
        Literal["reached_cutoff", "end_of_reviews", "max_pages", "not_available"] | None
    ) = None
