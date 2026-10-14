"""API configuration, read from environment variables prefixed REVIEW_ANALYSIS_."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REVIEW_ANALYSIS_", env_file=".env", extra="ignore"
    )

    snapshots_dir: Path = ROOT / "data" / "snapshots"
    # Live collection needs a writable disk and minutes of runtime. Serverless hosting (Vercel) has
    # neither, so it serves the committed snapshots with collection disabled.
    allow_collection: bool = True
    max_pages_per_storefront: int = 60  # caps a collection at ~1,200 reviews per storefront
    translate: bool = False  # offline MT for keywords needs the 'translate' extra (5 GB model)
    default_storefronts: str = "us,gb,ca,au,in,ie,nz"
