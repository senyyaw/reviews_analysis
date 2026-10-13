"""iTunes Lookup API: validates an app ID and returns its listing metadata.

Behaviour verified live (2026-10-13):
- unknown ID              -> 200 with resultCount 0
- non-numeric ID / bad cc -> 400 "Invalid value(s) for key(s): [...]"
- app not sold in cc      -> 200 with resultCount 0
"""

import re

from review_analysis.errors import AppNotFoundError, InvalidInputError, UpstreamError
from review_analysis.http import PoliteClient
from review_analysis.models import AppInfo

LOOKUP_URL = "https://itunes.apple.com/lookup"
STOREFRONT_RE = re.compile(r"^[a-z]{2}$")


def validate_app_id(app_id: int | str) -> int:
    text = str(app_id).strip()
    if not text.isdigit() or int(text) <= 0:
        raise InvalidInputError(f"app_id must be a positive integer, got {app_id!r}")
    return int(text)


def validate_storefront(storefront: str) -> str:
    cc = storefront.strip().lower()
    if not STOREFRONT_RE.match(cc):
        raise InvalidInputError(f"storefront must be a 2-letter country code, got {storefront!r}")
    return cc


def lookup_app(client: PoliteClient, app_id: int | str, storefront: str = "us") -> AppInfo:
    app_id = validate_app_id(app_id)
    cc = validate_storefront(storefront)
    response = client.get(LOOKUP_URL, params={"id": app_id, "country": cc})
    if response.status_code == 400:
        raise InvalidInputError(f"Apple rejected the lookup: {response.text[:200]}")
    if response.status_code != 200:
        raise UpstreamError(
            f"lookup returned HTTP {response.status_code}", status_code=response.status_code
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise UpstreamError("lookup returned malformed JSON") from exc
    results = payload.get("results") or []
    if not results:
        raise AppNotFoundError(f"app {app_id} not found in the '{cc}' storefront")
    r = results[0]
    return AppInfo(
        app_id=app_id,
        name=r.get("trackName", ""),
        seller=r.get("sellerName", ""),
        storefront=cc,
        listing_avg_rating=r.get("averageUserRating"),
        listing_rating_count=r.get("userRatingCount"),
        version=r.get("version"),
        current_version_release_date=r.get("currentVersionReleaseDate"),
        genre=r.get("primaryGenreName"),
    )
