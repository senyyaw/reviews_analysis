"""Apple's web reviews endpoint, the one apps.apple.com itself uses.

    GET https://apps.apple.com/api/apps/v1/catalog/{cc}/apps/{id}/reviews
        ?platform=web&limit=20&offset=N&sort=recent

Verified live (2026-10-13):
- no token needed; `platform` is required; `limit` is at most 20; any offset works;
- `sort=recent` returns newest first;
- past the last review          -> 404, errors[0].code == "40403"
- unknown storefront            -> 400, errors[0].code == "40009"
- throttling                    -> 429 "API capacity exceeded" (retried by PoliteClient)
- fields: date, rating, title, review, userName, isEdited, developerResponse{body}? (no app version)

The endpoint is undocumented and robots.txt disallows /api/*. We access it politely and say so in
the README; the committed snapshot means the project never depends on it at review time.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from review_analysis.errors import InvalidInputError, UpstreamError
from review_analysis.http import PoliteClient
from review_analysis.models import Review

BASE_URL = "https://apps.apple.com/api/apps/v1/catalog/{cc}/apps/{app_id}/reviews"
PAGE_SIZE = 20
END_OF_REVIEWS_CODE = "40403"
UNKNOWN_STOREFRONT_CODE = "40009"


@dataclass
class Page:
    """One fetched page. `raw` is kept so the snapshot holds exactly what Apple returned."""

    offset: int
    reviews: list[Review] = field(default_factory=list)
    invalid: int = 0
    end_of_reviews: bool = False
    raw: dict[str, Any] | None = None

    @property
    def items_seen(self) -> int:
        return len(self.reviews) + self.invalid


def author_hash(user_name: str) -> str:
    """Pseudonymise the public user name: stable for de-duplication, never the raw handle."""
    return hashlib.sha256(user_name.encode("utf-8")).hexdigest()[:16]


def _error_code(payload: Any) -> str | None:
    try:
        return str(payload["errors"][0]["code"])
    except (KeyError, IndexError, TypeError):
        return None


def parse_page(
    payload: dict[str, Any], *, app_id: int, storefront: str, offset: int, fetched_at: datetime
) -> Page:
    """Turn a JSON page into validated Reviews. Malformed items are counted, not fatal."""
    page = Page(offset=offset, raw=payload)
    for item in payload.get("data") or []:
        try:
            attrs = item["attributes"]
            dev = attrs.get("developerResponse") or {}
            page.reviews.append(
                Review(
                    review_id=str(item["id"]),
                    app_id=app_id,
                    storefront=storefront,
                    date=attrs["date"],
                    rating=attrs["rating"],
                    title=(attrs.get("title") or "").strip(),
                    body=(attrs.get("review") or "").strip(),
                    author_hash=author_hash(attrs.get("userName") or ""),
                    is_edited=bool(attrs.get("isEdited", False)),
                    developer_response=dev.get("body"),
                    source="apple_web",
                    fetched_at=fetched_at,
                )
            )
        except (KeyError, TypeError, ValidationError):
            page.invalid += 1
    return page


def fetch_page(client: PoliteClient, app_id: int, storefront: str, offset: int) -> Page:
    url = BASE_URL.format(cc=storefront, app_id=app_id)
    params: dict[str, str | int] = {
        "platform": "web",
        "limit": PAGE_SIZE,
        "offset": offset,
        "sort": "recent",
    }
    response = client.get(url, params=params)
    fetched_at = datetime.now(UTC)

    if response.status_code == 200:
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(f"malformed JSON at {storefront} offset {offset}") from exc
        return parse_page(
            payload, app_id=app_id, storefront=storefront, offset=offset, fetched_at=fetched_at
        )

    code = _error_code(_safe_json(response))
    if response.status_code == 404 and code == END_OF_REVIEWS_CODE:
        return Page(offset=offset, end_of_reviews=True)
    if response.status_code == 400 and code == UNKNOWN_STOREFRONT_CODE:
        raise InvalidInputError(f"unknown storefront '{storefront}'")
    raise UpstreamError(
        f"reviews endpoint returned HTTP {response.status_code} (code {code}) "
        f"at {storefront} offset {offset}",
        status_code=response.status_code,
    )


def _safe_json(response: Any) -> Any:
    try:
        return response.json()
    except ValueError:
        return None
