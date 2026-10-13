"""Download every written review inside a time window, storefront by storefront.

`sort=recent` returns reviews newest first, so we page forward until a page reaches past the
cutoff date. We then fetch `overscan_pages` more, because edited reviews may be slightly out of
order. Raw pages are cached on disk as they arrive, so an interrupted run resumes instead of
starting over.
"""

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from review_analysis.http import PoliteClient
from review_analysis.models import Review, StorefrontStats
from review_analysis.sources import apple_web

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]  # storefront, pages fetched, reviews in window


@dataclass
class PoolResult:
    app_id: int
    since: datetime
    collected_at: datetime
    reviews: list[Review]  # de-duplicated, newest first
    storefronts: list[StorefrontStats]
    duplicates_removed: int = 0
    warnings: list[str] = field(default_factory=list)


def collect_window(
    client: PoliteClient,
    app_id: int,
    storefronts: Iterable[str],
    since: datetime,
    *,
    raw_dir: Path | None = None,
    overscan_pages: int = 1,
    max_pages_per_storefront: int = 500,
    progress: ProgressFn | None = None,
) -> PoolResult:
    if since.tzinfo is None:
        raise ValueError("since must be timezone-aware")
    by_id: dict[str, Review] = {}
    duplicates = 0
    all_stats: list[StorefrontStats] = []
    warnings: list[str] = []

    for cc in storefronts:
        stats = StorefrontStats(storefront=cc)
        pages_past_cutoff = 0
        offset = 0
        while True:
            if stats.pages_fetched >= max_pages_per_storefront:
                stats.stop_reason = "max_pages"
                warnings.append(f"{cc}: stopped at max_pages={max_pages_per_storefront}")
                break
            page = _load_or_fetch(client, app_id, cc, offset, raw_dir)
            stats.pages_fetched += 1
            if page.end_of_reviews or page.items_seen == 0:
                stats.stop_reason = "end_of_reviews"
                break

            stats.reviews_seen += page.items_seen
            stats.invalid_skipped += page.invalid
            for review in page.reviews:
                if review.date < since:
                    continue
                if review.review_id in by_id:
                    duplicates += 1
                    continue
                by_id[review.review_id] = review
                stats.reviews_in_window += 1

            if page.reviews:
                oldest = min(r.date for r in page.reviews)
                stats.oldest_date_seen = min(oldest, stats.oldest_date_seen or oldest)
                if oldest < since:
                    pages_past_cutoff += 1
                    if pages_past_cutoff > overscan_pages:
                        stats.stop_reason = "reached_cutoff"
                        break
            if page.items_seen < apple_web.PAGE_SIZE:
                stats.stop_reason = "end_of_reviews"
                break
            offset += apple_web.PAGE_SIZE
            if progress:
                progress(cc, stats.pages_fetched, stats.reviews_in_window)

        log.info(
            "%s: %d in window from %d pages (%s)",
            cc,
            stats.reviews_in_window,
            stats.pages_fetched,
            stats.stop_reason,
        )
        if stats.invalid_skipped:
            warnings.append(f"{cc}: skipped {stats.invalid_skipped} malformed review(s)")
        all_stats.append(stats)

    reviews = sorted(by_id.values(), key=lambda r: (r.date, r.review_id), reverse=True)
    return PoolResult(
        app_id=app_id,
        since=since,
        collected_at=datetime.now(UTC),
        reviews=reviews,
        storefronts=all_stats,
        duplicates_removed=duplicates,
        warnings=warnings,
    )


def _load_or_fetch(
    client: PoliteClient, app_id: int, cc: str, offset: int, raw_dir: Path | None
) -> apple_web.Page:
    """Use the cached raw page if this run already fetched it; otherwise fetch and cache it."""
    cache = raw_dir / cc / f"offset_{offset:06d}.json" if raw_dir else None
    if cache and cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("end_of_reviews"):
            return apple_web.Page(offset=offset, end_of_reviews=True)
        fetched_at = datetime.fromisoformat(cached["fetched_at"])
        return apple_web.parse_page(
            cached["payload"], app_id=app_id, storefront=cc, offset=offset, fetched_at=fetched_at
        )

    page = apple_web.fetch_page(client, app_id, cc, offset)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        record = (
            {"end_of_reviews": True}
            if page.end_of_reviews
            else {"fetched_at": datetime.now(UTC).isoformat(), "payload": page.raw}
        )
        cache.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return page
