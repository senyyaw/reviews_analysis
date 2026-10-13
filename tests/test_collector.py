"""Collector tests against a fake storefront: N reviews, one per day, newest first."""

import re
from datetime import UTC, datetime, timedelta

import httpx
import respx

from review_analysis.collector import collect_window
from review_analysis.sources import apple_web

APP = 42
NOW = datetime(2026, 10, 13, tzinfo=UTC)


def fake_storefront(cc: str, n_reviews: int, *, duplicate_at_boundary: bool = False):
    """A respx side effect that serves `n_reviews` reviews dated one day apart, newest first."""

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        limit = int(request.url.params["limit"])
        start = offset
        # Simulate a new review arriving mid-crawl: everything shifts down by one, so the previous
        # page's last item shows up again at the top of this page.
        if duplicate_at_boundary and offset > 0:
            start -= 1
        if start >= n_reviews:
            return httpx.Response(404, json={"errors": [{"code": "40403"}]})
        items = []
        for i in range(start, min(start + limit, n_reviews)):
            items.append(
                {
                    "id": f"{cc}-{i}",
                    "type": "user-reviews",
                    "attributes": {
                        "date": (NOW - timedelta(days=i)).isoformat().replace("+00:00", "Z"),
                        "rating": 1 + i % 5,
                        "title": f"t{i}",
                        "review": f"body {i}",
                        "userName": f"u{i}",
                        "isEdited": False,
                    },
                }
            )
        return httpx.Response(200, json={"data": items})

    return handler


def route_for(cc: str):
    return respx.get(url__regex=re.escape(apple_web.BASE_URL.format(cc=cc, app_id=APP)))


@respx.mock
def test_stops_after_cutoff_plus_overscan(make_client):
    route = route_for("us").mock(side_effect=fake_storefront("us", 1000))
    since = NOW - timedelta(days=50)  # reviews 0..50 are in the window (51 reviews)
    with make_client() as client:
        pool = collect_window(client, APP, ["us"], since, overscan_pages=1)
    assert len(pool.reviews) == 51
    # pages: 0-19, 20-39, 40-59 (crosses the cutoff), 60-79 (overscan) -> stop
    assert route.call_count == 4
    assert pool.storefronts[0].stop_reason == "reached_cutoff"
    assert pool.reviews[0].date > pool.reviews[-1].date  # newest first


@respx.mock
def test_end_of_reviews_on_short_last_page(make_client):
    route_for("ie").mock(side_effect=fake_storefront("ie", 7))
    with make_client() as client:
        pool = collect_window(client, APP, ["ie"], NOW - timedelta(days=365))
    assert len(pool.reviews) == 7
    assert pool.storefronts[0].stop_reason == "end_of_reviews"


@respx.mock
def test_storefront_with_no_reviews(make_client):
    route_for("nz").mock(return_value=httpx.Response(404, json={"errors": [{"code": "40403"}]}))
    with make_client() as client:
        pool = collect_window(client, APP, ["nz"], NOW - timedelta(days=365))
    assert pool.reviews == []
    assert pool.storefronts[0].stop_reason == "end_of_reviews"


@respx.mock
def test_shifted_pages_are_deduplicated(make_client):
    route_for("us").mock(side_effect=fake_storefront("us", 60, duplicate_at_boundary=True))
    with make_client() as client:
        pool = collect_window(client, APP, ["us"], NOW - timedelta(days=365))
    ids = [r.review_id for r in pool.reviews]
    assert len(ids) == len(set(ids)) == 60
    assert pool.duplicates_removed == 1  # one shift -> one repeated item


@respx.mock
def test_multiple_storefronts_are_pooled(make_client):
    route_for("us").mock(side_effect=fake_storefront("us", 30))
    route_for("gb").mock(side_effect=fake_storefront("gb", 10))
    with make_client() as client:
        pool = collect_window(client, APP, ["us", "gb"], NOW - timedelta(days=365))
    assert len(pool.reviews) == 40
    assert {s.storefront: s.reviews_in_window for s in pool.storefronts} == {"us": 30, "gb": 10}


@respx.mock
def test_resume_uses_cached_pages(make_client, tmp_path):
    route = route_for("us").mock(side_effect=fake_storefront("us", 45))
    since = NOW - timedelta(days=365)
    with make_client() as client:
        first = collect_window(client, APP, ["us"], since, raw_dir=tmp_path)
    calls_first_run = route.call_count
    with make_client() as client:
        second = collect_window(client, APP, ["us"], since, raw_dir=tmp_path)
    assert route.call_count == calls_first_run  # nothing re-fetched
    assert [r.review_id for r in first.reviews] == [r.review_id for r in second.reviews]
