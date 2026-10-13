from datetime import UTC, datetime

import httpx
import pytest
import respx

from review_analysis.errors import AppNotFoundError, InvalidInputError, UpstreamError
from review_analysis.sources import apple_web
from review_analysis.sources.lookup import LOOKUP_URL, lookup_app, validate_storefront
from tests.conftest import load_fixture

NOTION = 1232780281
REVIEWS_US = apple_web.BASE_URL.format(cc="us", app_id=NOTION)


# --- lookup -----------------------------------------------------------------------------------


@respx.mock
def test_lookup_parses_listing(make_client):
    respx.get(LOOKUP_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("apple/lookup_ok.json"))
    )
    with make_client() as client:
        app = lookup_app(client, NOTION, "us")
    assert app.name == "Notion: Notes, Tasks, AI"
    assert app.listing_rating_count == 90377
    assert 4.7 < (app.listing_avg_rating or 0) < 4.8


@respx.mock
def test_lookup_unknown_app_is_not_found(make_client):
    respx.get(LOOKUP_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("apple/lookup_not_found.json"))
    )
    with make_client() as client, pytest.raises(AppNotFoundError):
        lookup_app(client, 1, "us")


@pytest.mark.parametrize("bad", ["abc", "-5", "0", "12a", ""])
def test_lookup_rejects_bad_app_id_before_calling_apple(make_client, bad):
    with make_client() as client, pytest.raises(InvalidInputError):
        lookup_app(client, bad, "us")
    assert client.requests_made == 0


@pytest.mark.parametrize("bad", ["usa", "u", "1a", ""])
def test_storefront_validation(bad):
    with pytest.raises(InvalidInputError):
        validate_storefront(bad)
    assert validate_storefront(" GB ") == "gb"


# --- web reviews endpoint ---------------------------------------------------------------------


def test_parse_real_page():
    page = apple_web.parse_page(
        load_fixture("apple/web_page_us_0.json"),
        app_id=NOTION,
        storefront="us",
        offset=0,
        fetched_at=datetime.now(UTC),
    )
    assert len(page.reviews) == 20 and page.invalid == 0
    first = page.reviews[0]
    assert first.review_id == "14656572111"
    assert first.title == "Heavy" and first.body == "680mb is crazy" and first.rating == 1
    assert first.date.tzinfo is not None
    assert first.text == "Heavy. 680mb is crazy"
    assert len(first.author_hash) == 16 and "user" not in first.author_hash


def test_parse_counts_malformed_items_instead_of_failing():
    payload = {
        "data": [
            {
                "id": "1",
                "attributes": {
                    "date": "2026-10-01T00:00:00Z",
                    "rating": 5,
                    "title": "ok",
                    "review": "fine",
                    "userName": "a",
                },
            },
            {"id": "2", "attributes": {"date": "2026-10-01T00:00:00Z", "rating": 9}},  # bad rating
            {"id": "3"},  # no attributes
        ]
    }
    page = apple_web.parse_page(
        payload, app_id=1, storefront="us", offset=0, fetched_at=datetime.now(UTC)
    )
    assert len(page.reviews) == 1 and page.invalid == 2


@respx.mock
def test_past_end_is_end_of_reviews(make_client):
    respx.get(REVIEWS_US).mock(
        return_value=httpx.Response(404, json=load_fixture("apple/web_past_end.json"))
    )
    with make_client() as client:
        page = apple_web.fetch_page(client, NOTION, "us", 999_999)
    assert page.end_of_reviews


@respx.mock
def test_unknown_storefront_is_invalid_input(make_client):
    url = apple_web.BASE_URL.format(cc="zz", app_id=NOTION)
    respx.get(url).mock(
        return_value=httpx.Response(400, json=load_fixture("apple/web_bad_storefront.json"))
    )
    with make_client() as client, pytest.raises(InvalidInputError):
        apple_web.fetch_page(client, NOTION, "zz", 0)


@respx.mock
def test_persistent_server_error_is_upstream_error(make_client):
    respx.get(REVIEWS_US).mock(return_value=httpx.Response(503))
    with make_client(max_retries=1) as client, pytest.raises(UpstreamError):
        apple_web.fetch_page(client, NOTION, "us", 0)


@respx.mock
def test_html_error_page_is_upstream_error(make_client):
    respx.get(REVIEWS_US).mock(return_value=httpx.Response(200, text="<html>oops</html>"))
    with make_client() as client, pytest.raises(UpstreamError):
        apple_web.fetch_page(client, NOTION, "us", 0)
