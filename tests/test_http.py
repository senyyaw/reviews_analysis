import httpx
import pytest
import respx

from review_analysis.errors import BudgetExceededError, UpstreamError

URL = "https://example.test/x"


@respx.mock
def test_retries_429_with_exponential_backoff_then_succeeds(make_client, clock):
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(429), httpx.Response(200, json={"ok": 1})]
    )
    with make_client() as client:
        response = client.get(URL)
    assert response.status_code == 200
    assert route.call_count == 3
    assert clock.sleeps == [5.0, 10.0]
    assert client.retries_made == 2


@respx.mock
def test_gives_up_after_max_retries(make_client):
    respx.get(URL).mock(return_value=httpx.Response(429, text="API capacity exceeded"))
    with make_client(max_retries=2) as client, pytest.raises(UpstreamError) as err:
        client.get(URL)
    assert err.value.status_code == 429
    assert client.requests_made == 3


@respx.mock
def test_client_errors_are_returned_not_retried(make_client):
    route = respx.get(URL).mock(return_value=httpx.Response(404))
    with make_client() as client:
        assert client.get(URL).status_code == 404
    assert route.call_count == 1


@respx.mock
def test_transport_errors_are_retried(make_client):
    respx.get(URL).mock(side_effect=[httpx.ConnectError("boom"), httpx.Response(200)])
    with make_client() as client:
        assert client.get(URL).status_code == 200


@respx.mock
def test_min_interval_spaces_requests(make_client, clock):
    respx.get(URL).mock(return_value=httpx.Response(200))
    with make_client(min_interval_s=3.0) as client:
        client.get(URL)
        client.get(URL)
    assert clock.sleeps == [3.0]


@respx.mock
def test_budget_stops_sustained_throttling(make_client):
    respx.get(URL).mock(return_value=httpx.Response(429))
    with make_client(budget_s=12.0) as client, pytest.raises(BudgetExceededError):
        client.get(URL)  # backoffs 5 + 10 > 12
