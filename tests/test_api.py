"""API tests: a real collection job against a simulated App Store, then every endpoint on it."""

import re
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from review_analysis.api.app import create_app
from review_analysis.api.jobs import JobManager
from review_analysis.api.settings import Settings
from review_analysis.http import PoliteClient, RetryPolicy
from review_analysis.sentiment import SentimentResult
from review_analysis.sources import apple_web
from review_analysis.sources.lookup import LOOKUP_URL
from tests.conftest import load_fixture
from tests.test_collector import fake_storefront

APP = 42


class SyncExecutor:
    """Runs submitted work immediately, so a job finishes inside the POST request in tests."""

    def submit(self, fn: Callable[[], None]) -> Future[None]:
        fut: Future[None] = Future()
        fn()
        fut.set_result(None)
        return fut


class FakeRouter:
    def predict(self, texts: list[str], langs: list[str]) -> list[SentimentResult]:
        out = []
        for t in texts:
            neg = "crash" in t or "review 1" in t or "t1" in t
            out.append(
                SentimentResult(
                    label="negative" if neg else "positive",
                    p_negative=0.9 if neg else 0.05,
                    p_neutral=0.05,
                    p_positive=0.05 if neg else 0.9,
                    p_complaint=0.9 if neg else 0.1,
                    complaint=neg,
                    model="fake",
                )
            )
        return out


class FakeClassifier:
    name = "fake-classifier"

    def predict(self, texts: list[str], kinds: list[str]) -> list[list[str]]:
        return [["c_stability"] if k == "complaint" else ["p_versatility"] for k in kinds]


def fast_client() -> PoliteClient:
    return PoliteClient(RetryPolicy(min_interval_s=0, jitter_s=0, backoff_base_s=0))


@pytest.fixture
def api(tmp_path: Path):  # type: ignore[no-untyped-def]
    lookup = load_fixture("apple/lookup_ok.json")
    lookup["results"][0]["trackName"] = "Test App"
    with respx.mock(assert_all_called=False) as mock:
        mock.get(LOOKUP_URL).mock(return_value=httpx.Response(200, json=lookup))
        for cc, n in (("us", 60), ("gb", 15)):
            url = apple_web.BASE_URL.format(cc=cc, app_id=APP)
            mock.get(url__regex=re.escape(url)).mock(side_effect=fake_storefront(cc, n))
        app = create_app(
            Settings(snapshots_dir=tmp_path),
            jobs=JobManager(SyncExecutor()),
            client_factory=fast_client,
            router_factory=FakeRouter,
            classifier_factory=FakeClassifier,
        )
        yield TestClient(app)


def collect(api: TestClient, **body: Any) -> dict[str, Any]:
    r = api.post(
        f"/v1/apps/{APP}/collections",
        json={"count": 30, "storefronts": ["us", "gb"], "window_days": 1000, **body},
    )
    assert r.status_code == 202, r.text
    assert r.headers["location"] == f"/v1/jobs/{r.json()['job_id']}"
    job = api.get(r.headers["location"]).json()
    assert job["state"] == "done", job
    return job


def test_collection_job_end_to_end(api: TestClient):
    job = collect(api)
    assert any("us:" in p for p in job["progress"]) and job["progress"][-1] == "done"
    sample_id, pool_id = job["collection_ids"]
    assert sample_id.endswith(".random_30_s42") and pool_id.endswith(".pool")

    listed = {c["collection_id"]: c for c in api.get("/v1/collections").json()}
    assert listed[sample_id]["n"] == 30 and listed[pool_id]["n"] == 75
    assert listed[sample_id]["analysed"] and listed[sample_id]["app"]["name"] == "Test App"


def test_reviews_download_json_csv_and_filters(api: TestClient):
    sample_id, pool_id = collect(api)["collection_ids"]
    body = api.get(f"/v1/collections/{pool_id}/reviews").json()
    assert body["total"] == 75 and len(body["items"]) == 75
    first = body["items"][0]
    assert {
        "review_id",
        "rating",
        "title",
        "body",
        "sentiment",
        "complaint",
        "lang",
    } <= first.keys()
    assert "user_name" not in first and "userName" not in first  # usernames are never exposed

    only_5 = api.get(f"/v1/collections/{pool_id}/reviews", params={"rating": 5}).json()
    assert only_5["total"] > 0 and all(r["rating"] == 5 for r in only_5["items"])
    page = api.get(f"/v1/collections/{pool_id}/reviews", params={"limit": 10, "offset": 5}).json()
    assert len(page["items"]) == 10 and page["offset"] == 5

    csv_resp = api.get(f"/v1/collections/{sample_id}/reviews", params={"format": "csv"})
    assert csv_resp.status_code == 200 and csv_resp.headers["content-type"].startswith("text/csv")
    assert f'filename="{sample_id}.csv"' in csv_resp.headers["content-disposition"]
    lines = csv_resp.text.strip().splitlines()
    assert lines[0].startswith("review_id,") and len(lines) == 31


def test_metrics_with_finite_population_correction(api: TestClient):
    sample_id, pool_id = collect(api)["collection_ids"]
    m = api.get(f"/v1/collections/{sample_id}/metrics").json()
    assert m["metrics"]["n"] == 30 and m["metrics"]["rating"]["fpc_applied"]
    assert m["representativeness"]["pool_n"] == 75
    assert m["listing"]["listing_rating_count"] > 0
    assert set(m["stars_vs_text"]["counts"]) == {"1", "2", "3", "4", "5"}
    pool = api.get(f"/v1/collections/{pool_id}/metrics").json()
    assert pool["representativeness"] is None and not pool["metrics"]["rating"]["fpc_applied"]


def test_insights_degrade_gracefully_on_small_collections(api: TestClient):
    sample_id, _ = collect(api)["collection_ids"]
    ins = api.get(f"/v1/collections/{sample_id}/insights").json()
    assert ins["sentiment"]["n"] == 30
    assert abs(sum(p["share"] for p in ins["sentiment"]["distribution"].values()) - 1) < 1e-9
    assert ins["drivers"]["available"] is False and "needs" in ins["drivers"]["reason"]
    assert any("Small collection" in n for n in ins["notes"])


def test_themes_only_for_apps_the_taxonomy_covers(api: TestClient, monkeypatch):
    # The test app is not in data/themes.json: no themes, a note, phrase-based recommendations.
    pool_id = collect(api)["collection_ids"][1]
    ins = api.get(f"/v1/collections/{pool_id}/insights").json()
    assert ins["themes"] is None and any("taxonomy" in n for n in ins["notes"])
    assert all(r["source"] == "phrases" for r in ins["recommendations"])

    # Once the taxonomy covers the app, the (fake) classifier tags its sentences.
    monkeypatch.setattr("review_analysis.themes.taxonomy_apps", lambda: {APP})
    pool_id = collect(api)["collection_ids"][1]
    ins = api.get(f"/v1/collections/{pool_id}/insights").json()
    themes = {t["id"]: t for t in ins["themes"]["complaint"]}
    assert themes["c_stability"]["n_reviews"] > 0
    assert ins["recommendations"][0]["theme"] == "c_stability"
    assert ins["recommendations"][0]["source"] == "themes"


def test_report_is_html_with_the_numbers(api: TestClient):
    sample_id, pool_id = collect(api)["collection_ids"]
    for cid in (sample_id, pool_id):
        r = api.get(f"/v1/collections/{cid}/report")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
        assert "Test App" in r.text and 'id="report-data"' in r.text
        assert f"/v1/collections/{cid}/reviews?format=csv" in r.text
    assert api.get("/v1/collections/missing.pool/report").status_code == 404


def test_report_escapes_review_text(api: TestClient, monkeypatch):
    monkeypatch.setattr("review_analysis.themes.taxonomy_apps", lambda: {APP})
    pool_id = collect(api)["collection_ids"][1]
    snap = api.app.state.store.resolve(pool_id)[0]  # type: ignore[attr-defined]
    # plant markup in a theme quote: it must come out escaped
    path = snap / "sentences.jsonl"
    path.write_text(path.read_text().replace("This is the body", "<script>alert(1)</script> body"))
    from review_analysis import analysis

    analysis._theme_files.cache_clear()
    for f in (snap / "cache").glob("*"):
        f.unlink()
    html = api.get(f"/v1/collections/{pool_id}/report").text
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)" in html


def test_compare_is_cached_and_validated(api: TestClient, monkeypatch):
    monkeypatch.setattr("review_analysis.themes.taxonomy_apps", lambda: {APP})
    job = collect(api)
    sample_id, pool_id = job["collection_ids"]
    recent = collect(api, strategy="recent", count=20)["collection_ids"][0]
    # different collection runs are different snapshots: refused
    r = api.get("/v1/compare", params={"base": pool_id, "target": recent})
    assert r.status_code == 422 and r.headers["content-type"] == "application/problem+json"
    r = api.get("/v1/compare", params={"base": sample_id, "target": sample_id})
    assert r.status_code == 409  # nothing in base is older than the target window


def test_root_redirects_to_docs(api: TestClient):
    r = api.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"


def test_collection_needs_the_nlp_extra(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("review_analysis.api.app.nlp_installed", lambda: False)
    api = TestClient(create_app(Settings(snapshots_dir=tmp_path), client_factory=fast_client))
    r = api.post(f"/v1/apps/{APP}/collections", json={})
    assert r.status_code == 503 and "nlp" in r.json()["detail"]


def test_serving_does_not_import_the_nlp_stack():
    """Read-only hosting installs only the core dependencies: importing the API must not need
    spaCy, lingua, onnxruntime or scikit-learn."""
    import subprocess
    import sys

    code = (
        "import sys; import review_analysis.api.app, review_analysis.report; "
        "heavy = {'spacy', 'lingua', 'onnxruntime', 'sklearn', 'tokenizers', 'vaderSentiment'}; "
        "print(sorted(heavy & set(sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_errors_are_problem_json(api: TestClient):
    r = api.get("/v1/collections/missing.pool")
    assert r.status_code == 404 and r.headers["content-type"] == "application/problem+json"
    assert r.json()["title"] == "Collection not found"
    r = api.post(f"/v1/apps/{APP}/collections", json={"count": 0})
    assert r.status_code == 422 and "count" in r.json()["detail"]
    r = api.post(f"/v1/apps/{APP}/collections", json={"storefronts": ["usa"]})
    assert r.status_code == 422
    assert api.get("/v1/jobs/nope").status_code == 404


def test_unknown_app_is_404_before_queueing(tmp_path: Path):
    with respx.mock:
        respx.get(LOOKUP_URL).mock(
            return_value=httpx.Response(200, json=load_fixture("apple/lookup_not_found.json"))
        )
        api = TestClient(create_app(Settings(snapshots_dir=tmp_path), client_factory=fast_client))
        r = api.post("/v1/apps/1/collections", json={})
    assert r.status_code == 404 and r.json()["title"] == "App not found"


def test_collection_can_be_disabled_for_read_only_hosting(tmp_path: Path):
    api = TestClient(create_app(Settings(snapshots_dir=tmp_path, allow_collection=False)))
    r = api.post(f"/v1/apps/{APP}/collections", json={})
    assert r.status_code == 503 and "read-only" in r.json()["detail"]
    assert api.get("/ready").json()["collection_enabled"] is False


def test_empty_window_is_a_valid_collection(api: TestClient):
    """No reviews in the window: the job succeeds, warns, and every endpoint still answers."""
    job = collect(api, window_days=1)  # the fake storefront's newest review is older than a day
    sample_id, pool_id = job["collection_ids"]
    assert any(
        "only 0 of 30" in w for w in api.get(f"/v1/collections/{sample_id}").json()["warnings"]
    )
    for cid in (sample_id, pool_id):
        assert api.get(f"/v1/collections/{cid}/metrics").json()["metrics"]["n"] == 0
        assert api.get(f"/v1/collections/{cid}/insights").status_code == 200
        report = api.get(f"/v1/collections/{cid}/report")
        assert report.status_code == 200 and "no written reviews" in report.text
