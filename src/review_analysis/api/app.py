"""FastAPI application.

    uv run review-analysis serve            # http://127.0.0.1:8000/docs

Endpoints (all JSON unless noted):
  GET  /health, /ready
  GET  /v1/apps/{app_id}                          App Store lookup (validates the id)
  POST /v1/apps/{app_id}/collections              start a collection job -> 202 + job
  GET  /v1/jobs/{job_id}                          job progress / result
  GET  /v1/collections                            all collections
  GET  /v1/collections/{collection_id}            one collection (parameters, sizes, links)
  GET  /v1/collections/{collection_id}/reviews    raw reviews, ?format=json|csv (download)
  GET  /v1/collections/{collection_id}/metrics    rating metrics with confidence intervals
  GET  /v1/collections/{collection_id}/insights   sentiment, topics, phrases, drivers, themes
  GET  /v1/collections/{collection_id}/report     the same as an HTML report with charts
  GET  /v1/compare?base=&target=                  emerging / fading themes and phrases
Errors are RFC 9457 problem+json.

Serving analysed snapshots needs only the core dependencies; results are computed once and cached
next to the snapshot. Collecting and analysing new reviews needs the `nlp` extra (spaCy, lingua,
ONNX models). Heavy modules are therefore imported inside the endpoints that use them.
"""

import csv
import importlib.util
import io
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from review_analysis import __version__
from review_analysis.api.jobs import Job, JobManager
from review_analysis.api.schemas import (
    CollectionSummary,
    CollectRequest,
    InsightsResponse,
    MetricsResponse,
    Problem,
    SentimentSummary,
)
from review_analysis.api.settings import Settings
from review_analysis.api.store import CollectionNotFoundError, CollectionStore, read_jsonl
from review_analysis.errors import (
    AppNotFoundError,
    InvalidInputError,
    ReviewAnalysisError,
    UpstreamError,
)
from review_analysis.http import PoliteClient
from review_analysis.metrics import compute_metrics, representativeness, wilson
from review_analysis.snapshot import load_pool, load_sample
from review_analysis.sources.lookup import lookup_app

AppId = Annotated[int, Path(gt=0, description="App Store id, e.g. 1232780281 (Notion)")]
CollectionId = Annotated[str, Path(description='e.g. "notion_2026-10-13.A_random"')]
PROBLEM = {"model": Problem, "content": {"application/problem+json": {}}}

CSV_FIELDS = [
    "review_id",
    "storefront",
    "date",
    "rating",
    "title",
    "body",
    "is_edited",
    "developer_response",
    "author_hash",
    "lang",
    "translated",
    "text_en",
    "sentiment",
    "p_negative",
    "p_neutral",
    "p_positive",
    "complaint",
    "p_complaint",
]


NLP_MODULES = ("spacy", "lingua", "onnxruntime", "sklearn", "tokenizers", "vaderSentiment")


def nlp_installed() -> bool:
    """True when the `nlp` extra is installed (needed to collect and analyse new reviews)."""
    return all(importlib.util.find_spec(m) is not None for m in NLP_MODULES)


class NotAvailableError(Exception):
    """A result that cannot be produced for this collection (maps to 409)."""


def _problem(status: int, title: str, detail: str, request: Request) -> JSONResponse:
    body = Problem(title=title, status=status, detail=detail, instance=str(request.url.path))
    return JSONResponse(
        body.model_dump(), status_code=status, media_type="application/problem+json"
    )


def create_app(
    settings: Settings | None = None,
    *,
    jobs: JobManager | None = None,
    client_factory: Callable[[], PoliteClient] = PoliteClient,
    router_factory: Callable[[], Any] | None = None,
    classifier_factory: Callable[[], Any] | None = None,
) -> FastAPI:
    settings = settings or Settings()
    store = CollectionStore(settings.snapshots_dir)
    job_manager = jobs or JobManager()
    state: dict[str, Any] = {"router": None, "classifier": None}

    def router() -> Any:
        if state["router"] is None:
            if router_factory is not None:
                state["router"] = router_factory()
            else:
                from review_analysis.sentiment import SentimentRouter

                state["router"] = SentimentRouter()
        return state["router"]

    def classifier() -> Any:
        if state["classifier"] is None:
            if classifier_factory is not None:
                state["classifier"] = classifier_factory()
            else:
                from review_analysis.theme_classifier import ThemeClassifier

                state["classifier"] = ThemeClassifier()
        return state["classifier"]

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
        yield

    app = FastAPI(
        title="App Store Review Analysis API",
        version=__version__,
        description="Collect Apple App Store reviews for any app, then get metrics with "
        "confidence intervals, sentiment and complaint detection (fine-tuned models), topics, "
        "distinctive phrases, rating drivers and themes. Raw data downloads as JSON or CSV.",
        lifespan=lifespan,
    )

    # --- errors -> RFC 9457 ----------------------------------------------------------------
    @app.exception_handler(CollectionNotFoundError)
    async def _nf(request: Request, exc: CollectionNotFoundError) -> JSONResponse:
        return _problem(404, "Collection not found", str(exc), request)

    @app.exception_handler(AppNotFoundError)
    async def _app_nf(request: Request, exc: AppNotFoundError) -> JSONResponse:
        return _problem(404, "App not found", str(exc), request)

    @app.exception_handler(InvalidInputError)
    async def _bad(request: Request, exc: InvalidInputError) -> JSONResponse:
        return _problem(422, "Invalid input", str(exc), request)

    @app.exception_handler(UpstreamError)
    async def _up(request: Request, exc: UpstreamError) -> JSONResponse:
        return _problem(502, "App Store unavailable", str(exc), request)

    @app.exception_handler(ReviewAnalysisError)
    async def _other(request: Request, exc: ReviewAnalysisError) -> JSONResponse:
        return _problem(400, "Request failed", str(exc), request)

    @app.exception_handler(NotAvailableError)
    async def _na(request: Request, exc: NotAvailableError) -> JSONResponse:
        return _problem(409, "Not available", str(exc), request)

    @app.exception_handler(ModuleNotFoundError)
    async def _no_nlp(request: Request, exc: ModuleNotFoundError) -> JSONResponse:
        return _problem(
            503,
            "Analysis stack not installed",
            f"This result is not cached yet and computing it needs the 'nlp' extra "
            f"(missing module: {exc.name}). Install with: uv sync --extra nlp",
            request,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return _problem(422, "Invalid input", detail, request)

    # --- health ------------------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready", tags=["meta"])
    def ready() -> dict[str, Any]:
        return {
            "status": "ok",
            "collections": len(store.collection_ids()),
            "collection_enabled": settings.allow_collection,
        }

    # --- apps & collection jobs ---------------------------------------------------------------
    @app.get("/v1/apps/{app_id}", tags=["collect"], responses={404: PROBLEM, 502: PROBLEM})
    def get_app(app_id: AppId, storefront: str = "us") -> dict[str, Any]:
        with client_factory() as client:
            return lookup_app(client, app_id, storefront).model_dump()

    @app.post(
        "/v1/apps/{app_id}/collections",
        tags=["collect"],
        status_code=202,
        response_model=Job,
        responses={404: PROBLEM, 422: PROBLEM, 503: PROBLEM},
    )
    def start_collection(app_id: AppId, body: CollectRequest, request: Request) -> Any:
        """Collect reviews for an app (background job). Downloads every written review in the
        window from the chosen storefronts, draws the sample, runs preprocessing and sentiment.
        Poll the returned job; when done it lists the new collection ids."""
        if not settings.allow_collection:
            return _problem(
                503,
                "Collection disabled",
                "This deployment serves pre-collected "
                "snapshots only (read-only hosting). Run the API locally or in "
                "Docker to collect new reviews.",
                request,
            )
        if router_factory is None and not nlp_installed():
            return _problem(
                503,
                "Analysis stack not installed",
                "Collecting needs the 'nlp' extra (language ID, spaCy, ONNX models): "
                "uv sync --extra nlp",
                request,
            )
        with client_factory() as client:  # fail fast on an unknown app, before queueing
            app_info = lookup_app(client, app_id, body.storefronts[0])
        from review_analysis import pipeline

        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        # unique per job: two requests in the same second must not share a snapshot directory
        snap = settings.snapshots_dir / f"app{app_id}_{stamp}-{uuid.uuid4().hex[:6]}"
        samples = [(body.strategy, body.count, body.seed)]

        def work(progress: Callable[[str], None]) -> list[str]:
            with client_factory() as client:
                _, defs = pipeline.collect_snapshot(
                    snap,
                    app_id,
                    body.storefronts,
                    body.window_days,
                    samples,
                    max_pages=settings.max_pages_per_storefront,
                    client=client,
                    progress=progress,
                )
            progress("preprocessing (cleaning, language, lemmas)")
            pipeline.preprocess_snapshot(snap, translate=settings.translate)
            progress("sentiment + complaint detection")
            pipeline.sentiment_snapshot(snap, router())
            progress("themes (sentence sentiment + theme classifier)")
            tagged = pipeline.themes_snapshot(snap, router(), classifier())
            if tagged.get("skipped"):
                progress("themes skipped: the taxonomy covers other apps; phrase insights only")
            progress("done")
            return [f"{snap.name}.{d.name}" for d in defs] + [f"{snap.name}.pool"]

        job = job_manager.submit(app_id, {"app": app_info.name, **body.model_dump()}, work)
        return JSONResponse(
            job.model_dump(mode="json"),
            status_code=202,
            headers={"Location": f"/v1/jobs/{job.job_id}"},
        )

    @app.get("/v1/jobs/{job_id}", tags=["collect"], response_model=Job, responses={404: PROBLEM})
    def get_job(job_id: str, request: Request) -> Any:
        job = job_manager.get(job_id)
        if job is None:
            return _problem(404, "Job not found", f"job '{job_id}' not found", request)
        return job

    # --- collections --------------------------------------------------------------------------
    def summary(collection_id: str) -> CollectionSummary:
        snap, sample = store.resolve(collection_id)
        man = store.manifest(snap)
        definition = store.sample_def(snap, sample)
        base = f"/v1/collections/{collection_id}"
        return CollectionSummary(
            collection_id=collection_id,
            snapshot=snap.name,
            sample=sample,
            app=man.app,
            strategy=definition.strategy if definition else "pool",
            n=definition.n if definition else man.pool_size,
            pool_size=man.pool_size,
            seed=definition.seed if definition else None,
            window_start=definition.window_start if definition else man.since,
            window_end=definition.window_end if definition else man.collected_at,
            collected_at=man.collected_at,
            storefronts=[s.storefront for s in man.storefronts],
            analysed=store.analysed(snap),
            warnings=man.warnings,
            links={
                "self": base,
                "reviews": f"{base}/reviews",
                "csv": f"{base}/reviews?format=csv",
                "metrics": f"{base}/metrics",
                "insights": f"{base}/insights",
                "report": f"{base}/report",
            },
        )

    @app.get("/v1/collections", tags=["collections"], response_model=list[CollectionSummary])
    def list_collections() -> list[CollectionSummary]:
        return [summary(c) for c in store.collection_ids()]

    @app.get(
        "/v1/collections/{collection_id}",
        tags=["collections"],
        response_model=CollectionSummary,
        responses={404: PROBLEM},
    )
    def get_collection(collection_id: CollectionId) -> CollectionSummary:
        return summary(collection_id)

    @app.get(
        "/v1/collections/{collection_id}/reviews",
        tags=["collections"],
        responses={404: PROBLEM, 200: {"content": {"text/csv": {}}}},
    )
    def get_reviews(
        collection_id: CollectionId,
        format: Literal["json", "csv"] = "json",
        include_analysis: bool = Query(
            True, description="add language, translation, sentiment and complaint columns"
        ),
        rating: Annotated[list[int] | None, Query()] = None,
        sentiment: Literal["negative", "neutral", "positive"] | None = None,
        lang: str | None = None,
        limit: int = Query(1000, ge=1, le=5000),
        offset: int = Query(0, ge=0),
    ) -> Any:
        """Raw review data for download. CSV is streamed with a filename."""
        snap, sample = store.resolve(collection_id)
        rows = _review_rows(snap, sample, include_analysis and store.analysed(snap))
        if rating:
            rows = [r for r in rows if r["rating"] in rating]
        if sentiment:
            rows = [r for r in rows if r.get("sentiment") == sentiment]
        if lang:
            rows = [r for r in rows if r.get("lang") == lang]
        total = len(rows)
        page = rows[offset : offset + limit]
        if format == "csv":
            fields = [f for f in CSV_FIELDS if any(f in r for r in page)] or CSV_FIELDS[:9]
            return StreamingResponse(
                _csv(page, fields),
                media_type="text/csv; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="{collection_id}.csv"',
                    "X-Total-Count": str(total),
                },
            )
        return {
            "collection_id": collection_id,
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": page,
        }

    def _review_rows(snap: Any, sample: str, with_analysis: bool) -> list[dict[str, Any]]:
        reviews = load_pool(snap) if sample == "pool" else load_sample(snap, sample)[1]
        processed = read_jsonl(snap / "processed.jsonl") if with_analysis else {}
        scores = read_jsonl(snap / "sentiment.jsonl") if with_analysis else {}
        rows = []
        for r in reviews:
            row: dict[str, Any] = r.model_dump(
                mode="json", exclude={"source", "fetched_at", "app_id"}
            )
            if with_analysis:
                p, s = processed[r.review_id], scores[r.review_id]
                row |= {
                    "lang": p["lang"],
                    "translated": p["translated"],
                    "text_en": p["text_en"] if p["translated"] else None,
                    "sentiment": s["label"],
                    "p_negative": round(s["p_negative"], 4),
                    "p_neutral": round(s["p_neutral"], 4),
                    "p_positive": round(s["p_positive"], 4),
                    "complaint": s["complaint"],
                    "p_complaint": round(s["p_complaint"], 4),
                }
            rows.append(row)
        return rows

    def metrics_of(collection_id: str) -> dict[str, Any]:
        snap, sample = store.resolve(collection_id)
        man = store.manifest(snap)

        def compute() -> dict[str, Any]:
            pool = load_pool(snap)
            reviews = pool if sample == "pool" else load_sample(snap, sample)[1]
            definition = store.sample_def(snap, sample)
            is_random = definition is not None and definition.strategy == "random"
            m = compute_metrics(reviews, population_size=len(pool) if is_random else None)
            out: dict[str, Any] = {
                "collection_id": collection_id,
                "metrics": m.model_dump(mode="json"),
                "listing": {
                    "listing_avg_rating": man.app.listing_avg_rating,
                    "listing_rating_count": man.app.listing_rating_count,
                    "written_reviews_mean": m.rating.mean if m.rating else None,
                    "note": "The listing average includes star-only ratings; written "
                    "reviews skew negative (people with a problem write).",
                },
                "representativeness": representativeness(reviews, pool).model_dump()
                if is_random and len(reviews) >= 2
                else None,
            }
            if store.analysed(snap):
                from review_analysis.metrics import stars_vs_text

                out["stars_vs_text"] = stars_vs_text(reviews, read_jsonl(snap / "sentiment.jsonl"))
            return out

        return dict(store.cached(snap, sample, "metrics", compute))

    @app.get(
        "/v1/collections/{collection_id}/metrics",
        tags=["analysis"],
        response_model=MetricsResponse,
        responses={404: PROBLEM},
    )
    def get_metrics(collection_id: CollectionId) -> MetricsResponse:
        """Average rating with 95% CI (finite-population corrected for random samples), star
        distribution with Wilson CIs, breakdowns by storefront and month, the listing-vs-written
        gap, and, for analysed snapshots, how star ratings relate to the text."""
        return MetricsResponse.model_validate(metrics_of(collection_id))

    def insights_of(collection_id: str) -> dict[str, Any]:
        snap, sample = store.resolve(collection_id)
        if not store.analysed(snap):
            raise NotAvailableError(
                "Not analysed yet: run preprocessing and sentiment for this snapshot first "
                "(the collection job does this)."
            )

        def compute() -> dict[str, Any]:
            from review_analysis import analysis

            data = analysis.load(snap)
            ids = data.ids(sample)
            labels = Counter(data.sentiment[r]["label"] for r in ids)
            complaints = sum(bool(data.sentiment[r]["complaint"]) for r in ids)
            n = len(ids)
            notes = []
            if n < 300:
                notes.append(
                    f"Small collection (n={n}): phrase and driver statistics need a "
                    "few hundred reviews; use the pool collection for those."
                )
            theme_stats = analysis.theme_insights(data, ids)
            keyword_block = analysis.keyword_insights(data, ids)
            negative_block = analysis.negative_keywords(data, ids)
            driver_block = analysis.driver_insights(data, ids)
            areas = analysis.improvement_areas(theme_stats)
            if not theme_stats:
                notes.append(
                    "No themes for this app: the theme taxonomy (data/themes.json) is built per "
                    "app and covers Notion. Problem areas come from the phrases that set "
                    "negative reviews apart; build a taxonomy to get named themes (README)."
                )
                areas = analysis.phrase_improvement_areas(
                    data, ids, keyword_block, driver_block, negative_block
                )
            return {
                "collection_id": collection_id,
                "sentiment": SentimentSummary(
                    n=n,
                    distribution={
                        k: wilson(labels[k], n) for k in ("negative", "neutral", "positive")
                    },
                    complaint=wilson(complaints, n),
                ).model_dump(),
                "most_mentioned": analysis.most_mentioned(
                    {r: data.terms[r] for r in ids if r in data.terms}, data.sentiment
                ),
                "negative_keywords": negative_block,
                "distinctive_terms": keyword_block,
                "drivers": driver_block,
                "themes": theme_stats,
                "improvement_areas": areas,
                "notes": notes,
            }

        return dict(store.cached(snap, sample, "insights", compute))

    @app.get(
        "/v1/collections/{collection_id}/insights",
        tags=["analysis"],
        response_model=InsightsResponse,
        responses={404: PROBLEM, 409: PROBLEM},
    )
    def get_insights(collection_id: CollectionId) -> Any:
        """NLP insights: sentiment and complaint shares (fine-tuned models), most-mentioned
        product aspects with tone, distinctive phrases of negative vs other reviews (Fightin'
        Words), rating drivers (logistic regression, bootstrap-stable), themes, and the top
        improvement areas (complaint themes ranked by prevalence x rating drag, with quotes)."""
        return InsightsResponse.model_validate(insights_of(collection_id))

    def comparison_of(base: str, target: str) -> dict[str, Any]:
        snap_a, sample_a = store.resolve(base)
        snap_b, sample_b = store.resolve(target)
        if snap_a != snap_b:
            raise InvalidInputError(
                "base and target must come from the same snapshot (same app and collection run)"
            )
        if not store.analysed(snap_a):
            raise NotAvailableError("this snapshot is not analysed yet")

        def compute() -> dict[str, Any]:
            from review_analysis import analysis

            data = analysis.load(snap_a)
            dates = {r.review_id: r.date for r in data.pool}
            target_ids = data.ids(sample_b)
            start = min(dates[r] for r in target_ids)
            in_target = set(target_ids)
            base_ids = [r for r in data.ids(sample_a) if dates[r] < start and r not in in_target]
            if not base_ids:
                raise NotAvailableError(f"no reviews of '{base}' are older than '{target}'")
            rows = analysis.theme_comparison(data, base_ids, target_ids)
            if not rows:
                raise NotAvailableError("themes are not assigned for this snapshot")
            return {
                "base": base,
                "target": target,
                "n_base": len(base_ids),
                "n_target": len(target_ids),
                "target_window_start": start.isoformat(),
                "themes": rows,
                "emerging_phrases": analysis.emerging_phrases(data, base_ids, target_ids),
            }

        return dict(store.cached(snap_a, f"{sample_a}~{sample_b}", "compare", compute))

    @app.get("/v1/compare", tags=["analysis"], responses={404: PROBLEM, 409: PROBLEM, 422: PROBLEM})
    def compare(
        base: str = Query(..., description="earlier collection, e.g. notion_2026-10-13.A_random"),
        target: str = Query(..., description="recent collection, e.g. notion_2026-10-13.B_recent"),
    ) -> Any:
        """Which complaint and praise themes are emerging, fading or persistent: theme prevalence
        in `target` vs `base` with two-proportion tests and Benjamini-Hochberg correction. Reviews
        of `base` that fall inside `target`'s time window are excluded, so the comparison is
        earlier vs recent. Taxonomy-free emerging phrases are included as a drift signal."""
        return comparison_of(base, target)

    def default_comparison(collection_id: str) -> dict[str, Any] | None:
        """For the report: this collection vs the snapshot's most-recent sample, if it has one."""
        snap, _ = store.resolve(collection_id)
        recent = [
            c
            for c in store.collection_ids()
            if c.startswith(f"{snap.name}.")
            and (d := store.sample_def(snap, c.rpartition(".")[2])) is not None
            and d.strategy == "recent"
        ]
        if not recent:
            return None
        target = recent[0]
        base = f"{snap.name}.pool" if target == collection_id else collection_id
        try:
            return comparison_of(base, target)
        except (NotAvailableError, InvalidInputError, ModuleNotFoundError):
            return None

    @app.get(
        "/v1/collections/{collection_id}/report",
        tags=["analysis"],
        response_class=HTMLResponse,
        responses={404: PROBLEM, 409: PROBLEM},
    )
    def get_report(collection_id: CollectionId) -> HTMLResponse:
        """The metrics and insights of a collection as a self-contained HTML report with
        interactive charts (open it in a browser)."""
        from review_analysis.report import render_report

        html = render_report(
            summary(collection_id).model_dump(mode="json"),
            metrics_of(collection_id),
            insights_of(collection_id),
            default_comparison(collection_id),
        )
        return HTMLResponse(html)

    def report_inputs(collection_id: str) -> tuple[dict[str, Any], ...]:
        """summary, metrics, insights and comparison of a collection (static site export)."""
        return (
            summary(collection_id).model_dump(mode="json"),
            metrics_of(collection_id),
            insights_of(collection_id),
            default_comparison(collection_id) or {},
        )

    app.state.report_inputs = report_inputs
    app.state.store = store

    return app


def _csv(rows: list[dict[str, Any]], fields: list[str]) -> Iterator[str]:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    yield buf.getvalue()
    for row in rows:
        buf.seek(0)
        buf.truncate(0)
        writer.writerow(row)
        yield buf.getvalue()
