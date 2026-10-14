"""Command-line entry point.

review-analysis collect --app-id 1232780281 --out data/snapshots/notion   # ~5 min
review-analysis preprocess --snapshot data/snapshots/notion [--translate]
review-analysis analyze --snapshot data/snapshots/notion                  # sentiment + themes
review-analysis figures --snapshot data/snapshots/notion --out reports    # PNG charts
review-analysis export-site --out docs                                    # HTML reports
review-analysis serve                                                     # REST API
"""

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from review_analysis import __version__
from review_analysis.collector import collect_window
from review_analysis.errors import ReviewAnalysisError
from review_analysis.http import PoliteClient
from review_analysis.sampling import most_recent, random_sample
from review_analysis.snapshot import SampleDef, write_snapshot
from review_analysis.sources.lookup import lookup_app, validate_app_id, validate_storefront

DEFAULT_STOREFRONTS = "us,gb,ca,au,in,ie,nz"


def _collect(args: argparse.Namespace) -> int:
    app_id = validate_app_id(args.app_id)
    storefronts = [validate_storefront(s) for s in args.storefronts.split(",") if s.strip()]
    now = datetime.now(UTC)
    since = now - timedelta(days=args.window_days)
    out = Path(args.out)

    with PoliteClient() as client:
        app = lookup_app(client, app_id, storefronts[0])
        print(
            f"App: {app.name} ({app.seller}), listing {app.listing_avg_rating} "
            f"from {app.listing_rating_count} ratings"
        )

        def progress(cc: str, pages: int, in_window: int) -> None:
            print(f"  {cc}: {pages} pages, {in_window} reviews in window", flush=True)

        pool = collect_window(
            client,
            app_id,
            storefronts,
            since,
            raw_dir=out / "raw",
            max_pages_per_storefront=args.max_pages,
            progress=progress,
        )
        print(f"Requests: {client.requests_made} ({client.retries_made} retries)")

    window = {"window_start": since, "window_end": now, "pool_size": len(pool.reviews)}
    a = random_sample(pool.reviews, args.random_n, args.seed)
    b = most_recent(pool.reviews, args.recent_n)
    samples = [
        SampleDef(
            name="A_random",
            strategy="random",
            n_requested=args.random_n,
            n=len(a),
            seed=args.seed,
            review_ids=[r.review_id for r in a],
            **window,
        ),
        SampleDef(
            name="B_recent",
            strategy="recent",
            n_requested=args.recent_n,
            n=len(b),
            seed=None,
            review_ids=[r.review_id for r in b],
            **window,
        ),
    ]
    for s in samples:
        if s.n < s.n_requested:
            pool.warnings.append(f"{s.name}: only {s.n} of {s.n_requested} reviews available")

    params = vars(args) | {"since": since.isoformat()}
    params.pop("func", None)
    write_snapshot(out, app, pool, samples, collector_version=__version__, params=params)

    print(
        f"Pool: {len(pool.reviews)} reviews since {since:%Y-%m-%d} "
        f"({pool.duplicates_removed} duplicates removed)"
    )
    for st in pool.storefronts:
        print(
            f"  {st.storefront}: {st.reviews_in_window} "
            f"({st.pages_fetched} pages, {st.stop_reason})"
        )
    for s in samples:
        print(f"Sample {s.name}: {s.n}")
    for w in pool.warnings:
        print(f"WARNING: {w}")
    print(f"Snapshot written to {out}")
    return 0


def _preprocess(args: argparse.Namespace) -> int:
    from review_analysis.preprocessing.pipeline import preprocess
    from review_analysis.preprocessing.translate import M2MTranslator, TranslationCache
    from review_analysis.snapshot import load_pool

    snap = Path(args.snapshot)
    cache = TranslationCache(snap / "translations.json")
    if args.retranslate:
        cache.entries.clear()
    translator = M2MTranslator(args.translation_model) if args.translate else None
    items, report = preprocess(load_pool(snap), cache, translator=translator)
    if translator is not None:
        cache.save()
    with (snap / "processed.jsonl").open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")
    (snap / "preprocessing_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    print(report.model_dump_json(indent=2))
    if report.untranslated:
        print(
            f"WARNING: {report.untranslated} non-English review(s) have no translation; "
            "run with --translate (needs: uv sync --extra translate)"
        )
    return 0


def _analyze(args: argparse.Namespace) -> int:
    import json

    from review_analysis import pipeline
    from review_analysis.sentiment import SentimentRouter

    snap = Path(args.snapshot)
    router = SentimentRouter()
    counts = pipeline.sentiment_snapshot(snap, router)
    print(f"sentiment for {sum(counts.values())} reviews: {counts}")
    if args.no_themes:
        return 0
    labels = snap / "theme_labels.jsonl"
    if labels.exists() and not args.retag_themes:
        sources = {json.loads(line).get("source") for line in labels.read_text().splitlines()}
        if not any(str(src).startswith("classifier") for src in sources):
            print(
                f"themes: kept the existing labels from {', '.join(map(str, sources))} "
                "(--retag-themes replaces them with the classifier)"
            )
            return 0
    stats = pipeline.themes_snapshot(snap, router)
    print(f"themes: {stats}")
    return 0


def _figures(args: argparse.Namespace) -> int:
    import json

    from review_analysis import analysis, viz

    snap, out = Path(args.snapshot), Path(args.out)
    data = analysis.build(snap)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data" / "insights.json").write_text(json.dumps(data, indent=1, default=float))
    for path in viz.render_all(data, out / "figures"):
        print(f"wrote {path}")
    return 0


def _export_site(args: argparse.Namespace) -> int:
    """Static HTML reports (and their data files) for every collection, e.g. for GitHub Pages."""
    import json

    from review_analysis.api.app import create_app
    from review_analysis.api.settings import Settings
    from review_analysis.report import render_index, render_report

    app = create_app(Settings(snapshots_dir=Path(args.snapshots), allow_collection=False))
    from fastapi.testclient import TestClient

    inputs, store = app.state.report_inputs, app.state.store
    client = TestClient(app)
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    entries = []
    for cid in store.collection_ids():
        if args.snapshot and not cid.startswith(f"{args.snapshot}."):
            continue
        summary, metrics, insights, comparison = inputs(cid)
        files = {
            "csv": f"data/{cid}.reviews.csv",
            "reviews": f"data/{cid}.reviews.csv",
            "metrics": f"data/{cid}.metrics.json",
            "insights": f"data/{cid}.insights.json",
        }
        csv = client.get(f"/v1/collections/{cid}/reviews", params={"format": "csv", "limit": 5000})
        (out / files["csv"]).write_text(csv.text, encoding="utf-8")
        (out / files["metrics"]).write_text(json.dumps(metrics, indent=1, default=float))
        (out / files["insights"]).write_text(json.dumps(insights, indent=1, default=float))
        html = render_report(summary, metrics, insights, comparison, links=files)
        (out / f"{cid}.html").write_text(html, encoding="utf-8")
        entries.append((summary, metrics, insights))
        print(f"wrote {out / cid}.html")
    (out / "index.html").write_text(render_index(entries, args.api_url), encoding="utf-8")
    (out / ".nojekyll").write_text("")
    print(f"wrote {out / 'index.html'}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from review_analysis.api.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="review-analysis")
    sub = parser.add_subparsers(required=True)

    c = sub.add_parser("collect", help="collect a review pool and draw samples A and B")
    c.add_argument("--app-id", required=True)
    c.add_argument("--storefronts", default=DEFAULT_STOREFRONTS)
    c.add_argument("--window-days", type=int, default=365)
    c.add_argument("--random-n", type=int, default=100, help="sample A size (Notion report: 300)")
    c.add_argument("--recent-n", type=int, default=100)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--max-pages", type=int, default=500, help="per storefront (20 reviews/page)")
    c.add_argument("--out", required=True)
    c.add_argument("-v", "--verbose", action="store_true")
    c.set_defaults(func=_collect)

    p = sub.add_parser("preprocess", help="clean, detect language, translate, lemmatise a snapshot")
    p.add_argument("--snapshot", required=True)
    p.add_argument(
        "--translate", action="store_true", help="translate non-English reviews (m2m100)"
    )
    p.add_argument("--translation-model", default="facebook/m2m100_1.2B")
    p.add_argument("--retranslate", action="store_true", help="ignore cached translations")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=_preprocess)

    a = sub.add_parser("analyze", help="sentiment + complaint per review, then sentence themes")
    a.add_argument("--snapshot", required=True)
    a.add_argument("--no-themes", action="store_true", help="sentiment only")
    a.add_argument(
        "--retag-themes",
        action="store_true",
        help="replace existing LLM theme labels with the classifier's",
    )
    a.add_argument("-v", "--verbose", action="store_true")
    a.set_defaults(func=_analyze)

    f = sub.add_parser("figures", help="insight data (JSON) + charts (PNG) for a snapshot")
    f.add_argument("--snapshot", required=True)
    f.add_argument("--out", default="reports")
    f.add_argument("-v", "--verbose", action="store_true")
    f.set_defaults(func=_figures)

    e = sub.add_parser("export-site", help="static HTML reports for every collection")
    e.add_argument("--snapshots", default="data/snapshots")
    e.add_argument("--snapshot", help="only this snapshot (directory name)")
    e.add_argument("--out", default="docs")
    e.add_argument("--api-url", default="", help="link the index page to a deployed API")
    e.add_argument("-v", "--verbose", action="store_true")
    e.set_defaults(func=_export_site)

    sv = sub.add_parser("serve", help="run the REST API (docs at /docs)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("-v", "--verbose", action="store_true")
    sv.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.func(args))
    except ReviewAnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
