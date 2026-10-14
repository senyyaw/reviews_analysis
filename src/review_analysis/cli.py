"""Command-line entry point.

review-analysis collect --app-id 1232780281 --out data/snapshots/notion
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
    from dataclasses import asdict

    from review_analysis.sentiment import SentimentRouter

    snap = Path(args.snapshot)
    rows = [
        json.loads(line) for line in (snap / "processed.jsonl").read_text().splitlines() if line
    ]
    results = SentimentRouter().predict([r["text_model"] for r in rows], [r["lang"] for r in rows])
    with (snap / "sentiment.jsonl").open("w", encoding="utf-8") as fh:
        for row, res in zip(rows, results, strict=True):
            fh.write(json.dumps({"review_id": row["review_id"], **asdict(res)}) + "\n")
    counts: dict[str, int] = {}
    for res in results:
        counts[res.label] = counts.get(res.label, 0) + 1
    print(
        f"sentiment for {len(results)} reviews: {counts}; "
        f"complaints: {sum(r.complaint for r in results)}; "
        f"multilingual model used for {sum(r.model == 'sentiment-multi' for r in results)}"
    )
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
    c.add_argument("--random-n", type=int, default=300)
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

    a = sub.add_parser("analyze", help="sentiment + complaint for every review in a snapshot")
    a.add_argument("--snapshot", required=True)
    a.add_argument("-v", "--verbose", action="store_true")
    a.set_defaults(func=_analyze)

    f = sub.add_parser("figures", help="insight data (JSON) + charts (PNG) for a snapshot")
    f.add_argument("--snapshot", required=True)
    f.add_argument("--out", default="reports")
    f.add_argument("-v", "--verbose", action="store_true")
    f.set_defaults(func=_figures)

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
