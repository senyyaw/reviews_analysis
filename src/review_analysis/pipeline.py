"""End-to-end steps shared by the CLI and the API: collect -> preprocess -> sentiment."""

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from review_analysis import __version__
from review_analysis.collector import collect_window
from review_analysis.http import PoliteClient
from review_analysis.models import AppInfo
from review_analysis.sampling import most_recent, random_sample
from review_analysis.snapshot import SampleDef, load_pool, write_snapshot
from review_analysis.sources.lookup import lookup_app, validate_app_id, validate_storefront

Strategy = Literal["random", "recent"]
Progress = Callable[[str], None]


def sample_name(strategy: Strategy, count: int, seed: int | None) -> str:
    return f"random_{count}_s{seed}" if strategy == "random" else f"recent_{count}"


def collect_snapshot(
    out_dir: Path,
    app_id: int | str,
    storefronts: list[str],
    window_days: int,
    samples: list[tuple[Strategy, int, int | None]],
    *,
    max_pages: int = 500,
    client: PoliteClient | None = None,
    progress: Progress | None = None,
) -> tuple[AppInfo, list[SampleDef]]:
    """Download every review in the window, draw the requested samples, write the snapshot."""
    app_id = validate_app_id(app_id)
    storefronts = [validate_storefront(s) for s in storefronts]
    now = datetime.now(UTC)
    since = now - timedelta(days=window_days)
    own_client = client is None
    client = client or PoliteClient()
    try:
        app = lookup_app(client, app_id, storefronts[0])
        if progress:
            progress(f"found {app.name}")
        pool = collect_window(
            client,
            app_id,
            storefronts,
            since,
            raw_dir=out_dir / "raw",
            max_pages_per_storefront=max_pages,
            progress=(lambda cc, pages, n: progress(f"{cc}: {n} reviews ({pages} pages)"))
            if progress
            else None,
        )
    finally:
        if own_client:
            client.close()

    defs = []
    for strategy, count, seed in samples:
        picked = (
            random_sample(pool.reviews, count, seed if seed is not None else 42)
            if strategy == "random"
            else most_recent(pool.reviews, count)
        )
        defs.append(
            SampleDef(
                name=sample_name(strategy, count, seed),
                strategy=strategy,
                n_requested=count,
                n=len(picked),
                seed=seed if strategy == "random" else None,
                pool_size=len(pool.reviews),
                window_start=since,
                window_end=now,
                review_ids=[r.review_id for r in picked],
            )
        )
        if len(picked) < count:
            pool.warnings.append(f"{defs[-1].name}: only {len(picked)} of {count} available")
    write_snapshot(
        out_dir,
        app,
        pool,
        defs,
        collector_version=__version__,
        params={
            "app_id": app_id,
            "storefronts": storefronts,
            "window_days": window_days,
            "samples": [list(s) for s in samples],
            "max_pages": max_pages,
            "since": since.isoformat(),
        },
    )
    return app, defs


def preprocess_snapshot(snap: Path, *, translate: bool = False) -> dict[str, object]:
    from review_analysis.preprocessing.pipeline import preprocess
    from review_analysis.preprocessing.translate import M2MTranslator, TranslationCache

    cache = TranslationCache(snap / "translations.json")
    translator = M2MTranslator() if translate else None
    items, report = preprocess(load_pool(snap), cache, translator=translator)
    if translator is not None:
        cache.save()
    with (snap / "processed.jsonl").open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")
    (snap / "preprocessing_report.json").write_text(report.model_dump_json(indent=2))
    return report.model_dump()


def sentiment_snapshot(snap: Path, router: object | None = None) -> dict[str, int]:
    from review_analysis.sentiment import SentimentRouter

    rows = [
        json.loads(line) for line in (snap / "processed.jsonl").read_text().splitlines() if line
    ]
    model = router if router is not None else SentimentRouter()
    results = model.predict([r["text_model"] for r in rows], [r["lang"] for r in rows])  # type: ignore[attr-defined]
    counts: dict[str, int] = {}
    with (snap / "sentiment.jsonl").open("w", encoding="utf-8") as fh:
        for row, res in zip(rows, results, strict=True):
            fh.write(json.dumps({"review_id": row["review_id"], **asdict(res)}) + "\n")
            counts[res.label] = counts.get(res.label, 0) + 1
    return counts


def themes_snapshot(
    snap: Path, router: object | None = None, classifier: object | None = None, min_words: int = 3
) -> dict[str, int]:
    """Sentence-level sentiment + theme tags for a snapshot (sentences.jsonl, theme_labels.jsonl).

    Same sentence rules as the research pipeline: original-language sentences of >= min_words
    words; complaint = negative or P(complaint) >= 0.5; praise = positive and no complaint.
    """
    from review_analysis.preprocessing.normalize import words
    from review_analysis.sentiment import SentimentRouter
    from review_analysis.theme_classifier import ThemeClassifier

    processed = [
        json.loads(line) for line in (snap / "processed.jsonl").read_text().splitlines() if line
    ]
    rows = []
    for p in processed:
        if p["flags"]["duplicate_of"]:
            continue
        for i, s in enumerate(p["sentences"]):
            if len(words(s)) >= min_words:
                rows.append(
                    {
                        "sid": f"{p['review_id']}:{i}",
                        "review_id": p["review_id"],
                        "idx": i,
                        "text": s,
                        "lang": p["lang"],
                        "rating": p["rating"],
                        "translation": p["text_en"] if p["translated"] else None,
                    }
                )
    model = router if router is not None else SentimentRouter()
    sentiment = model.predict([r["text"] for r in rows], [r["lang"] for r in rows])  # type: ignore[attr-defined]
    for r, s in zip(rows, sentiment, strict=True):
        r.update(label=s.label, p_complaint=round(s.p_complaint, 4))
        r["kind"] = (
            "complaint"
            if s.label == "negative" or s.complaint
            else "praise"
            if s.label == "positive"
            else "other"
        )
    tagged = [r for r in rows if r["kind"] in ("complaint", "praise")]
    clf = classifier if classifier is not None else ThemeClassifier()
    labels = clf.predict([r["text"] for r in tagged], [r["kind"] for r in tagged])  # type: ignore[attr-defined]
    with (snap / "sentences.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(
                json.dumps({k: v for k, v in r.items() if k != "lang"}, ensure_ascii=False) + "\n"
            )
    with (snap / "theme_labels.jsonl").open("w", encoding="utf-8") as fh:
        for r, themes in zip(tagged, labels, strict=True):
            fh.write(
                json.dumps(
                    {
                        "sid": r["sid"],
                        "themes": themes,
                        "source": getattr(clf, "name", "classifier"),
                    }
                )
                + "\n"
            )
    return {
        "sentences": len(rows),
        "tagged": len(tagged),
        "with_theme": sum(bool(t) for t in labels),
    }
