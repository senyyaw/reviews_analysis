"""Assemble the data behind every insight chart from a snapshot. It is pure data, saved as JSON.

The same JSON feeds the static PNG figures (viz.py) and the interactive HTML report. Each block
answers one product question (DESIGN.md §6):
- star_distribution     How do recent reviews (B) compare with the whole year (A)?
- monthly               When did ratings change? (pool, 12 months, with CIs)
- stars_vs_text         Where do complaints hide? (stars x text sentiment, complaint share)
- distinctive_terms     What do unhappy users talk about that happy users don't, and vice versa?
- drivers               Which product aspects raise or lower the odds of a 1-2 star rating?
"""

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from review_analysis.drivers import driver_analysis
from review_analysis.keywords import doc_terms, fightin_words, sensitivity, top_terms
from review_analysis.metrics import mean_ci, wilson
from review_analysis.models import Review
from review_analysis.snapshot import load_pool, load_sample

OTHERS_DIR = Path(__file__).resolve().parents[2] / "data/snapshots/others"
BACKGROUND_CACHE = OTHERS_DIR / "background_lemmas.jsonl"


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    return {
        (row := json.loads(line))["review_id"]: row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def background_lemmas() -> list[list[list[str]]]:
    """Track B lemmas of the other apps' reviews: the Fightin' Words prior. Cached on disk."""
    if BACKGROUND_CACHE.exists():
        return [
            json.loads(line)["lemma_sentences"]
            for line in BACKGROUND_CACHE.read_text().splitlines()
        ]
    from review_analysis.preprocessing.pipeline import preprocess
    from review_analysis.preprocessing.translate import TranslationCache

    reviews = [
        Review.model_validate_json(line)
        for f in sorted(OTHERS_DIR.glob("*/pool.jsonl"))
        for line in f.read_text().splitlines()
        if line
    ]
    items, _ = preprocess(reviews, TranslationCache(OTHERS_DIR / "_no_translations.json"))
    with BACKGROUND_CACHE.open("w", encoding="utf-8") as fh:
        for p in items:
            if p.lemma_sentences:
                fh.write(
                    json.dumps({"review_id": p.review_id, "lemma_sentences": p.lemma_sentences})
                    + "\n"
                )
    return [p.lemma_sentences for p in items if p.lemma_sentences]


def _prop(count: int, n: int) -> dict[str, float]:
    p = wilson(count, n)
    return {"count": count, "n": n, "share": p.share, "ci_low": p.ci_low, "ci_high": p.ci_high}


def star_distribution(samples: dict[str, list[Review]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, rows in samples.items():
        c = Counter(r.rating for r in rows)
        out[name] = {str(s): _prop(c[s], len(rows)) for s in range(1, 6)}
    return out


def monthly(pool: list[Review]) -> list[dict[str, Any]]:
    by_month: dict[str, list[int]] = defaultdict(list)
    for r in pool:
        by_month[r.date.strftime("%Y-%m")].append(r.rating)
    rows = []
    for month in sorted(by_month):
        ratings = by_month[month]
        m = mean_ci(ratings)
        rows.append(
            {
                "month": month,
                "n": len(ratings),
                "mean_rating": m.mean,
                "mean_ci": [m.ci_low, m.ci_high],
                "negative": _prop(sum(x <= 2 for x in ratings), len(ratings)),
            }
        )
    return rows


def stars_vs_text(pool: list[Review], sentiment: dict[str, dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, dict[str, int]] = {
        str(s): {"negative": 0, "neutral": 0, "positive": 0} for s in range(1, 6)
    }
    complaints: Counter[int] = Counter()
    totals: Counter[int] = Counter()
    for r in pool:
        s = sentiment[r.review_id]
        table[str(r.rating)][s["label"]] += 1
        totals[r.rating] += 1
        complaints[r.rating] += bool(s["complaint"])
    return {
        "counts": table,
        "complaint_share": {str(s): _prop(complaints[s], totals[s]) for s in range(1, 6)},
    }


def most_mentioned(
    terms: dict[str, set[str]], sentiment: dict[str, dict[str, Any]], k: int = 20
) -> dict[str, Any]:
    """What users talk about most: aspect phrases ranked by the share of reviews mentioning them,
    split by tone. This is the quantitative replacement for a word cloud."""
    df: Counter[str] = Counter(t for d in terms.values() for t in d if len(t.split()) <= 2)
    neg: Counter[str] = Counter(
        t for r, d in terms.items() if sentiment[r]["label"] == "negative" for t in d
    )
    ranked = [t for t, _ in df.most_common(k * 3)]
    kept: list[str] = []
    for t in ranked:  # drop a phrase covered by a longer one already kept (>= 80% of its reviews)
        if not any(f" {t} " in f" {o} " and df[o] >= 0.8 * df[t] for o in kept if o != t):
            kept.append(t)
        if len(kept) == k:
            break
    n = len(terms)
    return {
        "n_reviews": n,
        "terms": [
            {"term": t, **_prop(df[t], n), "negative": neg[t], "not_negative": df[t] - neg[t]}
            for t in kept
        ],
    }


def build(snapshot: Path) -> dict[str, Any]:
    pool = load_pool(snapshot)
    sample_a, reviews_a = load_sample(snapshot, "A_random")
    sample_b, reviews_b = load_sample(snapshot, "B_recent")
    processed = _read_jsonl(snapshot / "processed.jsonl")
    sentiment = _read_jsonl(snapshot / "sentiment.jsonl")
    report = json.loads((snapshot / "preprocessing_report.json").read_text())
    domain = frozenset(report["domain_stopwords"])

    usable = [
        p for p in processed.values() if p["lemma_sentences"] and not p["flags"]["duplicate_of"]
    ]
    terms = {p["review_id"]: doc_terms(p["lemma_sentences"], domain) for p in usable}
    neg = [terms[r] for r in terms if sentiment[r]["label"] == "negative"]
    rest = [terms[r] for r in terms if sentiment[r]["label"] != "negative"]
    background = [doc_terms(s, domain) for s in background_lemmas()]
    fw = fightin_words(neg, rest, background)
    neg_top, pos_top = top_terms(fw, k=12)

    rated = [r for r in terms if processed[r]["rating"] != 3]
    drv = driver_analysis([terms[r] for r in rated], [processed[r]["rating"] <= 2 for r in rated])

    return {
        "snapshot": snapshot.name,
        "n_pool": len(pool),
        "samples": {"A": {"n": sample_a.n, "seed": sample_a.seed}, "B": {"n": sample_b.n}},
        "star_distribution": star_distribution({"A": reviews_a, "B": reviews_b}),
        "monthly": monthly(pool),
        "stars_vs_text": stars_vs_text(pool, sentiment),
        "most_mentioned": most_mentioned(terms, sentiment),
        "distinctive_terms": {
            "method": "Fightin' Words (Monroe et al. 2008), informative Dirichlet prior from 14 "
            "other productivity apps; document-level counts; generic sentiment words removed",
            "group_a": "negative reviews (model)",
            "group_b": "neutral + positive reviews (model)",
            "n_a": len(neg),
            "n_b": len(rest),
            "n_background": len(background),
            "negative": [asdict(t) for t in neg_top],
            "positive": [asdict(t) for t in pos_top],
            "prior_sensitivity_top12_overlap": sensitivity(neg, rest, background, k=12),
        },
        "drivers": {
            "method": "L2 logistic regression, 1-2 vs 4-5 stars, aspect n-grams; C by 5-fold CV; "
            "200 bootstraps, sign stability >= 0.9",
            "cv_auc": drv.cv_auc,
            "c": drv.c,
            "n_low": drv.n_low,
            "n_high": drv.n_high,
            "n_features": drv.n_features,
            "towards_low": [asdict(d) for d in drv.drivers[:12]],
            "towards_high": [asdict(d) for d in drv.drivers[::-1][:12]],
        },
    }
