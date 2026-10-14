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
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from review_analysis import keywords, themes
from review_analysis.drivers import driver_analysis
from review_analysis.keywords import doc_terms, fightin_words, sensitivity, top_terms
from review_analysis.metrics import mean_ci, wilson
from review_analysis.models import Review
from review_analysis.preprocessing import lexical, normalize
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


@dataclass
class SnapshotData:
    """Everything the analysis needs from one snapshot, loaded once."""

    path: Path
    pool: list[Review]
    processed: dict[str, dict[str, Any]]
    sentiment: dict[str, dict[str, Any]]
    domain: frozenset[str]
    terms: dict[str, set[str]]  # review_id -> aspect n-grams (usable reviews only)

    def ids(self, sample: str) -> list[str]:
        """Review ids of a collection: "pool" or one of the snapshot's samples."""
        if sample == "pool":
            return [r.review_id for r in self.pool]
        definition, _ = load_sample(self.path, sample)
        return definition.review_ids


@lru_cache(maxsize=8)
def load(snapshot: Path) -> SnapshotData:
    processed = _read_jsonl(snapshot / "processed.jsonl")
    sentiment = _read_jsonl(snapshot / "sentiment.jsonl")
    report = json.loads((snapshot / "preprocessing_report.json").read_text())
    domain = frozenset(report["domain_stopwords"])
    terms = {
        rid: doc_terms(p["lemma_sentences"], domain)
        for rid, p in processed.items()
        if p["lemma_sentences"] and not p["flags"]["duplicate_of"]
    }
    return SnapshotData(snapshot, load_pool(snapshot), processed, sentiment, domain, terms)


@lru_cache(maxsize=1)
def _background(domain: frozenset[str]) -> list[set[str]]:
    return [doc_terms(s, domain) for s in background_lemmas()]


MIN_GROUP_KEYWORDS = 20  # documents per side for Fightin' Words
MIN_GROUP_DRIVERS = 30  # 1-2 star and 4-5 star reviews each for the driver model


def keyword_insights(data: SnapshotData, ids: list[str], k: int = 12) -> dict[str, Any]:
    terms = {r: data.terms[r] for r in ids if r in data.terms}
    neg = [t for r, t in terms.items() if data.sentiment[r]["label"] == "negative"]
    rest = [t for r, t in terms.items() if data.sentiment[r]["label"] != "negative"]
    if min(len(neg), len(rest)) < MIN_GROUP_KEYWORDS:
        return {
            "available": False,
            "reason": f"needs >= {MIN_GROUP_KEYWORDS} reviews per side",
            "n_a": len(neg),
            "n_b": len(rest),
        }
    background = _background(data.domain)
    fw = fightin_words(neg, rest, background)
    neg_top, pos_top = top_terms(fw, k=k)
    return {
        "available": True,
        "method": "Fightin' Words (Monroe et al. 2008), informative Dirichlet prior from 14 "
        "other productivity apps + this corpus; document-level counts; generic sentiment and "
        "filler words removed",
        "group_a": "negative reviews (model)",
        "group_b": "neutral + positive reviews (model)",
        "n_a": len(neg),
        "n_b": len(rest),
        "n_background": len(background),
        "negative": [asdict(t) for t in neg_top],
        "positive": [asdict(t) for t in pos_top],
        "prior_sensitivity_top12_overlap": sensitivity(neg, rest, background, k=k),
    }


def driver_insights(data: SnapshotData, ids: list[str], k: int = 12) -> dict[str, Any]:
    rated = [r for r in ids if r in data.terms and data.processed[r]["rating"] != 3]
    low = [data.processed[r]["rating"] <= 2 for r in rated]
    if min(sum(low), len(low) - sum(low)) < MIN_GROUP_DRIVERS:
        return {
            "available": False,
            "reason": f"needs >= {MIN_GROUP_DRIVERS} reviews with 1-2 and with 4-5 stars",
        }
    drv = driver_analysis([data.terms[r] for r in rated], low)
    return {
        "available": True,
        "method": "L2 logistic regression, 1-2 vs 4-5 stars, aspect n-grams; C by 5-fold CV; "
        "200 bootstraps, sign stability >= 0.9",
        "cv_auc": drv.cv_auc,
        "c": drv.c,
        "n_low": drv.n_low,
        "n_high": drv.n_high,
        "n_features": drv.n_features,
        "towards_low": [asdict(d) for d in drv.drivers[:k]],
        "towards_high": [asdict(d) for d in drv.drivers[::-1][:k]],
    }


def reviews_of(data: SnapshotData, ids: list[str]) -> list[Review]:
    by_id = {r.review_id: r for r in data.pool}
    return [by_id[r] for r in ids]


@lru_cache(maxsize=8)
def _theme_files(snapshot: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]] | None:
    """Sentence table and sentence theme labels of a snapshot, if themes have been assigned."""
    sent_path, label_path = snapshot / "sentences.jsonl", snapshot / "theme_labels.jsonl"
    if not (sent_path.exists() and label_path.exists()):
        return None
    sentences = {
        (row := json.loads(line))["sid"]: row for line in sent_path.read_text().splitlines() if line
    }
    labels = {
        (row := json.loads(line))["sid"]: row["themes"]
        for line in label_path.read_text().splitlines()
        if line
    }
    return sentences, labels


def theme_insights(data: SnapshotData, ids: list[str]) -> dict[str, Any] | None:
    files = _theme_files(data.path)
    if files is None:
        return None
    sentences, labels = files
    by_review = themes.review_themes(labels)
    ratings = {r: data.processed[r]["rating"] for r in ids}
    out = themes.theme_stats(ids, ratings, by_review, themes.load_taxonomy(), sentences, labels)
    covered = sum(bool(by_review.get(r)) for r in ids)
    out["coverage"] = {"reviews_with_a_theme": covered, "share": covered / max(1, len(ids))}
    sources = Counter(s for s in _label_sources(data.path))
    out["source"] = sources.most_common(1)[0][0] if sources else None
    out["method"] = (
        "Sentence-level themes from a human-approved taxonomy (data/themes.json); labels by "
        f"{out['source']}; impact = mean stars of reviews mentioning the theme minus the rest "
        "(Welch 95% CI)"
    )
    out["unassigned"] = unassigned_complaints(data, ids, sentences, labels)
    return out


@lru_cache(maxsize=8)
def _label_sources(snapshot: Path) -> tuple[str, ...]:
    path = snapshot / "theme_labels.jsonl"
    return tuple(
        json.loads(line).get("source") or "unknown"
        for line in path.read_text().splitlines()
        if line
    )


def unassigned_complaints(
    data: SnapshotData,
    ids: list[str],
    sentences: dict[str, dict[str, Any]],
    labels: dict[str, list[str]],
    k: int = 8,
) -> dict[str, Any]:
    """Drift early warning: complaint sentences that fit no theme in the taxonomy. A rising share,
    or a recurring word among them, signals a new problem the taxonomy does not cover yet; that is
    the trigger to re-run the TopicGPT extraction and refresh the taxonomy (README, maintenance)."""
    wanted = set(ids)
    rows = [
        s
        for sid, s in sentences.items()
        if s["review_id"] in wanted and s["kind"] == "complaint" and sid in labels
    ]
    orphans = [s for s in rows if not labels[s["sid"]]]
    stop = (
        lexical.BASE_STOPWORDS
        | lexical.KEEP_WORDS
        | data.domain
        | keywords.generic_sentiment_words()
    )
    words_: Counter[str] = Counter(
        w
        for s in orphans
        for w in {t.lower() for t in normalize.words(s["text"])}
        if len(w) > 2 and w not in stop and not w.isdigit() and "'" not in w and "\u2019" not in w
    )
    examples = sorted(
        (s for s in orphans if 25 <= len(s["text"]) <= 220),
        key=lambda s: (-s["p_complaint"], s["sid"]),
    )[:k]
    return {
        "complaint_sentences": len(rows),
        "unassigned": len(orphans),
        "share": len(orphans) / max(1, len(rows)),
        "top_words": [{"word": w, "sentences": n} for w, n in words_.most_common(12)],
        "examples": [
            {"text": s["text"], "rating": s["rating"], "review_id": s["review_id"]}
            for s in examples
        ],
        "note": "Complaint sentences no theme covers. A rising share or a new recurring word "
        "is the trigger to refresh the taxonomy.",
    }


def emerging_phrases(
    data: SnapshotData, base: list[str], target: list[str], k: int = 10
) -> dict[str, Any]:
    """Taxonomy-free change detection: phrases over-represented in recent vs earlier reviews
    (Fightin' Words, same prior as distinctive_terms). New problems appear here even if no theme
    covers them yet."""
    recent = [data.terms[r] for r in target if r in data.terms]
    earlier = [data.terms[r] for r in base if r in data.terms]
    if min(len(recent), len(earlier)) < MIN_GROUP_KEYWORDS:
        return {"available": False}
    fw = fightin_words(recent, earlier, _background(data.domain))
    up, down = top_terms(fw, k=k)
    return {
        "available": True,
        "n_recent": len(recent),
        "n_earlier": len(earlier),
        "rising": [asdict(t) for t in up],
        "falling": [asdict(t) for t in down],
    }


def theme_comparison(
    data: SnapshotData, base: list[str], target: list[str]
) -> list[dict[str, Any]]:
    files = _theme_files(data.path)
    if files is None:
        return []
    return themes.compare(base, target, themes.review_themes(files[1]), themes.load_taxonomy())


def recommendations(
    stats: dict[str, Any] | None,
    change: list[dict[str, Any]] | None = None,
    k: int = 5,
    min_reviews: int = 5,
) -> list[dict[str, Any]]:
    """Top complaint themes by priority (prevalence x rating drag), each with its evidence, the
    suggested action from the taxonomy and verbatim quotes. Generated by templates, so nothing is
    invented: every number comes from theme_stats and every quote is copied."""
    if not stats:
        return []
    actions = {t["id"]: t.get("action", "") for t in themes.load_taxonomy()["complaint"]}
    status = {c["id"]: c for c in (change or [])}
    out = []
    for t in stats["complaint"]:
        if t["n_reviews"] < min_reviews or not t["rating_impact"] or t["rating_impact"] >= 0:
            continue
        trend = status.get(t["id"])
        out.append(
            {
                "theme": t["id"],
                "name": t["name"],
                "evidence": f"{t['prevalence']:.0%} of reviews (95% CI {t['prevalence_ci'][0]:.0%}"
                f"–{t['prevalence_ci'][1]:.0%}); reviews mentioning it average "
                f"{t['mean_rating']:.1f}★, {abs(t['rating_impact']):.1f}★ below the rest"
                + (
                    f"; recent trend: {trend['status']} ({trend['base_share']:.0%} → "
                    f"{trend['target_share']:.0%})"
                    if trend
                    else ""
                ),
                "action": actions.get(t["id"], ""),
                "priority": t["priority"],
                "prevalence": t["prevalence"],
                "rating_impact": t["rating_impact"],
                "trend": trend["status"] if trend else None,
                "quotes": t["quotes"][:2],
            }
        )
        if len(out) == k:
            break
    return out


def themes_by_stars(data: SnapshotData, ids: list[str]) -> dict[str, Any] | None:
    files = _theme_files(data.path)
    if files is None:
        return None
    by_review = themes.review_themes(files[1])
    tax = themes.load_taxonomy()
    by_star: dict[int, list[str]] = defaultdict(list)
    for r in ids:
        by_star[data.processed[r]["rating"]].append(r)
    rows = []
    for kind in ("complaint", "praise"):
        for t in tax[kind]:
            rows.append(
                {
                    "id": t["id"],
                    "name": t["name"],
                    "kind": kind,
                    "shares": {
                        str(s): sum(t["id"] in by_review.get(r, set()) for r in by_star[s])
                        / max(1, len(by_star[s]))
                        for s in range(1, 6)
                    },
                }
            )
    return {"n_by_star": {str(s): len(by_star[s]) for s in range(1, 6)}, "themes": rows}


def build(snapshot: Path) -> dict[str, Any]:
    """The report bundle: pool-level insights plus the A-vs-B comparison of star ratings."""
    data = load(snapshot)
    sample_a, reviews_a = load_sample(snapshot, "A_random")
    sample_b, reviews_b = load_sample(snapshot, "B_recent")
    pool_ids = data.ids("pool")
    theme_stats = theme_insights(data, pool_ids)
    b_start = min(r.date for r in reviews_b)
    earlier = [r.review_id for r in reviews_a if r.date < b_start]
    change = theme_comparison(data, earlier, sample_b.review_ids)
    top4 = [t["id"] for t in (theme_stats or {"complaint": []})["complaint"][:4]]
    files = _theme_files(snapshot)
    return {
        "themes": theme_stats,
        "theme_change": {
            "base": "A (random, last 12 months) before B's window",
            "n_base": len(earlier),
            "target": "B (100 most recent)",
            "n_target": sample_b.n,
            "b_window_start": b_start.date().isoformat(),
            "themes": change,
        },
        "theme_monthly": themes.monthly_shares(
            pool_ids,
            {r.review_id: r.date for r in data.pool},
            themes.review_themes(files[1]) if files else {},
            top4,
        )
        if files
        else None,
        "theme_monthly_ids": top4,
        "emerging_phrases": emerging_phrases(data, earlier, sample_b.review_ids),
        "themes_by_stars": themes_by_stars(data, pool_ids),
        "recommendations": recommendations(theme_stats, change),
        "snapshot": snapshot.name,
        "n_pool": len(data.pool),
        "samples": {"A": {"n": sample_a.n, "seed": sample_a.seed}, "B": {"n": sample_b.n}},
        "star_distribution": star_distribution({"A": reviews_a, "B": reviews_b}),
        "monthly": monthly(data.pool),
        "stars_vs_text": stars_vs_text(data.pool, data.sentiment),
        "most_mentioned": most_mentioned(
            {r: data.terms[r] for r in pool_ids if r in data.terms}, data.sentiment
        ),
        "distinctive_terms": keyword_insights(data, pool_ids),
        "drivers": driver_insights(data, pool_ids),
    }
