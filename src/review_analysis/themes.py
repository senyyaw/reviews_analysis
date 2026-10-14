"""Theme analytics: how big each theme is, how much it hurts the rating, what changed, and quotes.

Input: sentence-level theme labels (sid = "<review_id>:<sentence index>" -> theme ids), assigned by
the LLM annotator against the approved taxonomy (data/themes.json). A review "mentions" a theme
when any of its sentences carries it.

For each theme in a set of reviews:
- prevalence: reviews mentioning it / all reviews, with a Wilson 95% CI;
- rating impact (the Artefact method): mean stars of reviews mentioning the theme minus mean stars
  of the other reviews. The CI is Welch's, so variances need not be equal. Negative means the
  theme drags the rating down;
- priority (complaint themes) = prevalence x (-impact), i.e. "how many people, how badly";
- quotes: verbatim sentences carrying only this theme, from different reviews. They are copied,
  never generated. Non-English quotes keep the original next to the translation.
compare(): theme prevalence in B (recent) vs the earlier reviews (A minus B's window), with a
two-proportion z-test per theme and Benjamini-Hochberg correction across themes (17 tests at
alpha .05 would otherwise produce about one false "emerging" theme by chance).
"""

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from scipy import stats

from review_analysis.metrics import wilson

THEMES_FILE = Path(__file__).resolve().parents[2] / "data" / "themes.json"


def load_taxonomy(path: Path = THEMES_FILE) -> dict[str, list[dict[str, str]]]:
    tax = json.loads(path.read_text(encoding="utf-8"))
    return {"complaint": tax["complaint"], "praise": tax["praise"]}


def taxonomy_apps(path: Path = THEMES_FILE) -> set[int]:
    """App Store ids the taxonomy was built for. Its themes are specific to those apps' reviews."""
    meta = json.loads(path.read_text(encoding="utf-8")).get("_meta", {})
    return {int(a["id"]) for a in meta.get("apps", [])}


def review_themes(sentence_labels: dict[str, list[str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for sid, themes in sentence_labels.items():
        out[sid.rsplit(":", 1)[0]].update(themes)
    return dict(out)


def _welch_diff(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    """Mean(a) - mean(b) with a Welch 95% CI."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return ma - mb, ma - mb, ma - mb
    df = (va / len(a) + vb / len(b)) ** 2 / (
        (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)
    )
    t = float(stats.t.ppf(0.975, df))
    return ma - mb, ma - mb - t * se, ma - mb + t * se


def theme_stats(
    ids: Sequence[str],
    ratings: dict[str, int],
    by_review: dict[str, set[str]],
    taxonomy: dict[str, list[dict[str, str]]],
    sentences: dict[str, dict[str, Any]],
    sentence_labels: dict[str, list[str]],
    n_quotes: int = 3,
) -> dict[str, Any]:
    n = len(ids)
    out: dict[str, Any] = {"n_reviews": n, "complaint": [], "praise": []}
    for kind in ("complaint", "praise"):
        for theme in taxonomy[kind]:
            tid = theme["id"]
            mention = [r for r in ids if tid in by_review.get(r, set())]
            other = [r for r in ids if tid not in by_review.get(r, set())]
            impact, lo, hi = _welch_diff([ratings[r] for r in mention], [ratings[r] for r in other])
            prev = wilson(len(mention), n)
            row = {
                "id": tid,
                "name": theme["name"],
                "definition": theme["definition"],
                "n_reviews": len(mention),
                "prevalence": prev.share,
                "prevalence_ci": [prev.ci_low, prev.ci_high],
                "mean_rating": (sum(ratings[r] for r in mention) / len(mention))
                if mention
                else None,
                "rating_impact": impact,
                "rating_impact_ci": [lo, hi],
                "quotes": _quotes(tid, set(mention), sentences, sentence_labels, kind, n_quotes),
            }
            if kind == "complaint":
                row["priority"] = prev.share * -impact if not math.isnan(impact) else 0.0
            out[kind].append(row)
        key = "priority" if kind == "complaint" else "n_reviews"
        out[kind].sort(key=lambda r: -(r[key] or 0))
    return out


def _quotes(
    tid: str,
    reviews: set[str],
    sentences: dict[str, dict[str, Any]],
    sentence_labels: dict[str, list[str]],
    kind: str,
    k: int,
) -> list[dict[str, Any]]:
    """Verbatim, single-theme sentences from distinct reviews, most typical first: the strongest
    complaint probability for complaints, the weakest for praise. Readable lengths only."""
    cands = [
        sentences[sid]
        for sid, themes in sentence_labels.items()
        if themes == [tid]
        and sid in sentences
        and sentences[sid]["review_id"] in reviews
        and 25 <= len(sentences[sid]["text"]) <= 220
    ]
    sign = -1 if kind == "complaint" else 1
    cands.sort(key=lambda s: (sign * s["p_complaint"], s["sid"]))
    picked, seen = [], set()
    for s in cands:
        if s["review_id"] in seen:
            continue
        seen.add(s["review_id"])
        picked.append(
            {
                "text": s["text"],
                "translation": s.get("translation"),
                "rating": s["rating"],
                "review_id": s["review_id"],
            }
        )
        if len(picked) == k:
            break
    return picked


def compare(
    base_ids: Sequence[str],
    target_ids: Sequence[str],
    by_review: dict[str, set[str]],
    taxonomy: dict[str, list[dict[str, str]]],
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Theme prevalence in target (recent) vs base (earlier), BH-corrected two-proportion tests."""
    rows: list[dict[str, Any]] = []
    for kind in ("complaint", "praise"):
        for theme in taxonomy[kind]:
            tid = theme["id"]
            a = sum(tid in by_review.get(r, set()) for r in base_ids)
            b = sum(tid in by_review.get(r, set()) for r in target_ids)
            na, nb = len(base_ids), len(target_ids)
            pa, pb = a / na, b / nb
            pooled = (a + b) / (na + nb)
            se = math.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb)) if 0 < pooled < 1 else 0
            z = (pb - pa) / se if se else 0.0
            p = float(2 * stats.norm.sf(abs(z))) if se else 1.0
            rows.append(
                {
                    "id": tid,
                    "kind": kind,
                    "name": theme["name"],
                    "base_share": pa,
                    "target_share": pb,
                    "base_ci": list(_ci(a, na)),
                    "target_ci": list(_ci(b, nb)),
                    "diff": pb - pa,
                    "z": z,
                    "p": p,
                }
            )
    # Benjamini-Hochberg across all themes.
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    m = len(rows)
    q = [0.0] * m
    running = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        running = min(running, rows[i]["p"] * m / rank)
        q[i] = running
    for row, qi in zip(rows, q, strict=True):
        row["q"] = qi
        row["status"] = _status(row, qi, alpha)
    return rows


def _status(row: dict[str, Any], q: float, alpha: float) -> str:
    """confirmed: q < alpha after BH; signal: p < alpha but not after correction, so worth
    watching but not proven with these sample sizes."""
    up = row["diff"] > 0
    confirmed, signal = q < alpha, row["p"] < alpha
    if row["kind"] == "complaint":
        if confirmed:
            return "emerging" if up else "fading"
        if signal:
            return "rising signal (unconfirmed)" if up else "falling signal (unconfirmed)"
        both_common = min(row["base_share"], row["target_share"]) >= 0.05
        return "persistent" if both_common else "no significant change"
    if confirmed:
        return "growing praise" if up else "declining praise"
    if signal:
        return "praise rising (unconfirmed)" if up else "praise falling (unconfirmed)"
    return "no significant change"


def _ci(k: int, n: int) -> tuple[float, float]:
    p = wilson(k, n)
    return p.ci_low, p.ci_high


def monthly_shares(
    ids: Sequence[str],
    dates: dict[str, datetime],
    by_review: dict[str, set[str]],
    theme_ids: Sequence[str],
) -> list[dict[str, Any]]:
    months: dict[str, list[str]] = defaultdict(list)
    for r in ids:
        months[dates[r].strftime("%Y-%m")].append(r)
    return [
        {
            "month": m,
            "n": len(rs),
            **{t: sum(t in by_review.get(r, set()) for r in rs) / len(rs) for t in theme_ids},
        }
        for m, rs in sorted(months.items())
    ]
