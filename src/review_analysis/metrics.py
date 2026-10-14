"""Rating metrics for a collection, with uncertainty.

Every estimate carries a 95% interval, so the report never shows a bare percentage from 100 reviews:
- mean rating: t-interval, with the finite-population correction when the sample was drawn without
  replacement from a known pool (sample A: 300 of 1,143);
- shares (star buckets, negative/neutral/positive): Wilson score interval, which behaves well for
  small counts and for shares near 0% or 100%, unlike the normal approximation.

For a census (the whole pool, or sample B = every review in the latest window) no correction is
applied: the interval then describes the review-generating process ("what share of reviews this app
tends to get"), which is the right reading when comparing periods. The counts themselves are exact.

`representativeness()` checks a random sample against the pool it was drawn from: is the pool mean
inside the sample's CI, and does a chi-square goodness-of-fit test reject the star distribution?
"""

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel
from scipy import stats

from review_analysis.models import Review

Z95 = 1.959963984540054
STARS = (1, 2, 3, 4, 5)


class Proportion(BaseModel):
    count: int
    n: int
    share: float
    ci_low: float
    ci_high: float


class MeanEstimate(BaseModel):
    n: int
    mean: float
    sd: float
    ci_low: float
    ci_high: float
    fpc_applied: bool


class GroupStat(BaseModel):
    key: str
    n: int
    mean_rating: float
    negative: Proportion  # 1-2 stars


class Metrics(BaseModel):
    n: int
    date_min: str | None
    date_max: str | None
    rating: MeanEstimate | None
    median_rating: float | None
    distribution: dict[int, Proportion]
    sentiment_by_stars: dict[str, Proportion]  # negative 1-2, neutral 3, positive 4-5
    by_storefront: list[GroupStat]
    by_month: list[GroupStat]
    median_length_chars: float | None
    developer_response: Proportion
    edited: Proportion


class Representativeness(BaseModel):
    pool_n: int
    pool_mean: float
    sample_mean: float
    difference: float
    pool_mean_within_sample_ci: bool
    chi2: float
    chi2_p_value: float
    max_star_share_gap: float  # largest |sample share - pool share| over the five star buckets


def wilson(count: int, n: int, z: float = Z95) -> Proportion:
    if n == 0:
        return Proportion(count=0, n=0, share=0.0, ci_low=0.0, ci_high=0.0)
    p = count / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return Proportion(
        count=count, n=n, share=p, ci_low=max(0.0, centre - half), ci_high=min(1.0, centre + half)
    )


def mean_ci(values: Sequence[float], population_size: int | None = None) -> MeanEstimate:
    n = len(values)
    if n == 0:
        raise ValueError("no values")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    fpc = population_size is not None and population_size > n
    if fpc and population_size is not None:
        se *= math.sqrt((population_size - n) / (population_size - 1))
    t = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    return MeanEstimate(
        n=n, mean=mean, sd=sd, ci_low=mean - t * se, ci_high=mean + t * se, fpc_applied=fpc
    )


def _group(reviews: Sequence[Review], key_of: str) -> list[GroupStat]:
    groups: dict[str, list[int]] = defaultdict(list)
    for r in reviews:
        key = r.storefront if key_of == "storefront" else r.date.strftime("%Y-%m")
        groups[key].append(r.rating)
    out = [
        GroupStat(
            key=k,
            n=len(v),
            mean_rating=statistics.fmean(v),
            negative=wilson(sum(x <= 2 for x in v), len(v)),
        )
        for k, v in groups.items()
    ]
    if key_of == "storefront":
        return sorted(out, key=lambda g: -g.n)
    return sorted(out, key=lambda g: g.key)


def compute_metrics(reviews: Sequence[Review], *, population_size: int | None = None) -> Metrics:
    """Metrics for one collection. Pass `population_size` for a random sample from a known pool."""
    n = len(reviews)
    ratings = [r.rating for r in reviews]
    counts = Counter(ratings)
    return Metrics(
        n=n,
        date_min=min(r.date for r in reviews).date().isoformat() if reviews else None,
        date_max=max(r.date for r in reviews).date().isoformat() if reviews else None,
        rating=mean_ci(ratings, population_size) if reviews else None,
        median_rating=statistics.median(ratings) if reviews else None,
        distribution={s: wilson(counts[s], n) for s in STARS},
        sentiment_by_stars={
            "negative": wilson(counts[1] + counts[2], n),
            "neutral": wilson(counts[3], n),
            "positive": wilson(counts[4] + counts[5], n),
        },
        by_storefront=_group(reviews, "storefront"),
        by_month=_group(reviews, "month"),
        median_length_chars=statistics.median(len(r.text) for r in reviews) if reviews else None,
        developer_response=wilson(sum(r.developer_response is not None for r in reviews), n),
        edited=wilson(sum(r.is_edited for r in reviews), n),
    )


def representativeness(sample: Sequence[Review], pool: Sequence[Review]) -> Representativeness:
    """Does the random sample look like the pool it came from?"""
    est = mean_ci([r.rating for r in sample], population_size=len(pool))
    pool_mean = statistics.fmean(r.rating for r in pool)
    s_counts, p_counts = Counter(r.rating for r in sample), Counter(r.rating for r in pool)
    observed = [s_counts[s] for s in STARS]
    expected = [p_counts[s] / len(pool) * len(sample) for s in STARS]
    chi2, p_value = stats.chisquare(observed, expected)
    gaps = [abs(s_counts[s] / len(sample) - p_counts[s] / len(pool)) for s in STARS]
    return Representativeness(
        pool_n=len(pool),
        pool_mean=pool_mean,
        sample_mean=est.mean,
        difference=est.mean - pool_mean,
        pool_mean_within_sample_ci=est.ci_low <= pool_mean <= est.ci_high,
        chi2=float(chi2),
        chi2_p_value=float(p_value),
        max_star_share_gap=max(gaps),
    )


def stars_vs_text(pool: Sequence[Review], sentiment: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Star rating x text sentiment crosstab, and the share of reviews with a complaint per star."""
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
        "complaint_share": {
            str(s): wilson(complaints[s], totals[s]).model_dump() for s in range(1, 6)
        },
    }
