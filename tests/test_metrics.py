from datetime import UTC, datetime, timedelta

import pytest

from review_analysis.metrics import compute_metrics, mean_ci, representativeness, wilson
from review_analysis.models import Review
from review_analysis.sampling import random_sample

NOW = datetime(2026, 10, 13, tzinfo=UTC)


def review(i: int, rating: int, *, storefront: str = "us", days_ago: int = 0) -> Review:
    return Review(
        review_id=f"r{i}",
        app_id=1,
        storefront=storefront,
        date=NOW - timedelta(days=days_ago),
        rating=rating,
        title="t",
        body="b" * 10,
        author_hash="x",
        source="apple_web",
        fetched_at=NOW,
    )


def test_wilson_matches_known_values():
    p = wilson(0, 10)  # zero successes: the normal approximation would give [0, 0]
    assert p.ci_low == 0.0 and p.ci_high == pytest.approx(0.2775, abs=1e-3)
    p = wilson(45, 100)
    assert (p.ci_low, p.ci_high) == (
        pytest.approx(0.3561, abs=1e-3),
        pytest.approx(0.5476, abs=1e-3),
    )


def test_finite_population_correction_narrows_the_interval():
    values = [1, 5] * 150
    without = mean_ci(values)
    with_fpc = mean_ci(values, population_size=1143)
    assert with_fpc.fpc_applied and not without.fpc_applied
    assert (with_fpc.ci_high - with_fpc.ci_low) < (without.ci_high - without.ci_low)
    census = mean_ci(values, population_size=len(values))  # whole population: no correction
    assert not census.fpc_applied


def test_metrics_shapes_and_groups():
    reviews = [
        review(0, 1, days_ago=40),
        review(1, 2),
        review(2, 5, storefront="gb"),
        review(3, 4),
        review(4, 3),
    ]
    m = compute_metrics(reviews)
    assert m.n == 5 and m.rating is not None and m.rating.mean == 3.0
    assert m.sentiment_by_stars["negative"].count == 2
    assert sum(p.count for p in m.distribution.values()) == 5
    assert [g.key for g in m.by_storefront] == ["us", "gb"]
    assert len(m.by_month) == 2 and m.by_month[0].key < m.by_month[1].key


def test_empty_collection_is_safe():
    m = compute_metrics([])
    assert m.n == 0 and m.rating is None and m.distribution[5].share == 0.0


def test_random_sample_is_representative_of_its_pool():
    pool = [review(i, 1 + (i * 7) % 5) for i in range(1143)]
    sample = random_sample(pool, 300, seed=42)
    check = representativeness(sample, pool)
    assert check.pool_mean_within_sample_ci
    assert check.chi2_p_value > 0.05
