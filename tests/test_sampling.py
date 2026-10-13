from datetime import UTC, datetime, timedelta

import pytest

from review_analysis.models import Review
from review_analysis.sampling import most_recent, random_sample

NOW = datetime(2026, 10, 13, tzinfo=UTC)


def make_pool(n: int) -> list[Review]:
    return [
        Review(
            review_id=f"r{i:04d}",
            app_id=1,
            storefront="us",
            date=NOW - timedelta(hours=i),
            rating=1 + i % 5,
            title="",
            body=f"b{i}",
            author_hash="x",
            source="apple_web",
            fetched_at=NOW,
        )
        for i in range(n)
    ]


def test_random_sample_is_reproducible_and_order_independent():
    pool = make_pool(500)
    a = random_sample(pool, 300, seed=42)
    b = random_sample(list(reversed(pool)), 300, seed=42)
    assert [r.review_id for r in a] == [r.review_id for r in b]
    assert len({r.review_id for r in a}) == 300


def test_different_seed_gives_different_sample():
    pool = make_pool(500)
    a = {r.review_id for r in random_sample(pool, 100, seed=1)}
    b = {r.review_id for r in random_sample(pool, 100, seed=2)}
    assert a != b


def test_sample_larger_than_pool_returns_whole_pool():
    assert len(random_sample(make_pool(30), 300, seed=42)) == 30


def test_most_recent_is_newest_first():
    pool = make_pool(150)
    recent = most_recent(list(reversed(pool)), 100)
    assert [r.review_id for r in recent] == [f"r{i:04d}" for i in range(100)]


@pytest.mark.parametrize("fn", [lambda p: random_sample(p, 0, 1), lambda p: most_recent(p, 0)])
def test_non_positive_n_is_rejected(fn):
    with pytest.raises(ValueError):
        fn(make_pool(5))
