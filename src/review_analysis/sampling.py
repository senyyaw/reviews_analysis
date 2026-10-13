"""Draw samples from a collected pool.

Sample A, `random_sample`, is a simple random sample: every review in the pool has the same
probability n/N of being chosen, so statistics computed on it need no weights. The pool is sorted
by review_id before drawing, which makes the result depend only on (pool contents, n, seed) and not
on the order the reviews were downloaded in.

Sample B, `most_recent`, is not random. It is every review in the most recent window: a census of
that window, not an estimate of it.
"""

import random
from collections.abc import Sequence

from review_analysis.models import Review


def random_sample(pool: Sequence[Review], n: int, seed: int) -> list[Review]:
    if n <= 0:
        raise ValueError("n must be positive")
    ordered = sorted(pool, key=lambda r: r.review_id)
    if n >= len(ordered):
        return ordered
    return random.Random(seed).sample(ordered, n)


def most_recent(pool: Sequence[Review], n: int) -> list[Review]:
    if n <= 0:
        raise ValueError("n must be positive")
    return sorted(pool, key=lambda r: (r.date, r.review_id), reverse=True)[:n]
