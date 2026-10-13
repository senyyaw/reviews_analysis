"""Duplicate handling.

- Exact duplicate: same author AND same normalised text. Only the earliest is kept; it is a repost.
- Same text from different authors ("Great app") is legitimate: different people, same opinion.
- Near-duplicate: character 3-5-gram TF-IDF cosine >= threshold between texts with at least
  MIN_WORDS_NEAR words. These are flagged and grouped (copy-paste campaigns), not removed.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIN_WORDS_NEAR = 5
NEAR_THRESHOLD = 0.90
_NON_ALNUM = re.compile(r"[\W_]+")


@dataclass(frozen=True)
class DedupeItem:
    review_id: str
    author_hash: str
    text: str
    n_words: int


def _key(text: str) -> str:
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def exact_duplicates(items: Sequence[DedupeItem]) -> dict[str, str]:
    """review_id -> review_id of the kept original. `items` must be in chronological order."""
    seen: dict[tuple[str, str], str] = {}
    duplicates: dict[str, str] = {}
    for it in items:
        key = (it.author_hash, _key(it.text))
        if key in seen:
            duplicates[it.review_id] = seen[key]
        else:
            seen[key] = it.review_id
    return duplicates


def near_duplicate_groups(items: Sequence[DedupeItem]) -> dict[str, int]:
    """review_id -> group number, only for reviews that belong to a group of size >= 2."""
    eligible = [it for it in items if it.n_words >= MIN_WORDS_NEAR]
    if len(eligible) < 2:
        return {}
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(
        [_key(it.text) for it in eligible]
    )
    sim = cosine_similarity(matrix)
    parent = list(range(len(eligible)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    rows, cols = (sim >= NEAR_THRESHOLD).nonzero()
    for i, j in zip(rows, cols, strict=True):
        if i < j:
            parent[find(i)] = find(j)
    roots: dict[int, list[int]] = {}
    for i in range(len(eligible)):
        roots.setdefault(find(i), []).append(i)
    groups = [members for members in roots.values() if len(members) > 1]
    return {eligible[i].review_id: g for g, members in enumerate(groups) for i in members}
