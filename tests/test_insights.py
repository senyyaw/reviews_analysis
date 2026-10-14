"""Keyword and driver analyses: synthetic corpora with a planted signal."""

import random

from review_analysis.drivers import driver_analysis
from review_analysis.keywords import (
    doc_terms,
    drop_redundant,
    fightin_words,
    generic_sentiment_words,
    is_aspect,
    top_terms,
)

DOMAIN = frozenset({"app", "notion", "use", "work"})


def corpus(seed: int = 0) -> tuple[list[set[str]], list[set[str]], list[set[str]]]:
    rng = random.Random(seed)
    filler = [f"word{i}" for i in range(40)]

    def doc(extra: list[str]) -> set[str]:
        return set(rng.sample(filler, 6)) | set(extra)

    neg = [doc(["ipad"] if i % 3 == 0 else ["sync"] if i % 5 == 0 else []) for i in range(300)]
    pos = [doc(["template"] if i % 4 == 0 else []) for i in range(300)]
    background = [
        doc(["sync"] if i % 4 == 0 else []) for i in range(1000)
    ]  # sync: common everywhere
    return neg, pos, background


def test_planted_terms_come_out_on_the_right_side():
    neg, pos, bg = corpus()
    a, b = top_terms(fightin_words(neg, pos, bg), k=5)
    assert a[0].term == "ipad"
    assert b[0].term == "template"
    assert not any(t.term.startswith("word") for t in a + b)  # random filler is not significant


def test_terms_absent_from_background_are_not_penalised():
    """Regression test: with a background-only prior, app-specific terms got near-zero pseudo-counts
    and a deflated z. "ipad" (100 vs 0) must rank above "sync" (40 vs 0)."""
    neg, pos, bg = corpus()
    z = {t.term: t.z for t in fightin_words(neg, pos, bg)}
    assert z["ipad"] > z["sync"] > 1.96


def test_ranking_is_stable_across_prior_strengths():
    neg, pos, bg = corpus()
    from review_analysis.keywords import sensitivity

    assert min(sensitivity(neg, pos, bg, k=3).values()) >= 0.6


def test_generic_sentiment_is_removed_but_product_signals_stay():
    generic = generic_sentiment_words()
    assert {"love", "great", "best", "hate", "good"} <= generic
    assert not {"crash", "lag", "easy", "free", "broken", "ugly", "sync"} & generic
    assert not is_aspect("love", DOMAIN) and not is_aspect("good app", DOMAIN)
    assert is_aspect("crash", DOMAIN) and is_aspect("not work", DOMAIN)


def test_doc_terms_are_document_level_and_filtered():
    terms = doc_terms([["love", "ipad"], ["ipad", "crash"], ["not", "work"]], DOMAIN)
    assert {"ipad", "crash", "ipad crash", "not work"} <= terms
    assert "love" not in terms and "love ipad" in terms  # the phrase keeps its aspect


def test_redundant_sub_phrases_are_dropped():
    from review_analysis.keywords import Term

    terms = [
        Term("offline mode", 5, 30, 1, 0.1, 0),
        Term("offline", 4, 32, 1, 0.1, 0),
        Term("sync", 3, 20, 1, 0.1, 0),
    ]
    assert [t.term for t in drop_redundant(terms)] == ["offline mode", "sync"]


def test_driver_analysis_recovers_planted_drivers():
    rng = random.Random(1)
    docs, low = [], []
    for _ in range(600):
        d = set(rng.sample([f"w{i}" for i in range(30)], 5))
        is_low = rng.random() < 0.5
        if is_low and rng.random() < 0.6:
            d.add("crash")
        if not is_low and rng.random() < 0.6:
            d.add("templates")
        docs.append(d)
        low.append(is_low)
    res = driver_analysis(docs, low)
    assert res.cv_auc > 0.7
    assert res.drivers[0].term == "crash" and res.drivers[0].odds_ratio > 1
    assert res.drivers[-1].term == "templates" and res.drivers[-1].odds_ratio < 1
