"""Distinctive phrases: what one group of reviews talks about that the other does not.

Method: "Fightin' Words" (Monroe, Colaresi & Quinn 2008, Political Analysis 16(4)), the weighted
log-odds ratio with an informative Dirichlet prior:

    delta_w = log((y_aw + a_w) / (n_a + a0 - y_aw - a_w))
            - log((y_bw + a_w) / (n_b + a0 - y_bw - a_w))
    var_w  ~= 1 / (y_aw + a_w) + 1 / (y_bw + a_w)
    z_w     = delta_w / sqrt(var_w)

- y counts DOCUMENTS containing the term (binary per review), so one long rant cannot dominate.
- The prior a_w = a0 * pi_w comes from a background corpus (reviews of 14 other productivity apps)
  plus the analysed reviews themselves. Every candidate gets prior mass in proportion to how common
  it is, and terms frequent in productivity-app reviews generally get extra shrinkage. a0 sets
  the prior's strength; the default is 10% of the analysed corpus, and `sensitivity()` checks
  that the ranking is stable.
- Candidates are Track B n-grams (1-3, within one sentence, domain stopwords only inside phrases;
  see preprocessing/lexical.py), with document frequency >= MIN_DF in the analysed reviews.
- Generic sentiment words are excluded, so the chart shows product ASPECTS rather than "love",
  "great" or "bad". This is the Artefact approach. A word counts as generic when VADER (Hutto &
  Gilbert 2014) rates it |valence| >= 2.5, or when it is on a short explicit list of weaker generic
  words. A blanket VADER filter is NOT used: VADER also scores product signals such as "crash"
  (-1.7), "lag" (-1.4), "lose" (-1.7), "easy" (1.9) and "free" (2.3), and those must stay.
- Redundant sub-phrases are dropped: "offline" is not listed separately when "offline mode" covers
  >= 80% of its documents.
"""

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from review_analysis.preprocessing import lexical

MIN_DF = 3
COVER_RATIO = 0.8
STRONG_VALENCE = 2.5
# Weak generic sentiment words (VADER |valence| < 2.5 but still say nothing about the product).
WEAK_GENERIC = frozenset(
    {
        "good",
        "nice",
        "bad",
        "terrible",
        "awful",
        "poor",
        "cool",
        "fine",
        "okay",
        "ok",
        "decent",
        "thank",
        "thanks",
        "recommend",
        "wow",
        "star",
        "stars",
        "ever",
        "really",
        "pretty",
        "lot",
        "garbage",
        "trash",
        "suck",
        "sucks",
        "joke",
        "disappointed",
        "disappointing",
        "happy",
        "glad",
    }
)
# Frequent filler words that are not product aspects ("time", "try", "way"...). They are
# excluded so that frequency charts show aspects. They stay usable inside phrases ("waste time").
FILLER = frozenset(
    {
        "time",
        "try",
        "way",
        "year",
        "day",
        "week",
        "month",
        "start",
        "thing",
        "people",
        "make",
        "get",
        "go",
        "come",
        "know",
        "think",
        "say",
        "see",
        "look",
        "feel",
        "give",
        "take",
        "keep",
        "seem",
        "let",
        "able",
        "actually",
        "literally",
        "honestly",
        "anymore",
        "bit",
        "lot",
    }
)


@lru_cache(maxsize=1)
def generic_sentiment_words() -> frozenset[str]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    lexicon = SentimentIntensityAnalyzer().lexicon
    strong = {w for w, v in lexicon.items() if abs(v) >= STRONG_VALENCE and w.isalpha()}
    return frozenset(strong | WEAK_GENERIC | FILLER)


def is_aspect(ngram: str, domain: frozenset[str]) -> bool:
    """A phrase about the product: has a word that is not generic sentiment, a domain word or a
    negation, or pairs a negation with a domain word ("not work", "not sync")."""
    generic = generic_sentiment_words()
    tokens = ngram.split()
    if any(t not in domain and t not in generic and t not in lexical.KEEP_WORDS for t in tokens):
        return True
    return (
        len(tokens) > 1
        and any(t in lexical.NEGATIONS for t in tokens)
        and any(t in domain and t not in generic for t in tokens)
    )


@dataclass(frozen=True)
class Term:
    term: str
    z: float  # > 0: distinctive of group A; < 0: of group B
    df_a: int
    df_b: int
    share_a: float  # df_a / number of documents in A
    share_b: float


def doc_terms(lemma_sentences: Sequence[Sequence[str]], domain: frozenset[str]) -> set[str]:
    """The set of informative n-grams in one review (document-level, binary)."""
    return {
        g
        for sent in lemma_sentences
        for g in lexical.ngrams(sent)
        if lexical.is_informative(g, domain) and is_aspect(g, domain)
    }


def fightin_words(
    docs_a: Sequence[set[str]],
    docs_b: Sequence[set[str]],
    background: Iterable[set[str]],
    *,
    prior_strength: float = 0.10,
    min_df: int = MIN_DF,
) -> list[Term]:
    ya: Counter[str] = Counter(t for d in docs_a for t in d)
    yb: Counter[str] = Counter(t for d in docs_b for t in d)
    bg: Counter[str] = Counter(t for d in background for t in d)
    na, nb = sum(ya.values()), sum(yb.values())
    a0 = prior_strength * (na + nb)
    vocab = [t for t in ya.keys() | yb.keys() if ya[t] + yb[t] >= min_df]
    # Prior distribution pi_w: background corpus PLUS the analysed corpus (Monroe et al. 2008;
    # Jurafsky et al. 2014). Using the background alone would give app-specific terms that are
    # rare in other apps' reviews a near-zero pseudo-count, so their variance would explode and they
    # would be systematically under-ranked. Including the analysed corpus avoids that.
    prior = {t: bg[t] + ya[t] + yb[t] for t in vocab}
    prior_total = sum(prior.values())
    terms = []
    for t in vocab:
        a_w = a0 * prior[t] / prior_total
        da = math.log((ya[t] + a_w) / (na + a0 - ya[t] - a_w))
        db = math.log((yb[t] + a_w) / (nb + a0 - yb[t] - a_w))
        z = (da - db) / math.sqrt(1 / (ya[t] + a_w) + 1 / (yb[t] + a_w))
        terms.append(
            Term(t, z, ya[t], yb[t], ya[t] / max(1, len(docs_a)), yb[t] / max(1, len(docs_b)))
        )
    return sorted(terms, key=lambda x: -x.z)


def drop_redundant(terms: Sequence[Term]) -> list[Term]:
    """Remove a phrase when a longer phrase containing it covers >= COVER_RATIO of its documents."""
    kept = []
    for t in terms:
        df = t.df_a + t.df_b
        covered = any(
            o.term != t.term
            and f" {t.term} " in f" {o.term} "
            and (o.df_a + o.df_b) >= COVER_RATIO * df
            and (o.z > 0) == (t.z > 0)
            for o in terms
        )
        if not covered:
            kept.append(t)
    return kept


def top_terms(
    terms: Sequence[Term], k: int = 15, z_min: float = 1.96
) -> tuple[list[Term], list[Term]]:
    """Top-k significant terms for each side (|z| > z_min, i.e. p < .05 two-sided)."""
    clean = drop_redundant([t for t in terms if abs(t.z) > z_min])
    side_a = [t for t in clean if t.z > 0][:k]
    side_b = sorted((t for t in clean if t.z < 0), key=lambda t: t.z)[:k]
    return side_a, side_b


def sensitivity(
    docs_a: Sequence[set[str]],
    docs_b: Sequence[set[str]],
    background: Sequence[set[str]],
    k: int = 15,
    strengths: Sequence[float] = (0.05, 0.10, 0.20),
) -> dict[float, float]:
    """Overlap of the top-k (both sides) at each prior strength with the default (0.10)."""

    def top(s: float) -> set[str]:
        a, b = top_terms(fightin_words(docs_a, docs_b, background, prior_strength=s), k)
        return {t.term for t in a + b}

    ref = top(0.10)
    return {s: len(top(s) & ref) / max(1, len(ref)) for s in strengths}
