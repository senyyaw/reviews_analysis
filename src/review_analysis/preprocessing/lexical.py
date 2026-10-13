"""Track B: the lexical representation used for keywords and phrases.

English text only (non-English reviews are translated first). Steps, justified in DESIGN.md §4:
lowercase -> product-term synonyms -> collapse elongations -> spaCy lemmas -> drop punctuation,
emoji, placeholders and stopwords (negations are KEPT) -> per-sentence token lists, so phrases never
cross a sentence boundary. No stemming and no spelling correction.
"""

import re
from collections.abc import Iterable, Sequence
from functools import lru_cache

import emoji
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.language import Language

from review_analysis.preprocessing.normalize import EMAIL_PLACEHOLDER, URL_PLACEHOLDER

# Negations and intensity words carry the complaint ("not sync", "too slow", "nothing works").
# spaCy's stop list, like sklearn's, contains them, so we take them back out.
NEGATIONS = frozenset(
    {
        "not",
        "no",
        "never",
        "nor",
        "cannot",
        "without",
        "nothing",
        "none",
        "nobody",
        "nowhere",
        "neither",
    }
)
KEEP_WORDS = NEGATIONS | {"too"}
BASE_STOPWORDS = frozenset(w.lower() for w in STOP_WORDS) - KEEP_WORDS
# Always-removed domain words: the app's own name and the generic word for it.
ALWAYS_DOMAIN_STOPWORDS = frozenset({"notion", "app", "application"})

# A small, explicit synonym map for product vocabulary, applied before tokenisation.
SYNONYMS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bi[\s-]?pads?\b"), "ipad"),
    (re.compile(r"\bi[\s-]?phones?\b"), "iphone"),
    (re.compile(r"\bmac\s?os\b"), "macos"),
    (re.compile(r"\boff[\s-]line\b"), "offline"),
    (re.compile(r"\bwi[\s-]?fi\b"), "wifi"),
    (re.compile(r"\bdark[\s-]mode\b"), "darkmode"),
    (re.compile(r"\ba\.?i\.?\b"), "ai"),
)
_ELONGATION_RE = re.compile(r"(\w)\1{2,}")  # "sooo" -> "soo", "Goood" -> "Good"


@lru_cache(maxsize=1)
def _nlp() -> Language:
    return spacy.load("en_core_web_sm", disable=["ner"])


def prepare(text_en: str) -> str:
    text = text_en.replace(URL_PLACEHOLDER, " ").replace(EMAIL_PLACEHOLDER, " ")
    text = emoji.replace_emoji(text, " ")
    text = text.lower()
    for pattern, replacement in SYNONYMS:
        text = pattern.sub(replacement, text)
    return _ELONGATION_RE.sub(r"\1\1", text)


def lemma_sentences(texts_en: Iterable[str]) -> list[list[list[str]]]:
    """For each text, a list of sentences, each a list of kept lemmas.

    Domain stopwords are NOT removed here; see is_informative().
    """
    out: list[list[list[str]]] = []
    for doc in _nlp().pipe((prepare(t) for t in texts_en), batch_size=64):
        sentences: list[list[str]] = []
        for sent in doc.sents:
            tokens = [
                lemma
                for tok in sent
                if not (tok.is_punct or tok.is_space or tok.like_url or tok.like_email)
                and (lemma := tok.lemma_.lower().strip())
                and any(ch.isalnum() for ch in lemma)
                and lemma not in BASE_STOPWORDS
                and (len(lemma) > 1 or lemma.isdigit())
            ]
            if tokens:
                sentences.append(tokens)
        out.append(sentences)
    return out


def derive_domain_stopwords(
    positive_docs: Sequence[set[str]],
    negative_docs: Sequence[set[str]],
    *,
    min_df: float = 0.08,
    max_ratio: float = 1.5,
) -> set[str]:
    """Words frequent in BOTH positive and negative reviews, at a similar rate: they describe the
    app in general rather than what people like or dislike. Data-driven instead of hand-picked.
    """

    def doc_freq(docs: Sequence[set[str]]) -> dict[str, float]:
        counts: dict[str, int] = {}
        for doc in docs:
            for w in doc:
                counts[w] = counts.get(w, 0) + 1
        return {w: c / max(1, len(docs)) for w, c in counts.items()}

    pos, neg = doc_freq(positive_docs), doc_freq(negative_docs)
    derived = {
        w
        for w in pos.keys() & neg.keys()
        if pos[w] >= min_df
        and neg[w] >= min_df
        and max(pos[w], neg[w]) / min(pos[w], neg[w]) <= max_ratio
    }
    return (derived | ALWAYS_DOMAIN_STOPWORDS) - KEEP_WORDS


def is_informative(ngram: str, domain_stopwords: set[str] | frozenset[str]) -> bool:
    """Domain stopwords are removed as keywords, but kept inside phrases.

    "work" alone says nothing, but "not work" is a complaint, so a phrase survives if it has a
    non-domain word or combines a negation with something else.
    """
    tokens = ngram.split()
    if any(t not in domain_stopwords and t not in KEEP_WORDS for t in tokens):
        return True
    return len(tokens) > 1 and any(t in NEGATIONS for t in tokens)


def ngrams(tokens: Sequence[str], n_max: int = 3) -> list[str]:
    """1..n_max-grams within one sentence."""
    return [
        " ".join(tokens[i : i + n]) for n in range(1, n_max + 1) for i in range(len(tokens) - n + 1)
    ]
