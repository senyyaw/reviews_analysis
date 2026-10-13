"""Sentence splitting for any language, used for per-sentence theme assignment.

spaCy's rule-based sentencizer on the multi-language blank pipeline: punctuation-based (including
CJK 。！？), maintained, and dependency-free beyond spaCy itself. Reviews often lack punctuation, so
normalize_text() has already turned line breaks into sentence ends.
"""

import re
from functools import lru_cache

import spacy
from spacy.language import Language

# Chinese/Japanese text has no spaces, so the tokenizer would keep "太可惜了。体验很不好" as one
# token and the sentencizer could never split it. A space after full-width marks fixes that.
_CJK_END_RE = re.compile(r"([。！？])(?=\S)")


@lru_cache(maxsize=1)
def _splitter() -> Language:
    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(text: str) -> list[str]:
    text = _CJK_END_RE.sub(r"\1 ", text)
    return [s.text.strip() for s in _splitter()(text).sents if s.text.strip()]
