"""Language identification with lingua, with corrections for low-confidence Latin-script text.

On our Notion pool, unrestricted detection labelled short English reviews as Ganda ("iPad. Laggy"),
Yoruba ("Wowie. Slay") and Latin (".. nice app"). Restricting to plausible languages still left
low-confidence misses such as "One star. Won't open" -> Dutch (0.17). Rules for Latin-script text
whose top guess is not English:

1. confidence >= ACCEPT_CONFIDENCE                 -> accept the detector's language;
2. else it contains an unambiguous English word     -> English (method "english_cue");
3. else fewer than MIN_WORDS words                  -> "und" (method "short_uncertain");
4. else                                             -> accept the detector's language.

Non-Latin scripts (Chinese, Korean, Arabic, Cyrillic, ...) are reliable and always accepted.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from lingua import Language, LanguageDetector, LanguageDetectorBuilder

from review_analysis.preprocessing.normalize import words

CANDIDATES = (
    Language.ENGLISH,
    Language.SPANISH,
    Language.FRENCH,
    Language.GERMAN,
    Language.ITALIAN,
    Language.PORTUGUESE,
    Language.DUTCH,
    Language.RUSSIAN,
    Language.UKRAINIAN,
    Language.POLISH,
    Language.TURKISH,
    Language.ARABIC,
    Language.HINDI,
    Language.CHINESE,
    Language.JAPANESE,
    Language.KOREAN,
    Language.INDONESIAN,
    Language.VIETNAMESE,
    Language.TAGALOG,
    Language.SWEDISH,
)
MIN_WORDS = 4
ACCEPT_CONFIDENCE = 0.90
# Words that are English and not common words of the other Latin-script candidates.
# ("me", "a", "no", "de" are deliberately absent: they are Spanish/French/Dutch too.)
ENGLISH_CUES = frozenset(
    {
        "the",
        "and",
        "is",
        "it",
        "its",
        "it's",
        "this",
        "that",
        "with",
        "for",
        "of",
        "to",
        "was",
        "are",
        "but",
        "so",
        "my",
        "you",
        "your",
        "i",
        "i'm",
        "im",
        "won't",
        "don't",
        "can't",
        "doesn't",
        "isn't",
        "not",
        "very",
        "just",
        "best",
        "good",
        "great",
        "love",
        "star",
        "stars",
        "ever",
        "works",
        "work",
        "thanks",
        "thank",
        "like",
        "lags",
    }
)

Method = Literal["detector", "english_cue", "short_uncertain", "no_letters"]


@dataclass(frozen=True)
class LanguageGuess:
    lang: str  # ISO 639-1, or "und" when there is nothing to detect
    confidence: float
    method: Method


@lru_cache(maxsize=1)
def _detector() -> LanguageDetector:
    return (
        LanguageDetectorBuilder.from_languages(*CANDIDATES).with_preloaded_language_models().build()
    )


def _is_latin_script(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and all(ord(ch) < 0x250 for ch in letters)  # Basic Latin .. Latin Ext-B


def detect_language(text: str) -> LanguageGuess:
    if not any(ch.isalpha() for ch in text):
        return LanguageGuess("und", 0.0, "no_letters")
    values = _detector().compute_language_confidence_values(text)
    top = values[0]
    lang = top.language.iso_code_639_1.name.lower()
    if lang != "en" and _is_latin_script(text) and top.value < ACCEPT_CONFIDENCE:
        tokens = {w.lower().replace("\u2019", "'") for w in words(text)}
        if tokens & ENGLISH_CUES:
            en = next((v.value for v in values if v.language is Language.ENGLISH), 0.0)
            return LanguageGuess("en", en, "english_cue")
        if len(tokens) < MIN_WORDS:
            return LanguageGuess("und", top.value, "short_uncertain")
    return LanguageGuess(lang, top.value, "detector")
