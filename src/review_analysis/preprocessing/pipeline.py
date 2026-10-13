"""Run all preprocessing for a pool of reviews and produce a report of what was done."""

import logging
from collections import Counter
from collections.abc import Sequence

import emoji
from pydantic import BaseModel, Field

from review_analysis.models import Review
from review_analysis.preprocessing import dedupe, lexical
from review_analysis.preprocessing.language import detect_language
from review_analysis.preprocessing.normalize import join_title_body, normalize_text, words
from review_analysis.preprocessing.sentences import split_sentences
from review_analysis.preprocessing.translate import M2MTranslator, Translation, TranslationCache

log = logging.getLogger(__name__)

SHORT_WORDS = 3


class Flags(BaseModel):
    short: bool = False  # fewer than SHORT_WORDS words: kept for ratings/sentiment, no keywords
    emoji_only: bool = False
    n_emoji: int = 0
    duplicate_of: str | None = None  # exact repost by the same author -> excluded from analysis
    near_duplicate_group: int | None = None  # flagged, not excluded
    untranslated: bool = False  # non-English but no translation available -> no keywords


class ProcessedReview(BaseModel):
    review_id: str
    rating: int
    text_raw: str
    text_model: str  # Track A: shared normalisation only
    lang: str
    lang_confidence: float
    lang_method: str
    translated: bool = False
    text_en: str | None = None  # English text used by Track B (original if English)
    sentences: list[str] = Field(default_factory=list)  # original language, for themes
    lemma_sentences: list[list[str]] = Field(default_factory=list)  # Track B lemmas
    flags: Flags = Field(default_factory=Flags)

    @property
    def excluded(self) -> bool:
        return self.flags.duplicate_of is not None

    @property
    def keyword_tokens(self) -> list[str]:
        return [t for sent in self.lemma_sentences for t in sent]


class PreprocessReport(BaseModel):
    n_reviews: int
    languages: dict[str, int]
    language_methods: dict[str, int]
    translated: int
    untranslated: int
    short: int
    emoji_only: int
    exact_duplicates: int
    near_duplicate_groups: int
    near_duplicate_reviews: int
    domain_stopwords: list[str]


def preprocess(
    reviews: Sequence[Review],
    cache: TranslationCache,
    *,
    translator: M2MTranslator | None = None,
) -> tuple[list[ProcessedReview], PreprocessReport]:
    chronological = sorted(reviews, key=lambda r: (r.date, r.review_id))
    out: list[ProcessedReview] = []
    for r in chronological:
        raw = join_title_body(r.title, r.body)
        clean = normalize_text(raw)
        guess = detect_language(clean)
        n_words = len(words(emoji.replace_emoji(clean, " ")))
        out.append(
            ProcessedReview(
                review_id=r.review_id,
                rating=r.rating,
                text_raw=raw,
                text_model=clean,
                lang=guess.lang,
                lang_confidence=round(guess.confidence, 4),
                lang_method=guess.method,
                sentences=split_sentences(clean),
                flags=Flags(
                    short=n_words < SHORT_WORDS,
                    emoji_only=n_words == 0 and emoji.emoji_count(clean) > 0,
                    n_emoji=emoji.emoji_count(clean),
                ),
            )
        )

    _translate(out, cache, translator)
    _mark_duplicates(out, chronological)
    domain_stopwords = _lemmatise(out)

    report = PreprocessReport(
        n_reviews=len(out),
        languages=dict(Counter(p.lang for p in out).most_common()),
        language_methods=dict(Counter(p.lang_method for p in out).most_common()),
        translated=sum(p.translated for p in out),
        untranslated=sum(p.flags.untranslated for p in out),
        short=sum(p.flags.short for p in out),
        emoji_only=sum(p.flags.emoji_only for p in out),
        exact_duplicates=sum(p.flags.duplicate_of is not None for p in out),
        near_duplicate_groups=len({p.flags.near_duplicate_group for p in out} - {None}),
        near_duplicate_reviews=sum(p.flags.near_duplicate_group is not None for p in out),
        domain_stopwords=sorted(domain_stopwords),
    )
    out.sort(key=lambda p: p.review_id)
    return out, report


def _translate(
    items: list[ProcessedReview], cache: TranslationCache, translator: M2MTranslator | None
) -> None:
    for p in items:
        if p.lang in ("en", "und"):
            p.text_en = p.text_model
    pending = [p for p in items if p.text_en is None and cache.get(p.review_id) is None]
    if pending and translator is not None:
        by_lang: dict[str, list[ProcessedReview]] = {}
        for p in pending:
            by_lang.setdefault(p.lang, []).append(p)
        for lang, group in by_lang.items():
            if lang not in translator.supported:
                log.warning(
                    "translator does not support %r; %d review(s) left untranslated",
                    lang,
                    len(group),
                )
                continue
            for p, text_en in zip(
                group, translator.translate([g.text_model for g in group], lang), strict=True
            ):
                cache.put(
                    p.review_id,
                    Translation(src_lang=lang, text_en=text_en, model=translator.model_id),
                )
    for p in items:
        if p.text_en is not None:
            continue
        cached = cache.get(p.review_id)
        if cached is not None:
            p.text_en, p.translated = cached.text_en, True
        else:
            p.flags.untranslated = True


def _mark_duplicates(items: list[ProcessedReview], chronological: Sequence[Review]) -> None:
    by_id = {p.review_id: p for p in items}
    dedupe_items = [
        dedupe.DedupeItem(
            r.review_id,
            r.author_hash,
            by_id[r.review_id].text_model,
            len(words(by_id[r.review_id].text_model)),
        )
        for r in chronological
    ]
    for review_id, original in dedupe.exact_duplicates(dedupe_items).items():
        by_id[review_id].flags.duplicate_of = original
    for review_id, group in dedupe.near_duplicate_groups(dedupe_items).items():
        by_id[review_id].flags.near_duplicate_group = group


def _lemmatise(items: list[ProcessedReview]) -> set[str]:
    usable = [p for p in items if p.text_en is not None and not p.excluded and not p.flags.short]
    for p, sents in zip(
        usable, lexical.lemma_sentences(p.text_en or "" for p in usable), strict=True
    ):
        p.lemma_sentences = sents
    domain = lexical.derive_domain_stopwords(
        [set(p.keyword_tokens) for p in usable if p.rating >= 4],
        [set(p.keyword_tokens) for p in usable if p.rating <= 2],
    )
    return domain
