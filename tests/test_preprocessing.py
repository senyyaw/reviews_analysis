"""Preprocessing tests. Most examples are real reviews from the Notion pool."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from review_analysis.models import Review
from review_analysis.preprocessing import dedupe, lexical
from review_analysis.preprocessing.language import detect_language
from review_analysis.preprocessing.normalize import join_title_body, normalize_text
from review_analysis.preprocessing.pipeline import preprocess
from review_analysis.preprocessing.sentences import split_sentences
from review_analysis.preprocessing.translate import Translation, TranslationCache

NOW = datetime(2026, 10, 13, tzinfo=UTC)

# --- shared normalisation -------------------------------------------------------------------


def test_title_and_body_are_joined_once():
    assert join_title_body("Heavy", "680mb is crazy") == "Heavy. 680mb is crazy"
    assert join_title_body("Love it!", "Great") == "Love it! Great"
    assert join_title_body("", "only body") == "only body"


def test_normalisation_repairs_but_never_removes_content():
    text = normalize_text(
        "Don&#39;t   sync\u200b!!\n\uff49\uff30\uff41\uff44 \U0001f621 680mb don\u00e2\u20ac\u2122t"
    )
    assert text == "Don't sync!! iPad 😡 680mb don't"


def test_line_breaks_become_sentence_boundaries():
    assert normalize_text("Love it\nBut slow on iPad") == "Love it. But slow on iPad"


def test_links_and_emails_are_replaced():
    text = normalize_text("see https://notion.so/x or mail me@x.com")
    assert "<URL>" in text and "<EMAIL>" in text and "notion.so" not in text


def test_multi_part_emoji_survive():
    assert "\U0001f469\u200d\U0001f4bb" in normalize_text("coding \U0001f469\u200d\U0001f4bb")


# --- language identification ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "iPad. Laggy",  # was Ganda before the fixes
        "One star. Won't open",  # was Dutch
        "Shvt app. lags like a fuvkin joke",  # was Swedish
        "Claude and notion is a banger combination",  # was French
        "The best app ever. Amaiziiiiiiiinggg!!!!",
    ],
)
def test_short_or_informal_english_is_english(text):
    assert detect_language(text).lang == "en"


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Me encanta esta app. Realmente hace mi trabajo más fácil", "es"),
        ("Une perte de temps. Le pire logiciel de correction, il est inutilisable.", "fr"),
        ("跟ipadpro2022的横屏不适配. 如上", "zh"),
        ("Дайте Русский язык!!! Если вы хотите заработать много денег", "ru"),
    ],
)
def test_other_languages_are_detected(text, lang):
    assert detect_language(text).lang == lang


def test_short_ambiguous_text_is_undetermined_not_guessed():
    assert detect_language("Wowie. Slay").lang == "und"
    assert detect_language("🔥🔥🔥").method == "no_letters"


# --- sentences ------------------------------------------------------------------------------


def test_sentence_split_handles_cjk_punctuation():
    assert len(split_sentences("Good. Slow on iPad! 太可惜了。体验很不好")) == 4


# --- Track B (lexical) ----------------------------------------------------------------------


def tokens_of(text: str) -> list[str]:
    return [t for sent in lexical.lemma_sentences([text])[0] for t in sent]


def test_negations_are_kept():
    tokens = tokens_of("It doesn't sync and I can't open pages")
    assert tokens.count("not") == 2
    assert "sync" in tokens and "page" in tokens


def test_numbers_are_kept_and_emoji_and_links_removed():
    tokens = tokens_of("680mb is crazy 😡 see <URL>")
    assert "680" in tokens or "680mb" in tokens
    assert "url" not in tokens and not any("😡" in t for t in tokens)


def test_synonyms_and_elongations_are_normalised():
    tokens = tokens_of("Sooo laggy on i pad and iPads, need off-line")
    assert tokens.count("ipad") == 2 and "offline" in tokens
    assert "sooo" not in tokens


def test_phrases_do_not_cross_sentences():
    sents = lexical.lemma_sentences(["Sync is slow. Pages crash."])[0]
    grams = {g for s in sents for g in lexical.ngrams(s)}
    assert "sync slow" in grams and "page crash" in grams
    assert "slow page" not in grams


def test_domain_stopwords_are_kept_inside_phrases():
    domain = {"work", "app", "use"}
    assert not lexical.is_informative("work", domain)
    assert lexical.is_informative("not work", domain)
    assert lexical.is_informative("offline mode", domain)
    assert not lexical.is_informative("use app", domain)


def test_domain_stopwords_are_derived_from_both_groups():
    positive = [{"use", "love", "page"}, {"use", "love"}, {"use", "great"}]
    negative = [{"use", "crash", "page"}, {"use", "crash"}, {"use", "slow"}]
    derived = lexical.derive_domain_stopwords(positive, negative, min_df=0.3)
    assert "use" in derived and "notion" in derived
    assert "love" not in derived and "crash" not in derived


# --- duplicates -----------------------------------------------------------------------------


def test_exact_duplicate_needs_same_author():
    items = [
        dedupe.DedupeItem("1", "a", "Great app!", 2),
        dedupe.DedupeItem("2", "b", "great app", 2),  # different author: legitimate
        dedupe.DedupeItem("3", "a", "Great  app", 2),  # same author, same text: repost
    ]
    assert dedupe.exact_duplicates(items) == {"3": "1"}


def test_near_duplicates_are_grouped():
    base = "The app keeps crashing every time I open a long page on my iPad"
    items = [
        dedupe.DedupeItem("1", "a", base, 14),
        dedupe.DedupeItem("2", "b", base + "!!", 14),
        dedupe.DedupeItem("3", "c", "Love the templates and the calendar view", 7),
    ]
    groups = dedupe.near_duplicate_groups(items)
    assert groups["1"] == groups["2"] and "3" not in groups


# --- pipeline -------------------------------------------------------------------------------


def review(
    review_id: str, title: str, body: str, rating: int, *, author: str = "x", days_ago: int = 0
) -> Review:
    return Review(
        review_id=review_id,
        app_id=1,
        storefront="us",
        date=NOW - timedelta(days=days_ago),
        rating=rating,
        title=title,
        body=body,
        author_hash=author,
        source="apple_web",
        fetched_at=NOW,
    )


def test_pipeline_end_to_end_with_cached_translation(tmp_path: Path):
    cache = TranslationCache(tmp_path / "t.json")
    cache.put(
        "3", Translation(src_lang="es", text_en="Somewhat disappointing. It freezes.", model="test")
    )
    reviews = [
        review("1", "Heavy", "680mb is crazy and it doesn't sync", 1, author="a"),
        review("2", "Love it", "Best notes app, templates are great", 5, author="b"),
        review("3", "Algo desepcionante", "Se congela luego de algunas acciones", 2, author="c"),
        review("4", "Heavy", "680mb is crazy and it doesn't sync", 1, author="a", days_ago=-1),
        review("5", "Wow", "", 5, author="d"),
    ]
    items, report = preprocess(reviews, cache)
    by_id = {p.review_id: p for p in items}

    assert by_id["1"].text_model == "Heavy. 680mb is crazy and it doesn't sync"
    assert by_id["4"].flags.duplicate_of == "1" and by_id["4"].excluded
    assert by_id["3"].lang == "es" and by_id["3"].translated
    assert by_id["3"].text_en == "Somewhat disappointing. It freezes."
    assert "freeze" in by_id["3"].keyword_tokens
    assert by_id["5"].flags.short and by_id["5"].keyword_tokens == []
    assert "not" in by_id["1"].keyword_tokens
    assert report.exact_duplicates == 1 and report.translated == 1 and report.untranslated == 0


def test_missing_translation_is_flagged_not_guessed(tmp_path: Path):
    items, report = preprocess(
        [review("1", "Algo desepcionante", "Se congela luego de algunas acciones", 2)],
        TranslationCache(tmp_path / "none.json"),
    )
    assert items[0].flags.untranslated and items[0].keyword_tokens == []
    assert report.untranslated == 1
