"""Shared normalisation, applied before both tracks.

It only repairs and standardises text (HTML entities, broken encodings, Unicode forms, invisible
characters, links). It never removes words, case, punctuation, emoji or numbers. Those decisions are
track-specific.
"""

import html
import re
import unicodedata

import ftfy

URL_PLACEHOLDER = "<URL>"
EMAIL_PLACEHOLDER = "<EMAIL>"

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Zero-width spaces/joiners, BOM, soft hyphen. U+200D (ZWJ) is kept: it glues multi-part emoji.
_INVISIBLE_RE = re.compile("[\u200b\u200c\u2060\ufeff\u00ad]")
_SENTENCE_END = (".", "!", "?", "\u2026", "\u3002", "\uff01", "\uff1f")  # incl. CJK 。！？
ZWJ = "\u200d"  # zero-width joiner: glues multi-part emoji, so it is kept
_SPACES_RE = re.compile("[ \t\u00a0]+")
_WORD_RE = re.compile("[^\\W_]+(?:['\u2019][^\\W_]+)?")


def join_title_body(title: str, body: str) -> str:
    """'Heavy' + '680mb is crazy' -> 'Heavy. 680mb is crazy'. Titles often carry the verdict."""
    title, body = title.strip(), body.strip()
    if not title:
        return body
    if not body:
        return title
    sep = " " if title.endswith(_SENTENCE_END) else ". "
    return f"{title}{sep}{body}"


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = ftfy.fix_text(text)  # mojibake such as "donâ€™t" -> "don't"
    text = unicodedata.normalize("NFKC", text)  # full-width letters, ligatures, compatibility forms
    text = _INVISIBLE_RE.sub("", text)
    text = "".join(
        ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C" or ch == ZWJ
    )
    text = _URL_RE.sub(URL_PLACEHOLDER, text)
    text = _EMAIL_RE.sub(EMAIL_PLACEHOLDER, text)
    # A line break usually separates ideas: make it a sentence boundary if it is not one already.
    lines = [_SPACES_RE.sub(" ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    text = " ".join(
        line if i == len(lines) - 1 or line.endswith(_SENTENCE_END) else f"{line}."
        for i, line in enumerate(lines)
    )
    return text.strip()


def words(text: str) -> list[str]:
    """Word-like tokens (letters/digits, with inner apostrophes). Used for counts and flags."""
    return _WORD_RE.findall(text)
