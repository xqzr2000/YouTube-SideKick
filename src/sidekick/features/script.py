"""Writing-system check for titles: is a title written mostly in a non-Latin script?

The profile's ``language: en`` rule is applied in Stage B on title + description, so a video
that isn't enriched (over the enrichment budget) is never language-checked. The script of the
title catches the clear cases before any page is fetched, without a model: a title written
mostly in Japanese, Chinese, Korean, Cyrillic, Arabic, Devanagari, Thai… can't be an English
video. Latin-script titles are left to fastText in Stage B (French or German can't be told from
English by script).

Unlike the language rule this check also applies to music (Russell, 2026-11-06): a German or
French concert still passes, a song titled in Japanese doesn't.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable

# Languages normally written in a non-Latin script. When the profile allows any of them, the
# script check is off (a Japanese title may be exactly what you want).
NON_LATIN_LANGS = frozenset(
    "am ar be bg bn el fa gu he hi hy ja ka km kn ko lo mk ml mn mr my ne pa ru si sr ta te th "
    "uk ur yi zh".split()
)
_NAMES = {"HIRAGANA": "japanese", "KATAKANA": "japanese", "CJK": "cjk", "HANGUL": "korean"}
MIN_LETTERS = 4
MIN_SHARE = 0.5
# A Latin letter is a sound; a kanji, kana or Hangul block is a syllable or a whole word. Weighted
# by ~3, "【MV】夜に駆ける / YOASOBI" counts as mostly Japanese while "The Great Guitars - 東京
# 1980" or "BTS (방탄소년단) Dynamite Official MV" stay mostly Latin.
_WEIGHT = {"japanese": 3, "cjk": 3, "korean": 3}


def _script(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    name = unicodedata.name(ch, "")
    if not name or "LATIN" in name:
        return "latin"
    first = name.split()[0]
    return _NAMES.get(first, first.lower())


def non_latin(text: str | None, *, min_letters: int = MIN_LETTERS, share: float = MIN_SHARE):
    """The main non-Latin script of ``text`` if at least ``share`` of its letters are written
    in one (e.g. ``"japanese"``, ``"cyrillic"``), else None. Digits, punctuation and emoji
    don't count; CJK, kana and Hangul characters weigh 3 (a syllable or word each, against one
    Latin letter); less than ``min_letters`` letters' weight is too little to judge."""
    counts: Counter[str] = Counter()
    for sc in map(_script, text or ""):
        if sc:
            counts[sc] += _WEIGHT.get(sc, 1)
    total = sum(counts.values())
    other = total - counts["latin"]
    if total < min_letters or other / total < share:
        return None
    # Japanese titles mix kana with kanji; any kana makes the whole title Japanese
    if counts["japanese"]:
        return "japanese"
    return max((s for s in counts if s != "latin"), key=lambda s: counts[s])


def applies(languages: Iterable[str]) -> bool:
    """The check runs when the profile names languages and all of them use the Latin script."""
    langs = {lang.lower() for lang in languages}
    return bool(langs) and not (langs & NON_LATIN_LANGS)
