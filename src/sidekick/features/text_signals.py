"""Hand-built text signals for Stage C: a clickbait score for titles, a promo score for
descriptions. Both are v0, interpretable, and return their parts so every drop can be explained
and the weights can later be replaced by a learned model (M6).

Clickbait (title), each part in [0, 1]:

- ``caps``: share of 4+-letter words written in ALL CAPS, saturating at 30 %. Four letters and
  at least one vowel, so lecture acronyms like "MIT", "PCA", "DQN", "MCMC" don't count.
- ``punct``: ``!`` and ``?`` count, saturating at 2.
- ``emoji``: any emoji / pictograph.
- ``bait``: a bait phrase ("you won't believe", "freaks out", "like this", "destroys", ...).

``score = 0.4·caps + 0.2·punct + 0.2·emoji + 0.4·bait``, clipped to 1. Examples from real
up-next lists: "🚨Trump FREAKS OUT OVER BLUE WAVE!!!" → 1.0; "Nobody Explained Higgs Boson
Like THIS!" → ~0.77; "StatQuest: PCA main ideas in only 5 minutes!!!" → 0.2.

Promo (description, links kept, since a shop or sponsor URL is itself evidence): the number of
distinct promo *groups* found, /3, clipped to 1. Groups: sponsor read, discount code, affiliate
link, app/platform pitch, booking/hire pitch, merch/tickets. The first 1500 characters only:
sponsor lines sit at the top, and long descriptions end in boilerplate.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_WORD_RE = re.compile(r"[^\W\d_]{4,}")
_VOWEL_RE = re.compile(r"[AEIOUY]")
_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),  # pictographs, emoticons, transport, symbols & pictographs ext.
    (0x2600, 0x27BF),  # misc symbols, dingbats
    (0x2B00, 0x2BFF),  # arrows/stars (⭐, ⬆)
)
BAIT_PHRASES = (
    "you won't believe",
    "you wont believe",
    "won't believe",
    "freaks out",
    "freak out",
    "destroys",
    "destroyed",
    "obliterates",
    "humiliates",
    "melts down",
    "meltdown",
    "goes off on",
    "slams",
    "shocking",
    "insane",
    "mind blowing",
    "mind-blowing",
    "nobody explained",
    "no one explained",
    "nobody talks about",
    "nobody is talking",
    "like this",
    "the truth about",
    "exposed",
    "gone wrong",
    "what happens next",
    "will blow your mind",
    "the answer will",
    "you need to know",
    "before it's too late",
    "this changes everything",
)
_BAIT_RES = [re.compile(rf"(?<!\w){re.escape(p)}(?!\w)") for p in BAIT_PHRASES]

PROMO_CHARS = 1500
PROMO_GROUPS: dict[str, tuple[str, ...]] = {
    "sponsor": (
        r"\bsponsored by\b",
        r"\bthis video is sponsored\b",
        r"\bthanks to .{1,40}? for sponsoring\b",
        r"\bpaid promotion\b",
        r"\btoday'?s sponsor\b",
    ),
    "discount": (
        r"\b\d{1,2}\s?% off\b",
        r"\b(?:promo|discount|coupon) code\b",
        r"\buse (?:my )?code\b",
    ),
    "affiliate": (r"\baffiliate\b", r"amzn\.to/", r"\bamazon\.com/shop\b", r"\bgeni\.us/"),
    "app_pitch": (
        r"\bdownload the .{0,25}?app\b",
        r"\bon the .{0,30}?app\b",
        r"\bfree trial\b",
        r"\bstart (?:your|a) (?:free )?subscription\b",
    ),
    "booking": (
        r"\bbook .{0,40}? for your (?:event|show|party|company)\b",
        r"\bfor bookings?\b",
        r"\bbooking inquiries\b",
        r"\bhire .{0,20}? for your\b",
    ),
    "merch": (r"\bmerch\b", r"\bshop now\b", r"\bbuy tickets\b", r"\bget tickets\b"),
}
_PROMO_RES = {g: [re.compile(p, re.I) for p in pats] for g, pats in PROMO_GROUPS.items()}


def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    if any(lo <= cp <= hi for lo, hi in _EMOJI_RANGES):
        return unicodedata.category(ch) in ("So", "Sk", "Sm") or cp >= 0x1F000
    return False


def clickbait(title: str) -> tuple[float, dict[str, Any]]:
    """``(score, parts)`` for a title. Empty title → 0."""
    t = title or ""
    words = _WORD_RE.findall(t)
    shouty = sum(w.isupper() and bool(_VOWEL_RE.search(w)) for w in words)
    caps_ratio = shouty / len(words) if words else 0.0
    n_punct = t.count("!") + t.count("?")
    n_emoji = sum(_is_emoji(ch) for ch in t)
    low = " ".join(t.casefold().split()).replace("’", "'")
    bait = [p for p, rx in zip(BAIT_PHRASES, _BAIT_RES, strict=True) if rx.search(low)]
    parts = {
        "caps": round(min(caps_ratio / 0.3, 1.0), 3),
        "punct": round(min(n_punct / 2, 1.0), 3),
        "emoji": 1.0 if n_emoji else 0.0,
        "bait": 1.0 if bait else 0.0,
    }
    score = 0.4 * parts["caps"] + 0.2 * parts["punct"] + 0.2 * parts["emoji"]
    score += 0.4 * parts["bait"]
    return round(min(score, 1.0), 3), {**parts, "bait_phrases": bait[:3]}


def promo(description: str) -> tuple[float, list[str]]:
    """``(score, groups found)`` for a description (raw text, links included)."""
    d = (description or "")[:PROMO_CHARS].replace("’", "'")
    groups = [g for g, rxs in _PROMO_RES.items() if any(rx.search(d) for rx in rxs)]
    return round(min(len(groups) / 3, 1.0), 3), groups
