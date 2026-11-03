"""Small helpers for walking YouTube's embedded JSON and parsing display text."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

_MULT = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
_WORDS = {"thousand": "K", "million": "M", "billion": "B"}


def extract_json(html: str, var_name: str) -> Any | None:
    """Return the JSON object assigned to ``var_name`` in the page, or None.

    Uses ``raw_decode`` so braces inside strings and trailing ``;</script>`` are handled.
    """
    for m in re.finditer(re.escape(var_name) + r"\s*=\s*", html):
        start = html.find("{", m.end())
        if start == -1 or start - m.end() > 5:
            continue
        try:
            obj, _ = json.JSONDecoder().raw_decode(html, start)
        except json.JSONDecodeError:
            continue
        return obj
    return None


def find_all(node: Any, key: str) -> Iterator[Any]:
    """Yield every value stored under ``key`` anywhere in a nested dict/list structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                yield v
            else:
                yield from find_all(v, key)
    elif isinstance(node, list):
        for item in node:
            yield from find_all(item, key)


def text_of(node: Any) -> str:
    """Text from a YouTube text node: ``{"simpleText": ...}`` or ``{"runs": [...]}``."""
    if not isinstance(node, dict):
        return ""
    if "simpleText" in node:
        return str(node["simpleText"])
    if "runs" in node:
        return "".join(str(r.get("text", "")) for r in node["runs"] if isinstance(r, dict))
    return ""


def parse_duration(text: str) -> int | None:
    """``"1:02:03"`` -> 3723, ``"4:05"`` -> 245. None if the text is not a clock time."""
    parts = text.strip().split(":")
    if not parts or not all(p.isdigit() for p in parts) or len(parts) > 3:
        return None
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + int(p)
    return seconds


def parse_count(text: str) -> int | None:
    """``"1,234 views"`` -> 1234, ``"1.2M"`` -> 1200000, ``"33 thousand views"`` -> 33000,
    ``"No views"`` -> 0."""
    t = text.strip()
    for word, suffix in _WORDS.items():
        t = re.sub(rf"\s*{word}\b", suffix, t, flags=re.I)
    if not t:
        return None
    if t.lower().startswith("no "):
        return 0
    m = re.search(r"([\d][\d,]*\.?\d*)\s*([KMB])?", t)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    return int(num * _MULT.get(m.group(2) or "", 1))


_AGE_UNITS = {
    "y": 365.0,
    "year": 365.0,
    "mo": 30.0,
    "month": 30.0,
    "w": 7.0,
    "week": 7.0,
    "d": 1.0,
    "day": 1.0,
    "h": 1 / 24,
    "hour": 1 / 24,
    "m": 1 / 1440,
    "min": 1 / 1440,
    "minute": 1 / 1440,
    "s": 1 / 86400,
    "second": 1 / 86400,
}
_AGE_RE = re.compile(
    r"(\d+)\s*(years?|y|months?|mo|weeks?|w|days?|d|hours?|h|minutes?|min|m|seconds?|s)\s+ago", re.I
)


def parse_age_days(text: str) -> float | None:
    """Approximate age in days from relative text.

    ``"9y ago"`` -> 3285, ``"1mo ago"`` -> 30, ``"Streamed 2 years ago"`` -> 730, ``"5 days ago"``
    -> 5. Months are 30 days and years 365, so this is only for features and charts; the exact
    date comes from the watch page.
    """
    m = _AGE_RE.search(text or "")
    if not m:
        return None
    unit = m.group(2).lower()
    if unit not in _AGE_UNITS:
        unit = unit.rstrip("s")
    return int(m.group(1)) * _AGE_UNITS[unit]
