"""Small helpers for walking YouTube's embedded JSON and parsing display text."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

_MULT = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


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
    """``"1,234 views"`` -> 1234, ``"1.2M views"`` -> 1200000, ``"No views"`` -> 0."""
    t = text.strip()
    if not t:
        return None
    if t.lower().startswith("no "):
        return 0
    m = re.search(r"([\d][\d,]*\.?\d*)\s*([KMB])?", t)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    return int(num * _MULT.get(m.group(2) or "", 1))
