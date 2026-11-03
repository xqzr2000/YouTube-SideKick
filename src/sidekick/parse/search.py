"""Parse a YouTube search results page into candidates."""

from __future__ import annotations

from sidekick.models import SearchPage
from sidekick.parse.renderers import candidate_from_renderer
from sidekick.parse.util import extract_json, find_all

# Renderer keys that hold Shorts shelves/items. Shorts are always excluded, but we count them.
_SHORTS_KEYS = ("reelItemRenderer", "shortsLockupViewModel")


def extract_search_blobs(html: str) -> dict:
    return {"data": extract_json(html, "ytInitialData")}


def parse_search(html: str) -> SearchPage:
    return parse_search_data(extract_search_blobs(html)["data"])


def parse_search_data(data: dict | None) -> SearchPage:
    if not isinstance(data, dict):
        return SearchPage(candidates=[])

    seen: set[str] = set()
    candidates = []
    for r in find_all(data, "videoRenderer"):
        cand = candidate_from_renderer(r)
        if cand and cand.video_id not in seen:
            seen.add(cand.video_id)
            cand.source = "query"
            candidates.append(cand)

    shorts = sum(1 for key in _SHORTS_KEYS for _ in find_all(data, key))
    return SearchPage(candidates=candidates, shorts_skipped=shorts)
