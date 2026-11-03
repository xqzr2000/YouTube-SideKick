"""Parse a YouTube search results page into candidates."""

from __future__ import annotations

from sidekick.models import SearchPage
from sidekick.parse.renderers import candidate_from_renderer
from sidekick.parse.util import extract_json, find_all

# Renderer keys that hold Shorts shelves/items. Shorts are always excluded, but we count them.
_SHORTS_KEYS = ("reelItemRenderer", "shortsLockupViewModel")


def parse_search(html: str) -> SearchPage:
    data = extract_json(html, "ytInitialData")
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
