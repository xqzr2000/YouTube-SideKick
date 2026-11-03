"""Parse YouTube's newer ``lockupViewModel`` items (the up-next layout on current watch pages).

Shape (only the parts we use)::

    lockupViewModel:
      contentId: "<video id>"
      contentType: "LOCKUP_CONTENT_TYPE_VIDEO" | "..._PLAYLIST" | ...
      contentImage.thumbnailViewModel.overlays[...]
        ...thumbnailBadgeViewModel: {text: "12:34" | "LIVE" | "SHORTS", badgeStyle: "..."}
      metadata.lockupMetadataViewModel:
        title: {content: "<title>"}
        metadata.contentMetadataViewModel.metadataRows[]:
          metadataParts[]: {text: {content: "<channel>" | "1.2M views" | "2 years ago"}}

The parser searches by key name rather than fixed paths, so small layout moves don't break it.
"""

from __future__ import annotations

import re
from typing import Any

from sidekick.models import Candidate
from sidekick.parse.util import find_all, parse_count, parse_duration

_VIEWS_RE = re.compile(r"\b(views?|watching|waiting)\b", re.I)
_AGE_RE = re.compile(r"\b(ago|streamed|premiered)\b", re.I)
_UPCOMING_RE = re.compile(r"\b(upcoming|premieres|scheduled for)\b", re.I)


def _content(node: Any) -> str:
    if isinstance(node, dict):
        if "content" in node:
            return str(node["content"])
        if "simpleText" in node:
            return str(node["simpleText"])
    return ""


def _metadata_rows(meta: dict[str, Any]) -> list[list[str]]:
    rows = []
    for cm in find_all(meta, "contentMetadataViewModel"):
        for row in cm.get("metadataRows", []) or []:
            parts = [_content(p.get("text")) for p in row.get("metadataParts", []) or []]
            parts = [p for p in parts if p]
            if parts:
                rows.append(parts)
        break
    return rows


def _badges(lockup: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(b.get("text", "")), str(b.get("badgeStyle", "")))
        for b in find_all(lockup, "thumbnailBadgeViewModel")
        if isinstance(b, dict)
    ]


def is_video_lockup(lockup: dict[str, Any]) -> bool:
    ctype = str(lockup.get("contentType", ""))
    return bool(lockup.get("contentId")) and (ctype == "" or ctype.endswith("_VIDEO"))


def candidate_from_lockup(lockup: dict[str, Any]) -> Candidate | None:
    if not is_video_lockup(lockup):
        return None
    meta = next(find_all(lockup, "lockupMetadataViewModel"), {})
    title = _content(meta.get("title")) if isinstance(meta, dict) else ""
    rows = _metadata_rows(meta) if isinstance(meta, dict) else []
    parts = [p for row in rows for p in row]

    # Channel: first part of the first row that isn't a view count or an age.
    channel = ""
    for p in rows[0] if rows else []:
        if not _VIEWS_RE.search(p) and not _AGE_RE.search(p):
            channel = p
            break
    views_text = next((p for p in parts if _VIEWS_RE.search(p)), "")
    age_text = next((p for p in parts if _AGE_RE.search(p)), "")

    badges = _badges(lockup)
    duration = next((d for d in (parse_duration(t) for t, _ in badges) if d is not None), None)
    badge_blob = " ".join(f"{t} {s}" for t, s in badges).upper()

    is_upcoming = (
        "UPCOMING" in badge_blob
        or any(_UPCOMING_RE.search(p) for p in parts)
        or "waiting" in views_text.lower()
    )
    is_live = ("LIVE" in badge_blob or "watching" in views_text.lower()) and not is_upcoming
    is_short = (
        "SHORTS" in badge_blob
        or any(True for _ in find_all(lockup, "reelWatchEndpoint"))
        or any("/shorts/" in str(u) for u in find_all(lockup, "url"))
    )

    return Candidate(
        video_id=str(lockup["contentId"]),
        title=title,
        channel=channel,
        duration_s=duration,
        views=None if (is_live or is_upcoming) else parse_count(views_text),
        published_text=age_text,
        is_live=is_live,
        is_upcoming=is_upcoming,
        is_short=is_short,
    )
