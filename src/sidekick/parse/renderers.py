"""Turn video renderers (up-next ``compactVideoRenderer`` and search ``videoRenderer``)
into ``Candidate`` objects. Both share almost the same shape."""

from __future__ import annotations

from typing import Any

from sidekick.models import Candidate
from sidekick.parse.util import find_all, parse_count, parse_duration, text_of


def _overlay_styles(r: dict[str, Any]) -> set[str]:
    return {
        str(o.get("style", "")).upper() for o in find_all(r, "thumbnailOverlayTimeStatusRenderer")
    }


def _badge_flags(r: dict[str, Any]) -> str:
    parts = []
    for b in find_all(r, "metadataBadgeRenderer"):
        parts.append(str(b.get("style", "")))
        parts.append(str(b.get("label", "")))
    return " ".join(parts).upper()


def candidate_from_renderer(r: dict[str, Any]) -> Candidate | None:
    video_id = r.get("videoId")
    if not video_id:
        return None

    styles = _overlay_styles(r)
    badges = _badge_flags(r)
    length_text = text_of(r.get("lengthText"))
    views_text = text_of(r.get("viewCountText"))

    is_upcoming = "upcomingEventData" in r or "UPCOMING" in styles
    is_live = (
        "LIVE" in styles
        or "LIVE_NOW" in badges
        or "LIVE" in badges.split()
        or "watching" in views_text.lower()
    ) and not is_upcoming
    is_short = "SHORTS" in styles or any(True for _ in find_all(r, "reelWatchEndpoint"))

    channel = text_of(r.get("longBylineText")) or text_of(r.get("ownerText"))
    channel = channel or text_of(r.get("shortBylineText"))

    return Candidate(
        video_id=str(video_id),
        title=text_of(r.get("title")),
        channel=channel,
        duration_s=parse_duration(length_text),
        views=None if (is_live or is_upcoming) else parse_count(views_text),
        published_text=text_of(r.get("publishedTimeText")),
        is_live=is_live,
        is_upcoming=is_upcoming,
        is_short=is_short,
    )
