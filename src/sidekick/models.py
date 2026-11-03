"""Plain data models shared by the parser, store and (later) the ranking pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    """A video seen in an up-next list or a search result (cheap fields only)."""

    video_id: str
    title: str = ""
    channel: str = ""
    duration_s: int | None = None
    views: int | None = None
    published_text: str = (
        ""  # relative text such as "3 years ago"; exact date comes from enrichment
    )
    is_live: bool = False
    is_upcoming: bool = False
    is_short: bool = False
    source: str = ""  # upnext | query | snowball (set by the caller)


@dataclass
class VideoMeta:
    """Exact metadata from a watch page (enrichment)."""

    video_id: str
    title: str = ""
    channel: str = ""
    channel_id: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    duration_s: int | None = None
    views: int | None = None
    publish_date: str = ""  # ISO date from the page microformat, if present
    category: str = ""
    is_live: bool = False
    is_upcoming: bool = False


@dataclass
class WatchPage:
    meta: VideoMeta | None
    upnext: list[Candidate]
    lockups_skipped: int = 0  # up-next items in a layout the parser does not handle yet


@dataclass
class SearchPage:
    candidates: list[Candidate]
    shorts_skipped: int = 0
