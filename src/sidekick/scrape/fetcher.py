"""Polite, signed-out fetcher for YouTube watch and search pages.

- One ``httpx`` client, browser-like User-Agent (override with ``SIDEKICK_USER_AGENT``).
- A 1-2 s jittered delay between requests and a page cap per compute (default 60).
- Consent pages, bot checks and HTTP 429 raise ``BlockedError`` and start a back-off window
  (default 30 min) recorded in ``~/.sidekick/blocked_until``; fetches are refused until it ends.
- Every page's JSON blobs (not the full HTML) go to the raw log for replay.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import quote_plus

import httpx

from sidekick.config import Paths, get_paths
from sidekick.models import SearchPage, WatchPage
from sidekick.parse.search import extract_search_blobs, parse_search_data
from sidekick.parse.watch import extract_watch_blobs, parse_watch_data
from sidekick.store import rawlog

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
_BLOCK_MARKERS = (
    "consent.youtube.com",
    "Before you continue to YouTube",
    "unusual traffic from your computer network",
    "/sorry/index",
)


class BlockedError(RuntimeError):
    """YouTube served a consent/bot page or rate-limited us."""


class PageCapReached(RuntimeError):
    """The per-compute page budget is used up."""


@dataclass
class FetchConfig:
    min_delay: float = 1.0
    max_delay: float = 2.0
    page_cap: int = 60
    timeout: float = 20.0
    backoff_minutes: int = 30
    user_agent: str = ""

    def __post_init__(self) -> None:
        self.user_agent = self.user_agent or os.environ.get("SIDEKICK_USER_AGENT", DEFAULT_UA)


@dataclass
class ThumbCheck:
    video_id: str
    url: str
    status: int
    content_type: str
    size: int


def midpoint_thumbnail_url(video_id: str) -> str:
    """YouTube's auto-generated frame from the middle of the video (``hq2``; ``hq1``/``hq3``
    are the 25%/75% frames). Used instead of the click-optimized custom thumbnail."""
    return f"https://i.ytimg.com/vi/{video_id}/hq2.jpg"


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def search_url(query: str) -> str:
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"


class Fetcher:
    def __init__(
        self,
        config: FetchConfig | None = None,
        *,
        client: httpx.Client | None = None,
        paths: Paths | None = None,
        sleep: Callable[[float], None] = time.sleep,
        log_raw: bool = True,
    ) -> None:
        self.config = config or FetchConfig()
        self.paths = paths or get_paths()
        self.sleep = sleep
        self.log_raw = log_raw
        self.pages_fetched = 0
        self._last_request = False
        self.client = client or httpx.Client(
            headers={
                "User-Agent": self.config.user_agent,
                "Accept-Language": "en-US,en;q=0.9",
            },
            follow_redirects=True,
            timeout=self.config.timeout,
        )

    # -- context manager ------------------------------------------------------------------
    def __enter__(self) -> Fetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.client.close()

    # -- back-off --------------------------------------------------------------------------
    @property
    def _blocked_file(self):
        return self.paths.home / "blocked_until"

    def blocked_until(self) -> datetime | None:
        try:
            until = datetime.fromisoformat(self._blocked_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None
        return until if until > datetime.now(UTC) else None

    def _start_backoff(self, reason: str) -> None:
        until = datetime.now(UTC) + timedelta(minutes=self.config.backoff_minutes)
        self.paths.home.mkdir(parents=True, exist_ok=True)
        self._blocked_file.write_text(until.isoformat())
        raise BlockedError(f"{reason}; backing off until {until:%H:%M} UTC")

    # -- requests --------------------------------------------------------------------------
    def _wait(self) -> None:
        if self._last_request:
            self.sleep(random.uniform(self.config.min_delay, self.config.max_delay))
        self._last_request = True

    def _get(self, url: str, *, counts_toward_cap: bool = True) -> httpx.Response:
        if until := self.blocked_until():
            raise BlockedError(f"backing off until {until:%H:%M} UTC")
        if counts_toward_cap and self.pages_fetched >= self.config.page_cap:
            raise PageCapReached(f"page cap of {self.config.page_cap} reached")
        self._wait()
        r = self.client.get(url)
        if counts_toward_cap:
            self.pages_fetched += 1
        if r.status_code == 429:
            self._start_backoff("HTTP 429 Too Many Requests")
        return r

    def _get_page(self, url: str) -> str:
        r = self._get(url)
        final = str(r.url)
        head = r.text[:200_000]
        if any(m in final or m in head for m in _BLOCK_MARKERS):
            self._start_backoff(f"consent or bot page at {final}")
        r.raise_for_status()
        return r.text

    def watch(self, video_id: str) -> WatchPage:
        html = self._get_page(watch_url(video_id))
        blobs = extract_watch_blobs(html)
        if self.log_raw:
            rawlog.append("watch", video_id, blobs, paths=self.paths)
        return parse_watch_data(blobs["player"], blobs["data"])

    def search(self, query: str) -> SearchPage:
        html = self._get_page(search_url(query))
        blobs = extract_search_blobs(html)
        if self.log_raw:
            rawlog.append("search", query, blobs, paths=self.paths)
        return parse_search_data(blobs["data"])

    def check_thumbnail(self, video_id: str) -> ThumbCheck:
        """Fetch the midpoint thumbnail once to confirm the URL pattern works (not capped)."""
        url = midpoint_thumbnail_url(video_id)
        r = self._get(url, counts_toward_cap=False)
        return ThumbCheck(
            video_id=video_id,
            url=url,
            status=r.status_code,
            content_type=r.headers.get("content-type", ""),
            size=len(r.content),
        )
