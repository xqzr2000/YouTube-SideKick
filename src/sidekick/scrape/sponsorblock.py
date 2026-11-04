"""SponsorBlock public API: community-marked sponsor / self-promo segments. No account needed.

We use the *hash-prefix* endpoint (``/api/skipSegments/<first 4 hex of sha256(id)>``): the
server returns every video sharing that prefix, so it never learns which exact video we asked
about. A 404 means "no segments for any video with this prefix".

Results feed the ads/sponsorship features (``sponsor_frac``, ``selfpromo_frac``). A network
problem gives ``None`` (unknown), never an exception, and after a few consecutive failures the
client stops calling for the rest of the compute so a dead API can't eat the 90 s budget.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import httpx

API_BASE = "https://sponsor.ajay.app/api/skipSegments"
CATEGORIES = ("sponsor", "selfpromo")
USER_AGENT = "YouTube-SideKick/0.1 (+https://github.com/xqzr2000/YouTube-SideKick)"
MAX_FAILURES = 3


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    category: str


def hash_prefix(video_id: str) -> str:
    return hashlib.sha256(video_id.encode()).hexdigest()[:4]


def covered_fraction(segments: list[Segment], duration_s: int | None, category: str) -> float:
    """Share of the video covered by ``category`` segments (overlaps merged), 0..1."""
    if not duration_s:
        return 0.0
    spans = sorted(
        (max(0.0, s.start), min(float(duration_s), s.end))
        for s in segments
        if s.category == category and s.end > s.start
    )
    total = 0.0
    cur_start = cur_end = None
    for a, b in spans:
        if cur_end is None or a > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = a, b
        else:
            cur_end = max(cur_end, b)
    if cur_end is not None:
        total += cur_end - cur_start
    return round(min(1.0, total / duration_s), 4)


class SponsorBlock:
    def __init__(self, client: httpx.Client | None = None, *, timeout: float = 8.0):
        self.client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=True
        )
        self.failures = 0
        self.calls = 0

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> SponsorBlock:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def gave_up(self) -> bool:
        return self.failures >= MAX_FAILURES

    def segments(self, video_id: str) -> list[Segment] | None:
        """Segments for one video; ``[]`` if none are marked; ``None`` if unknown (API down)."""
        if self.gave_up:
            return None
        self.calls += 1
        try:
            r = self.client.get(
                f"{API_BASE}/{hash_prefix(video_id)}",
                params={"categories": json.dumps(list(CATEGORIES))},
            )
        except httpx.HTTPError:
            self.failures += 1
            return None
        if r.status_code == 404:
            self.failures = 0
            return []
        if r.status_code != 200:
            self.failures += 1
            return None
        self.failures = 0
        try:
            entries = r.json()
        except ValueError:
            return None
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict) and entry.get("videoID") == video_id:
                out = []
                for seg in entry.get("segments", []):
                    try:
                        start, end = seg["segment"][:2]
                        out.append(Segment(float(start), float(end), str(seg["category"])))
                    except (KeyError, TypeError, ValueError):
                        continue
                return out
        return []
