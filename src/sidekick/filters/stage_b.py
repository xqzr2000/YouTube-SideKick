"""Stage B: enrichment for Stage A survivors (spec §5.6).

For each survivor (up to ``ENRICH_CAP``, in first-seen order, so up-next comes first):

1. **Watch page** (signed out): exact views, publish date, duration, description, keywords,
   category. A video enriched within ``ENRICH_TTL`` is read back from the database instead,
   so a refresh minutes later costs almost no page fetches.
2. **SponsorBlock**: share of the video marked as sponsor / self-promo (cached on the video
   row with the same TTL).
3. **Language** on title + description, which is much more reliable than the title alone.

Stage B drops a video only for hard facts the exact data reveals: ``lang:<code>`` (confident,
not in your languages), ``live`` / ``upcoming``, and ``duration<min`` / ``duration>max`` on the
exact duration. The language rule doesn't apply to music: a German or French concert is still
worth watching. It is skipped when the candidate's category is in ``LANG_EXEMPT_CATEGORIES``
(Music), and also when the *current* video's category is, because uploaders often file concerts
under Entertainment. For those the language guess is only a feature, marked ``lang_exempt``
(``"category"`` or ``"context"``). Everything else (sponsor share, age, views) is a
*feature* for Stage C and ranking, logged with the decision.

Videos past the cap, or whose page failed, are kept and marked ``enriched: false``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from sidekick.features.lang import LangDetector
from sidekick.filters.stage_a import LANG_MIN_CONF, Decision
from sidekick.models import Candidate, VideoMeta
from sidekick.profile.parser import Profile
from sidekick.scrape.fetcher import PageCapReached
from sidekick.scrape.sponsorblock import SponsorBlock, covered_fraction

STAGE = "B"
ENRICH_CAP = 50
ENRICH_TTL = timedelta(hours=24)
DESC_CHARS_FOR_LID = 1000
# YouTube categories where the spoken/written language says little about the video.
LANG_EXEMPT_CATEGORIES = frozenset({"Music"})

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_TAG_RE = re.compile(r"[#@]\w+")


@dataclass
class EnrichStats:
    fetched: int = 0
    cached: int = 0
    failed: int = 0
    over_cap: int = 0
    sponsor_calls: int = 0
    sponsor_unknown: int = 0


def lid_text(meta: VideoMeta) -> str:
    """Title + start of the description, without links and hashtags (they confuse LID)."""
    desc = _TAG_RE.sub(" ", _URL_RE.sub(" ", meta.description[:DESC_CHARS_FOR_LID]))
    return f"{meta.title} {desc}"


def age_days(publish_date: str, now: datetime | None = None) -> float | None:
    if not publish_date:
        return None
    try:
        dt = datetime.fromisoformat(publish_date)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return round(((now or datetime.now(UTC)) - dt).total_seconds() / 86400, 2)


def _fresh(ts: str | None, ttl: timedelta) -> bool:
    if not ts:
        return False
    return datetime.now(UTC) - datetime.fromisoformat(ts) <= ttl


def _sponsor_features(
    meta: VideoMeta, store, sponsorblock: SponsorBlock | None, stats: EnrichStats, ttl
) -> dict[str, Any]:
    row = store.get_video(meta.video_id) if store else None
    if row and _fresh(row.get("sponsor_checked_at"), ttl):
        return {
            "sponsor_frac": row["sponsor_frac"],
            "selfpromo_frac": row["selfpromo_frac"],
            "sponsor_segments": row["sponsor_segments"],
        }
    if sponsorblock is None:
        return {"sponsor_frac": None, "selfpromo_frac": None, "sponsor_segments": None}
    segs = sponsorblock.segments(meta.video_id)
    if segs is None:
        stats.sponsor_unknown += 1
        return {"sponsor_frac": None, "selfpromo_frac": None, "sponsor_segments": None}
    feats = {
        "sponsor_frac": covered_fraction(segs, meta.duration_s, "sponsor"),
        "selfpromo_frac": covered_fraction(segs, meta.duration_s, "selfpromo"),
        "sponsor_segments": len(segs),
    }
    if store:
        store.set_sponsor(
            meta.video_id,
            sponsor_frac=feats["sponsor_frac"],
            selfpromo_frac=feats["selfpromo_frac"],
            n_segments=feats["sponsor_segments"],
        )
    return feats


def run_stage_b(
    survivors: Sequence[Candidate],
    profile: Profile,
    *,
    fetcher,
    store=None,
    sponsorblock: SponsorBlock | None = None,
    detector: LangDetector | None = None,
    cap: int = ENRICH_CAP,
    ttl: timedelta = ENRICH_TTL,
    lang_min_conf: float = LANG_MIN_CONF,
    current_category: str | None = None,
) -> tuple[list[Decision], EnrichStats]:
    """Enrich survivors and decide. ``BlockedError`` propagates: the compute stops as blocked.

    ``current_category`` is the current video's YouTube category; a music context switches the
    language rule off for every candidate.
    """
    rules = profile.rules
    allowed = set(rules.languages)
    music_context = current_category in LANG_EXEMPT_CATEGORIES
    stats = EnrichStats()
    decisions: list[Decision] = []
    capped = False

    for i, c in enumerate(survivors):
        if i >= cap or capped:
            stats.over_cap += 1
            decisions.append(Decision(c, kept=True, features={"enriched": False}))
            continue

        meta = store.get_meta(c.video_id, max_age=ttl) if store else None
        if meta is not None:
            stats.cached += 1
            cached = True
        else:
            cached = False
            try:
                meta = fetcher.watch(c.video_id).meta
            except PageCapReached:
                capped = True
                stats.over_cap += 1
                decisions.append(Decision(c, kept=True, features={"enriched": False}))
                continue
            except httpx.HTTPError:
                meta = None
            if meta is not None and meta.video_id != c.video_id:
                meta = None  # redirected / removed video: don't file its data under this id
            if meta is None:
                stats.failed += 1
                decisions.append(Decision(c, kept=True, features={"enriched": False}))
                continue
            stats.fetched += 1
            if store:
                store.upsert_meta(meta)

        feats: dict[str, Any] = {
            "enriched": True,
            "cached": cached,
            "duration_s": meta.duration_s,
            "views": meta.views,
            "age_days": age_days(meta.publish_date),
            "category": meta.category,
            "n_keywords": len(meta.keywords),
            "desc_chars": len(meta.description),
        }
        feats.update(_sponsor_features(meta, store, sponsorblock, stats, ttl))

        reasons: list[str] = []
        if meta.is_live:
            reasons.append("live")
        if meta.is_upcoming:
            reasons.append("upcoming")
        d = meta.duration_s
        if d is not None and rules.min_duration_s is not None and d < rules.min_duration_s:
            reasons.append("duration<min")
        if d is not None and rules.max_duration_s is not None and d > rules.max_duration_s:
            reasons.append("duration>max")
        if detector is not None:
            guess = detector.detect(lid_text(meta))
            if guess is not None:
                feats["lang"] = guess.lang
                feats["lang_conf"] = round(guess.conf, 4)
                if store:
                    store.set_language(meta.video_id, guess.lang, feats["lang_conf"])
                if allowed and guess.lang not in allowed and guess.conf >= lang_min_conf:
                    if meta.category in LANG_EXEMPT_CATEGORIES:
                        feats["lang_exempt"] = "category"
                    elif music_context:
                        feats["lang_exempt"] = "context"
                    else:
                        reasons.append(f"lang:{guess.lang}")
        decisions.append(Decision(c, kept=not reasons, reasons=reasons, features=feats))

    if sponsorblock is not None:
        stats.sponsor_calls = sponsorblock.calls
    if store:
        store.commit()
    return decisions, stats
