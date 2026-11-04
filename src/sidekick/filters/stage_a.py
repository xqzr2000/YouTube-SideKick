"""Stage A: cheap hard filters, using only fields from up-next and search results (spec §5.3).

One decision per unique video per compute (a video seen in up-next *and* a search is judged
once). Every check runs, so a dropped video carries *all* its reasons (``"short;lang:de"``),
which keeps NB02's "what gets filtered and why" counts honest. Kept videos have no reason.

Reasons:

========================  ==============================================================
``current_video``         the video being watched (searches often return it)
``short``                 YouTube Short
``live`` / ``upcoming``   live stream / scheduled premiere
``downvoted``             you 👎'd it before
``channel:<name>``        channel in ``exclude_channels`` (case-insensitive, exact name)
``keyword:<kw>``          keyword in ``exclude_keywords`` found in the title (whole words)
``duration<min``          shorter than ``min_duration`` (unknown duration is kept)
``duration>max``          longer than ``max_duration``
========================  ==============================================================

Language is **not** a Stage A reason. The title's fastText guess is recorded as a feature
(``title_lang``, ``title_lang_conf``) and Stage B decides on title + description. Titles alone
were too unreliable: on 2026-11-04 "MIT 6.S191: Reinforcement Learning" came back German at
≥ 0.80 and was dropped, though that lecture series is one the user rates good.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from typing import Any

from sidekick.features.lang import LangDetector
from sidekick.models import Candidate
from sidekick.profile.parser import Profile

STAGE = "A"
LANG_MIN_CONF = 0.80


@dataclass
class Decision:
    candidate: Candidate
    kept: bool
    reasons: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def video_id(self) -> str:
        return self.candidate.video_id

    @property
    def reason(self) -> str | None:
        return ";".join(self.reasons) or None


def _norm(s: str) -> str:
    return " ".join(s.casefold().split())


def _keyword_patterns(keywords: Iterable[str]) -> list[tuple[str, re.Pattern[str]]]:
    return [
        (kw, re.compile(rf"(?<!\w){re.escape(_norm(kw))}(?!\w)")) for kw in keywords if _norm(kw)
    ]


def unique_candidates(cands: Iterable[Candidate]) -> list[Candidate]:
    """First sighting of each video, in order."""
    seen: set[str] = set()
    out = []
    for c in cands:
        if c.video_id not in seen:
            seen.add(c.video_id)
            out.append(c)
    return out


def run_stage_a(
    cands: Iterable[Candidate],
    profile: Profile,
    *,
    current_video_id: str | None = None,
    downvoted: Collection[str] = (),
    detector: LangDetector | None = None,
) -> list[Decision]:
    rules = profile.rules
    blocked_channels = {_norm(ch) for ch in rules.exclude_channels if _norm(ch)}
    keyword_res = _keyword_patterns(rules.exclude_keywords)
    allowed_langs = set(rules.languages)
    downvoted = set(downvoted)

    decisions = []
    for c in unique_candidates(cands):
        reasons: list[str] = []
        feats: dict[str, Any] = {
            "duration_s": c.duration_s,
            "views": c.views,
            "is_short": c.is_short,
            "is_live": c.is_live,
            "is_upcoming": c.is_upcoming,
        }
        if c.video_id == current_video_id:
            reasons.append("current_video")
        if c.is_short:
            reasons.append("short")
        if c.is_live:
            reasons.append("live")
        if c.is_upcoming:
            reasons.append("upcoming")
        if c.video_id in downvoted:
            reasons.append("downvoted")
        if c.channel and _norm(c.channel) in blocked_channels:
            reasons.append(f"channel:{c.channel}")
        title = _norm(c.title)
        for kw, rx in keyword_res:
            if rx.search(title):
                reasons.append(f"keyword:{kw}")
        if c.duration_s is not None:
            if rules.min_duration_s is not None and c.duration_s < rules.min_duration_s:
                reasons.append("duration<min")
            if rules.max_duration_s is not None and c.duration_s > rules.max_duration_s:
                reasons.append("duration>max")
        if detector is not None and c.title:
            guess = detector.detect(c.title)
            if guess is not None:
                feats["title_lang"] = guess.lang
                feats["title_lang_conf"] = round(guess.conf, 4)
        elif allowed_langs:
            feats["title_lang"] = None  # no model: language not checked
        decisions.append(Decision(c, kept=not reasons, reasons=reasons, features=feats))
    return decisions


def log_decisions(
    store, compute_id: str, decisions: Iterable[Decision], *, stage: str = STAGE
) -> None:
    with store.conn:
        for d in decisions:
            store.log_decision(
                compute_id,
                d.video_id,
                stage=stage,
                kept=d.kept,
                reason=d.reason,
                features=d.features,
            )
