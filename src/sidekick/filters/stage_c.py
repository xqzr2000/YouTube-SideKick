"""Interest gate (spec §5.4) and Stage C soft filters (spec §5.7).

Both work in one *similarity space* per compute (``SimSpace``):

- representation: ``e5-small-qp`` if the e5 model is downloaded, else TF-IDF fitted on the
  database (much weaker; see the 2026-11-04 ``just sim --eval``);
- **centered** vectors (the mean of every cached video vector subtracted, then re-normalized)
  for comparisons with profile lines and downvoted videos, where raw e5 cosines bunch up;
- **raw** vectors for similarity to the current video, which the labels ranked better raw.

**Gate.** ``gate_score`` = best centered similarity of the current video to any Interests
line. At or above τ → ``interest`` mode, else ``filter-only``. In M3 the mode is only decided
and logged; backfill (M4) is what acts on it.

**Stage C** runs on Stage B survivors and drops a video when any signal reaches its threshold
(precision-first, see ``thresholds``):

==========================  ==========================================================
``dont_show:<line>``        centered similarity to a Don't-show-me line
``like_downvoted``          centered similarity to a video you 👎'd
``clickbait``               title clickbait score (``features.text_signals``)
``promo``                   description promo score
==========================  ==========================================================

Every score is logged as a feature whether it drops the video or not, so thresholds can be
re-tuned from the log without re-running anything.

**News context** (decided 2026-11-05, Russell: while watching news, show news, not pundits).
When the current video is news (:func:`sidekick.topic.news_context`: YouTube category
"News & Politics" or "news" in the channel/title), a Don't-show-me match only drops a candidate
that is **not itself news** (late-night shows are filed under Entertainment) or whose title
reads like **commentary** (clickbait ≥ ``thresholds.PUNDIT_CLICKBAIT``; reason ``pundit:<line>``).
Straight news reports are kept and marked ``dont_show_spared = "news_report"``. On 2026-11-05
real titles, pundit channels scored clickbait 0.36–0.40 ("Trump DOOMED the GOP", "BREAKING:
MAJOR UPDATE…") and newscasts 0.00. Outside a news context nothing changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sidekick.embed import center, corpus, encode_videos
from sidekick.embed import e5 as e5mod
from sidekick.embed.tfidf import TfidfRep
from sidekick.features.text_signals import clickbait, promo
from sidekick.filters.stage_a import Decision
from sidekick.filters.thresholds import Thresholds
from sidekick.models import Candidate
from sidekick.profile.parser import Profile
from sidekick.store.db import Store
from sidekick.topic import news_context

STAGE = "C"
MIN_CACHED_FOR_MEAN = 20  # below this, the mean comes from the vectors at hand


@dataclass
class SimSpace:
    rep: Any
    mean: np.ndarray
    interests: list[str]
    interest_vecs: np.ndarray  # centered
    dont_show: list[str]
    dont_show_vecs: np.ndarray  # centered
    downvoted_vecs: np.ndarray  # centered
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.rep.name

    def videos(self, store: Store, ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """``(raw, centered)`` vectors for ``ids`` (cached for e5)."""
        raw, st = encode_videos(store, self.rep, list(ids))
        for k, v in st.items():
            self.stats[k] = self.stats.get(k, 0) + v
        if raw.size == 0:
            return raw, raw
        return raw, center(raw, self.mean)


def _cached_mean(store: Store, model: str) -> tuple[np.ndarray | None, int]:
    rows = store.conn.execute("SELECT vector FROM embeddings WHERE model = ?", (model,)).fetchall()
    if len(rows) < MIN_CACHED_FOR_MEAN:
        return None, len(rows)
    m = np.vstack([np.frombuffer(r[0], dtype="<f4") for r in rows]).mean(axis=0, keepdims=True)
    return m.astype(np.float32), len(rows)


def build_space(
    store: Store,
    profile: Profile,
    *,
    e5=None,
    paths=None,
    extra_ids: Sequence[str] = (),
) -> SimSpace:
    """The similarity space for one compute. ``extra_ids`` (the current video and the
    survivors) are embedded up front so a young database still gets a sensible mean."""
    interests = list(profile.interests)
    dont_show = [d.text for d in profile.dont_show if d.text]
    if e5 is None:
        e5 = e5mod.load(paths)
    if e5 is not None:
        rep = e5 if getattr(e5, "name", "") == e5mod.NAME_QP else e5.with_passages()
    else:
        rep = TfidfRep().fit(corpus(store, interests + dont_show))

    space = SimSpace(rep, np.zeros((1, 1), np.float32), interests, None, dont_show, None, None)
    space.stats = {"cached": 0, "computed": 0}
    raw_extra = None
    if extra_ids:
        raw_extra, st = encode_videos(store, rep, list(extra_ids))
        space.stats.update(st)
    mean = None
    n_cached = 0
    if rep.cacheable:
        mean, n_cached = _cached_mean(store, rep.name)
    if mean is None:
        if rep.cacheable:
            pool = raw_extra
        else:  # TF-IDF: the mean over the whole (cheap) corpus
            ids = [r[0] for r in store.conn.execute("SELECT video_id FROM videos")]
            pool, _ = encode_videos(store, rep, ids)
        mean = (
            pool.mean(axis=0, keepdims=True)
            if pool is not None and len(pool)
            else np.zeros((1, rep.encode(["x"]).shape[1]), np.float32)
        )
    space.mean = mean.astype(np.float32)
    space.stats["mean_from"] = n_cached

    def lines(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, space.mean.shape[1]), np.float32)
        return center(rep.encode(texts), space.mean)

    space.interest_vecs = lines(interests)
    space.dont_show_vecs = lines(dont_show)
    down = sorted(store.downvoted_video_ids())
    space.downvoted_vecs = (
        space.videos(store, down)[1] if down else np.zeros((0, space.mean.shape[1]), np.float32)
    )
    return space


@dataclass
class GateResult:
    mode: str  # interest | filter-only
    score: float | None
    best_interest: str | None
    tau: float
    rep: str


def interest_gate(store: Store, space: SimSpace, current_id: str, tau: float) -> GateResult:
    if not space.interests or store.get_video(current_id) is None:
        return GateResult("filter-only", None, None, tau, space.name)
    _, c = space.videos(store, [current_id])
    sims = (c @ space.interest_vecs.T).ravel()
    j = int(np.argmax(sims))
    score = round(float(sims[j]), 4)
    mode = "interest" if score >= tau else "filter-only"
    return GateResult(mode, score, space.interests[j], tau, space.name)


def _max(vec: np.ndarray, mat: np.ndarray) -> tuple[float | None, int | None]:
    if mat is None or mat.shape[0] == 0:
        return None, None
    s = mat @ vec
    j = int(np.argmax(s))
    return round(float(s[j]), 4), j


def run_stage_c(
    store: Store,
    space: SimSpace,
    survivors: Sequence[Candidate],
    *,
    current_id: str | None,
    thresholds: Thresholds,
) -> list[Decision]:
    """Score and decide each survivor. Kept videos carry all scores as features."""
    if not survivors:
        return []
    th = thresholds.for_rep(space.name)
    ids = [c.video_id for c in survivors]
    raw, cen = space.videos(store, ids)
    cur_raw = None
    cur = store.get_video(current_id) if current_id else None
    if cur is not None:
        cur_raw = space.videos(store, [current_id])[0][0]
    news = news_context(cur) if cur else ""
    out = []
    for i, c in enumerate(survivors):
        v = store.get_video(c.video_id) or {}
        feats: dict[str, Any] = {"rep": space.name}
        reasons: list[str] = []
        ds, dj = _max(cen[i], space.dont_show_vecs)
        feats["dont_show_sim"] = ds
        if dj is not None:
            feats["dont_show_line"] = space.dont_show[dj]
        dv, _ = _max(cen[i], space.downvoted_vecs)
        feats["downvoted_sim"] = dv
        isim, ij = _max(cen[i], space.interest_vecs)
        feats["interest_sim"] = isim
        if ij is not None:
            feats["interest_line"] = space.interests[ij]
        if cur_raw is not None:
            feats["seed_sim"] = round(float(raw[i] @ cur_raw), 4)
        cb, cb_parts = clickbait(v.get("title") or c.title)
        pr, pr_groups = promo(v.get("description") or "")
        feats.update(clickbait=cb, clickbait_parts=cb_parts, promo=pr, promo_groups=pr_groups)

        if news:
            feats["news_context"] = news
            feats["cand_news"] = news_context(v or {"title": c.title, "channel": c.channel})
        if ds is not None and ds >= th["dont_show"]:
            line = space.dont_show[dj][:40]
            if not news or not feats["cand_news"]:
                reasons.append(f"dont_show:{line}")
            elif cb >= thresholds.pundit_clickbait:
                reasons.append(f"pundit:{line}")
            else:
                feats["dont_show_spared"] = "news_report"
        if dv is not None and dv >= th["downvoted"]:
            reasons.append("like_downvoted")
        if cb >= thresholds.clickbait:
            reasons.append("clickbait")
        if pr >= thresholds.promo:
            reasons.append("promo")
        out.append(Decision(c, kept=not reasons, reasons=reasons, features=feats))
    return out
