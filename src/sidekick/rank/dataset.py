"""The M6 evaluation dataset: logged ranking pools joined with hand labels.

A **pool** is everything one compute ranked: its decisions at stage ``R`` (shown or not). For
each candidate in the pool the dataset gathers what was logged when it was ranked, so a new
ranker can be scored on exactly the choices v0 faced, with no new fetching:

- the v0 parts and context from the stage R features (topic, seed, interest, recurrence, ...);
- the Stage C scores from the same compute (Don't-show, downvoted, clickbait, promo);
- the Stage B facts if the video was enriched (sponsor share, language confidence, ...);
- a few video facts from the ``videos`` table (duration, age);
- the label: the newest candidate label for (current video, candidate), as good / meh / bad
  → gain 2 / 1 / 0.

Feature values that weren't logged (a video that wasn't enriched has no sponsor share) are
``nan``; models must handle missing values (LightGBM does; for logistic regression, M6c adds
an "is missing" indicator and imputes).

``load_pools(which="newest")`` takes the newest ranked compute per current video, the same
choice as E4 part 1, so each current video counts once. ``which="all"`` returns every ranked
compute (more rows for training; pools of one current video overlap).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sidekick.features import script as scr
from sidekick.recommend import STAGE_RANK
from sidekick.recommend.evaluate import latest_recommends
from sidekick.store.db import Store

GAIN = {"good": 2, "meh": 1, "bad": 0}

# name -> one-line meaning; the order is the column order of ``Pool.matrix``
FEATURES: dict[str, str] = {
    # v0 parts (0-1), as logged
    "topic": "topic-anchor match to the current video (series / story / keyphrases)",
    "seed": "e5 similarity to the current video, as a percentile within the pool",
    "interest": "e5 similarity to the best Interests line, percentile (0 in filter-only mode)",
    "recurrence": "how many lists surfaced it (1 list = 0, 3+ = 1)",
    "same_channel": "1 if same channel as the current video",
    "persistence": "earlier computes for this current video that also had it (3+ = 1)",
    "views": "log10(views) / 7, clipped",
    # raw similarities behind the percentiles
    "seed_sim": "raw e5 cosine to the current video",
    "interest_sim": "centered e5 cosine to the best Interests line",
    # topic-match parts
    "topic_tok": "IDF-weighted token overlap with the topic anchor",
    "topic_ent": "shared named entities",
    "topic_series": "shared series / course words",
    # where it came from
    "n_lists": "number of lists that surfaced it",
    "from_upnext": "in the signed-out up-next list",
    "from_tab": "in the signed-in up-next list from the tab",
    "from_search": "from a backfill search",
    "from_snowball": "from a snowball (up-next of another candidate)",
    "upnext_pos": "best position in an up-next list, 0-1 (1 = first; nan if not in one)",
    "enriched": "its watch page was fetched (Stage B)",
    # Stage C scores
    "dont_show_sim": "centered e5 similarity to the closest Don't-show line",
    "downvoted_sim": "similarity to the closest 👎 video",
    "clickbait": "hand-weighted clickbait score of the title",
    "promo": "promotional phrases in the description",
    # Stage B facts (nan if not enriched)
    "sponsor_frac": "SponsorBlock sponsor share of the video",
    "lang_conf": "fastText confidence of the detected language",
    "desc_chars_log": "log10(1 + description length)",
    # video facts
    "duration_log": "log10(duration in seconds)",
    "age_days_log": "log10(1 + age in days)",
    # context of the compute
    "interest_mode": "1 if the current video passed the interest gate",
    "ctx_lecture": "1 if lecture context (course caps)",
    "ctx_news": "1 if the current video is news (Stage C's news context)",
    "seed_foreign": "1 if the current video's title is mostly non-Latin script",
}
NAN = float("nan")


@dataclass
class Row:
    video_id: str
    label: str | None
    v0_score: float
    v0_rank: int | None  # 1-10 if shown, else None
    features: dict[str, float]
    group: str = ""  # diversity group as logged (course channel, performer, or channel)

    @property
    def gain(self) -> int | None:
        return GAIN.get(self.label) if self.label else None


@dataclass
class Pool:
    compute_id: str
    seed_id: str
    seed_title: str
    ts: str
    trigger: str
    rows: list[Row] = field(default_factory=list)
    kind: str = "other"  # lecture | other: which diversity cap applied
    rep: str = ""  # representation Stage C used (embeddings model name), for MMR replay

    @property
    def labeled(self) -> list[Row]:
        return [r for r in self.rows if r.label is not None]

    @property
    def coverage(self) -> float:
        return len(self.labeled) / len(self.rows) if self.rows else 0.0

    def matrix(self, rows: list[Row] | None = None) -> list[list[float]]:
        return [[r.features.get(f, NAN) for f in FEATURES] for r in (rows or self.rows)]


def _num(x: Any) -> float:
    if x is None or isinstance(x, (dict, list, str)):
        return NAN
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    return v


def _log10(x: Any, plus_one: bool = False) -> float:
    v = _num(x)
    if math.isnan(v) or v < 0 or (v == 0 and not plus_one):
        return NAN
    return round(math.log10(v + 1 if plus_one else v), 4)


def _age_days(published_at: str | None) -> float:
    if not published_at:
        return NAN
    try:
        pub = datetime.fromisoformat(published_at)
    except ValueError:
        return NAN
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - pub).total_seconds() / 86400)


def _stage_feats(store: Store, compute_id: str, stage: str) -> dict[str, dict]:
    return {
        vid: json.loads(f or "{}")
        for vid, f in store.conn.execute(
            "SELECT video_id, features FROM decisions WHERE compute_id = ? AND stage = ?",
            (compute_id, stage),
        )
    }


def _best_positions(store: Store, compute_id: str) -> dict[str, float]:
    """Best position in the up-next lists (signed-out or tab), scaled so 1 = first."""
    out: dict[str, float] = {}
    for source in ("upnext", "tab"):
        rows = store.conn.execute(
            "SELECT video_id, position FROM candidates WHERE compute_id = ? AND source = ? "
            "AND position IS NOT NULL",
            (compute_id, source),
        ).fetchall()
        n = max((p for _, p in rows), default=0)
        for vid, pos in rows:
            if n:
                val = 1.0 - (pos - 1) / n
                out[vid] = max(out.get(vid, 0.0), round(val, 4))
    return out


def features_for(
    r: dict, c: dict, b: dict, video: dict, upnext_pos: float, *, mode: str, seed_foreign: bool
) -> dict[str, float]:
    """One candidate's feature dict from its logged stage R / C / B features."""
    lists = r.get("lists") or []
    tp = r.get("topic_parts") or {}
    kind = r.get("kind") or ""
    return {
        **{k: _num(r.get(k)) for k in ("topic", "seed", "interest", "recurrence")},
        "same_channel": _num(r.get("same_channel")),
        "persistence": _num(r.get("persistence")),
        "views": _num(r.get("views")),
        "seed_sim": _num(r.get("seed_sim", c.get("seed_sim"))),
        "interest_sim": _num(r.get("interest_sim", c.get("interest_sim"))),
        "topic_tok": _num(tp.get("tok")),
        "topic_ent": _num(tp.get("ent")),
        "topic_series": _num(tp.get("series")),
        "n_lists": float(len(lists)) if lists else NAN,
        "from_upnext": float("upnext" in lists) if lists else NAN,
        "from_tab": float("tab" in lists) if lists else NAN,
        "from_search": float(any(x.startswith("query:") for x in lists)) if lists else NAN,
        "from_snowball": float(any(x.startswith("snowball:") for x in lists)) if lists else NAN,
        "upnext_pos": upnext_pos,
        "enriched": float(bool(r["enriched"])) if "enriched" in r else NAN,
        "dont_show_sim": _num(c.get("dont_show_sim")),
        "downvoted_sim": _num(c.get("downvoted_sim")),
        "clickbait": _num(c.get("clickbait")),
        "promo": _num(c.get("promo")),
        "sponsor_frac": _num(b.get("sponsor_frac")),
        "lang_conf": _num(b.get("lang_conf")),
        "desc_chars_log": _log10(b.get("desc_chars"), plus_one=True),
        "duration_log": _log10(video.get("duration_s")),
        "age_days_log": _log10(_age_days(video.get("published_at")), plus_one=True),
        "interest_mode": float(mode == "interest"),
        "ctx_lecture": float(kind == "lecture"),
        "ctx_news": float(bool(c.get("news_context"))),  # Stage C's news context (M6c fix)
        "seed_foreign": float(seed_foreign),
    }


def _labels(store: Store) -> dict[tuple[str, str], str]:
    rows = store.conn.execute(
        """SELECT c.current_video_id, l.video_id, l.label FROM labels l
           JOIN computes c ON c.compute_id = l.compute_id
           WHERE l.kind = 'candidate' ORDER BY l.ts, l.label_id"""
    ).fetchall()
    return {(seed, vid): lab for seed, vid, lab in rows}


def _ranked_computes(store: Store) -> list[tuple[str, str]]:
    return store.conn.execute(
        """SELECT c.compute_id, c.current_video_id FROM computes c
           WHERE c.current_video_id IS NOT NULL
             AND EXISTS (SELECT 1 FROM decisions d WHERE d.compute_id = c.compute_id
                         AND d.stage = ?)
           ORDER BY c.ts, c.rowid""",
        (STAGE_RANK,),
    ).fetchall()


def load_pools(store: Store, *, which: str = "newest") -> list[Pool]:
    """Ranking pools with features and labels, oldest compute first."""
    if which not in ("newest", "all"):
        raise ValueError("which must be 'newest' or 'all'")
    pairs = latest_recommends(store) if which == "newest" else _ranked_computes(store)
    labels = _labels(store)
    pools = []
    for cid, seed in pairs:
        comp = store.conn.execute(
            "SELECT ts, trigger, mode FROM computes WHERE compute_id = ?", (cid,)
        ).fetchone()
        seed_v = store.get_video(seed) or {}
        foreign = scr.non_latin(seed_v.get("title")) is not None
        c_feats = _stage_feats(store, cid, "C")
        b_feats = _stage_feats(store, cid, "B")
        pos = _best_positions(store, cid)
        pool = Pool(cid, seed, seed_v.get("title") or seed, comp[0], comp[1] or "")
        for vid, score, feats in store.conn.execute(
            "SELECT video_id, score, features FROM decisions WHERE compute_id = ? AND stage = ? "
            "ORDER BY rowid",
            (cid, STAGE_RANK),
        ):
            if vid == seed:
                continue
            r = json.loads(feats or "{}")
            mode = r.get("mode") or comp[2] or "filter-only"
            f = features_for(
                r,
                c_feats.get(vid, {}),
                b_feats.get(vid, {}),
                store.get_video(vid) or {},
                pos.get(vid, NAN),
                mode=mode,
                seed_foreign=foreign,
            )
            rank = r.get("rank")
            pool.kind = r.get("kind") or pool.kind
            pool.rep = pool.rep or (c_feats.get(vid) or {}).get("rep", "")
            pool.rows.append(
                Row(
                    vid,
                    labels.get((seed, vid)),
                    _num(score),
                    int(rank) if rank else None,
                    f,
                    group=str(r.get("group") or ""),
                )
            )
        if pool.rows:
            pools.append(pool)
    pools.sort(key=lambda p: (p.ts, p.compute_id))
    return pools


def to_records(pools: list[Pool]) -> list[dict[str, Any]]:
    """Flat rows (one per candidate) for a CSV or a DataFrame."""
    out = []
    for p in pools:
        for r in p.rows:
            out.append(
                {
                    "compute_id": p.compute_id,
                    "seed_id": p.seed_id,
                    "seed_title": p.seed_title,
                    "ts": p.ts,
                    "trigger": p.trigger,
                    "video_id": r.video_id,
                    "label": r.label or "",
                    "gain": r.gain if r.gain is not None else "",
                    "v0_score": r.v0_score,
                    "v0_rank": r.v0_rank or "",
                    **r.features,
                }
            )
    return out
