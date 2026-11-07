"""Precision@10 of SideKick's top 10 against YouTube's own up-next, from hand labels.

For the newest ``just recommend`` compute of each seed, two lists of 10:

- **SideKick:** the videos it showed (decisions stage ``R``, ``kept = 1``), in rank order;
- **YouTube:** the first 10 videos of that compute's up-next, skipping what nobody would be
  shown anyway: the current video, Shorts, live streams and premieres.

A video counts by its newest candidate label in that seed's context (``just label``;
``just label --top10`` asks exactly about these two lists, mixed and unmarked). Precision@10 =
good ÷ labeled; the bad share is reported too, because removing bad videos is half of what
SideKick is for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from sidekick.embed.evaluate import candidate_labels
from sidekick.store.db import Store

K = 10
NEVER_SHOWN = ("current_video", "short", "live", "upcoming")


def latest_recommends(store: Store) -> list[tuple[str, str]]:
    """``(compute_id, seed_id)``: the newest compute with a ranking (stage R) per seed."""
    rows = store.conn.execute(
        """SELECT c.compute_id, c.current_video_id FROM computes c
           WHERE c.current_video_id IS NOT NULL
             AND EXISTS (SELECT 1 FROM decisions d WHERE d.compute_id = c.compute_id
                         AND d.stage = 'R')
           ORDER BY c.ts, c.rowid"""
    ).fetchall()
    newest: dict[str, str] = {}
    for cid, seed in rows:
        newest[seed] = cid
    return [(cid, seed) for seed, cid in newest.items()]


def sidekick_list(store: Store, compute_id: str, k: int = K) -> list[str]:
    rows = store.conn.execute(
        "SELECT video_id, features FROM decisions WHERE compute_id = ? AND stage = 'R' "
        "AND kept = 1 ORDER BY rowid",
        (compute_id,),
    ).fetchall()
    ranked = sorted(
        enumerate(rows), key=lambda x: (json.loads(x[1][1] or "{}").get("rank") or 1e9, x[0])
    )
    return [r[0] for _, r in ranked][:k]


def youtube_list(store: Store, compute_id: str, seed_id: str, k: int = K) -> list[str]:
    never = {
        vid
        for vid, reason in store.conn.execute(
            "SELECT video_id, reason FROM decisions WHERE compute_id = ? AND kept = 0 "
            "AND reason IS NOT NULL",
            (compute_id,),
        )
        if any(r in NEVER_SHOWN for r in reason.split(";"))
    }
    out: list[str] = []
    for (vid,) in store.conn.execute(
        "SELECT video_id FROM candidates WHERE compute_id = ? AND source = 'upnext' "
        "ORDER BY position",
        (compute_id,),
    ):
        v = store.get_video(vid) or {}
        if vid == seed_id or vid in never or v.get("is_short") or vid in out:
            continue
        out.append(vid)
        if len(out) == k:
            break
    return out


@dataclass
class ListScore:
    n: int = 0
    labeled: int = 0
    good: int = 0
    meh: int = 0
    bad: int = 0

    def add(self, label: str | None) -> None:
        self.n += 1
        if label in ("good", "meh", "bad"):
            self.labeled += 1
            setattr(self, label, getattr(self, label) + 1)

    @property
    def precision(self) -> float:
        return self.good / self.labeled if self.labeled else float("nan")

    @property
    def bad_share(self) -> float:
        return self.bad / self.labeled if self.labeled else float("nan")


@dataclass
class SeedRow:
    seed_id: str
    compute_id: str
    title: str
    sidekick: ListScore = field(default_factory=ListScore)
    youtube: ListScore = field(default_factory=ListScore)
    overlap: int = 0


def evaluate(store: Store, k: int = K) -> list[SeedRow]:
    labels = {(s, v): lab for s, v, lab in candidate_labels(store)}
    out = []
    for cid, seed in latest_recommends(store):
        sk = sidekick_list(store, cid, k)
        yt = youtube_list(store, cid, seed, k)
        row = SeedRow(seed, cid, (store.get_video(seed) or {}).get("title") or "")
        for v in sk:
            row.sidekick.add(labels.get((seed, v)))
        for v in yt:
            row.youtube.add(labels.get((seed, v)))
        row.overlap = len(set(sk) & set(yt))
        out.append(row)
    return out


def to_label(store: Store, k: int = K) -> list[tuple[str, str, str]]:
    """``(seed_id, video_id, compute_id)`` in either list that has no label in its seed yet."""
    labels = {(s, v) for s, v, _ in candidate_labels(store)}
    out = []
    for cid, seed in latest_recommends(store):
        seen: set[str] = set()
        for v in sidekick_list(store, cid, k) + youtube_list(store, cid, seed, k):
            if v not in seen and (seed, v) not in labels:
                seen.add(v)
                out.append((seed, v, cid))
    return out


def _pct(x: float) -> str:
    return "  –" if x != x else f"{x:4.0%}"


def report(store: Store, k: int = K) -> list[str]:
    rows = evaluate(store, k)
    if not rows:
        return []
    lines = [
        f"top-{k} precision (newest `just recommend` per seed; good ÷ labeled, bad ÷ labeled):",
        "  seed         SideKick: P@10  bad  labeled   YouTube: P@10  bad  labeled   overlap",
    ]
    tot_s, tot_y = ListScore(), ListScore()
    for r in rows:
        for tot, s in ((tot_s, r.sidekick), (tot_y, r.youtube)):
            for name in ("n", "labeled", "good", "meh", "bad"):
                setattr(tot, name, getattr(tot, name) + getattr(s, name))
        lines.append(
            f"  {r.seed_id}  {_pct(r.sidekick.precision):>14} {_pct(r.sidekick.bad_share)} "
            f"{r.sidekick.labeled:>4}/{r.sidekick.n:<4} {_pct(r.youtube.precision):>15} "
            f"{_pct(r.youtube.bad_share)} {r.youtube.labeled:>4}/{r.youtube.n:<4} "
            f"{r.overlap:>6}   {r.title[:40]}"
        )
    lines.append(
        f"  {'all':<11}  {_pct(tot_s.precision):>14} {_pct(tot_s.bad_share)} "
        f"{tot_s.labeled:>4}/{tot_s.n:<4} {_pct(tot_y.precision):>15} "
        f"{_pct(tot_y.bad_share)} {tot_y.labeled:>4}/{tot_y.n:<4}"
    )
    missing = len(to_label(store, k))
    if missing:
        lines.append(f"  {missing} video(s) in these lists still unlabeled: `just label --top10`")
    return lines
