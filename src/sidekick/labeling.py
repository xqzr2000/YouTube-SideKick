"""Label queues for the M6 labeling app (``notebooks/05_labeling.py``).

M6 compares rankers offline by re-ranking logged pools: for each compute, every candidate that
reached the ranking step (decisions stage ``R``, shown or not). NDCG@10 and P@10 of a new
ranker are only trustworthy if those pools are fully labeled; an unlabeled video that a new
ranker moves into its top 10 is a hole in the measurement. So the app works **seed by seed**
and tries to **complete each seed's pool** before moving on, rather than spreading a few
labels over every seed as ``just label`` does.

Two queues:

- ``ranked``: the union of the ranked pools of every compute for a seed (the current video).
  This is what M6b/M6c evaluate on.
- ``filtered``: videos the filters dropped for that seed (Stage A rule drops, Stage B, Stage C),
  minus the format drops SideKick never shows (current video, Shorts, live, upcoming) and
  minus anything that reached ranking in another compute. Labels here measure false
  removals.

A label is contextual, as in ``just label``: *while watching the seed, would you want this
shown next?* It is stored per (seed, video) through the compute it came from, so one label
covers every compute of that seed. The seed question (*does the current video match your
Interests?*) is asked first for a seed that has no seed label yet.

Order within a seed is a fixed shuffle (stable between reruns, the same for everyone), so
the queue never reveals SideKick's rank or what was shown. Seeds come newest first, so panel
use gets labeled while it's fresh.

A third queue, ``relabel`` (E5, taste drift), asks again about a fixed random sample of the
**first week's** candidate labels, without showing the old answer, to measure how stable the
labels are over time. The sample is drawn the same way every time (fixed seed, fixed window),
so it never needs storing; the answers go to the ``relabels`` table, never to ``labels``, so
nothing that evaluates on labels changes. Items are grouped by current video (one context at
a time, current videos in a fixed shuffle).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sidekick.label import _NEVER_SHOWN, _fmt_duration, _fmt_views, latest_seed_label
from sidekick.parse.util import parse_age_days
from sidekick.recommend import STAGE_RANK
from sidekick.scrape.fetcher import midpoint_thumbnail_url
from sidekick.store.db import Store

QUEUES = ("ranked", "filtered", "relabel")
LABELS = ("good", "meh", "bad")
PANEL_TRIGGERS = ("open", "refresh", "click")

# E5 relabel sample, fixed in advance (2026-11-27, before any relabel was made): 50 candidate
# labels drawn uniformly from those made in the first 7 days of labeling, with this RNG seed.
RELABEL_N = 50
RELABEL_WINDOW_DAYS = 7
RELABEL_SEED = "e5-relabel"


@dataclass(frozen=True)
class Item:
    seed_id: str
    video_id: str
    compute_id: str | None  # newest compute of this seed that had the video in this queue
    kind: str = "candidate"  # or "seed": the seed question for ``seed_id``; or "relabel"
    label_id: int | None = None  # relabel only: the old label being repeated


@dataclass
class SeedProgress:
    seed_id: str
    title: str
    last_ts: str
    panel: bool  # any compute for this seed came from the side panel
    seed_label: str | None
    ranked: int = 0
    ranked_labeled: int = 0
    shown: int = 0
    shown_labeled: int = 0
    filtered: int = 0
    filtered_labeled: int = 0
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(LABELS, 0))

    @property
    def complete(self) -> bool:
        return self.ranked > 0 and self.ranked_labeled == self.ranked


def ranked_pools(store: Store) -> dict[str, dict[str, tuple[str, bool]]]:
    """``{seed: {video: (newest compute_id, shown in any compute)}}`` over stage R decisions."""
    rows = store.conn.execute(
        """SELECT c.current_video_id, d.video_id, MAX(d.compute_id), MAX(d.kept)
           FROM decisions d JOIN computes c ON c.compute_id = d.compute_id
           WHERE d.stage = ? AND c.current_video_id IS NOT NULL AND d.video_id != c.current_video_id
           GROUP BY c.current_video_id, d.video_id""",
        (STAGE_RANK,),
    ).fetchall()
    out: dict[str, dict[str, tuple[str, bool]]] = {}
    for seed, vid, cid, shown in rows:
        out.setdefault(seed, {})[vid] = (cid, bool(shown))
    return out


def filtered_pools(
    store: Store, ranked: dict[str, dict[str, tuple[str, bool]]] | None = None
) -> dict[str, dict[str, str]]:
    """``{seed: {video: newest compute_id}}`` of filter drops that SideKick could have shown."""
    ranked = ranked_pools(store) if ranked is None else ranked
    rows = store.conn.execute(
        """SELECT c.current_video_id, d.video_id, d.compute_id, d.reason
           FROM decisions d JOIN computes c ON c.compute_id = d.compute_id
           WHERE d.stage IN ('A', 'B', 'C') AND d.kept = 0 AND c.current_video_id IS NOT NULL
             AND d.video_id != c.current_video_id
           ORDER BY d.compute_id"""
    ).fetchall()
    never: set[tuple[str, str]] = set()
    out: dict[str, dict[str, str]] = {}
    for seed, vid, cid, reason in rows:
        if any(r in _NEVER_SHOWN for r in (reason or "").split(";")):
            never.add((seed, vid))
            continue
        out.setdefault(seed, {})[vid] = cid  # ordered by compute id, so the newest wins
    shorts = {r[0] for r in store.conn.execute("SELECT video_id FROM videos WHERE is_short = 1")}
    for seed in list(out):
        pool = out[seed]
        for vid in list(pool):
            if (seed, vid) in never or vid in shorts or vid in ranked.get(seed, {}):
                del pool[vid]
        if not pool:
            del out[seed]
    return out


def labeled(store: Store) -> dict[tuple[str, str], str]:
    """``{(seed, video): label}`` of candidate labels; the newest label per pair counts."""
    rows = store.conn.execute(
        """SELECT c.current_video_id, l.video_id, l.label FROM labels l
           JOIN computes c ON c.compute_id = l.compute_id
           WHERE l.kind = 'candidate' ORDER BY l.ts, l.label_id"""
    ).fetchall()
    return {(seed, vid): lab for seed, vid, lab in rows}


def _seed_meta(store: Store) -> dict[str, tuple[str, bool]]:
    """``{seed: (last compute ts, any panel compute)}``."""
    rows = store.conn.execute(
        f"""SELECT current_video_id, MAX(ts),
                   MAX(trigger IN ({",".join("?" * len(PANEL_TRIGGERS))}))
            FROM computes WHERE current_video_id IS NOT NULL GROUP BY current_video_id""",
        PANEL_TRIGGERS,
    ).fetchall()
    return {r[0]: (r[1], bool(r[2])) for r in rows}


def progress(store: Store) -> list[SeedProgress]:
    """Per seed with a ranked or filtered pool, newest first."""
    ranked = ranked_pools(store)
    filtered = filtered_pools(store, ranked)
    done = labeled(store)
    meta = _seed_meta(store)
    out = []
    for seed in set(ranked) | set(filtered):
        ts, panel = meta.get(seed, ("", False))
        p = SeedProgress(
            seed_id=seed,
            title=(store.get_video(seed) or {}).get("title") or seed,
            last_ts=ts,
            panel=panel,
            seed_label=latest_seed_label(store, seed),
        )
        for vid, (_, shown) in ranked.get(seed, {}).items():
            lab = done.get((seed, vid))
            p.ranked += 1
            p.shown += shown
            if lab:
                p.ranked_labeled += 1
                p.shown_labeled += shown
                p.counts[lab] += 1
        for vid in filtered.get(seed, {}):
            p.filtered += 1
            lab = done.get((seed, vid))
            if lab:
                p.filtered_labeled += 1
                p.counts[lab] += 1
        out.append(p)
    out.sort(key=lambda p: p.last_ts, reverse=True)
    return out


def _ts(s: str) -> datetime:
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def relabel_sample(
    store: Store,
    n: int = RELABEL_N,
    *,
    window_days: int = RELABEL_WINDOW_DAYS,
    rng_seed: str = RELABEL_SEED,
) -> list[Item]:
    """The fixed E5 sample: ``n`` candidate labels from the first ``window_days`` of labeling,
    grouped by current video. Deterministic: the window is in the past, so the same database
    always gives the same sample."""
    rows = store.conn.execute(
        """SELECT l.label_id, l.ts, c.current_video_id, l.video_id, l.compute_id FROM labels l
           JOIN computes c ON c.compute_id = l.compute_id
           WHERE l.kind = 'candidate' AND c.current_video_id IS NOT NULL
             AND l.video_id != c.current_video_id
           ORDER BY l.label_id"""
    ).fetchall()
    if not rows:
        return []
    end = min(_ts(r[1]) for r in rows) + timedelta(days=window_days)
    newest: dict[tuple[str, str], tuple] = {}
    for r in rows:  # newest label per (seed, video) in the window: the one in force back then
        if _ts(r[1]) < end:
            newest[(r[2], r[3])] = r
    pool = sorted(newest.values(), key=lambda r: r[0])
    picked = random.Random(rng_seed).sample(pool, min(n, len(pool)))
    seeds = sorted({r[2] for r in picked})
    random.Random(f"{rng_seed}:seeds").shuffle(seeds)
    order = {s: i for i, s in enumerate(seeds)}
    picked.sort(key=lambda r: (order[r[2]], random.Random(f"{rng_seed}:{r[0]}").random()))
    return [Item(r[2], r[3], r[4], kind="relabel", label_id=r[0]) for r in picked]


def relabeled_ids(store: Store) -> set[int]:
    return {r[0] for r in store.conn.execute("SELECT DISTINCT label_id FROM relabels")}


def _queue(store: Store, queue: str, seed: str | None) -> list[Item]:
    if queue not in QUEUES:
        raise ValueError(f"queue must be one of {QUEUES}")
    if queue == "relabel":
        done = relabeled_ids(store)
        return [
            it
            for it in relabel_sample(store)
            if it.label_id not in done and (seed is None or it.seed_id == seed)
        ]
    if queue == "ranked":
        pool = {vid: cid for vid, (cid, _) in ranked_pools(store).get(seed, {}).items()}
    else:
        pool = filtered_pools(store).get(seed, {})
    done = labeled(store)
    vids = sorted(v for v in pool if (seed, v) not in done)
    random.Random(f"{seed}:{queue}").shuffle(vids)  # stable, and hides rank and shown-or-not
    return [Item(seed, v, pool[v]) for v in vids]


def next_item(
    store: Store,
    queue: str = "ranked",
    *,
    seed: str | None = None,
    skip: frozenset[tuple[str, str]] = frozenset(),
) -> Item | None:
    """The next thing to ask: the seed question for a seed without a seed label, else the next
    unlabeled, unskipped video of that seed's queue. ``seed=None`` takes the newest seed whose
    queue isn't finished. The ``relabel`` queue has no seed question and keeps its own order."""
    if queue == "relabel":
        items = [it for it in _queue(store, queue, seed) if (it.seed_id, it.video_id) not in skip]
        return items[0] if items else None
    if seed is None:
        seed_list = [p.seed_id for p in progress(store)]
    else:
        seed_list = [seed]
    for s in seed_list:
        items = [it for it in _queue(store, queue, s) if (s, it.video_id) not in skip]
        if not items:
            continue
        if latest_seed_label(store, s) is None and (s, s) not in skip:
            return Item(s, s, None, kind="seed")
        return items[0]
    return None


def remaining(store: Store, queue: str, seed: str) -> int:
    """Items left for ``seed`` (for ``relabel``: left in the whole sample)."""
    return len(_queue(store, queue, None if queue == "relabel" else seed))


def save(store: Store, item: Item, label: str, *, reason: str = "", session: str) -> int:
    """Store one label; returns its id (for undo: a ``relabels`` id for a relabel item, see
    :func:`undo`)."""
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    if item.kind == "seed":
        return store.add_label(item.seed_id, label, kind="seed", session=session)
    if item.kind == "relabel":
        return store.add_relabel(item.label_id, label, reason=reason.strip(), session=session)
    return store.add_label(
        item.video_id,
        label,
        kind="candidate",
        compute_id=item.compute_id,
        reason=reason.strip(),
        session=session,
    )


def undo(store: Store, item_kind: str, row_id: int) -> None:
    """Take back what :func:`save` stored for an item of this kind."""
    if item_kind == "relabel":
        store.delete_relabel(row_id)
    else:
        store.delete_label(row_id)


def recent_reasons(store: Store, n: int = 10) -> list[str]:
    """Most used reasons so far, to keep wording consistent between sessions."""
    rows = store.conn.execute(
        """SELECT lower(trim(reason)) AS r, COUNT(*) AS k FROM labels
           WHERE kind = 'candidate' AND reason IS NOT NULL AND trim(reason) != ''
           GROUP BY r ORDER BY k DESC, r LIMIT ?""",
        (n,),
    ).fetchall()
    return [r[0] for r in rows]


def card(store: Store, video_id: str) -> dict[str, Any]:
    """What the app shows for a video. No scores, ranks or filter decisions, on purpose."""
    v = store.get_video(video_id) or {}
    views = store.conn.execute(
        "SELECT views FROM video_snapshots WHERE video_id = ? AND views IS NOT NULL "
        "ORDER BY source = 'watch' DESC, ts DESC LIMIT 1",
        (video_id,),
    ).fetchone()
    age = None
    if v.get("published_at"):
        try:
            pub = datetime.fromisoformat(v["published_at"])
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=UTC)
            age = (datetime.now(UTC) - pub).days
        except ValueError:
            age = None
    if age is None:
        age = parse_age_days(v.get("published_text") or "")
    desc = " ".join((v.get("description") or "").split())
    return {
        "video_id": video_id,
        "title": v.get("title") or "(no title)",
        "channel": v.get("channel") or "?",
        "duration": _fmt_duration(v.get("duration_s")),
        "views": _fmt_views(views[0] if views else None),
        "age": f"{age:.0f} d old" if age is not None else "age ?",
        "description": desc[:400] + ("…" if len(desc) > 400 else ""),
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "thumbnail": midpoint_thumbnail_url(video_id),
    }
