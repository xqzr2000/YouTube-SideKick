"""E6: why does YouTube keep recommending what you don't want?

Spec §8, E6: a persistence analysis of unwanted videos in YouTube's own up-next lists. Every
compute logged YouTube's lists for the current video: the signed-out list from the fetched
watch page (``source = upnext``) and, for panel computes, the signed-in list from the tab
(``source = tab``). This module asks, from those logs alone (no new requests):

1. **Stickiness:** when a video is in YouTube's list for a current video, how often is it
   still there the next time that same current video is computed? Unwanted vs wanted.
2. **Spread:** under how many *different* current videos does YouTube show it? A video
   pushed regardless of context (a late-night clip under a statistics lecture) is the
   "keeps recommending" pattern.
3. **Channels:** which channels' unwanted videos turn up under the most different contexts.
4. **Position, freshness, popularity:** where YouTube puts unwanted videos in its list, and
   how old and how viewed they are compared with wanted ones.
5. **Signed in vs signed out:** does the personalized list carry more unwanted videos?
6. **After a 👎:** does YouTube keep showing a video (or its channel) after you downvoted it?

Each sighting (compute, video in YouTube's list) gets a class from what's known about it in
*that* context:

- ``unwanted``: labeled bad for this current video, dropped by Stage C in this compute
  (Don't-show / pundit / promo / clickbait / like a 👎'd video), or downvoted (👎) ever;
- ``wanted``: labeled good for this current video;
- ``meh``: labeled meh; ``unknown``: none of the above.

Rates come with 95 % bootstrap intervals over current videos (resampling current videos and
pooling their counts), because sightings under one current video aren't independent.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import median

import numpy as np

from sidekick.config import ensure_home
from sidekick.store.db import Store

YT_SOURCES = ("upnext", "tab")
CLASSES = ("unwanted", "wanted", "meh", "unknown")


@dataclass
class Sighting:
    compute_id: str
    ts: str
    seed: str
    video_id: str
    source: str
    position: int
    cls: str = "unknown"
    why: str = ""  # label / drop reason / downvote


@dataclass
class Data:
    sightings: list[Sighting] = field(default_factory=list)
    videos: dict[str, dict] = field(default_factory=dict)
    downvotes: dict[str, str] = field(default_factory=dict)  # video -> first 👎 ts


def _drop_kind(reason: str) -> str:
    r = reason or ""
    for prefix, kind in (
        ("dont_show", "don't-show"),
        ("pundit", "pundit"),
        ("like_downvoted", "like a 👎"),
        ("clickbait", "clickbait"),
        ("promo", "promo"),
    ):
        if prefix in r:
            return kind
    return ""


def load(store: Store) -> Data:
    d = Data()
    labels = {
        (seed, vid): (lab, reason)
        for seed, vid, lab, reason in store.conn.execute(
            """SELECT c.current_video_id, l.video_id, l.label, l.reason FROM labels l
               JOIN computes c ON c.compute_id = l.compute_id
               WHERE l.kind = 'candidate' ORDER BY l.ts, l.label_id"""
        )
    }
    drops = {
        (cid, vid): _drop_kind(reason)
        for cid, vid, reason in store.conn.execute(
            "SELECT compute_id, video_id, reason FROM decisions WHERE stage = 'C' AND kept = 0"
        )
    }
    for vid, ts in store.conn.execute(
        "SELECT video_id, MIN(ts) FROM feedback WHERE type = 'downvote' GROUP BY video_id"
    ):
        d.downvotes[vid] = ts
    rows = store.conn.execute(
        f"""SELECT k.compute_id, c.ts, c.current_video_id, k.video_id, k.source,
                   MIN(k.position)
            FROM candidates k JOIN computes c ON c.compute_id = k.compute_id
            WHERE k.source IN ({",".join("?" * len(YT_SOURCES))})
              AND c.current_video_id IS NOT NULL AND k.video_id != c.current_video_id
            GROUP BY k.compute_id, k.video_id, k.source
            ORDER BY c.ts, k.compute_id""",
        YT_SOURCES,
    ).fetchall()
    for cid, ts, seed, vid, source, pos in rows:
        s = Sighting(cid, ts, seed, vid, source, int(pos))
        lab, reason = labels.get((seed, vid), (None, None))
        drop = drops.get((cid, vid), "")
        if lab == "bad" or drop or vid in d.downvotes:
            s.cls = "unwanted"
            s.why = (
                f"labeled bad ({reason})"
                if lab == "bad" and reason
                else "labeled bad"
                if lab == "bad"
                else f"dropped: {drop}"
                if drop
                else "👎"
            )
        elif lab == "good":
            s.cls = "wanted"
        elif lab == "meh":
            s.cls = "meh"
        d.sightings.append(s)
    for vid in {s.video_id for s in d.sightings} | {s.seed for s in d.sightings}:
        d.videos[vid] = store.get_video(vid) or {}
        snap = store.conn.execute(
            "SELECT views FROM video_snapshots WHERE video_id = ? AND views IS NOT NULL "
            "ORDER BY source = 'watch' DESC, ts DESC LIMIT 1",
            (vid,),
        ).fetchone()
        d.videos[vid]["_views"] = snap[0] if snap else None
    return d


# -- statistics ---------------------------------------------------------------------------------


def cluster_rate(
    counts: dict[str, tuple[int, int]], n: int = 2000, seed: int = 0
) -> tuple[float, float, float, int]:
    """Pooled hits / total over clusters (current videos), with a bootstrap interval that
    resamples clusters. Returns (rate, lo, hi, total)."""
    keys = [k for k, (_, tot) in counts.items() if tot]
    if not keys:
        return (math.nan, math.nan, math.nan, 0)
    hits = np.array([counts[k][0] for k in keys], dtype=float)
    tots = np.array([counts[k][1] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    rates = hits[idx].sum(axis=1) / np.maximum(tots[idx].sum(axis=1), 1)
    return (
        float(hits.sum() / tots.sum()),
        float(np.percentile(rates, 2.5)),
        float(np.percentile(rates, 97.5)),
        int(tots.sum()),
    )


def stickiness(d: Data) -> dict[str, dict[str, tuple[int, int]]]:
    """Per class, per current video: (still there next compute, sightings with a next
    compute). YouTube's two lists are merged per compute."""
    by_compute: dict[str, set[str]] = defaultdict(set)
    seed_of: dict[str, str] = {}
    ts_of: dict[str, str] = {}
    cls: dict[tuple[str, str], str] = {}
    for s in d.sightings:
        by_compute[s.compute_id].add(s.video_id)
        seed_of[s.compute_id], ts_of[s.compute_id] = s.seed, s.ts
        if cls.get((s.compute_id, s.video_id)) != "unwanted":
            cls[(s.compute_id, s.video_id)] = s.cls
    chains: dict[str, list[str]] = defaultdict(list)
    for cid in sorted(by_compute, key=lambda c: (ts_of[c], c)):
        chains[seed_of[cid]].append(cid)
    out: dict[str, dict[str, list[int]]] = {c: defaultdict(lambda: [0, 0]) for c in CLASSES}
    for seed, cids in chains.items():
        for a, b in zip(cids, cids[1:], strict=False):
            for vid in by_compute[a]:
                c = cls[(a, vid)]
                out[c][seed][1] += 1
                out[c][seed][0] += vid in by_compute[b]
    return {c: {k: (v[0], v[1]) for k, v in per.items()} for c, per in out.items()}


def video_class(d: Data) -> dict[str, str]:
    """A video's overall class: unwanted if it was unwanted anywhere, else wanted if labeled
    good anywhere, else meh / unknown."""
    seen: dict[str, set[str]] = defaultdict(set)
    for s in d.sightings:
        seen[s.video_id].add(s.cls)
    order = ("unwanted", "wanted", "meh", "unknown")
    return {v: next(c for c in order if c in cs) for v, cs in seen.items()}


def spread(d: Data) -> dict[str, Counter]:
    """Per video: number of distinct current videos it appeared under, grouped by class."""
    ctx: dict[str, set[str]] = defaultdict(set)
    for s in d.sightings:
        ctx[s.video_id].add(s.seed)
    vc = video_class(d)
    out: dict[str, Counter] = {c: Counter() for c in CLASSES}
    for v, seeds in ctx.items():
        out[vc[v]][len(seeds)] += 1
    return out


def channels(d: Data, top: int = 10) -> list[dict]:
    """Channels whose unwanted videos appeared under the most different current videos."""
    per: dict[str, dict] = defaultdict(
        lambda: {"contexts": set(), "unwanted_contexts": set(), "sightings": 0, "unwanted": 0}
    )
    for s in d.sightings:
        ch = d.videos.get(s.video_id, {}).get("channel") or "?"
        p = per[ch]
        p["contexts"].add(s.seed)
        p["sightings"] += 1
        if s.cls == "unwanted":
            p["unwanted"] += 1
            p["unwanted_contexts"].add(s.seed)
    rows = [
        {
            "channel": ch,
            "unwanted_contexts": len(p["unwanted_contexts"]),
            "contexts": len(p["contexts"]),
            "unwanted": p["unwanted"],
            "sightings": p["sightings"],
        }
        for ch, p in per.items()
        if p["unwanted"]
    ]
    rows.sort(key=lambda r: (-r["unwanted_contexts"], -r["unwanted"], r["channel"]))
    return rows[:top]


def _age_days(v: dict, at: str) -> float | None:
    pub = v.get("published_at")
    if not pub:
        return None
    try:
        p, t = datetime.fromisoformat(pub), datetime.fromisoformat(at)
    except ValueError:
        return None
    p = p if p.tzinfo else p.replace(tzinfo=UTC)
    t = t if t.tzinfo else t.replace(tzinfo=UTC)
    return max(0.0, (t - p).total_seconds() / 86400)


def profile_by_class(d: Data) -> dict[str, dict[str, float | None]]:
    """Median up-next position, age at the time of the sighting, and views, per class."""
    acc: dict[str, dict[str, list[float]]] = {
        c: {"position": [], "age": [], "views": []} for c in CLASSES
    }
    for s in d.sightings:
        v = d.videos.get(s.video_id, {})
        a = acc[s.cls]
        a["position"].append(s.position)
        age = _age_days(v, s.ts)
        if age is not None:
            a["age"].append(age)
        if v.get("_views"):
            a["views"].append(float(v["_views"]))
    return {
        c: {k: (median(vals) if vals else None) for k, vals in a.items()}
        | {"n": len(a["position"])}
        for c, a in acc.items()
    }


def by_source(d: Data) -> dict[str, dict[str, tuple[int, int]]]:
    """Unwanted share per list, per current video: (unwanted, sightings)."""
    out: dict[str, dict[str, list[int]]] = {src: defaultdict(lambda: [0, 0]) for src in YT_SOURCES}
    for s in d.sightings:
        out[s.source][s.seed][1] += 1
        out[s.source][s.seed][0] += s.cls == "unwanted"
    return {src: {k: (v[0], v[1]) for k, v in per.items()} for src, per in out.items()}


def paired_sources(d: Data) -> list[tuple[str, float, float]]:
    """For computes that logged both lists: (compute, unwanted share signed-out, signed-in)."""
    per: dict[str, dict[str, list[int]]] = defaultdict(lambda: {s: [0, 0] for s in YT_SOURCES})
    for s in d.sightings:
        per[s.compute_id][s.source][1] += 1
        per[s.compute_id][s.source][0] += s.cls == "unwanted"
    out = []
    for cid, v in per.items():
        if v["upnext"][1] and v["tab"][1]:
            out.append((cid, v["upnext"][0] / v["upnext"][1], v["tab"][0] / v["tab"][1]))
    return out


def after_downvote(d: Data) -> list[dict]:
    """Per 👎'd video: YouTube sightings of it, and of its channel's other videos, after the
    first 👎."""
    out = []
    for vid, ts in d.downvotes.items():
        ch = d.videos.get(vid, {}).get("channel")
        same = [s for s in d.sightings if s.video_id == vid and s.ts > ts]
        chan = [
            s
            for s in d.sightings
            if s.ts > ts
            and s.video_id != vid
            and ch
            and d.videos.get(s.video_id, {}).get("channel") == ch
        ]
        out.append(
            {
                "video_id": vid,
                "title": d.videos.get(vid, {}).get("title") or vid,
                "channel": ch or "?",
                "again": len(same),
                "again_computes": len({s.compute_id for s in same}),
                "channel_after": len(chan),
            }
        )
    return sorted(out, key=lambda r: -r["again"])


def top_unwanted(d: Data, top: int = 10) -> list[dict]:
    per: dict[str, dict] = defaultdict(
        lambda: {"computes": set(), "contexts": set(), "why": Counter()}
    )
    for s in d.sightings:
        if s.cls == "unwanted":
            p = per[s.video_id]
            p["computes"].add(s.compute_id)
            p["contexts"].add(s.seed)
            p["why"][s.why.split(" (")[0]] += 1
    rows = []
    for vid, p in per.items():
        v = d.videos.get(vid, {})
        rows.append(
            {
                "title": v.get("title") or vid,
                "channel": v.get("channel") or "?",
                "computes": len(p["computes"]),
                "contexts": len(p["contexts"]),
                "why": p["why"].most_common(1)[0][0],
            }
        )
    rows.sort(key=lambda r: (-r["computes"], -r["contexts"], r["title"]))
    return rows[:top]


# -- report --------------------------------------------------------------------------------------


def _pct(t: tuple[float, float, float, int]) -> str:
    r, lo, hi, n = t
    return "–" if math.isnan(r) else f"{r * 100:.0f} % [{lo * 100:.0f}, {hi * 100:.0f}] (n {n})"


def _table(header: list[str], rows: list[list[str]], md: bool) -> list[str]:
    if md:
        return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + [
            "| " + " | ".join(r) + " |" for r in rows
        ]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows, strict=True)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return [fmt.format(*header)] + [fmt.format(*r) for r in rows]


def report(store: Store, *, md: bool = False) -> list[str]:
    d = load(store)
    if not d.sightings:
        return ["No YouTube up-next lists logged yet."]
    seeds = {s.seed for s in d.sightings}
    cls_n = Counter(s.cls for s in d.sightings)
    vc = Counter(video_class(d).values())
    src = ", ".join(f"{k} {v}" for k, v in Counter(s.source for s in d.sightings).items())
    lines = [
        "== Data ==",
        f"{len(d.sightings)} sightings in YouTube's lists ({src}), "
        f"{len({s.compute_id for s in d.sightings})} computes, {len(seeds)} current videos, "
        f"{sum(vc.values())} distinct videos",
        "sightings by class: " + " · ".join(f"{c} {cls_n[c]}" for c in CLASSES),
        "videos by class:    " + " · ".join(f"{c} {vc[c]}" for c in CLASSES),
        "",
        "== 1. Stickiness: still in YouTube's list next time the same current video is computed ==",
    ]
    st = stickiness(d)
    lines += _table(["class", "returns"], [[c, _pct(cluster_rate(st[c]))] for c in CLASSES], md)

    lines += ["", "== 2. Spread: number of different current videos a video appeared under =="]
    sp = spread(d)
    rows = []
    for c in CLASSES:
        tot = sum(sp[c].values())
        multi = sum(n for k, n in sp[c].items() if k >= 2)
        mean = sum(k * n for k, n in sp[c].items()) / tot if tot else math.nan
        rows.append([c, str(tot), f"{multi / tot * 100:.0f} %" if tot else "–", f"{mean:.2f}"])
    lines += _table(["class", "videos", "under ≥ 2 current videos", "mean"], rows, md)

    lines += ["", "== 3. Channels whose unwanted videos turn up under the most current videos =="]
    lines += _table(
        ["channel", "unwanted under", "seen under", "unwanted sightings", "all sightings"],
        [
            [
                r["channel"][:34],
                str(r["unwanted_contexts"]),
                str(r["contexts"]),
                str(r["unwanted"]),
                str(r["sightings"]),
            ]
            for r in channels(d)
        ],
        md,
    )

    lines += ["", "== 4. Where and what: medians per class =="]
    pr = profile_by_class(d)
    lines += _table(
        ["class", "sightings", "up-next position", "age (days)", "views"],
        [
            [
                c,
                str(pr[c]["n"]),
                "–" if pr[c]["position"] is None else f"{pr[c]['position']:.0f}",
                "–" if pr[c]["age"] is None else f"{pr[c]['age']:.0f}",
                "–" if pr[c]["views"] is None else f"{pr[c]['views']:,.0f}",
            ]
            for c in CLASSES
        ],
        md,
    )

    lines += ["", "== 5. Signed out vs signed in: unwanted share of each list =="]
    bs = by_source(d)
    lines += _table(
        ["list", "unwanted share"],
        [
            ["signed-out (upnext)", _pct(cluster_rate(bs["upnext"]))],
            ["signed-in (tab)", _pct(cluster_rate(bs["tab"]))],
        ],
        md,
    )
    ps = paired_sources(d)
    if ps:
        diff = [b - a for _, a, b in ps]
        lines.append(
            f"computes with both lists: {len(ps)}; mean signed-in − signed-out unwanted share "
            f"{np.mean(diff) * 100:+.0f} points"
        )

    lines += ["", "== 6. After a 👎 =="]
    ad = after_downvote(d)
    if not ad:
        lines.append("No 👎 yet.")
    else:
        lines += _table(
            ["video", "channel", "seen again (computes)", "channel's other videos after"],
            [
                [
                    r["title"][:40],
                    r["channel"][:24],
                    f"{r['again']} ({r['again_computes']})",
                    str(r["channel_after"]),
                ]
                for r in ad
            ],
            md,
        )

    lines += ["", "== Most persistent unwanted videos =="]
    lines += _table(
        ["title", "channel", "computes", "current videos", "why unwanted"],
        [
            [r["title"][:44], r["channel"][:24], str(r["computes"]), str(r["contexts"]), r["why"]]
            for r in top_unwanted(d)
        ],
        md,
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="just e6", description=__doc__.split("\n")[0])
    ap.add_argument("--md", action="store_true", help="Markdown tables")
    args = ap.parse_args(argv)
    with Store.open(ensure_home()) as store:
        for line in report(store, md=args.md):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
