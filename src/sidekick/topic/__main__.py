"""``just topic``: look at the topic anchor SideKick extracts, on your own data.

Examples::

    just topic                  # the topic of every current video (seed) in the database
    just topic 5eW3gNkNsA8      # one video's topic + its candidates ranked by topic match
    just topic 5eW3gNkNsA8 --k 30 --no-e5
    just topic --download       # fetch spaCy en_core_web_sm (part of `just models`)

Nothing here filters or ranks for real; it prints what the extractor found and how each
candidate matches, next to your label and the current filter decision, so the extraction can
be checked on real news pages before anything depends on it.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from sidekick.config import ensure_home
from sidekick.embed import _get_cached, _put_cached
from sidekick.embed import e5 as e5mod
from sidekick.embed.text import text_hash
from sidekick.store.db import Store
from sidekick.topic import Topic, TopicContext, extract, match
from sidekick.topic import nlp as nlpmod

E5_TOPIC = "e5-topic"  # embeddings.model for topic texts (cached by text hash)


def seeds(store: Store) -> list[str]:
    rows = store.conn.execute(
        "SELECT current_video_id FROM computes WHERE current_video_id IS NOT NULL "
        "GROUP BY current_video_id ORDER BY MIN(ts)"
    )
    return [r[0] for r in rows]


def candidates(store: Store, seed: str) -> list[str]:
    rows = store.conn.execute(
        """SELECT k.video_id, MIN(k.position) AS pos
           FROM candidates k JOIN computes c ON c.compute_id = k.compute_id
           WHERE c.current_video_id = ? AND k.video_id != ?
           GROUP BY k.video_id ORDER BY pos""",
        (seed, seed),
    )
    return [r[0] for r in rows]


def labels(store: Store, seed: str) -> dict[str, str]:
    """Latest candidate label per video, in this seed's context."""
    rows = store.conn.execute(
        """SELECT l.video_id, l.label FROM labels l JOIN computes c ON c.compute_id = l.compute_id
           WHERE l.kind = 'candidate' AND c.current_video_id = ? ORDER BY l.ts""",
        (seed,),
    )
    return {r[0]: r[1] for r in rows}


def drops(store: Store, seed: str) -> dict[str, str]:
    """Drop reason per video, from the latest compute for this seed that decided it."""
    rows = store.conn.execute(
        """SELECT d.video_id, d.compute_id, d.stage, d.kept, d.reason FROM decisions d
           JOIN computes c ON c.compute_id = d.compute_id
           WHERE c.current_video_id = ? ORDER BY c.ts, d.stage""",
        (seed,),
    )
    latest: dict[str, tuple[str, str]] = {}
    for r in rows:
        vid, cid = r["video_id"], r["compute_id"]
        if vid not in latest or latest[vid][0] != cid:
            latest[vid] = (cid, "")  # a newer compute decided it: start over
        if not r["kept"] and not latest[vid][1]:
            latest[vid] = (cid, f"{r['stage']}:{(r['reason'] or '')[:28]}")
    return {vid: why for vid, (_, why) in latest.items() if why}


def topic_vectors(store: Store, e5, topics: list[Topic]) -> np.ndarray:
    """e5 (``query: ``) vectors of topic texts, cached in ``embeddings`` as ``e5-topic``."""
    hashes = [text_hash(t.text) for t in topics]
    vecs = [
        _get_cached(store, E5_TOPIC, t.video_id, h) for t, h in zip(topics, hashes, strict=True)
    ]
    todo = [i for i, v in enumerate(vecs) if v is None]
    if todo:
        new = e5.encode([topics[i].text for i in todo])
        with store.conn:
            for i, vec in zip(todo, new, strict=True):
                _put_cached(store, E5_TOPIC, topics[i].video_id, hashes[i], vec)
                vecs[i] = vec
    return np.vstack(vecs) if vecs else np.zeros((0, e5mod.DIM), dtype=np.float32)


def print_topic(t: Topic, out=print) -> None:
    s = t.summary()
    out(f"  lead ({s['lead_source']}): {s['lead'][:160]}")
    out(f"  entities:   {', '.join(s['entities']) or '-'}")
    out(f"  keyphrases: {', '.join(s['keyphrases']) or '-'}")
    out(f"  series:     {', '.join(s['series']) or '-'}")
    out(f"  news:       {s['news']}    query: {s['query']!r}")


def _fmt(x) -> str:
    return "   -" if x is None else f"{x:4.2f}"


def show_one(store: Store, ctx: TopicContext, vid: str, *, k: int, e5, out=print) -> int:
    anchor = extract(ctx, vid)
    if anchor is None:
        print(f"{vid} isn't in the database; run `just fetch {vid}` first", file=sys.stderr)
        return 1
    out(f"\n{anchor.channel} | {anchor.title}")
    print_topic(anchor, out)
    ids = candidates(store, vid)
    if not ids:
        out("\n  (no candidates logged for this video)")
        return 0
    cands = [c for c in (extract(ctx, i) for i in ids) if c is not None]
    sims = [None] * len(cands)
    if e5 is not None:
        t0 = time.perf_counter()
        m = topic_vectors(store, e5, [anchor, *cands])
        sims = [float(x) for x in m[1:] @ m[0]]
        out(f"\n  e5 topic vectors: {len(cands) + 1} in {time.perf_counter() - t0:.1f} s")
    lab, drop = labels(store, vid), drops(store, vid)
    rows = []
    for c, sim in zip(cands, sims, strict=True):
        p = match(anchor, c)
        rows.append((p["score"], sim, p, c))
    rows.sort(key=lambda r: (-r[0], -(r[1] or 0)))
    out(
        f"\n  {'score':>5} {'ent':>4} {'tok':>4} {'e5':>4} {'ser':>4} {'days':>4}  "
        f"{'label':5} {'filter':30} title"
    )
    for score, sim, p, c in rows[:k]:
        days = "   -" if p["days"] is None else f"{p['days']:4.0f}"
        out(
            f"  {score:5.2f} {_fmt(p['ent'])} {_fmt(p['tok'])} {_fmt(sim)} {_fmt(p['series'])} "
            f"{days}  {lab.get(c.video_id, ''):5} {drop.get(c.video_id, ''):30} "
            f"{c.channel[:16]} | {c.title[:60]}"
        )
    if len(rows) > k:
        out(f"  … {len(rows) - k} more (--k)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="just topic", description=__doc__.split("\n")[0])
    ap.add_argument("ids", nargs="*", help="video ids (default: every seed, topic only)")
    ap.add_argument("--k", type=int, default=20, help="candidates to show per video")
    ap.add_argument("--no-e5", action="store_true", help="skip e5 topic similarity")
    ap.add_argument("--download", action="store_true", help="download spaCy en_core_web_sm")
    args = ap.parse_args(argv)
    paths = ensure_home()

    if args.download:
        p = nlpmod.model_dir(paths)
        if nlpmod.is_downloaded(paths):
            print(f"already there: {p}")
        else:
            print(f"downloading {nlpmod.MODEL_URL}")
            print(f"saved {nlpmod.download_model(paths)}")
        tagger = nlpmod.load_tagger(paths)
        sample = "B.C. Premier David Eby responds to port strike as talks stall in Vancouver"
        print(f"  tagger: {tagger.name}")
        print(f"  {sample!r} -> {[(e.text, e.label) for e in tagger.entities(sample)]}")
        return 0

    tagger = nlpmod.load_tagger(paths)
    if tagger.name == "heuristic":
        print("spaCy model not found; using the heuristic tagger (run `just models`)")
    with Store.open(paths) as store:
        ctx = TopicContext(store, tagger)
        if not args.ids:
            for vid in seeds(store):
                t = extract(ctx, vid)
                if t:
                    print(f"\n{vid}  {t.channel} | {t.title[:70]}")
                    print_topic(t)
            return 0
        e5 = None if args.no_e5 else e5mod.load(paths)
        rc = 0
        for vid in args.ids:
            vid = _video_id(vid)
            rc |= show_one(store, ctx, vid, k=args.k, e5=e5)
        return rc


def _video_id(s: str) -> str:
    """Accept a bare id or a watch URL (with &list=… etc.)."""
    if "v=" in s:
        return s.split("v=", 1)[1].split("&", 1)[0][:11]
    if "youtu.be/" in s:
        return s.split("youtu.be/", 1)[1][:11]
    return s


if __name__ == "__main__":
    sys.exit(main())
