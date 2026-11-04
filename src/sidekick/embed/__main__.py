"""``just sim``: look at what the text representations think is similar, on your own data.

Examples::

    just sim                         # interest-gate preview + videos closest to Don't show me
    just sim bFZ-0FH5hfs             # nearest videos in the database to this one
    just sim "political late-night"  # same, for any text
    just sim --rep e5-small-q        # one representation only
    just sim --eval                  # score every representation against your labels

``python -m sidekick.embed --download`` (part of ``just models``) fetches the e5 model.

Nothing here filters or ranks; it only prints similarities so the representations can be
eyeballed before Stage C and the interest gate use them (and before E1 measures them).
e5 vectors for every video in the database are computed (and cached) on the way, and the
time it takes is printed, since that cost counts against the 90 s compute budget.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np

from sidekick.config import ensure_home
from sidekick.embed import available, encode_videos
from sidekick.embed import e5 as e5mod
from sidekick.profile import load_profile
from sidekick.store.db import Store

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _title(store: Store, vid: str, n: int = 60) -> str:
    v = store.get_video(vid) or {}
    return f"{(v.get('channel') or '?')[:18]} | {(v.get('title') or vid)[:n]}"


def _seeds(store: Store) -> list[str]:
    rows = store.conn.execute(
        "SELECT current_video_id FROM computes WHERE current_video_id IS NOT NULL "
        "GROUP BY current_video_id ORDER BY MIN(ts)"
    )
    return [r[0] for r in rows]


def _all_ids(store: Store) -> list[str]:
    return [r[0] for r in store.conn.execute("SELECT video_id FROM videos ORDER BY video_id")]


def gate_preview(store, reps, interests, out=print) -> None:
    out("\nInterest gate preview: max similarity of each current video to your Interests")
    seeds = _seeds(store)
    if not seeds or not interests:
        out("  (needs computes and at least one Interest line)")
        return
    header = "  " + "".join(f"{r.name:>22}" for r in reps) + "  current video"
    out(header)
    cols = []
    for r in reps:
        vecs, _ = encode_videos(store, r, seeds)
        sims = vecs @ r.encode(interests).T
        cols.append(sims)
    for i, s in enumerate(seeds):
        cells = ""
        for sims in cols:
            j = int(np.argmax(sims[i]))
            name = interests[j][:14] if sims[i, j] > 0.005 else "-"
            cells += f"{sims[i, j]:>6.2f} {name:<15}"
        out(f"  {cells}  {_title(store, s, 45)}")


def dont_show_preview(store, reps, lines, k, out=print) -> None:
    out(f"\nClosest {k} videos to your Don't show me lines (Stage C will act on this)")
    if not lines:
        out("  (no Don't show me lines)")
        return
    ids = _all_ids(store)
    for r in reps:
        vecs, _ = encode_videos(store, r, ids)
        line_vecs = r.encode(lines)
        sims = vecs @ line_vecs.T
        best = sims.max(axis=1)
        which = sims.argmax(axis=1)
        out(f"  [{r.name}]")
        for i in np.argsort(-best)[:k]:
            out(f"    {best[i]:5.2f}  {lines[which[i]][:24]:<25} {_title(store, ids[i])}")


def neighbours(store, reps, query, profile, k, out=print) -> None:
    ids = _all_ids(store)
    is_video = bool(_ID_RE.match(query)) and store.get_video(query) is not None
    label = _title(store, query) if is_video else repr(query)
    out(f"\nNearest videos to {label}")
    lines = [("interest", t) for t in profile.interests] + [
        ("dont-show", d.text) for d in profile.dont_show
    ]
    for r in reps:
        vecs, _ = encode_videos(store, r, ids)
        q = encode_videos(store, r, [query])[0] if is_video else r.encode([query])
        sims = (vecs @ q.T).ravel()
        out(f"  [{r.name}]")
        for i in [i for i in np.argsort(-sims) if ids[i] != query][:k]:
            out(f"    {sims[i]:5.2f}  {_title(store, ids[i])}")
        if lines:
            ls = (r.encode([t for _, t in lines]) @ q.T).ravel()
            parts = ", ".join(
                f"{kind}:{t[:20]}={s:.2f}" for (kind, t), s in zip(lines, ls, strict=True)
            )
            out(f"    profile: {parts}")


def warm_e5(store, reps, out=print) -> None:
    e5 = next((r for r in reps if r.cacheable), None)
    if e5 is None:
        out("e5: model not downloaded (run `just models`); showing tfidf and lsa only")
        return
    t0 = time.perf_counter()
    _, stats = encode_videos(store, e5, _all_ids(store))
    dt = time.perf_counter() - t0
    per = dt / stats["computed"] if stats["computed"] else 0.0
    out(
        f"{e5.name}: {stats['computed']} embedded + {stats['cached']} cached in {dt:.1f} s"
        + (f" ({per * 1000:.0f} ms/video)" if stats["computed"] else "")
    )


def download(out=print) -> int:
    paths = ensure_home()
    t0 = time.perf_counter()
    rep = e5mod.download(paths)
    out(f"e5 model ready in {e5mod.cache_dir(paths)} ({time.perf_counter() - t0:.0f} s)")
    texts = [
        "17. Bayesian Statistics",
        "Bayesian inference explained: priors and posteriors",
        "Statistik für Anfänger: Bayes-Theorem",
        "Late-night host roasts the president",
    ]
    v = rep.encode(texts)
    out(f"  dim={v.shape[1]}; similarity to {texts[0]!r}:")
    for t, s in zip(texts[1:], (v[1:] @ v[0]).tolist(), strict=True):
        out(f"    {s:5.2f}  {t}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="just sim", description=__doc__.split("\n")[0])
    ap.add_argument("query", nargs="?", help="video id or any text (default: overview)")
    ap.add_argument("--rep", help="only this representation (tfidf, lsa, e5-small-q)")
    ap.add_argument("--k", type=int, default=8, help="how many neighbours (default 8)")
    ap.add_argument("--download", action="store_true", help="download the e5 model and exit")
    ap.add_argument("--eval", action="store_true", help="score representations against your labels")
    args = ap.parse_args(argv)
    if args.download:
        return download()

    paths = ensure_home()
    if not paths.db.exists():
        print(f"no database yet at {paths.db} (`just fetch <id>` creates it)")
        return 1
    profile = load_profile(paths)
    lines = profile.interests + [d.text for d in profile.dont_show]
    with Store.open(paths) as store:
        reps = available(store, lines, paths=paths)
        if args.rep:
            reps = [r for r in reps if r.name == args.rep]
            if not reps:
                print(f"unknown or unavailable representation {args.rep!r}", file=sys.stderr)
                return 1
        if args.eval:
            from sidekick.embed.evaluate import evaluate, report

            e5 = next((r for r in reps if r.name == e5mod.NAME), None)
            if e5 is not None and not args.rep:
                reps.append(e5.with_passages())
            for r in reps:
                if r.cacheable:
                    warm_e5(store, [r])
            res = evaluate(store, reps, profile.interests, [d.text for d in profile.dont_show])
            print("\n".join(report(res)))
            return 0
        warm_e5(store, reps)
        if args.query:
            neighbours(store, reps, args.query, profile, args.k)
        else:
            gate_preview(store, reps, profile.interests)
            dont_show_preview(store, reps, [d.text for d in profile.dont_show], args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
