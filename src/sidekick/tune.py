"""``just tune``: set the gate τ and Stage C thresholds from your labels (precision-first).

For each Stage C signal (Don't-show-me similarity, similarity to downvoted videos, clickbait,
promo) the new threshold is ``max(score of any video labeled good) + 0.01``, never below the
floor in ``thresholds.FLOORS``. So no video you called good would be dropped, and a small label
set can't make a filter trigger-happy. The gate τ is the midpoint between the lowest "yes" seed
and the highest "no" seed (when they separate).

Labels whose current video is **news** don't count for the Don't-show-me threshold: in a news
context Stage C keeps news reports that match a Don't-show line and only drops non-news and
commentary (see ``filters.stage_c``), so "PBS NewsHour, good" while watching news says nothing
about where the line should sit elsewhere. Before 2026-11-05 those labels would have pushed the
threshold to about 0.36 and switched the filter off almost everywhere.

It prints, per signal: the old and new threshold, how many bad / meh / good labeled videos it
would catch, and how many *unlabeled* Stage B survivors in the database it would drop (with a
few titles), since that's where false removals would come from. ``--write`` saves the result
to ``~/.sidekick/thresholds.json``; without it nothing changes.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime

from sidekick.config import get_paths
from sidekick.embed.evaluate import candidate_labels, seed_labels
from sidekick.filters import thresholds as thmod
from sidekick.filters.stage_c import build_space, run_stage_c
from sidekick.models import Candidate
from sidekick.profile import load_profile
from sidekick.store.db import Store

SIM_SIGNALS = ("dont_show", "downvoted")
TEXT_SIGNALS = ("clickbait", "promo")
FEATURE = {"dont_show": "dont_show_sim", "downvoted": "downvoted_sim"}
MARGIN = 0.01


def _ceil2(x: float) -> float:
    return math.ceil(round(x * 100, 6)) / 100


def _cand(store: Store, vid: str) -> Candidate:
    v = store.get_video(vid) or {}
    return Candidate(vid, title=v.get("title") or "", channel=v.get("channel") or "")


def score_labeled(store, space, th) -> list[tuple[str, str, dict]]:
    """``(video_id, label, features)`` for every labeled candidate, in its seed's context."""
    by_seed: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for seed, vid, lab in candidate_labels(store):
        by_seed[seed].append((vid, lab))
    out = []
    for seed, items in by_seed.items():
        cands = [_cand(store, v) for v, _ in items]
        decs = run_stage_c(store, space, cands, current_id=seed, thresholds=th)
        out += [(v, lab, d.features) for (v, lab), d in zip(items, decs, strict=True)]
    return out


def unlabeled_survivors(store: Store, labeled: set[str]) -> list[str]:
    rows = store.conn.execute(
        "SELECT DISTINCT video_id FROM decisions WHERE stage = 'B' AND kept = 1 ORDER BY video_id"
    )
    return [r[0] for r in rows if r[0] not in labeled]


def tune(store: Store, profile, th: thmod.Thresholds, *, paths=None, e5=None):
    labels = candidate_labels(store)
    seeds = seed_labels(store)
    ids = sorted({v for _, v, _ in labels} | set(seeds))
    space = build_space(store, profile, e5=e5, paths=paths, extra_ids=ids)
    rep = space.name
    cur = th.for_rep(rep)
    scored = score_labeled(store, space, th)
    pool_ids = unlabeled_survivors(store, {v for v, _, _ in scored})
    pool = run_stage_c(
        store, space, [_cand(store, v) for v in pool_ids], current_id=None, thresholds=th
    )

    new = thmod.Thresholds(
        sim={k: dict(v) for k, v in th.sim.items()},
        clickbait=th.clickbait,
        promo=th.promo,
        pundit_clickbait=th.pundit_clickbait,
        source=datetime.now(UTC).isoformat(timespec="seconds"),
        notes={},
    )
    new.sim.setdefault(rep, dict(cur))
    report = []
    for sig in SIM_SIGNALS + TEXT_SIGNALS:
        key = FEATURE.get(sig, sig)
        old = cur[sig] if sig in SIM_SIGNALS else getattr(th, sig)
        use = [
            (v, lab, f)
            for v, lab, f in scored
            if not (sig == "dont_show" and f.get("news_context"))
        ]
        skipped = len(scored) - len(use)
        vals = {
            lab: [f[key] for _, l2, f in use if l2 == lab and f.get(key) is not None]
            for lab in ("good", "meh", "bad")
        }
        if not any(vals.values()):
            report.append({"signal": sig, "old": old, "new": old, "note": "no scores (skipped)"})
            continue
        top_good = max(vals["good"]) if vals["good"] else None
        t = max(_ceil2(top_good + MARGIN) if top_good is not None else old, thmod.FLOORS[sig])
        caught = {lab: sum(x >= t for x in xs) for lab, xs in vals.items()}
        totals = {lab: len(xs) for lab, xs in vals.items()}
        bad_titles = [
            (store.get_video(v) or {}).get("title", v)
            for v, lab, f in use
            if lab == "bad" and f.get(key) is not None and f[key] >= t
        ]
        pool_hits = sorted(
            (
                (d.features.get(key), d.candidate.title)
                for d in pool
                if d.features.get(key) is not None and d.features[key] >= t
            ),
            reverse=True,
        )
        if sig in SIM_SIGNALS:
            new.sim[rep][sig] = t
        else:
            setattr(new, sig, t)
        report.append(
            {
                "signal": sig,
                "old": old,
                "new": t,
                "top_good": top_good,
                "floor": thmod.FLOORS[sig],
                "caught": caught,
                "totals": totals,
                "bad_titles": bad_titles[:4],
                "pool_n": len(pool),
                "pool_hits": pool_hits[:4],
                "pool_count": len(pool_hits),
                "news_skipped": skipped,
            }
        )
    # gate
    gate_scores = []
    for seed, lab in seeds.items():
        if store.get_video(seed) is None or not space.interests:
            continue
        _, c = space.videos(store, [seed])
        gate_scores.append((lab, float((c @ space.interest_vecs.T).max())))
    yes = [s for lab, s in gate_scores if lab == "good"]
    no = [s for lab, s in gate_scores if lab == "bad"]
    gate = {"old": cur["gate"], "new": cur["gate"], "yes": yes, "no": no}
    if yes and no and min(yes) > max(no):
        gate["new"] = round((min(yes) + max(no)) / 2, 3)
        new.sim[rep]["gate"] = gate["new"]
    elif yes and no:
        gate["note"] = "yes and no seeds overlap: tau unchanged"
    else:
        gate["note"] = "needs both yes and no seeds: tau unchanged"
    new.notes = {"rep": rep, "n_candidate_labels": len(scored), "n_seed_labels": len(gate_scores)}
    return new, rep, report, gate


def _fmt(report, gate, rep) -> list[str]:
    lines = [f"representation: {rep}", ""]
    g = gate
    yes = ", ".join(f"{x:.2f}" for x in sorted(g["yes"])) or "-"
    no = ", ".join(f"{x:.2f}" for x in sorted(g["no"])) or "-"
    lines.append(
        f"interest gate tau: {g['old']:.3f} -> {g['new']:.3f}   yes seeds {yes} | "
        f"no seeds {no}" + (f"   ({g['note']})" if g.get("note") else "")
    )
    lines.append("")
    for r in report:
        if "note" in r:
            lines.append(f"{r['signal']:<10} {r['old']:.2f} -> {r['new']:.2f}   ({r['note']})")
            continue
        c, n = r["caught"], r["totals"]
        tg = "-" if r["top_good"] is None else f"{r['top_good']:.2f}"
        lines.append(
            f"{r['signal']:<10} {r['old']:.2f} -> {r['new']:.2f}   (top good {tg}, floor "
            f"{r['floor']:.2f})   catches bad {c['bad']}/{n['bad']}, meh {c['meh']}/{n['meh']}, "
            f"good {c['good']}/{n['good']}; unlabeled survivors dropped "
            f"{r['pool_count']}/{r['pool_n']}"
        )
        if r.get("news_skipped"):
            lines.append(
                f"             ({r['news_skipped']} labels from news contexts not used: there "
                "Stage C keeps news reports and drops only non-news / commentary)"
            )
        for t in r["bad_titles"]:
            lines.append(f"             bad caught:  {t[:70]}")
        for s, t in r["pool_hits"]:
            lines.append(f"             unlabeled:   {s:.2f}  {t[:64]}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="just tune", description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="save to ~/.sidekick/thresholds.json")
    args = ap.parse_args(argv)
    paths = get_paths()
    if not paths.db.exists():
        print(f"no database yet at {paths.db} (`just fetch <id>` creates it)")
        return 1
    th = thmod.load(paths)
    with Store.open(paths) as store:
        if not candidate_labels(store):
            print("no labels yet: run `just label` first")
            return 1
        new, rep, report, gate = tune(store, load_profile(paths), th, paths=paths)
    print(f"current thresholds: {th.source}")
    print("\n".join(_fmt(report, gate, rep)))
    if args.write:
        p = thmod.save(new, paths)
        print(f"\nsaved {p}")
    else:
        print("\n(dry run: `just tune --write` saves these)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
