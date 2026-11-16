"""``just eval``: how good is the ranking, on fully labeled pools?

Examples::

    just eval                          # newest pool per current video, complete pools only
    just eval --min-coverage 0.9       # also pools with up to 10 % unlabeled
    just eval --csv ~/.sidekick/m6_dataset.csv     # the dataset, for pandas / marimo
    just eval --md                     # Markdown tables for reports/E4-ranker.md

It scores, on the same pools, what SideKick showed (v0 + MMR + caps), v0's score alone, a
random order, and a few single features, with 95 % bootstrap intervals over current videos.
Then it lists every current video and screens each logged feature by AUC (good vs rest). The
learned ranker (M6c) has to beat the first two lines on held-out current videos.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from sidekick.config import ensure_home
from sidekick.rank import metrics as M
from sidekick.rank.dataset import FEATURES, load_pools, to_records
from sidekick.store.db import Store

BASELINES: list[tuple[str, M.Ranker | None]] = [
    ("v0 shipped (v0 + MMR + caps)", M.v0_shipped),
    ("v0 score only", M.v0_score),
    ("random order", None),
    ("topic only", M.by_feature("topic")),
    ("seed similarity only", M.by_feature("seed_sim")),
    ("interest only", M.by_feature("interest_sim")),
]


def _fmt(t: tuple[float, float, float], pct: bool = False) -> str:
    m, lo, hi = t
    if math.isnan(m):
        return "–"
    if pct:
        return f"{m * 100:.0f} % [{lo * 100:.0f}, {hi * 100:.0f}]"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def _table(header: list[str], rows: list[list[str]], md: bool) -> list[str]:
    if md:
        out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        return out + ["| " + " | ".join(r) + " |" for r in rows]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows, strict=True)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return [fmt.format(*header)] + [fmt.format(*r) for r in rows]


def report(store: Store, *, min_coverage: float = 1.0, which: str = "newest", md: bool = False):
    pools = load_pools(store, which=which)
    ev = M.evaluable(pools, min_coverage=min_coverage)
    lines = ["== Data =="]
    n_lab = sum(len(p.labeled) for p in pools)
    lines.append(
        f"{len(pools)} pools ({which} per current video), {sum(len(p.rows) for p in pools)} "
        f"candidates, {n_lab} labeled; {len(ev)} pools evaluable "
        f"(coverage ≥ {min_coverage:.0%} and at least one good or meh)"
    )
    skipped = [p for p in pools if p not in ev]
    for p in skipped:
        lines.append(
            f"  skipped: {p.seed_title[:50]}  labeled {len(p.labeled)}/{len(p.rows)}"
            + ("" if p.labeled and any(r.gain for r in p.labeled) else " (no good/meh)")
        )
    if not ev:
        lines.append("Nothing to evaluate yet: label whole pools with `just label-app`.")
        return lines
    g = [r.gain for p in ev for r in p.labeled]
    lines.append(
        f"labels in evaluated pools: good {g.count(2)} · meh {g.count(1)} · bad {g.count(0)}"
    )

    lines += ["", f"== Rankers on {len(ev)} pools (mean [95 % interval over current videos]) =="]
    results: dict[str, list[dict]] = {}
    rows = []
    for name, ranker in BASELINES:
        res = M.random_per_pool(ev) if ranker is None else M.per_pool(ev, ranker)
        results[name] = res
        rows.append(
            [
                name,
                _fmt(M.bootstrap([r["ndcg"] for r in res])),
                _fmt(M.bootstrap([r["p"] for r in res]), pct=True),
                _fmt(M.bootstrap([r["bad"] for r in res]), pct=True),
            ]
        )
    lines += _table(["ranker", "NDCG@10", "P@10 (good)", "bad@10"], rows, md)

    shipped, score = results[BASELINES[0][0]], results[BASELINES[1][0]]
    rnd = results["random order"]
    d_mmr = M.paired_diff([r["ndcg"] for r in shipped], [r["ndcg"] for r in score])
    d_rnd = M.paired_diff([r["ndcg"] for r in shipped], [r["ndcg"] for r in rnd])
    lines += [
        "",
        "== Paired differences in NDCG@10 ==",
        f"v0 shipped − v0 score only: {_fmt(d_mmr)}  (what MMR + caps cost or add)",
        f"v0 shipped − random order:  {_fmt(d_rnd)}  (how much v0 knows)",
    ]

    lines += ["", "== Per current video (NDCG@10) =="]
    per = []
    for i, p in enumerate(ev):
        gg = [r.gain for r in p.labeled]
        per.append(
            [
                p.seed_title[:48],
                p.trigger,
                str(len(p.rows)),
                f"{gg.count(2)}/{gg.count(1)}/{gg.count(0)}",
                f"{shipped[i]['ndcg']:.3f}",
                f"{score[i]['ndcg']:.3f}",
                f"{rnd[i]['ndcg']:.3f}",
            ]
        )
    lines += _table(
        ["current video", "trigger", "pool", "g/m/b", "v0 shipped", "v0 score", "random"], per, md
    )

    lines += ["", "== Feature screening: AUC good vs rest, pooled (0.5 = no signal) =="]
    fa = [
        [name, "–" if math.isnan(auc) else f"{auc:.2f}", str(n), FEATURES[name]]
        for name, auc, n in M.feature_auc(ev)
        if n
    ]
    lines += _table(["feature", "AUC", "n", "meaning"], fa, md)
    lines.append(
        "(Pooled across current videos, so a feature that only says *which* current video it "
        "is can look strong here; M6c checks features on held-out current videos.)"
    )
    return lines


def write_csv(store: Store, path: Path, which: str) -> int:
    recs = to_records(load_pools(store, which=which))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if recs:
            w = csv.DictWriter(f, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(recs)
    return len(recs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="just eval", description=__doc__.split("\n")[0])
    ap.add_argument("--min-coverage", type=float, default=1.0, help="labeled share (0-1)")
    ap.add_argument("--all", action="store_true", help="every ranked compute, not the newest")
    ap.add_argument("--md", action="store_true", help="Markdown tables")
    ap.add_argument("--csv", type=Path, help="write the dataset (one row per candidate)")
    args = ap.parse_args(argv)
    which = "all" if args.all else "newest"
    paths = ensure_home()
    with Store.open(paths) as store:
        if args.csv:
            n = write_csv(store, args.csv.expanduser(), which)
            print(f"wrote {n} rows to {args.csv}")
            return 0
        for line in report(store, min_coverage=args.min_coverage, which=which, md=args.md):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
