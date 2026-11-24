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


def learn_report(store: Store, *, min_coverage: float = 1.0, md: bool = False) -> list[str]:
    """M6c: learned rankers and pickers on held-out current videos."""
    from sidekick.rank import learn as Ln
    from sidekick.rank import rerank as R

    pools = M.evaluable(load_pools(store), min_coverage=min_coverage)
    if len(pools) < 3:
        return ["", "Learning needs at least 3 fully labeled pools."]
    vectors = R.Vectors(store)
    lines = [
        "",
        f"== Replay check: v0 + caps + MMR reproduces the logged top 10 on "
        f"{R.agreement_with_log(pools, vectors):.0%} of pools ==",
    ]

    def table(ev: Ln.Eval) -> list[str]:
        rows = []
        for name, res in ev.results.items():
            rows.append(
                [
                    name,
                    _fmt(M.bootstrap([r["ndcg"] for r in res])),
                    _fmt(M.bootstrap([r["p"] for r in res]), pct=True),
                    _fmt(M.bootstrap([r["bad"] for r in res]), pct=True),
                ]
            )
        return _table(["ranker", "NDCG@10", "P@10 (good)", "bad@10"], rows, md)

    loo = Ln.leave_one_out(pools, vectors)
    lines += ["", f"== Leave one current video out ({len(loo.seeds)} folds) =="]
    lines += table(loo)
    nd = {k: [r["ndcg"] for r in v] for k, v in loo.results.items()}
    lines += ["", "== Paired differences in NDCG@10 (leave one out) =="]
    for a, b in (
        ("v0 + caps", "v0 + caps + MMR (shipped)"),
        ("v0 score only", "v0 + caps"),
        ("logistic regression + caps", "v0 + caps"),
        ("LightGBM lambdarank + caps", "v0 + caps"),
        ("LightGBM lambdarank + caps", "logistic regression + caps"),
    ):
        lines.append(f"{a} − {b}: {_fmt(M.paired_diff(nd[a], nd[b]))}")
    ws = ["plain" if w is None else f"{w:g}" for w in loo.windows]
    lines.append(f"MMR window chosen per fold (on its training pools): {', '.join(ws)}")

    ts = Ln.time_split_eval(pools, vectors)
    n_test = len(ts.seeds)
    lines += ["", f"== Time split check: {len(pools) - n_test} older train, {n_test} newer test =="]
    lines += table(ts)

    lines += ["", "== What the models use (mean over leave-one-out folds) =="]
    lr = Ln.mean_importance(loo.importance["logistic regression"])[:12]
    gb = Ln.mean_importance(loo.importance["LightGBM lambdarank"])[:12]
    rows = [
        [a[0], f"{a[1]:+.2f} ± {a[2]:.2f}", b[0], f"{b[1]:.0%} ± {b[2]:.0%}"]
        for a, b in zip(lr, gb, strict=True)
    ]
    lines += _table(
        ["logistic regression", "coefficient (per SD)", "LightGBM", "share of gain"], rows, md
    )
    return lines


def caps_report(store: Store, *, min_coverage: float = 1.0, md: bool = False) -> list[str]:
    """M6c part 2: how loose can the diversity caps get before they cost anything?"""
    from sidekick.rank import learn as Ln

    pools = M.evaluable(load_pools(store), min_coverage=min_coverage)
    if len(pools) < 3:
        return ["", "The cap grid needs at least 3 fully labeled pools."]
    n_lec = sum(p.kind == "lecture" for p in pools)
    lines = [
        "",
        f"== Cap grid on all {len(pools)} pools ({n_lec} in a lecture context), v0 scores, "
        "no MMR (descriptive) ==",
    ]
    rows = []
    for c in Ln.cap_grid():
        s = Ln.score_caps(pools, c)
        rows.append(
            [
                Ln.cap_label(c),
                f"{s['ndcg']:.3f}",
                f"{s['p'] * 100:.0f} %",
                f"{s['bad'] * 100:.0f} %",
                f"{s['groups']:.1f}",
                f"{s['largest']:.1f}",
            ]
        )
    lines += _table(
        ["caps", "NDCG@10", "P@10", "bad@10", "groups in top 10", "largest group"], rows, md
    )
    ev = Ln.caps_leave_one_out(pools)
    lines += [
        "",
        f"== Caps chosen inside each leave-one-out fold (tightest within "
        f"{Ln.TOLERANCE} NDCG of the best on its training pools) ==",
    ]
    counts: dict[str, int] = {}
    for c in ev.chosen:
        counts[Ln.cap_label(c)] = counts.get(Ln.cap_label(c), 0) + 1
    lines.append(
        "chosen: " + "; ".join(f"{k} ×{v}" for k, v in sorted(counts.items(), key=lambda t: -t[1]))
    )
    rows = []
    for name, res in ev.results.items():
        rows.append(
            [
                name,
                _fmt(M.bootstrap([r["ndcg"] for r in res])),
                _fmt(M.bootstrap([r["p"] for r in res]), pct=True),
                _fmt(M.bootstrap([r["bad"] for r in res]), pct=True),
            ]
        )
    lines += _table(["held-out", "NDCG@10", "P@10 (good)", "bad@10"], rows, md)
    nd = {k: [r["ndcg"] for r in v] for k, v in ev.results.items()}
    lines.append(
        "chosen − M5 caps: "
        + _fmt(M.paired_diff(nd["chosen caps (nested)"], nd["M5 caps (performer 2, course 5)"]))
    )
    lines.append(
        "no caps − chosen: " + _fmt(M.paired_diff(nd["no caps"], nd["chosen caps (nested)"]))
    )
    return lines


def query_report(store: Store, *, md: bool = False) -> list[str]:
    """How good were the search queries? Labels of each query's results, by query length."""
    from sidekick.e6 import cluster_rate

    labels = {
        (seed, vid): lab
        for seed, vid, lab in store.conn.execute(
            """SELECT c.current_video_id, l.video_id, l.label FROM labels l
               JOIN computes c ON c.compute_id = l.compute_id
               WHERE l.kind = 'candidate' ORDER BY l.ts, l.label_id"""
        )
    }
    rows = store.conn.execute(
        """SELECT DISTINCT c.current_video_id, k.compute_id, k.query, k.video_id
           FROM candidates k JOIN computes c ON c.compute_id = k.compute_id
           WHERE k.source = 'query' AND k.query IS NOT NULL"""
    ).fetchall()
    per_q: dict[tuple[str, str], dict[str, int]] = {}
    seed_of: dict[tuple[str, str], str] = {}
    for seed, cid, query, vid in rows:
        lab = labels.get((seed, vid))
        if lab is None:
            continue
        key = (cid, query)
        seed_of[key] = seed
        per_q.setdefault(key, {"good": 0, "meh": 0, "bad": 0})[lab] += 1
    lines = ["", "== Search queries: labels of their results, by query length =="]
    if not per_q:
        return lines + ["No labeled search results yet."]

    def bucket(q: str) -> str:
        n = len(q.split())
        return "1 word" if n == 1 else "2 words" if n == 2 else "3+ words"

    names = ("1 word", "2 words", "3+ words")
    counts: dict[str, dict[str, dict[str, tuple[int, int]]]] = {
        n: {"good": {}, "bad": {}} for n in names
    }
    n_queries = dict.fromkeys(names, 0)
    for key, c in per_q.items():
        name, seed, total = bucket(key[1]), seed_of[key], sum(c.values())
        n_queries[name] += 1
        for lab in ("good", "bad"):
            hits, tot = counts[name][lab].get(seed, (0, 0))
            counts[name][lab][seed] = (hits + c[lab], tot + total)

    def pct(t: tuple[float, float, float, int]) -> str:
        return (
            "–" if math.isnan(t[0]) else f"{t[0] * 100:.0f} % [{t[1] * 100:.0f}, {t[2] * 100:.0f}]"
        )

    table = []
    for name in names:
        good, bad = cluster_rate(counts[name]["good"]), cluster_rate(counts[name]["bad"])
        table.append([name, str(n_queries[name]), pct(good), pct(bad), str(good[3])])
    lines += _table(["query length", "queries", "good", "bad", "labeled results"], table, md)
    worst = sorted(
        ((k, c) for k, c in per_q.items() if sum(c.values()) >= 5),
        key=lambda kc: -kc[1]["bad"] / sum(kc[1].values()),
    )[:8]
    lines += ["", "Worst queries (≥ 5 labeled results):"]
    lines += _table(
        ["query", "good / meh / bad"],
        [[q[:50], f"{c['good']} / {c['meh']} / {c['bad']}"] for (_, q), c in worst],
        md,
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
    ap.add_argument("--learn", action="store_true", help="M6c: learned rankers, leave one out")
    ap.add_argument("--caps", action="store_true", help="M6c: replay a grid of diversity caps")
    ap.add_argument("--queries", action="store_true", help="label quality of search queries")
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
        if args.learn:
            for line in learn_report(store, min_coverage=args.min_coverage, md=args.md):
                print(line)
        if args.caps:
            for line in caps_report(store, min_coverage=args.min_coverage, md=args.md):
                print(line)
        if args.queries:
            for line in query_report(store, md=args.md):
                print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
