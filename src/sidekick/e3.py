"""``just e3``: experiment E3, the enrichment budget against the 90 s limit, from the logs.

Almost all of a compute's time is fetching pages (each watch or search page costs the 1–2 s
polite delay plus the request; SponsorBlock rides along), so total time is close to a straight
line in pages fetched. Every compute already logs ``pages_fetched`` and ``duration_ms``, so
E3 needs no new sweep of YouTube requests:

1. **Timing fit.** Least squares ``seconds = a + b · pages`` over every finished compute
   (``just fetch`` and ``just recommend``, warm and cold alike: a cache hit fetches no page,
   so the line holds for both). The *one-sided 95 % prediction bound* (not the fitted mean)
   says how many pages still fit under the limit, because the 90 s is a promise about the
   slow computes, not the typical one.
2. **Budget.** A cold compute fetches the current page, its search pages (1–2 in round 1,
   up to 2 more in rounds 2–3) and one watch page per enriched candidate, so the largest
   enrichment budget is ``max pages − 1 − searches``. Shown for the typical search count
   (the median logged) and the worst case (4).
3. **What a smaller budget costs.** Replayed on every logged ``just recommend`` compute: the
   enrichment budget is handed out in pre-score order, round by round, so each Stage B
   decision's position says whether it would still be enriched at budget *k*. For each *k*:
   how many of the 10 shown videos would lose enrichment (no description, no exact duration
   or language check, weaker similarity), how many of those were labeled good in that seed's
   context, and how many pages it saves. A video that loses enrichment can still be shown;
   this counts the ones put at risk, not the ones lost.

``--md`` prints the same as Markdown for ``reports/E3-timing.md``.

The ``--cold`` runs that check the fit (``just recommend <id> --cold``) skip the 24 h page
cache and are logged with ``trigger = cli-cold``, so they show up separately here. They still
use cached embeddings and loaded models: the first time a video is ever seen costs about
0.13 s more for its e5 vector, which the fit puts in the slope.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field

from sidekick.config import ensure_home
from sidekick.embed.evaluate import candidate_labels
from sidekick.store.db import Store

LIMIT_S = 90.0
BUDGETS = (10, 15, 20, 25, 30)
WORST_SEARCHES = 4  # round 1: topic + interest query; rounds 2-3: one Rocchio query each
TOP_K = 10


@dataclass
class Fit:
    n: int
    a: float = math.nan  # seconds at 0 pages (models, parsing, scoring)
    b: float = math.nan  # seconds per page
    r2: float = math.nan
    s: float = math.nan  # residual standard deviation
    x_mean: float = math.nan
    sxx: float = math.nan
    t95: float = math.nan  # one-sided 95 % t quantile, n - 2 degrees of freedom

    def mean(self, pages: float) -> float:
        return self.a + self.b * pages

    def upper(self, pages: float) -> float:
        """One-sided 95 % prediction bound for one new compute with this many pages."""
        se = self.s * math.sqrt(1 + 1 / self.n + (pages - self.x_mean) ** 2 / self.sxx)
        return self.mean(pages) + self.t95 * se

    def max_pages(self, limit: float = LIMIT_S) -> int:
        """Most pages whose 95 % bound is still under ``limit`` (−1 if none)."""
        p = -1
        while p < 500 and self.upper(p + 1) <= limit:
            p += 1
        return p


def _t95(df: int) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(0.95, df))
    except Exception:  # scipy comes with scikit-learn; this is only a fallback
        return 1.645 + 2.0 / max(df, 1)


def timing_rows(store: Store) -> list[dict]:
    """Finished computes with pages and time, plus whether they ranked (``just recommend``)."""
    rows = store.conn.execute(
        """SELECT c.compute_id, c.trigger, c.pages_fetched, c.duration_ms, c.current_video_id,
                  EXISTS (SELECT 1 FROM decisions d WHERE d.compute_id = c.compute_id
                          AND d.stage = 'R') AS ranked
           FROM computes c
           WHERE c.status = 'ok' AND c.pages_fetched IS NOT NULL AND c.duration_ms IS NOT NULL
             AND c.trigger LIKE 'cli%'
           ORDER BY c.ts"""
    ).fetchall()
    return [
        {
            "compute_id": r[0],
            "trigger": r[1],
            "pages": int(r[2]),
            "seconds": r[3] / 1000,
            "video": r[4],
            "kind": "recommend" if r[5] else "fetch",
        }
        for r in rows
    ]


def fit(rows: list[dict]) -> Fit:
    n = len(rows)
    if n < 3:
        return Fit(n=n)
    xs = [float(r["pages"]) for r in rows]
    ys = [r["seconds"] for r in rows]
    xm, ym = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - xm) ** 2 for x in xs)
    if sxx == 0:
        return Fit(n=n)
    b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys, strict=True)) / sxx
    a = ym - b * xm
    res = [y - (a + b * x) for x, y in zip(xs, ys, strict=True)]
    sse = sum(e * e for e in res)
    sst = sum((y - ym) ** 2 for y in ys)
    return Fit(
        n=n,
        a=a,
        b=b,
        r2=1 - sse / sst if sst else math.nan,
        s=math.sqrt(sse / (n - 2)),
        x_mean=xm,
        sxx=sxx,
        t95=_t95(n - 2),
    )


def searches_per_compute(store: Store) -> list[int]:
    """Search pages per logged ``just recommend`` compute."""
    rows = store.conn.execute(
        """SELECT c.compute_id, (SELECT COUNT(*) FROM queries q WHERE q.compute_id = c.compute_id)
           FROM computes c
           WHERE EXISTS (SELECT 1 FROM decisions d WHERE d.compute_id = c.compute_id
                         AND d.stage = 'R')"""
    ).fetchall()
    return [int(r[1]) for r in rows]


@dataclass
class Replay:
    computes: int = 0
    shown: int = 0
    good_shown: int = 0
    # budget k -> totals over computes
    at_risk: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    good_at_risk: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    pages_saved: dict[int, int] = field(default_factory=lambda: defaultdict(int))


def replay(store: Store, budgets=BUDGETS) -> Replay:
    """Which shown videos each budget would have left unenriched, over logged recommends."""
    good = {(s, v) for s, v, lab in candidate_labels(store) if lab == "good"}
    out = Replay()
    cids = store.conn.execute(
        """SELECT DISTINCT c.compute_id, c.current_video_id FROM computes c
           JOIN decisions d ON d.compute_id = c.compute_id AND d.stage = 'R'
           ORDER BY c.ts"""
    ).fetchall()
    for cid, seed in cids:
        b_rows = store.conn.execute(
            "SELECT video_id, features FROM decisions WHERE compute_id = ? AND stage = 'B' "
            "ORDER BY rowid",
            (cid,),
        ).fetchall()
        # each Stage B decision took the next budget slot (rounds run in order, each sorted by
        # pre-score), so its 1-based order is its position in the budget
        pos: dict[str, int] = {}
        fetched: set[str] = set()
        for i, (vid, feats) in enumerate(b_rows, start=1):
            f = json.loads(feats or "{}")
            pos.setdefault(vid, i)
            if f.get("enriched", True) and not f.get("cached", False):
                fetched.add(vid)
        shown = [
            r[0]
            for r in store.conn.execute(
                "SELECT video_id FROM decisions WHERE compute_id = ? AND stage = 'R' AND kept = 1",
                (cid,),
            )
        ]
        out.computes += 1
        out.shown += len(shown)
        out.good_shown += sum((seed, v) in good for v in shown)
        for k in budgets:
            risk = [v for v in shown if pos.get(v, 0) > k]
            out.at_risk[k] += len(risk)
            out.good_at_risk[k] += sum((seed, v) in good for v in risk)
            out.pages_saved[k] += sum(pos[v] > k for v in fetched)
    return out


def report(store: Store, *, md: bool = False, limit: float = LIMIT_S) -> list[str]:
    rows = timing_rows(store)
    f = fit(rows)
    srch = searches_per_compute(store)
    typical = int(statistics.median(srch)) if srch else 1
    rp = replay(store)
    lines: list[str] = []
    h = (lambda s: f"## {s}") if md else (lambda s: f"== {s} ==")

    lines.append(h("Timing fit (seconds = a + b · pages)"))
    by_kind: dict[str, int] = defaultdict(int)
    for r in rows:
        by_kind[r["kind"] + (" cold" if r["trigger"] == "cli-cold" else "")] += 1
    lines.append(
        f"{f.n} finished computes ("
        + ", ".join(f"{k} {n}" for k, n in sorted(by_kind.items()))
        + ")"
    )
    if math.isnan(f.b):
        lines.append("not enough computes with different page counts to fit a line (need 3+)")
        return lines
    lines.append(
        f"a = {f.a:.1f} s, b = {f.b:.2f} s/page, R² = {f.r2:.2f}, residual SD = {f.s:.1f} s"
    )
    pmax = f.max_pages(limit)
    lines.append(
        f"most pages under {limit:.0f} s (95 % prediction bound): {pmax} "
        f"(fitted mean there {f.mean(pmax):.0f} s, bound {f.upper(pmax):.0f} s)"
    )
    worst = sorted(rows, key=lambda r: r["seconds"] - f.mean(r["pages"]), reverse=True)[:3]
    lines.append(
        "slowest vs. the line: "
        + "; ".join(
            f"{r['compute_id']} {r['pages']} pages {r['seconds']:.0f} s "
            f"({r['seconds'] - f.mean(r['pages']):+.0f} s)"
            for r in worst
        )
    )
    cold = [r for r in rows if r["trigger"] == "cli-cold"]
    if cold:
        lines.append(
            "cold checks: "
            + "; ".join(
                f"{r['video']} {r['pages']} pages {r['seconds']:.0f} s "
                f"(line {f.mean(r['pages']):.0f}, bound {f.upper(r['pages']):.0f})"
                for r in cold
            )
        )

    lines.append("")
    lines.append(h("Enrichment budget"))
    lines.append(
        f"cold pages = 1 current + searches + one per enriched candidate; searches per logged "
        f"recommend: median {typical} (max {max(srch) if srch else '-'}), worst case "
        f"{WORST_SEARCHES}"
    )
    lines.append(
        f"largest budget under {limit:.0f} s: {max(0, pmax - 1 - typical)} with {typical} "
        f"search(es), {max(0, pmax - 1 - WORST_SEARCHES)} with {WORST_SEARCHES}"
    )

    lines.append("")
    lines.append(h(f"What a smaller budget costs ({rp.computes} logged `just recommend` computes)"))
    if not rp.computes:
        lines.append("no `just recommend` computes logged yet")
        return lines
    lines.append(
        f"{rp.shown} shown videos, {rp.good_shown} of them labeled good in their seed's context"
    )
    head = (
        "budget",
        "cold s (mean / bound)",
        "shown left unenriched",
        "labeled-good left unenriched",
        "pages saved (as logged)",
    )
    body = []
    for k in BUDGETS:
        p = 1 + typical + k
        body.append(
            (
                str(k),
                f"{f.mean(p):.0f} / {f.upper(p):.0f}",
                f"{rp.at_risk[k]} / {rp.shown}",
                f"{rp.good_at_risk[k]} / {rp.good_shown}",
                str(rp.pages_saved[k]),
            )
        )
    if md:
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        lines += ["| " + " | ".join(r) + " |" for r in body]
    else:
        w = [max(len(head[i]), *(len(r[i]) for r in body)) for i in range(len(head))]
        lines.append("  ".join(x.ljust(w[i]) for i, x in enumerate(head)))
        lines += ["  ".join(x.ljust(w[i]) for i, x in enumerate(r)) for r in body]
    lines.append(
        "(cold seconds assume the median search count; 'unenriched' videos can still be shown, "
        "with less information)"
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="just e3", description=__doc__.split("\n")[0])
    ap.add_argument("--md", action="store_true", help="Markdown for reports/E3-timing.md")
    ap.add_argument("--limit", type=float, default=LIMIT_S, help="time budget in seconds")
    args = ap.parse_args(argv)
    paths = ensure_home()
    with Store.open(paths) as store:
        for line in report(store, md=args.md, limit=args.limit):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
