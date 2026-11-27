"""E5: taste drift (spec §8). ``just e5``.

The spec asks for a time-decay half-life on 👎 feedback and Don't-show lines. On 2026-11-27 the
database had 13 👎 over 9 days, far too few to estimate one, so E5 is split in three:

1. **Measure label stability now** (the ``relabel`` queue of the labeling app): 50 candidate
   labels from the first week of labeling are answered again, blind. Agreement, Cohen's κ and
   the mean shift (new − old on good/meh/bad = 2/1/0) say whether taste moved (a systematic
   shift) or the labels are just noisy (disagreement without a direction).
2. **A half-life switch, off by default** (:func:`decay_weight`, ``half_life_days`` in
   ``~/.sidekick/thresholds.json``): Stage C multiplies each 👎'd video's similarity, and each
   dated Don't-show line's similarity (lines appended by a 👎), by ``0.5 ** (age / h)``.
   Undated, hand-written lines never decay. ``None`` (the default) leaves Stage C exactly as
   before.
3. **A pre-registered rule** for switching it on, fixed before looking at any decayed scores:

   - *Data condition:* at least :data:`RULE_MIN_DOWNVOTES` 👎 spread over at least
     :data:`RULE_MIN_SPAN_DAYS` days.
   - *Test:* for each half-life in :data:`HALF_LIVES` (None = no decay), the AUC of the decayed
     👎-similarity for bad vs good candidate labels, using only 👎 given before each label's
     compute and leaving out the 👎'd videos themselves.
   - *Decision:* switch on the best half-life only if its AUC beats no decay by at least
     :data:`RULE_MIN_GAIN` **and** the 95 % paired cluster-bootstrap interval of the gain
     (resampling current videos) is above 0. Otherwise stay off.

   Until the data condition is met, ``just e5`` reports the rule as *not met* and shows decayed
   AUCs only with ``--preview`` (labeled as not for deciding).

The rule tests 👎'd videos only, not the dated Don't-show lines: both come from the same 👎 and
share one half-life, and the videos' vectors are cached, so the test needs no model.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from sidekick.config import get_paths
from sidekick.embed import center
from sidekick.embed import e5 as e5mod
from sidekick.store.db import Store

LABELS = ("good", "meh", "bad")  # same as sidekick.labeling.LABELS (not imported: Stage C
# imports this module, and labeling imports recommend, which imports Stage C)
GAIN = {"good": 2, "meh": 1, "bad": 0}
HALF_LIVES: tuple[float | None, ...] = (None, 60.0, 30.0, 14.0, 7.0)
RULE_MIN_DOWNVOTES = 40
RULE_MIN_SPAN_DAYS = 28
RULE_MIN_GAIN = 0.02
N_BOOT = 2000


# -- the switch ---------------------------------------------------------------------------------


def parse_ts(s: str) -> datetime:
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def age_days(then: str | None, now: datetime) -> float | None:
    """Days from ``then`` (ISO date or datetime) to ``now``; None if missing or unreadable."""
    if not then:
        return None
    try:
        return max((now - parse_ts(then)).total_seconds() / 86400, 0.0)
    except ValueError:
        return None


def decay_weight(age: float | None, half_life_days: float | None) -> float:
    """``0.5 ** (age / h)``; 1.0 when there is no half-life or no age (undated lines)."""
    if half_life_days is None or age is None:
        return 1.0
    return 0.5 ** (age / half_life_days)


# -- part 1: label stability --------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    seed: str
    video: str
    old: str
    new: str
    old_ts: str
    new_ts: str

    @property
    def gap_days(self) -> float:
        return (parse_ts(self.new_ts) - parse_ts(self.old_ts)).total_seconds() / 86400


def relabel_pairs(store: Store) -> list[Pair]:
    """(old, newest relabel) per sampled label that has been relabeled."""
    from sidekick.labeling import relabel_sample

    sample = {it.label_id: it for it in relabel_sample(store)}
    if not sample:
        return []
    rows = store.conn.execute(
        """SELECT r.label_id, l.label, l.ts, r.label, r.ts FROM relabels r
           JOIN labels l ON l.label_id = r.label_id ORDER BY r.ts, r.relabel_id"""
    ).fetchall()
    newest = {r[0]: r for r in rows if r[0] in sample}
    return [
        Pair(sample[lid].seed_id, sample[lid].video_id, old, new, ots, nts)
        for lid, old, ots, new, nts in newest.values()
    ]


def confusion(pairs: list[Pair]) -> dict[str, Counter]:
    """``{old: Counter(new)}``."""
    out: dict[str, Counter] = {lab: Counter() for lab in LABELS}
    for p in pairs:
        out[p.old][p.new] += 1
    return out


def agreement(pairs: list[Pair]) -> float:
    return sum(p.old == p.new for p in pairs) / len(pairs) if pairs else math.nan


def cohen_kappa(pairs: list[Pair], *, weighted: bool = False) -> float:
    """Cohen's κ of old vs new label. ``weighted``: linear weights on the order
    good > meh > bad, so good↔bad counts as twice the disagreement of good↔meh."""
    n = len(pairs)
    if not n:
        return math.nan
    k = len(LABELS)
    idx = {lab: i for i, lab in enumerate(LABELS)}
    obs = np.zeros((k, k))
    for p in pairs:
        obs[idx[p.old], idx[p.new]] += 1
    obs /= n
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    i, j = np.indices((k, k))
    w = np.abs(i - j) / (k - 1) if weighted else (i != j).astype(float)
    de = float((w * exp).sum())
    if de == 0:  # every answer the same both times: κ is undefined
        return math.nan
    return 1 - float((w * obs).sum()) / de


def mean_shift(pairs: list[Pair]) -> float:
    """Mean of new − old on 2/1/0. Positive = more lenient now, negative = stricter."""
    return float(np.mean([GAIN[p.new] - GAIN[p.old] for p in pairs])) if pairs else math.nan


def cluster_boot(
    groups: dict[str, list[Any]], stat, n: int = N_BOOT, seed: int = 0
) -> tuple[float, float, float]:
    """``stat`` on all items, with a 95 % interval from resampling whole groups (current
    videos), since labels under one current video aren't independent."""
    keys = sorted(groups)
    allitems = [x for k in keys for x in groups[k]]
    point = stat(allitems)
    if len(keys) < 2:
        return point, math.nan, math.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        v = stat([x for i in pick for x in groups[keys[i]]])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return point, math.nan, math.nan
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def by_seed(pairs: list[Pair]) -> dict[str, list[Pair]]:
    g: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        g[p.seed].append(p)
    return dict(g)


# -- part 2/3: 👎 timeline and the half-life rule -----------------------------------------------


def downvote_span(store: Store) -> tuple[int, float]:
    """(number of 👎, days between the first and the last)."""
    rows = [r[0] for r in store.conn.execute("SELECT ts FROM feedback WHERE type = 'downvote'")]
    if not rows:
        return 0, 0.0
    ts = sorted(parse_ts(t) for t in rows)
    return len(rows), (ts[-1] - ts[0]).total_seconds() / 86400


def rule_met(store: Store) -> tuple[bool, int, float]:
    n, span = downvote_span(store)
    return n >= RULE_MIN_DOWNVOTES and span >= RULE_MIN_SPAN_DAYS, n, span


def _vectors(store: Store, model: str = e5mod.NAME_QP) -> tuple[dict[str, np.ndarray], int]:
    rows = store.conn.execute(
        "SELECT video_id, vector FROM embeddings WHERE model = ?", (model,)
    ).fetchall()
    if not rows:
        return {}, 0
    ids = [r[0] for r in rows]
    mat = np.vstack([np.frombuffer(r[1], dtype="<f4") for r in rows])
    cen = center(mat, mat.mean(axis=0, keepdims=True))
    return dict(zip(ids, cen, strict=True)), len(ids)


@dataclass
class Scored:
    seed: str
    bad: bool  # True = bad label, False = good (meh left out)
    sims: list[tuple[float, float]]  # (similarity to a 👎'd video, age of that 👎 in days)

    def score(self, h: float | None) -> float:
        return max(s * decay_weight(a, h) for s, a in self.sims)


def downvote_dataset(store: Store, model: str = e5mod.NAME_QP) -> list[Scored]:
    """Good and bad candidate labels whose compute came after at least one 👎 with a cached
    vector, with the similarity to each such 👎'd video and that 👎's age at compute time."""
    vecs, _ = _vectors(store, model)
    downs = [
        (v, parse_ts(t))
        for v, t in store.conn.execute(
            "SELECT video_id, ts FROM feedback WHERE type = 'downvote' ORDER BY ts"
        )
        if v in vecs
    ]
    if not downs:
        return []
    downed = {v for v, _ in downs}
    rows = store.conn.execute(
        """SELECT c.current_video_id, c.ts, l.video_id, l.label FROM labels l
           JOIN computes c ON c.compute_id = l.compute_id
           WHERE l.kind = 'candidate' AND l.label IN ('good', 'bad')
           ORDER BY l.ts, l.label_id"""
    ).fetchall()
    newest = {(seed, vid): (cts, lab) for seed, cts, vid, lab in rows}  # newest label wins
    out = []
    for (seed, vid), (cts, lab) in newest.items():
        if vid in downed or vid not in vecs:
            continue
        at = parse_ts(cts)
        sims = [
            (float(vecs[vid] @ vecs[d]), (at - t).total_seconds() / 86400)
            for d, t in downs
            if t < at
        ]
        if sims:
            out.append(Scored(seed, lab == "bad", sims))
    return out


def auc(items: list[Scored], h: float | None) -> float:
    """P(score of a bad label > score of a good one), ties count half."""
    bad = [x.score(h) for x in items if x.bad]
    good = [x.score(h) for x in items if not x.bad]
    if not bad or not good:
        return math.nan
    b = np.array(bad)[:, None]
    g = np.array(good)[None, :]
    return float(((b > g).sum() + 0.5 * (b == g).sum()) / (b.size * g.size))


def halflife_table(items: list[Scored]) -> list[dict[str, Any]]:
    """AUC per half-life, and its gain over no decay with a paired cluster-bootstrap
    interval over current videos."""
    groups: dict[str, list[Scored]] = defaultdict(list)
    for x in items:
        groups[x.seed].append(x)
    out = []
    for h in HALF_LIVES:
        a, lo, hi = cluster_boot(groups, lambda xs, h=h: auc(xs, h))
        g, glo, ghi = cluster_boot(groups, lambda xs, h=h: auc(xs, h) - auc(xs, None))
        out.append(
            {"h": h, "auc": a, "lo": lo, "hi": hi, "gain": g, "gain_lo": glo, "gain_hi": ghi}
        )
    return out


def decide(table: list[dict[str, Any]]) -> float | None:
    """The pre-registered decision: the best half-life if it clears the bar, else None."""
    best = max(
        (r for r in table if r["h"] is not None and not math.isnan(r["auc"])),
        key=lambda r: r["auc"],
        default=None,
    )
    if best is None:
        return None
    if best["gain"] >= RULE_MIN_GAIN and best["gain_lo"] > 0:
        return best["h"]
    return None


# -- report -------------------------------------------------------------------------------------


def _ci(t: tuple[float, float, float], fmt: str = "{:.2f}") -> str:
    p, lo, hi = t
    if math.isnan(p):
        return "–"
    if math.isnan(lo):
        return fmt.format(p)
    return f"{fmt.format(p)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def _table(header: list[str], rows: list[list[str]], md: bool) -> list[str]:
    if md:
        return [
            "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header),
            *("| " + " | ".join(r) + " |" for r in rows),
        ]
    w = [max(len(str(x)) for x in col) for col in zip(header, *rows, strict=True)]
    return ["  ".join(str(x).ljust(n) for x, n in zip(r, w, strict=True)) for r in [header, *rows]]


def report(store: Store, *, md: bool = False, preview: bool = False) -> list[str]:
    out: list[str] = []
    h2 = (lambda s: f"## {s}") if md else (lambda s: f"\n== {s} ==")

    from sidekick.labeling import relabel_sample

    out.append(h2("Label stability (relabel queue)"))
    sample = relabel_sample(store)
    pairs = relabel_pairs(store)
    out.append(
        f"sample {len(sample)} labels from the first week; relabeled {len(pairs)}"
        + ("" if len(pairs) >= len(sample) else " (finish them in `just label-app`, queue relabel)")
    )
    if pairs:
        g = by_seed(pairs)
        gaps = [p.gap_days for p in pairs]
        out.append(
            f"gap {min(gaps):.0f}–{max(gaps):.0f} days (median {float(np.median(gaps)):.0f}); "
            f"{len(g)} current videos; intervals resample current videos"
        )
        rows = [
            ["agreement", _ci(cluster_boot(g, agreement), "{:.0%}")],
            ["Cohen's κ", _ci(cluster_boot(g, cohen_kappa))],
            ["weighted κ (linear)", _ci(cluster_boot(g, lambda p: cohen_kappa(p, weighted=True)))],
            ["mean shift new − old (2/1/0)", _ci(cluster_boot(g, mean_shift), "{:+.2f}")],
        ]
        out += _table(["measure", "value [95 % CI]"], rows, md)
        cm = confusion(pairs)
        out.append("")
        out += _table(
            ["old \\ new", *LABELS, "n"],
            [[o, *(str(cm[o][n]) for n in LABELS), str(sum(cm[o].values()))] for o in LABELS],
            md,
        )
        moved = Counter(
            "up" if GAIN[p.new] > GAIN[p.old] else "down" for p in pairs if p.new != p.old
        )
        out.append(f"\nchanged: {moved['up']} up (more lenient), {moved['down']} down (stricter)")

    out.append(h2("👎 over time and the half-life rule"))
    met, n, span = rule_met(store)
    weeks = Counter(
        parse_ts(r[0]).strftime("%G-W%V")
        for r in store.conn.execute("SELECT ts FROM feedback WHERE type = 'downvote'")
    )
    out.append(
        f"{n} 👎 over {span:.0f} days"
        + (f" ({', '.join(f'{w}: {k}' for w, k in sorted(weeks.items()))})" if weeks else "")
    )
    out.append(
        f"rule: ≥ {RULE_MIN_DOWNVOTES} 👎 over ≥ {RULE_MIN_SPAN_DAYS} days → "
        + ("**met**" if met else "not met yet")
        + "; half-life stays off unless the test below clears the bar"
    )
    if met or preview:
        items = downvote_dataset(store)
        nb = sum(x.bad for x in items)
        out.append(
            ("" if met else "PREVIEW, not for deciding (rule not met). ")
            + f"{len(items)} labels after a 👎 ({nb} bad, {len(items) - nb} good), "
            f"{len({x.seed for x in items})} current videos"
        )
        if items:
            table = halflife_table(items)
            out += _table(
                ["half-life", "AUC bad > good [95 %]", "gain vs none [95 %]"],
                [
                    [
                        "none" if r["h"] is None else f"{r['h']:.0f} d",
                        _ci((r["auc"], r["lo"], r["hi"])),
                        "–"
                        if r["h"] is None
                        else _ci((r["gain"], r["gain_lo"], r["gain_hi"]), "{:+.3f}"),
                    ]
                    for r in table
                ],
                md,
            )
            if met:
                h = decide(table)
                out.append(
                    "decision: stay off"
                    if h is None
                    else f"decision: half_life_days = {h:.0f} (set it in thresholds.json)"
                )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="just e5", description=__doc__.split("\n\n")[0])
    ap.add_argument("--md", action="store_true", help="markdown tables (for reports/)")
    ap.add_argument(
        "--preview", action="store_true", help="show decayed AUCs before the rule is met"
    )
    args = ap.parse_args(argv)
    paths = get_paths()
    if not paths.db.exists():
        print(f"no database yet at {paths.db}")
        return 1
    with Store.open(paths) as store:
        print("\n".join(report(store, md=args.md, preview=args.preview)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
