"""Ranking metrics on labeled pools, baselines, splits and bootstrap intervals.

**NDCG@10** (normalized discounted cumulative gain) rewards putting good videos high:

    DCG@k  = sum over the first k positions i (1-based) of gain_i / log2(i + 1)
    NDCG@k = DCG@k of the ranker's order / DCG@k of the best possible order of the same pool

Gains are linear (good 2, meh 1, bad 0). NDCG is 1 for a perfect order and is computed per
pool, then averaged over pools. A pool with no good or meh video has no ideal order to
compare against and is skipped.

**Unlabeled videos** are dropped *after* ranking ("condensed lists", Sakai 2007): the ranker
orders the whole pool, then only labeled videos are kept, in that order. With complete pools
nothing is dropped; with nearly complete ones, an unlabeled video neither helps nor hurts.

**Uncertainty** is a bootstrap over pools (one per current video), because videos labeled in
one context and one sitting aren't independent. Differences between two rankers are paired:
the bootstrap resamples pools and recomputes both on the same sample.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator, Sequence

import numpy as np

from sidekick.rank.dataset import FEATURES, Pool, Row

K = 10
Ranker = Callable[[Pool], list[Row]]  # returns the pool's rows, best first


# -- metrics -----------------------------------------------------------------------------------


def dcg(gains: Sequence[float], k: int = K) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg(ordered_gains: Sequence[float], k: int = K) -> float | None:
    ideal = dcg(sorted(ordered_gains, reverse=True), k)
    return None if ideal == 0 else dcg(ordered_gains, k) / ideal


def precision(ordered_gains: Sequence[float], k: int = K) -> float | None:
    top = ordered_gains[:k]
    return sum(g == 2 for g in top) / len(top) if top else None


def bad_share(ordered_gains: Sequence[float], k: int = K) -> float | None:
    top = ordered_gains[:k]
    return sum(g == 0 for g in top) / len(top) if top else None


def condensed_gains(ordered: list[Row]) -> list[int]:
    return [r.gain for r in ordered if r.gain is not None]


# -- baselines ---------------------------------------------------------------------------------


def v0_shipped(pool: Pool) -> list[Row]:
    """What SideKick showed: the 10 picks in rank order (v0 + MMR + caps), then the rest by v0."""
    shown = sorted((r for r in pool.rows if r.v0_rank), key=lambda r: r.v0_rank)
    rest = sorted((r for r in pool.rows if not r.v0_rank), key=lambda r: -_nz(r.v0_score))
    return shown + rest


def v0_score(pool: Pool) -> list[Row]:
    """v0's score alone, without MMR or group caps (ties in a fixed shuffled order)."""
    return sorted(random_order(pool, seed=99), key=lambda r: -_nz(r.v0_score))


def by_feature(name: str, *, reverse: bool = False) -> Ranker:
    """Sort by one logged feature (missing values last); ``reverse`` = lower is better."""

    def rank(pool: Pool) -> list[Row]:
        def key(r: Row):
            v = r.features.get(name, math.nan)
            return (math.isnan(v), v if reverse else -v)

        # start from a fixed shuffle so ties (and missing values) don't inherit v0's order
        return sorted(random_order(pool, seed=99), key=key)

    rank.__name__ = f"feature:{name}"
    return rank


def random_order(pool: Pool, *, seed: int = 0) -> list[Row]:
    rows = list(pool.rows)
    random.Random(f"{pool.compute_id}:{seed}").shuffle(rows)
    return rows


def _nz(x: float) -> float:
    return 0.0 if x is None or math.isnan(x) else x


# -- evaluation --------------------------------------------------------------------------------


def evaluable(pools: list[Pool], *, min_coverage: float = 1.0) -> list[Pool]:
    """Pools labeled at least ``min_coverage`` with at least one good or meh video."""
    return [
        p
        for p in pools
        if p.labeled and p.coverage >= min_coverage and any(r.gain for r in p.labeled)
    ]


def per_pool(pools: list[Pool], ranker: Ranker, k: int = K) -> list[dict]:
    out = []
    for p in pools:
        g = condensed_gains(ranker(p))
        out.append(
            {
                "seed_id": p.seed_id,
                "ndcg": ndcg(g, k),
                "p": precision(g, k),
                "bad": bad_share(g, k),
            }
        )
    return out


def random_per_pool(pools: list[Pool], k: int = K, n: int = 200) -> list[dict]:
    """Expected metrics of a random order (mean over ``n`` shuffles)."""
    out = []
    for p in pools:
        vals = [condensed_gains(random_order(p, seed=s)) for s in range(n)]
        out.append(
            {
                "seed_id": p.seed_id,
                "ndcg": float(np.mean([ndcg(g, k) for g in vals])),
                "p": float(np.mean([precision(g, k) for g in vals])),
                "bad": float(np.mean([bad_share(g, k) for g in vals])),
            }
        )
    return out


def bootstrap(values: Sequence[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95 % percentile interval, resampling the values (one per pool)."""
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return (float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_diff(a: Sequence[float], b: Sequence[float], n: int = 2000, seed: int = 0):
    """Mean of a - b over pools with both values, with a paired bootstrap interval."""
    d = [x - y for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    return bootstrap(d, n, seed)


# -- feature screening -------------------------------------------------------------------------


def feature_auc(pools: list[Pool]) -> list[tuple[str, float, int]]:
    """Per feature: AUC of good vs not-good over labeled rows (0.5 = no signal; below 0.5 =
    higher values mean worse), and how many rows had a value. Pooled over current videos, so
    it's a screening tool, not an evaluation."""
    from sklearn.metrics import roc_auc_score

    rows = [r for p in pools for r in p.labeled]
    out = []
    for name in FEATURES:
        xs, ys = [], []
        for r in rows:
            v = r.features.get(name, math.nan)
            if not math.isnan(v):
                xs.append(v)
                ys.append(int(r.gain == 2))
        auc = roc_auc_score(ys, xs) if len(set(ys)) == 2 and len(set(xs)) > 1 else math.nan
        out.append((name, float(auc), len(xs)))
    return sorted(out, key=lambda t: -abs(t[1] - 0.5) if not math.isnan(t[1]) else 0)


# -- splits (for M6c) --------------------------------------------------------------------------


def seeds_in_time_order(pools: list[Pool]) -> list[str]:
    first: dict[str, str] = {}
    for p in sorted(pools, key=lambda p: p.ts):
        first.setdefault(p.seed_id, p.ts)
    return sorted(first, key=lambda s: first[s])


def time_split(pools: list[Pool], test_frac: float = 0.3) -> tuple[list[Pool], list[Pool]]:
    """Older current videos train, newer ones test; never a random split of rows."""
    seeds = seeds_in_time_order(pools)
    n_test = max(1, round(len(seeds) * test_frac)) if len(seeds) > 1 else 0
    test = set(seeds[len(seeds) - n_test :])
    return [p for p in pools if p.seed_id not in test], [p for p in pools if p.seed_id in test]


def leave_one_seed_out(pools: list[Pool]) -> Iterator[tuple[str, list[Pool], list[Pool]]]:
    for s in seeds_in_time_order(pools):
        yield s, [p for p in pools if p.seed_id != s], [p for p in pools if p.seed_id == s]
