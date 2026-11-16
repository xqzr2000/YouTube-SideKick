"""Replaying the "pick 10" step on logged pools, for any score.

SideKick shows 10 videos picked from a scored pool by MMR with group caps
(``recommend.diversity``). To compare pickers offline, this module redoes that step from what
was logged: the scores (v0, or a learned model's), each candidate's diversity group and the
pool's context (stage R features), and the e5 vectors from the embeddings cache, centered the
same way Stage C centers them. No model is loaded and nothing is fetched.

Variants, from least to most diversity:

- ``order``: by score only (no caps, no MMR);
- ``caps``: by score, with the group caps (course ≤ 5 in a lecture context, performer ≤ 2);
- ``mmr``: MMR (λ 0.75) with caps, as shipped;
- ``mmr window δ``: MMR that only considers videos within δ of the best remaining score.

The picked 10 come first; the rest of the pool follows by score, so NDCG@10 only depends on
the picks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from sidekick.rank.dataset import Pool, Row
from sidekick.recommend import diversity as dv
from sidekick.store.db import Store

K = 10


def _model_mean(store: Store, model: str) -> np.ndarray | None:
    rows = store.conn.execute("SELECT vector FROM embeddings WHERE model = ?", (model,)).fetchall()
    if not rows:
        return None
    return np.vstack([np.frombuffer(r[0], dtype="<f4") for r in rows]).mean(axis=0)


class Vectors:
    """Centered, normalized cached vectors per pool (zero rows for videos with none)."""

    def __init__(self, store: Store):
        self.store = store
        self._means: dict[str, np.ndarray | None] = {}

    def for_pool(self, pool: Pool) -> np.ndarray | None:
        model = pool.rep
        if not model:
            return None
        if model not in self._means:
            self._means[model] = _model_mean(self.store, model)
        mean = self._means[model]
        if mean is None:
            return None
        out = np.zeros((len(pool.rows), mean.shape[0]), dtype=np.float32)
        for i, r in enumerate(pool.rows):
            row = self.store.conn.execute(
                "SELECT vector FROM embeddings WHERE model = ? AND video_id = ?",
                (model, r.video_id),
            ).fetchone()
            if row:
                v = np.frombuffer(row[0], dtype="<f4") - mean
                n = float(np.linalg.norm(v))
                out[i] = v / n if n else v
        return out


def pick(
    pool: Pool,
    scores: Sequence[float],
    vecs: np.ndarray | None,
    *,
    caps: bool = True,
    mmr: bool = True,
    window: float | None = None,
    k: int = K,
    cap_values: dict[str, int | None] | None = None,
) -> list[Row]:
    """The pool's rows in shown order under one picking rule. ``cap_values`` overrides the
    shipped caps per context (``{"lecture": 8, "other": None}``; None = no cap)."""
    idx = list(range(len(pool.rows)))
    by_score = sorted(idx, key=lambda i: (-scores[i], pool.rows[i].video_id))
    if not caps and not mmr:
        return [pool.rows[i] for i in by_score]
    groups = [r.group or r.video_id for r in pool.rows]
    table = {**dv.CAPS, **(cap_values or {})}
    cap_v = table.get(pool.kind, table["other"])
    cap = cap_v if caps and cap_v is not None else 10**6
    order, _ = dv.mmr_select(
        list(scores),
        vecs if mmr else None,
        groups,
        k=k,
        cap=cap,
        same_group_redundant=pool.kind != "lecture",
        window=window,
    )
    seen = set(order)
    return [pool.rows[i] for i in order] + [pool.rows[i] for i in by_score if i not in seen]


Scorer = Callable[[Pool], list[float]]


def v0_scores(pool: Pool) -> list[float]:
    return [0.0 if r.v0_score != r.v0_score else r.v0_score for r in pool.rows]


def picker(
    scorer: Scorer,
    vectors: Vectors | None,
    *,
    caps: bool = True,
    mmr: bool = True,
    window: float | None = None,
    cap_values: dict[str, int | None] | None = None,
) -> Callable[[Pool], list[Row]]:
    """A ranker (pool → ordered rows) for ``metrics.per_pool``."""

    def rank(pool: Pool) -> list[Row]:
        vecs = vectors.for_pool(pool) if (vectors is not None and mmr) else None
        return pick(
            pool, scorer(pool), vecs, caps=caps, mmr=mmr, window=window, cap_values=cap_values
        )

    return rank


def agreement_with_log(pools: list[Pool], vectors: Vectors) -> float:
    """Share of pools whose replayed v0 + caps + MMR top 10 equals the logged top 10 (as a
    set), a check that the replay reproduces what SideKick showed."""
    if not pools:
        return float("nan")
    hits = 0
    for p in pools:
        rows = picker(v0_scores, vectors)(p)[:K]
        hits += {r.video_id for r in rows} == {r.video_id for r in p.rows if r.v0_rank}
    return hits / len(pools)


def variety(ordered: list[Row], k: int = K) -> tuple[int, int]:
    """List-level variety of a top k: (distinct groups, size of the largest group). Item
    labels can't see this; it's reported next to NDCG so the trade-off is visible."""
    counts: dict[str, int] = {}
    for r in ordered[:k]:
        g = r.group or r.video_id
        counts[g] = counts.get(g, 0) + 1
    return len(counts), max(counts.values(), default=0)
