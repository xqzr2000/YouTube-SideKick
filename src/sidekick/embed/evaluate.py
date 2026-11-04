"""``just sim --eval``: score each representation against your hand labels (an early look at E1).

Three threshold-free checks per representation, each *raw* and *centered* (the mean video
vector subtracted, then re-normalized; see ``embed.center``):

- **Gate**: each current video's best similarity to the Interests, grouped by its seed label.
  ``gap`` = lowest "yes" − highest "no". Positive means some τ separates them perfectly; the
  bigger, the safer. ("partly" seeds are shown but not used.)
- **Don't-show AUC**: how well max similarity to the Don't-show-me lines ranks candidates you
  labeled *bad* above those you labeled *good* (meh left out). 0.5 = chance, 1.0 = perfect.
- **Seed-sim AUC**: how well similarity to the *current video* ranks *good* candidates above
  *meh + bad*. Your reasons ("drifted from the course series") suggest this matters most.

AUC is the Mann-Whitney statistic: the share of (positive, negative) pairs ordered correctly,
ties counted as half. With ~40 labels these are rough; they guide choices, they don't settle
them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from sidekick.embed import center, encode_videos
from sidekick.store.db import Store


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score of a random positive > score of a random negative), ties = 1/2. NaN if empty."""
    p = np.asarray(pos, dtype=float)
    n = np.asarray(neg, dtype=float)
    if p.size == 0 or n.size == 0:
        return float("nan")
    diff = p[:, None] - n[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def seed_labels(store: Store) -> dict[str, str]:
    """Newest seed label per current video."""
    rows = store.conn.execute(
        """SELECT video_id, label FROM labels l WHERE kind = 'seed' AND label_id = (
               SELECT label_id FROM labels m WHERE m.kind = 'seed' AND m.video_id = l.video_id
               ORDER BY ts DESC, label_id DESC LIMIT 1)"""
    )
    return {r[0]: r[1] for r in rows}


def candidate_labels(store: Store) -> list[tuple[str, str, str]]:
    """``(seed_id, video_id, label)`` for every candidate label (newest per pair)."""
    rows = store.conn.execute(
        """SELECT c.current_video_id, l.video_id, l.label, l.ts, l.label_id
           FROM labels l JOIN computes c ON c.compute_id = l.compute_id
           WHERE l.kind = 'candidate' ORDER BY l.ts, l.label_id"""
    )
    latest: dict[tuple[str, str], str] = {}
    for seed, vid, label, _, _ in rows:
        latest[(seed, vid)] = label
    return [(s, v, lab) for (s, v), lab in latest.items()]


@dataclass
class Result:
    rep: str
    variant: str  # raw | centered
    gate: list[tuple[str, str, float]] = field(default_factory=list)  # (seed, label, score)
    gate_gap: float = float("nan")
    auc_dont: float = float("nan")
    auc_seed: float = float("nan")
    n: dict[str, int] = field(default_factory=dict)


def evaluate(store: Store, reps: list, interests: list[str], dont_show: list[str]) -> list[Result]:
    seeds = seed_labels(store)
    cands = candidate_labels(store)
    ids = [r[0] for r in store.conn.execute("SELECT video_id FROM videos ORDER BY video_id")]
    index = {v: i for i, v in enumerate(ids)}
    out = []
    for rep in reps:
        base, _ = encode_videos(store, rep, ids)
        lines_i = rep.encode(interests) if interests else np.zeros((0, base.shape[1]))
        lines_d = rep.encode(dont_show) if dont_show else np.zeros((0, base.shape[1]))
        mean = base.mean(axis=0, keepdims=True)
        for variant in ("raw", "centered"):
            if variant == "raw":
                V, Q, D = base, lines_i, lines_d
            else:
                V = center(base, mean)
                Q = center(lines_i, mean) if len(lines_i) else lines_i
                D = center(lines_d, mean) if len(lines_d) else lines_d
            res = Result(rep.name, variant)
            # gate
            if len(Q):
                for seed, label in seeds.items():
                    if seed in index:
                        res.gate.append((seed, label, float((V[index[seed]] @ Q.T).max())))
                yes = [s for _, lab, s in res.gate if lab == "good"]
                no = [s for _, lab, s in res.gate if lab == "bad"]
                if yes and no:
                    res.gate_gap = min(yes) - max(no)
            # don't-show: bad above good
            pairs = [(s, v, lab) for s, v, lab in cands if v in index and s in index]
            if len(D):
                dsim = {v: float((V[index[v]] @ D.T).max()) for _, v, _ in pairs}
                res.auc_dont = auc(
                    [dsim[v] for _, v, lab in pairs if lab == "bad"],
                    [dsim[v] for _, v, lab in pairs if lab == "good"],
                )
            # seed similarity: good above meh + bad
            ssim = [(float(V[index[v]] @ V[index[s]]), lab) for s, v, lab in pairs]
            res.auc_seed = auc(
                [x for x, lab in ssim if lab == "good"], [x for x, lab in ssim if lab != "good"]
            )
            res.n = {
                "seeds_yes": sum(lab == "good" for lab in seeds.values()),
                "seeds_no": sum(lab == "bad" for lab in seeds.values()),
                "good": sum(lab == "good" for *_, lab in pairs),
                "meh": sum(lab == "meh" for *_, lab in pairs),
                "bad": sum(lab == "bad" for *_, lab in pairs),
            }
            out.append(res)
    return out


def _f(x: float) -> str:
    return "   -" if np.isnan(x) else f"{x:+.2f}" if x < 0 else f"{x:.2f}"


def report(results: list[Result]) -> list[str]:
    if not results:
        return ["nothing to evaluate"]
    n = results[0].n
    lines = [
        f"labels: seeds {n['seeds_yes']} yes / {n['seeds_no']} no; "
        f"candidates {n['good']} good / {n['meh']} meh / {n['bad']} bad",
        "",
        f"  {'representation':<14} {'variant':<9} {'gate yes(min)':>13} {'no(max)':>8} "
        f"{'gap':>6}   {'AUC dont-show':>13} {'AUC seed-sim':>13}",
    ]
    for r in results:
        yes = [s for _, lab, s in r.gate if lab == "good"]
        no = [s for _, lab, s in r.gate if lab == "bad"]
        lines.append(
            f"  {r.rep:<14} {r.variant:<9} {_f(min(yes)) if yes else '   -':>13} "
            f"{_f(max(no)) if no else '   -':>8} {_f(r.gate_gap):>6}   "
            f"{_f(r.auc_dont):>13} {_f(r.auc_seed):>13}"
        )
    lines += [
        "",
        "gate gap > 0: a threshold separates yes from no seeds. AUC: 0.5 chance, 1.0 perfect.",
    ]
    return lines
