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
ties counted as half. With a few dozen labels these are rough, so each AUC also gets a 95 %
**bootstrap interval** (positives and negatives resampled separately, 2000 draws, fixed seed).
Two overlapping intervals mean the data can't tell the two apart yet.

Also reported for Don't-show-me: **recall at zero false removals** (``r0``), the share of bad
videos scoring strictly above every good one. That is what a precision-first threshold
(``just tune``) could catch with this representation, ignoring the floor.

``--md`` prints the same as a Markdown table for ``reports/``.
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


BOOT = 2000


def auc_ci(
    pos: Sequence[float], neg: Sequence[float], *, n_boot: int = BOOT, seed: int = 0
) -> tuple[float, float]:
    """95 % percentile bootstrap interval for ``auc``, resampling each class separately."""
    p = np.asarray(pos, dtype=float)
    n = np.asarray(neg, dtype=float)
    if p.size == 0 or n.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    pi = rng.integers(0, p.size, size=(n_boot, p.size))
    ni = rng.integers(0, n.size, size=(n_boot, n.size))
    ps, ns = p[pi], n[ni]  # (B, P), (B, N)
    diff = ps[:, :, None] - ns[:, None, :]
    stats = ((diff > 0).sum(axis=(1, 2)) + 0.5 * (diff == 0).sum(axis=(1, 2))) / (p.size * n.size)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def recall_at_zero_fp(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Share of positives scoring strictly above the highest negative. NaN if either is empty."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    top = max(neg)
    return float(np.mean([x > top for x in pos]))


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
    auc_dont_ci: tuple[float, float] = (float("nan"), float("nan"))
    auc_seed_ci: tuple[float, float] = (float("nan"), float("nan"))
    r0_dont: float = float("nan")
    n: dict[str, int] = field(default_factory=dict)
    # per labeled candidate, for plots: (seed_id, video_id, label, dont_show_sim, seed_sim)
    scores: list[tuple[str, str, str, float | None, float]] = field(default_factory=list)


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
            dsim: dict[str, float] = {}
            if len(D):
                dsim = {v: float((V[index[v]] @ D.T).max()) for _, v, _ in pairs}
                bad = [dsim[v] for _, v, lab in pairs if lab == "bad"]
                good = [dsim[v] for _, v, lab in pairs if lab == "good"]
                res.auc_dont = auc(bad, good)
                res.auc_dont_ci = auc_ci(bad, good)
                res.r0_dont = recall_at_zero_fp(bad, good)
            # seed similarity: good above meh + bad
            ssim = [(float(V[index[v]] @ V[index[s]]), lab) for s, v, lab in pairs]
            good_s = [x for x, lab in ssim if lab == "good"]
            rest_s = [x for x, lab in ssim if lab != "good"]
            res.auc_seed = auc(good_s, rest_s)
            res.auc_seed_ci = auc_ci(good_s, rest_s)
            res.scores = [
                (s, v, lab, dsim.get(v), x) for (s, v, lab), (x, _) in zip(pairs, ssim, strict=True)
            ]
            res.n = {
                "seeds_yes": sum(lab == "good" for lab in seeds.values()),
                "seeds_no": sum(lab == "bad" for lab in seeds.values()),
                "good": sum(lab == "good" for *_, lab in pairs),
                "meh": sum(lab == "meh" for *_, lab in pairs),
                "bad": sum(lab == "bad" for *_, lab in pairs),
            }
            out.append(res)
    return out


def markdown(results: list[Result]) -> list[str]:
    """The evaluation as a Markdown table (for ``reports/``)."""
    if not results:
        return ["_nothing to evaluate_"]
    n = results[0].n

    def ci(x: float, c: tuple[float, float]) -> str:
        return "–" if np.isnan(x) else f"{x:.2f} [{c[0]:.2f}, {c[1]:.2f}]"

    lines = [
        f"Labels: seeds {n['seeds_yes']} yes / {n['seeds_no']} no; candidates "
        f"{n['good']} good / {n['meh']} meh / {n['bad']} bad.",
        "",
        "| representation | variant | gate yes (min) | gate no (max) | gap | "
        "AUC don't-show [95 % CI] | r0 don't-show | AUC seed-sim [95 % CI] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        yes = [s for _, lab, s in r.gate if lab == "good"]
        no = [s for _, lab, s in r.gate if lab == "bad"]
        lines.append(
            f"| {r.rep} | {r.variant} | {f'{min(yes):.2f}' if yes else '–'} | "
            f"{f'{max(no):.2f}' if no else '–'} | {_f(r.gate_gap).strip()} | "
            f"{ci(r.auc_dont, r.auc_dont_ci)} | {_f(r.r0_dont).strip()} | "
            f"{ci(r.auc_seed, r.auc_seed_ci)} |"
        )
    return lines


def _f(x: float) -> str:
    return "   -" if np.isnan(x) else f"{x:+.2f}" if x < 0 else f"{x:.2f}"


def _ci(x: float, c: tuple[float, float]) -> str:
    return "-" if np.isnan(x) else f"{x:.2f} [{c[0]:.2f},{c[1]:.2f}]"


def report(results: list[Result]) -> list[str]:
    if not results:
        return ["nothing to evaluate"]
    n = results[0].n
    lines = [
        f"labels: seeds {n['seeds_yes']} yes / {n['seeds_no']} no; "
        f"candidates {n['good']} good / {n['meh']} meh / {n['bad']} bad",
        "",
        f"  {'representation':<14} {'variant':<9} {'gate yes(min)':>13} {'no(max)':>8} "
        f"{'gap':>6}   {'AUC dont-show [95% CI]':>23} {'r0':>5} {'AUC seed-sim [95% CI]':>22}",
    ]
    for r in results:
        yes = [s for _, lab, s in r.gate if lab == "good"]
        no = [s for _, lab, s in r.gate if lab == "bad"]
        lines.append(
            f"  {r.rep:<14} {r.variant:<9} {_f(min(yes)) if yes else '   -':>13} "
            f"{_f(max(no)) if no else '   -':>8} {_f(r.gate_gap):>6}   "
            f"{_ci(r.auc_dont, r.auc_dont_ci):>23} {_f(r.r0_dont):>5} "
            f"{_ci(r.auc_seed, r.auc_seed_ci):>22}"
        )
    lines += [
        "",
        "gate gap > 0: a threshold separates yes from no seeds. AUC: 0.5 chance, 1.0 perfect;",
        "overlapping intervals = not distinguishable yet. r0: share of bad videos above every",
        "good one on Don't-show-me (what a precision-first threshold could catch).",
    ]
    return lines
