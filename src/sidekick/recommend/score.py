"""Scores for ranking (M4, v0): a cheap pre-score that decides what gets enriched, and the
hand-weighted v0 score that orders the final list.

**Pre-score** (before Stage B, cheap fields only: title, channel, list positions). Enrichment
is the slow part of a compute (~2.5 s per uncached watch page), so the enrichment budget goes to
the candidates most likely to be shown::

    pre = 0.60 · topic + 0.25 · recurrence + 0.15 · position

- ``topic``: :func:`sidekick.topic.match` of the candidate's title/channel against the current
  video's topic anchor (0–1).
- ``recurrence``: how many of this compute's lists surfaced it (up-next, each query), as
  ``min(1, (n − 1) / 2)``; HackerTube's "recommended in several places" signal.
- ``position``: ``1 − (best position − 1) / list length``; YouTube's own order, as a weak prior.

**v0 score** (after Stage C, on the survivors). Weights are hand-set; every part is logged so
M6 can learn them::

    filter-only mode:  0.40 topic + 0.35 seed + 0.10 recurrence + 0.05 same channel
                       + 0.05 persistence + 0.05 views
    interest mode:     0.35 topic + 0.30 seed + 0.15 interest + 0.10 recurrence
                       + 0.04 same channel + 0.03 persistence + 0.03 views

- ``seed``: Stage C's raw similarity to the current video, as a **percentile within this
  pool**, so the e5 bunching (0.74–0.98) and the TF-IDF fallback land on the same 0–1 scale.
- ``interest``: Stage C's centered similarity to the best Interests line, also a percentile;
  only in interest mode.
- ``persistence``: in how many *earlier* computes for the same current video it was a
  candidate, as ``min(1, n / 3)``.
- ``views``: ``log10(views) / 7`` clipped to 1; deliberately weak (spec: rank by the user's own
  signal, not engagement).

Diversity is a plain per-channel cap for now (MMR comes next).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

PRE_WEIGHTS = {"topic": 0.60, "recurrence": 0.25, "position": 0.15}
WEIGHTS = {
    "filter-only": {
        "topic": 0.40,
        "seed": 0.35,
        "recurrence": 0.10,
        "same_channel": 0.05,
        "persistence": 0.05,
        "views": 0.05,
    },
    "interest": {
        "topic": 0.35,
        "seed": 0.30,
        "interest": 0.15,
        "recurrence": 0.10,
        "same_channel": 0.04,
        "persistence": 0.03,
        "views": 0.03,
    },
}


def recurrence(n_lists: int) -> float:
    return min(1.0, max(0, n_lists - 1) / 2)


def position(best_pos: int | None, list_len: int | None) -> float:
    if not best_pos or not list_len:
        return 0.0
    return max(0.0, 1.0 - (best_pos - 1) / list_len)


def views_part(views: int | None) -> float:
    if not views or views <= 0:
        return 0.0
    return min(1.0, math.log10(views) / 7)


def persistence(n_earlier: int) -> float:
    return min(1.0, n_earlier / 3)


def prescore(topic: float, n_lists: int, best_pos: int | None, list_len: int | None) -> float:
    parts = {
        "topic": topic,
        "recurrence": recurrence(n_lists),
        "position": position(best_pos, list_len),
    }
    return round(sum(PRE_WEIGHTS[k] * v for k, v in parts.items()), 4)


def percentiles(values: Sequence[float | None]) -> list[float]:
    """Rank-based 0–1 scale within one pool (ties share the average rank; None → 0)."""
    known = sorted(v for v in values if v is not None)
    n = len(known)
    if n == 0:
        return [0.0] * len(values)
    if n == 1:
        return [1.0 if v is not None else 0.0 for v in values]
    out = []
    for v in values:
        if v is None:
            out.append(0.0)
            continue
        lo = sum(x < v for x in known)
        eq = sum(x == v for x in known)
        out.append(round((lo + (eq - 1) / 2) / (n - 1), 4))
    return out


def v0(parts: dict[str, float], mode: str) -> float:
    w = WEIGHTS["interest" if mode == "interest" else "filter-only"]
    return round(sum(w[k] * parts.get(k, 0.0) for k in w), 4)
