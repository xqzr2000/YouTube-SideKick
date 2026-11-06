"""Picking the 10 to show: MMR with group caps (spec §5.8).

Decided 2026-11-05 (Russell), after the first real ``just recommend`` runs:

- **Lecture context** (the current video is in YouTube's Education category, or its title
  says lecture / course / module / seminar / class / tutorial): the same course may fill up to
  **5** slots. Your labels say "same course = good", and a cap of 3 had pushed out two lectures
  you labeled good. The group is the channel (a course's lectures share one).
- **Everything else:** a single performer or show gets at most **2** slots. The group is the
  person named in the title when there is one (spaCy PERSON), else the channel, so a comedian
  posting through two channels (Dry Bar Clips, Dry Bar Comedy) still counts once.

Within the caps, **maximal marginal relevance** picks greedily::

    mmr = λ · v0 − (1 − λ) · max similarity to what's already picked      (λ = 0.75)

with centered e5 (or TF-IDF) vectors, where near-duplicates score high and unrelated videos
near 0. If the caps leave slots empty (a pool that is all one course), the best capped videos
fill them, so a list is never shorter than it has to be.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from sidekick.topic import text as tx
from sidekick.topic.nlp import Tagger

LAMBDA = 0.75
CAPS = {"lecture": 5, "other": 2}
_LECTURE_RE = re.compile(
    r"(?i:\b(?:lecture|lectures|course|module|seminar|class|tutorial|lesson)\b)"
    r"|\b[A-Z]{2,4}\s?\d{2,4}[A-Z]?\b"  # course codes: "CS 224", "CS224R", "STAT 110"
)


def context_kind(category: str | None, title: str | None) -> str:
    """``lecture`` or ``other`` for the current video."""
    if (category or "") == "Education" or _LECTURE_RE.search(title or ""):
        return "lecture"
    return "other"


def group_key(title: str, channel: str, kind: str, tagger: Tagger | None) -> str:
    """The diversity group of a candidate: its channel in a lecture context, else the person
    named in its title (when there is one) or its channel."""
    if kind == "lecture" or tagger is None:
        return f"channel:{channel.casefold()}"
    for seg in tx.title_segments(title):
        for e in tagger.entities(seg):
            if e.label == "PERSON":
                name = " ".join(tx.strip_entity(e.text).casefold().split())
                if len(name.split()) >= 2:  # "Chad Thornsberry", not a lone "Trump"
                    return f"person:{name}"
    return f"channel:{channel.casefold()}"


def mmr_select(
    scores: Sequence[float],
    vecs: np.ndarray | None,
    groups: Sequence[str],
    *,
    k: int,
    cap: int,
    lam: float = LAMBDA,
) -> tuple[list[int], dict[int, dict]]:
    """Indices in pick order, plus per-index info (``mmr``, ``redundancy``, ``capped``)."""
    n = len(scores)
    order: list[int] = []
    info: dict[int, dict] = {}
    count: dict[str, int] = defaultdict(int)
    left = set(range(n))
    sims = vecs @ vecs.T if vecs is not None and len(vecs) == n and n else None
    capped: set[int] = set()
    while left and len(order) < k:
        best, best_val, best_red = None, -1e9, 0.0
        for i in sorted(left):
            if count[groups[i]] >= cap:
                capped.add(i)
                continue
            red = max((float(sims[i, j]) for j in order), default=0.0) if sims is not None else 0.0
            val = lam * scores[i] - (1 - lam) * max(0.0, red)
            if val > best_val + 1e-12:
                best, best_val, best_red = i, val, red
        if best is None:
            break
        order.append(best)
        info[best] = {"mmr": round(best_val, 4), "redundancy": round(best_red, 4)}
        count[groups[best]] += 1
        left.discard(best)
    # caps left slots empty: fill with the best capped ones
    for i in sorted(left, key=lambda i: -scores[i]):
        if len(order) >= k:
            break
        order.append(i)
        info[i] = {"mmr": None, "redundancy": None, "filled_past_cap": True}
        left.discard(i)
    for i in left:
        info[i] = {"capped": i in capped or count[groups[i]] >= cap}
    return order, info
