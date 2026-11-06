"""Backfill queries (spec §5.5).

**Round 1** (this PR):

- ``topic``: the current video's topic query (:meth:`sidekick.topic.Topic.query`: entities,
  then keyphrases, written as in the title; in news the place is added).
- ``interest_x`` (interest mode only): the best-matching Interests line crossed with the topic's
  first keyphrase that the line doesn't already contain, e.g. "reinforcement learning partially
  observable". It pulls the backfill toward what you said you care about without leaving the
  current video's subject. Skipped when the topic query already contains the line's words
  (the two searches would return the same videos and only inflate recurrence).

Rounds 2–3 (Rocchio expansion from the best survivors, snowball) come in the next PR.
"""

from __future__ import annotations

from sidekick.topic import Topic
from sidekick.topic import text as tx

MAX_WORDS = 6


def round1(anchor: Topic | None, mode: str, best_interest: str | None) -> list[tuple[str, str]]:
    """``[(query, method)]``; empty when the current video has no usable topic."""
    out: list[tuple[str, str]] = []
    if anchor is None or anchor.lead_source == "none":
        return out
    base = anchor.query(MAX_WORDS)
    if base:
        out.append((base, "topic"))
    if mode == "interest" and best_interest:
        have = set(tx.tokens(best_interest))
        if have and have <= set(tx.tokens(base)):
            return out  # the topic query already says it: a second search would only repeat it
        extra = ""
        for phrase, _ in anchor.keyphrases:
            toks = [t for t in phrase.split() if t not in have]
            if toks:
                extra = " ".join(anchor.surface.get(t, t) for t in toks[:3])
                break
        q = f"{best_interest} {extra}".strip()
        if q.casefold() != base.casefold():
            out.append((q, "interest_x"))
    return out
