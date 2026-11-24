"""Picking the 10 to show: MMR with group caps (spec §5.8).

**Since 2026-11-24 the default is plain v0 score order: no caps, MMR off** (M6c dropped the
performer cap and MMR; the pre-registered lecture-pool check dropped the course cap). See
``CAPS`` below and ``reports/E4-ranker.md``. The rest of this docstring describes the M4
design, which stays available through ``Config``.

Decided 2026-11-05 (Russell), after the first real ``just recommend`` runs:

- **Lecture context** (the current video is in YouTube's Education category, or its title
  says lecture / course / module / seminar / class / tutorial): the same course may fill up to
  **5** slots. Your labels say "same course = good", and a cap of 3 had pushed out two lectures
  you labeled good. The group is the channel (a course's lectures share one).
- **Everything else:** a single performer or show gets at most **2** slots. The group is the
  person named in the title when there is one (spaCy PERSON), else the channel, so a comedian
  posting through two channels (Dry Bar Clips, Dry Bar Comedy) still counts once.

**Finding the performer** (fixed after the second real run, where 6 videos of one comedian
got through a cap of 2): spaCy returns the name with different tails depending on the title
("Chad Thornsberry - Full Special", "Chad Thornsberry Stand"), so each entity is cut to its
leading capitalized name words (2–3, stopping at words like Full / Special / Stand / Comedy).
Every name found anywhere in the pool is then looked up in *every* title, so a title where
spaCy found nothing still joins its performer's group, and then in the channel name, so an
upload with the name only in its channel ("Running A 5k" on "Chad Thornsberry - Topic", third
real run) joins it too.

Within the caps, **maximal marginal relevance** picks greedily::

    mmr = λ · v0 − (1 − λ) · max similarity to what's already picked      (λ = 0.75)

with centered e5 (or TF-IDF) vectors, where near-duplicates score high and unrelated videos
near 0. In a lecture context, lectures of the *same course* don't count as redundant with each
other (they are different lectures, and you want up to 5 of them); the penalty applies across
courses. If the caps leave slots empty (a pool that is all one course), the best capped videos
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
# M6c (2026-11-15, Russell chose "let the data pick"): on 13 fully labeled pools, the
# performer cap of 2 cost NDCG@10 0.070 [0.014, 0.140] and tripled the bad share; every
# leave-one-out round chose no performer cap. MMR is off by default (``Config.mmr``): it added
# +0.003 on top of the caps.
# 2026-11-24, pre-registered rule met (8 lecture pools, judged on lecture pools only): every
# fold chose no course cap; held out +0.130 NDCG@10 [0.035, 0.224] over course cap 5.
CAPS: dict[str, int | None] = {"lecture": None, "other": None}  # None = no cap
CAPS_M5 = {"lecture": 5, "other": 2}  # what M4-M5 shipped, kept for comparisons
_LECTURE_RE = re.compile(
    r"(?i:\b(?:lecture|lectures|course|module|seminar|class|tutorial|lesson)\b)"
    r"|\b[A-Z]{2,4}\s?\d{2,4}[A-Z]?\b"  # course codes: "CS 224", "CS224R", "STAT 110"
)


def context_kind(category: str | None, title: str | None) -> str:
    """``lecture`` or ``other`` for the current video."""
    if (category or "") == "Education" or _LECTURE_RE.search(title or ""):
        return "lecture"
    return "other"


NAME_STOP = frozenset(
    """
    full special specials stand standup stand-up comedy comedian live show shows double
    feature best official clips clip dry bar the of and ft feat vs podcast interview
    highlights episode part new set tour hour
    """.split()
)
_NAME_WORD_RE = re.compile(r"^[A-Z][\w'’.-]*$")


def person_name(text: str) -> str:
    """ "Chad Thornsberry - Full Special" → "chad thornsberry"; "" unless 2+ name words."""
    words = []
    for w in re.split(r"[\s|•·:,()\[\]\-–—]+", tx.strip_entity(text)):
        if not w:
            continue
        if not _NAME_WORD_RE.match(w) or w.casefold() in NAME_STOP or len(words) == 3:
            break
        words.append(w.strip(".'’").casefold())
    return " ".join(words) if len(words) >= 2 else ""  # "Chad Thornsberry", not a lone "Trump"


def _names_in(title: str, tagger: Tagger) -> list[str]:
    out = []
    for seg in tx.title_segments(title):
        for e in tagger.entities(seg):
            if e.label == "PERSON":
                n = person_name(e.text)
                if n:
                    out.append(n)
    return out


def assign_groups(
    titles: Sequence[str], channels: Sequence[str], kind: str, tagger: Tagger | None
) -> list[str]:
    """Diversity group per candidate: the channel in a lecture context; elsewhere the performer
    (a name found in any title of the pool and present in this title, longest first, or failing
    that in this channel, e.g. "Chad Thornsberry - Topic"), else the channel."""
    if kind == "lecture" or tagger is None:
        return [f"channel:{c.casefold()}" for c in channels]
    found = [_names_in(t, tagger) for t in titles]
    known = sorted({n for ns in found for n in ns}, key=lambda n: (-len(n), n))
    pats = [(n, re.compile(rf"\b{re.escape(n)}\b", re.I)) for n in known]
    out = []
    for t, c in zip(titles, channels, strict=True):
        hit = next((n for n, rx in pats if rx.search(t)), None) or next(
            (n for n, rx in pats if rx.search(c)), None
        )
        out.append(f"person:{hit}" if hit else f"channel:{c.casefold()}")
    return out


def group_key(title: str, channel: str, kind: str, tagger: Tagger | None) -> str:
    """One candidate's group (see :func:`assign_groups`, which also matches names across the
    pool)."""
    return assign_groups([title], [channel], kind, tagger)[0]


def mmr_select(
    scores: Sequence[float],
    vecs: np.ndarray | None,
    groups: Sequence[str],
    *,
    k: int,
    cap: int,
    lam: float = LAMBDA,
    same_group_redundant: bool = True,
    window: float | None = None,
) -> tuple[list[int], dict[int, dict]]:
    """Indices in pick order, plus per-index info (``mmr``, ``redundancy``, ``capped``).
    ``same_group_redundant=False`` (lecture context) measures redundancy only against picks
    from other groups. ``window`` (M6c): each step only considers videos whose score is within
    ``window`` of the best one still available, so variety can reorder close calls but can't
    promote a weak outlier (``None`` = plain MMR, ``0`` = score order with caps)."""
    n = len(scores)
    order: list[int] = []
    info: dict[int, dict] = {}
    count: dict[str, int] = defaultdict(int)
    left = set(range(n))
    sims = vecs @ vecs.T if vecs is not None and len(vecs) == n and n else None
    capped: set[int] = set()
    while left and len(order) < k:
        best, best_val, best_red = None, -1e9, 0.0
        open_ = [i for i in left if count[groups[i]] < cap]
        floor = (
            max((scores[i] for i in open_), default=0.0) - window if window is not None else None
        )
        for i in sorted(left):
            if count[groups[i]] >= cap:
                capped.add(i)
                continue
            if floor is not None and scores[i] < floor - 1e-12:
                continue
            red = (
                max(
                    (
                        float(sims[i, j])
                        for j in order
                        if same_group_redundant or groups[j] != groups[i]
                    ),
                    default=0.0,
                )
                if sims is not None
                else 0.0
            )
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
