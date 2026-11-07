"""Backfill queries (spec §5.5).

**Round 1** (this PR):

- ``topic``: the current video's topic query (:meth:`sidekick.topic.Topic.query`: entities,
  then keyphrases, written as in the title; in news the place is added).
- ``interest_x`` (interest mode only): the best-matching Interests line crossed with the topic's
  first keyphrase that the line doesn't already contain, e.g. "reinforcement learning partially
  observable". It pulls the backfill toward what you said you care about without leaving the
  current video's subject. Skipped when the topic query already contains the line's words
  (the two searches would return the same videos and only inflate recurrence).

**Round 1 for a non-English video** (title mostly in a non-Latin script while the profile's
languages are Latin-script; Russell, 2026-11-06: "about the content, but only in English"). The
topic query would be in the video's own language and bring back more of the same, so instead:

- ``english``: the English words the video's own metadata already has: title first, then
  tags, then the start of the description (acronyms like "PCA" count; channel words, URLs and
  video filler like "MV" or "official" don't).
- ``crosslingual``: e5 is multilingual, so the current video's vector sits near English videos
  on the same subject. The English titles already in the database closest to it vote for their
  words (weighted by similarity, a word needs ≥ 2 titles, channel words excluded), and the top
  terms make the query. No translation model; experimental, judged on labels.

**Rounds 2–3** (only while fewer than ``min_pool`` videos survive and there's time left):

- ``rocchio``: pseudo-relevance feedback. The topic query plus the 2 heaviest terms shared by
  at least 2 of the best 3 survivors so far (their topic weights, averaged), e.g.
  "tunnel turnaround Vancouver" + "massey replacement". No negative feedback.
- snowball lists (not queries): the up-next lists of the best survivors whose watch pages this
  compute already fetched during enrichment, so they cost no extra requests.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

from sidekick.features import script as scr
from sidekick.models import VideoMeta
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


def rocchio(
    anchor: Topic | None, relevant: Sequence[Topic], *, n_terms: int = 2, min_docs: int = 2
) -> str:
    """The topic query expanded with terms the best survivors share; "" when nothing to add."""
    if anchor is None or anchor.lead_source == "none" or not relevant:
        return ""
    base = anchor.query(4)
    have = set(tx.tokens(base)) | set(anchor.weights)
    mass: Counter[str] = Counter()
    docs: Counter[str] = Counter()
    surface: dict[str, str] = {}
    for t in relevant:
        total = sum(t.weights.values()) or 1.0
        for tok, w in t.weights.items():
            mass[tok] += w / total / len(relevant)
            docs[tok] += 1
            surface.setdefault(tok, t.surface.get(tok, tok))
    new = [
        tok
        for tok, _ in mass.most_common()
        if docs[tok] >= min_docs and tok not in have and tok not in tx.STOP
    ][:n_terms]
    if not new:
        return ""
    return " ".join([base, *(surface[t] for t in new)]).strip()


# words that say "video" rather than what it's about
VIDEO_FILLER = frozenset(
    "mv pv official video music audio lyric lyrics feat ft ver version full hd 4k 8k live "
    "shorts subscribe channel youtube http https www com clip trailer teaser episode ep part "
    "vol eng sub subs subtitles cc".split()
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#'-]*[A-Za-z0-9+#]|[A-Za-z]")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def _latin_words(text: str, skip: set[str]) -> list[str]:
    out = []
    for w in _WORD_RE.findall(_URL_RE.sub(" ", text or "")):
        lw = w.lower()
        acronym = w.isupper() and 2 <= len(w) <= 6
        if (len(lw) < 3 and not acronym) or lw in tx.STOP or lw in VIDEO_FILLER or lw in skip:
            continue
        out.append(w)
    return out


def english_words(meta: VideoMeta, *, max_words: int = 3) -> list[str]:
    """English words from the video's own title, tags and description start, in that order."""
    skip = {w.lower() for w in _WORD_RE.findall(meta.channel or "")}
    seen: set[str] = set()
    out: list[str] = []
    for src in (meta.title, " ".join(meta.keywords or []), (meta.description or "")[:400]):
        for w in _latin_words(src, skip):
            if w.lower() not in seen:
                seen.add(w.lower())
                out.append(w)
                if len(out) >= max_words:
                    return out
    return out


def english_vocab(
    store, model: str, *, exclude: Iterable[str] = ()
) -> list[tuple[str, str, np.ndarray]]:
    """``(title, channel, vector)`` of every cached video with a Latin-script title."""
    ex = set(exclude)
    rows = store.conn.execute(
        "SELECT e.video_id, v.title, v.channel, e.vector FROM embeddings e "
        "JOIN videos v ON v.video_id = e.video_id WHERE e.model = ?",
        (model,),
    ).fetchall()
    return [
        (title, channel or "", np.frombuffer(vec, dtype="<f4"))
        for vid, title, channel, vec in rows
        if vid not in ex and title and scr.non_latin(title) is None
    ]


def crosslingual_terms(
    cur_vec: np.ndarray,
    vocab: Sequence[tuple[str, str, np.ndarray]],
    *,
    k: int = 8,
    n_terms: int = 3,
    min_docs: int = 2,
    exclude: Iterable[str] = (),
) -> list[str]:
    """Words shared by the English titles closest to ``cur_vec`` (similarity-weighted)."""
    if not len(vocab):
        return []
    mat = np.vstack([v for _, _, v in vocab]).astype(np.float32)
    q = cur_vec.astype(np.float32)
    sims = mat @ q / (np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) or 1.0) + 1e-9)
    have = {w.lower() for w in exclude}
    mass: Counter[str] = Counter()
    docs: Counter[str] = Counter()
    surface: dict[str, str] = {}
    for i in np.argsort(-sims)[:k]:
        title, channel, _ = vocab[int(i)]
        skip = have | {w.lower() for w in _WORD_RE.findall(channel)}
        for w in {w.lower(): w for w in _latin_words(title, skip)}.values():
            lw = w.lower()
            mass[lw] += float(sims[i])
            docs[lw] += 1
            surface.setdefault(lw, w)
    return [surface[w] for w, _ in mass.most_common() if docs[w] >= min_docs][:n_terms]


def english_round1(
    meta: VideoMeta,
    cur_vec: np.ndarray | None,
    vocab: Sequence[tuple[str, str, np.ndarray]],
) -> list[tuple[str, str]]:
    """Round-1 queries for a non-English current video: ``english`` and ``crosslingual``."""
    out: list[tuple[str, str]] = []
    words = english_words(meta)
    if len(words) >= 2 or (words and words[0].isupper()):
        out.append((" ".join(words), "english"))
    if cur_vec is not None:
        terms = crosslingual_terms(cur_vec, vocab, exclude=words)
        if terms:
            q = " ".join((words[:1] + terms)[:4])
            if all(q.casefold() != x.casefold() for x, _ in out):
                out.append((q, "crosslingual"))
    return out
