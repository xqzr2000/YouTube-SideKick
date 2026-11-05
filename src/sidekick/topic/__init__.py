"""The topic anchor: what the current video is *about*, so recommendations stay on it.

Decided 2026-11-05 (Russell): every compute gets a topic anchor; in a **news context** it is
strict and means **the lead story only**, and pundit/commentary videos should not be shown
there (that rule comes with the next PR, once there are "same story?" labels).

Extraction, for one video (no transcripts in v1):

1. **Clean.** Drop the channel's boilerplate description lines and its title template
   (see :mod:`sidekick.topic.text`). Template words are kept apart as ``series``.
2. **Lead.** If the title still says something specific (2+ content words left), the lead is
   the title plus the first content sentence of the description (``title+desc``). Otherwise
   (a full newscast, "… at Six for Tuesday"), the lead is the first chapter that isn't an
   intro/headlines/weather/sports (``chapter``), else the first specific description line
   (``description``), else the title.
3. **Entities and keyphrases** from the lead only, with spaCy (:mod:`sidekick.topic.nlp`).
4. **Weighted tokens**: each token weighs its IDF over our own videos (so "rezoning" counts
   more than "council"): 1× in the headline (title, chapter or lead line), 0.6× in the
   description sentence of a ``title+desc`` lead, 0.3× in keywords and the next lines.
   Keyphrases rank by mean token weight × √length.

Matching a candidate against the anchor (:func:`match`) gives interpretable parts, all in
[0, 1]: ``tok`` (weighted share of the anchor's tokens found in the candidate), ``ent`` (same,
for entity tokens only), ``series`` (share of the anchor's template words in the candidate's
title), plus the publish-date gap in days. ``score`` (v0) is the mean of ``tok`` and ``ent``;
``just topic`` also shows e5 similarity between the two topic texts. Which parts separate
"same story" from the rest is for the labels to say, not this docstring.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sidekick.store.db import Store
from sidekick.topic import text as tx
from sidekick.topic.nlp import SKIP_LABELS, HeuristicTagger, Tagger

NEWS_CATEGORY = "News & Politics"
_NEWS_NAME_RE = re.compile(r"\bnews\b|newshour|newscast|\bnewsroom\b|\bnews ?night\b", re.I)
LEAD_DESC_WEIGHT = 0.6  # first description sentence of a title+desc lead
SECONDARY_WEIGHT = 0.3
SECONDARY_LINES = 2
MIN_SPECIFIC = 2  # content words a title needs, after the template, to be its own lead


@dataclass
class Topic:
    video_id: str
    title: str
    channel: str
    lead: str  # the text the topic was taken from
    lead_source: str  # title | title+desc | chapter | description | none (no usable topic)
    entities: list[tuple[str, str]] = field(default_factory=list)  # (text, label)
    keyphrases: list[tuple[str, float]] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)  # token -> weight
    entity_tokens: set[str] = field(default_factory=set)
    series: list[str] = field(default_factory=list)
    is_news: bool = False
    news_reason: str = ""
    published_at: str | None = None
    tagger: str = ""

    @property
    def text(self) -> str:
        """What gets embedded for topic similarity: the lead, dates removed."""
        return tx.DATE_RE.sub(" ", self.lead).strip()

    def query(self, max_words: int = 6) -> str:
        """A search query for this topic (for M4 backfill): entities first, then keyphrases."""
        words: list[str] = []
        seen: set[str] = set()
        for text_, _ in self.entities:
            for w in text_.split():
                n = tx.norm_token(w)
                if n and n not in seen and n not in tx.STOP:
                    words.append(w)
                    seen.add(n)
        for phrase, _ in self.keyphrases:
            for n in phrase.split():
                if n not in seen:
                    words.append(n)
                    seen.add(n)
        return " ".join(words[:max_words])

    def summary(self) -> dict[str, Any]:
        return {
            "lead_source": self.lead_source,
            "lead": self.lead,
            "entities": [f"{t} ({lab})" for t, lab in self.entities],
            "keyphrases": [p for p, _ in self.keyphrases],
            "series": self.series,
            "news": self.news_reason or "no",
            "query": self.query(),
            "tagger": self.tagger,
        }


class TopicContext:
    """Corpus statistics and per-channel caches shared by every extraction in one run."""

    def __init__(self, store: Store, tagger: Tagger | None = None):
        self.store = store
        self.tagger = tagger or HeuristicTagger()
        rows = store.conn.execute("SELECT title, keywords, description FROM videos").fetchall()
        df: Counter[str] = Counter()
        for r in rows:
            df.update(set(tx.tokens(_doc_text(dict(r)))))
        self.n_docs = len(rows)
        self.df = df
        self._channels: dict[str, tuple[set[str], set[str]]] = {}

    def idf(self, token: str) -> float:
        return math.log((self.n_docs + 1) / (self.df.get(token, 0) + 1)) + 1.0

    def channel(self, channel: str) -> tuple[set[str], set[str]]:
        """``(boilerplate lines, title template)`` for a channel, from our database."""
        if channel not in self._channels:
            rows = self.store.conn.execute(
                "SELECT title, description FROM videos WHERE channel = ?", (channel,)
            ).fetchall()
            boiler = tx.channel_boilerplate(r["description"] or "" for r in rows)
            template = tx.title_template([r["title"] or "" for r in rows])
            self._channels[channel] = (boiler, template)
        return self._channels[channel]


def _doc_text(v: dict[str, Any]) -> str:
    return " ".join(
        [v.get("title") or "", " ".join(_keywords(v)), (v.get("description") or "")[:1000]]
    )


def _keywords(v: dict[str, Any]) -> list[str]:
    kw = v.get("keywords") or []
    if isinstance(kw, str):
        try:
            kw = json.loads(kw)
        except ValueError:
            kw = [kw]
    return [str(k) for k in kw]


def news_context(v: dict[str, Any]) -> str:
    """Why a video counts as news (``category`` / ``name``), or "" if it doesn't."""
    if (v.get("category") or "") == NEWS_CATEGORY:
        return "category"
    if _NEWS_NAME_RE.search(v.get("channel") or "") or _NEWS_NAME_RE.search(v.get("title") or ""):
        return "name"
    return ""


def _pick_lead(v: dict[str, Any], boiler: set[str], drop: set[str]) -> tuple[str, str, list[str]]:
    """``(lead, source, secondary lines)``."""
    title = v.get("title") or ""
    desc = v.get("description") or ""
    chaps = tx.chapters(desc)
    lines = [ln for ln in tx.content_lines(desc, boiler) if not (chaps and tx.is_chapter_line(ln))]

    def specific(s: str, n: int = MIN_SPECIFIC) -> bool:
        return len([t for t in tx.tokens(s) if t not in drop]) >= n

    if specific(title):
        first = next((tx.first_sentence(ln) for ln in lines if specific(ln, 3)), "")
        if first:
            return f"{title}. {first}", "title+desc", lines[1 : 1 + SECONDARY_LINES]
        return title, "title", lines[:SECONDARY_LINES]
    for _, ct in chaps:
        if not tx.is_generic_chapter(ct) and specific(ct):
            return ct, "chapter", lines[:SECONDARY_LINES]
    for i, ln in enumerate(lines):
        if specific(ln, 3):
            return tx.first_sentence(ln), "description", lines[i + 1 : i + 1 + SECONDARY_LINES]
    # A template-only title with nothing usable behind it: no topic to anchor on.
    return title, "none", []


def extract(ctx: TopicContext, video: dict[str, Any] | str) -> Topic | None:
    """The topic of one video (a ``videos`` row or its id); None if it isn't in the database."""
    v = ctx.store.get_video(video) if isinstance(video, str) else video
    if not v:
        return None
    channel = v.get("channel") or ""
    boiler, template = ctx.channel(channel)
    chan_tokens = set(tx.tokens(channel))
    drop = template | chan_tokens
    lead, source, secondary = _pick_lead(v, boiler, drop)

    # Headline words count fully; the description's first sentence a bit less; keywords and
    # the next lines less again.
    weights: dict[str, float] = {}
    head, rest = (v.get("title") or "", lead) if source == "title+desc" else (lead, "")
    for part, w in ((head, 1.0), (rest, LEAD_DESC_WEIGHT)):
        for t in tx.tokens(part):
            if t not in drop:
                weights[t] = max(weights.get(t, 0.0), w * ctx.idf(t))
    for t in tx.tokens(" ".join(_keywords(v) + secondary)):
        if t not in drop and t not in weights:
            weights[t] = SECONDARY_WEIGHT * ctx.idf(t)

    ents: list[tuple[str, str]] = []
    ent_tokens: set[str] = set()
    seen: set[str] = set()
    for e in ctx.tagger.entities(lead):
        if e.label in SKIP_LABELS:
            continue
        toks = [t for t in tx.tokens(e.text) if t not in drop]
        key = " ".join(toks)
        if not toks or key in seen:
            continue
        seen.add(key)
        ents.append((e.text.removeprefix("the ").removeprefix("The "), e.label))
        ent_tokens.update(toks)

    phrases: dict[tuple[str, ...], float] = {}
    for p in ctx.tagger.phrases(lead):
        toks = tuple(dict.fromkeys(t for t in tx.tokens(p) if t not in drop))
        if toks and toks not in phrases:  # mean weight × √length: multiword, not just long
            phrases[toks] = sum(weights.get(t, 0.0) for t in toks) / len(toks) ** 0.5
    keyphrases = sorted(((" ".join(k), round(s, 2)) for k, s in phrases.items()), key=_by_score)

    reason = news_context(v)
    return Topic(
        video_id=v["video_id"],
        title=v.get("title") or "",
        channel=channel,
        lead=lead,
        lead_source=source,
        entities=ents[:8],
        keyphrases=keyphrases[:6],
        weights=weights,
        entity_tokens=ent_tokens,
        series=sorted(template),
        is_news=bool(reason),
        news_reason=reason,
        published_at=v.get("published_at"),
        tagger=getattr(ctx.tagger, "name", ""),
    )


def _by_score(kv: tuple[str, float]) -> tuple[float, str]:
    return (-kv[1], kv[0])


def _days(a: str | None, b: str | None) -> float | None:
    try:
        return abs((date.fromisoformat(a[:10]) - date.fromisoformat(b[:10])).days)
    except (TypeError, ValueError):
        return None


def match(anchor: Topic, cand: Topic) -> dict[str, Any]:
    """How well ``cand`` stays on ``anchor``'s topic: interpretable parts and a v0 score."""
    have = set(cand.weights) | set(tx.tokens(cand.title))
    total = sum(anchor.weights.values())
    tok = sum(w for t, w in anchor.weights.items() if t in have) / total if total else 0.0
    ent = None
    if anchor.entity_tokens:
        ew = {t: anchor.weights.get(t, 1.0) for t in anchor.entity_tokens}
        ent = sum(w for t, w in ew.items() if t in have) / sum(ew.values())
    series = None
    if anchor.series:
        ct = set(tx.tokens(cand.title))
        series = sum(t in ct for t in anchor.series) / len(anchor.series)
    parts = [tok] + ([ent] if ent is not None else [])
    return {
        "score": round(sum(parts) / len(parts), 3),
        "tok": round(tok, 3),
        "ent": None if ent is None else round(ent, 3),
        "series": None if series is None else round(series, 3),
        "same_channel": anchor.channel == cand.channel,
        "days": _days(anchor.published_at, cand.published_at),
        "cand_news": cand.is_news,
    }
