"""The topic anchor: what the current video is *about*, so recommendations stay on it.

Decided 2026-11-05 (Russell): every compute gets a topic anchor; in a **news context** it is
strict and means **the lead story only**, and pundit/commentary videos should not be shown
there (that rule comes with a later PR, once there are "same story?" labels). Elsewhere the
anchor is lighter: the series the video belongs to counts as part of its topic.

Extraction, for one video (no transcripts in v1):

1. **Clean.** Drop the channel's boilerplate description lines, and split the title into
   segments (``"Story | Show | Channel"``), dropping the segments that are only channel name /
   title template / dates (see :mod:`sidekick.topic.text`). Template words are kept as
   ``series``.
2. **Lead.** *Outside news:* the cleaned title plus the first specific description sentence
   (``title+desc``), or the cleaned title alone (``title``). *In news:* the same if the
   cleaned title still says something specific (2+ content words); otherwise, for a full
   newscast ("… at Six for Tuesday"), the first chapter that isn't an intro/headlines/
   weather/sports (``chapter``), else the first specific description line (``description``),
   else ``none`` (no usable topic).
3. **Entities and keyphrases** from the lead only, with spaCy (:mod:`sidekick.topic.nlp`).
   ``places`` (GPE/LOC entities anywhere in the title or channel name, e.g. "Vancouver" in
   "CTV News Vancouver at Six") are kept for the search query, since a local story is defined
   by where it is.
4. **Weighted tokens**: each token weighs its IDF over our own videos (so "rezoning" counts
   more than "council"): 1× in the headline (title, chapter or lead line), 0.6× in the
   description sentence of a ``title+desc`` lead, 0.3× in tags and the next lines. Channel-name
   words never count. *In news,* template words and tags repeated across the channel's videos
   ("BC news") don't count either; *outside news,* template words count 0.3× (same series).
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

PLACE_LABELS = frozenset({"GPE", "LOC"})
# Place-like words in news-program names that don't locate a story ("ABC World News").
NOT_PLACES = frozenset({"world", "nation", "national", "international", "global", "earth"})

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
    places: list[str] = field(default_factory=list)  # news only: GPE/LOC in title / channel
    surface: dict[str, str] = field(default_factory=dict)  # normalized token -> word as written
    is_news: bool = False
    news_reason: str = ""
    published_at: str | None = None
    tagger: str = ""

    @property
    def text(self) -> str:
        """What gets embedded for topic similarity: the lead, dates removed."""
        return tx.DATE_RE.sub(" ", self.lead).strip()

    def query(self, max_words: int = 6) -> str:
        """A search query for this topic (for M4 backfill): entities, then keyphrases; in news,
        the place is always included ("tunnel turnaround Vancouver")."""
        if self.lead_source == "none":
            return ""
        place_words = []
        if self.is_news:
            for p in self.places:
                place_words += p.split()
        words: list[str] = []
        seen: set[str] = {tx.norm_token(w) for w in place_words}
        budget = max(1, max_words - len(place_words))
        for text_, _ in self.entities:
            for w in (tx.strip_entity(x) for x in text_.split()):
                n = tx.norm_token(w)
                if n and n not in seen and n not in tx.STOP:
                    words.append(w)
                    seen.add(n)
        for phrase, _ in self.keyphrases:
            for n in phrase.split():
                if n not in seen:
                    words.append(self.surface.get(n, n))  # "rule", not the stem "baye"
                    seen.add(n)
        return " ".join((words[:budget] + place_words)[:max_words])

    def summary(self) -> dict[str, Any]:
        return {
            "lead_source": self.lead_source,
            "lead": self.lead,
            "entities": [f"{t} ({lab})" for t, lab in self.entities],
            "keyphrases": [p for p, _ in self.keyphrases],
            "series": self.series,
            "places": self.places,
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
        self._channels: dict[str, ChannelInfo] = {}

    def idf(self, token: str) -> float:
        return math.log((self.n_docs + 1) / (self.df.get(token, 0) + 1)) + 1.0

    def channel(self, channel: str) -> ChannelInfo:
        """Boilerplate lines, title template and repeated tags of a channel, from our DB."""
        if channel not in self._channels:
            rows = self.store.conn.execute(
                "SELECT title, keywords, description FROM videos WHERE channel = ?", (channel,)
            ).fetchall()
            self._channels[channel] = ChannelInfo(
                boiler=tx.channel_boilerplate(r["description"] or "" for r in rows),
                template=tx.title_template([r["title"] or "" for r in rows]),
                tags=tx.channel_keywords(_keywords(dict(r)) for r in rows),
                tokens=set(tx.tokens(channel)),
            )
        return self._channels[channel]


@dataclass(frozen=True)
class ChannelInfo:
    boiler: set[str]  # normalized description lines on 2+ of the channel's videos
    template: set[str]  # title words in most of the channel's titles
    tags: set[str]  # tags (casefolded) on 2+ of the channel's videos
    tokens: set[str]  # words of the channel name


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


_PROGRAM_RE = re.compile(
    r"\b(?:at (?:six|five|noon|eleven|\d{1,2})|tonight|nightly|hour|newshour|prime|live|"
    r"full (?:broadcast|episode|show)|newscast|bulletin|evening|morning|update)\b",
    re.I,
)


def is_show_segment(seg: str, show: set[str], is_news: bool) -> bool:
    """A title segment that names the show, not the story: only channel/template words, or
    (in news) a news-program name ("CTV News Vancouver at Six", "ABC World News Tonight with
    David Muir Full Broadcast", or any "… News …" segment with a date)."""
    if not any(t not in show for t in tx.tokens(seg)):
        return True
    if not is_news or not _NEWS_NAME_RE.search(seg):
        return False
    return bool(tx.DATE_RE.search(seg) or _PROGRAM_RE.search(seg))


def clean_title(title: str, show: set[str], is_news: bool = False) -> str:
    """The title without segments that only name the show / channel or a date."""
    segs = tx.title_segments(title, news=is_news)
    keep = [seg for seg in segs if not is_show_segment(seg, show, is_news)]
    if keep:
        return " | ".join(keep)
    return "" if is_news else (title or "")  # a news title that only names the show


_CAP_RUN_RE = re.compile(r"[A-Z][\w.'’-]*(?:\s+[A-Z][\w.'’-]*)*")


def _ngrams(words: list[str]) -> list[str]:
    """All contiguous word spans, longest first."""
    n = len(words)
    return [" ".join(words[i : i + k]) for k in range(n, 0, -1) for i in range(n - k + 1)]


def find_places(tagger: Tagger, title: str, channel: str, show: set[str]) -> list[str]:
    """Places (GPE/LOC) named in a news video's title segments or channel name.

    spaCy reads "CTV News Vancouver" as one ORG and finds nothing in "Vancouver at Six", so
    besides the normal entities, the capitalized runs in show-name segments and in the channel
    name are probed as "in <span>" (spans longest first), where the recognizer reliably says
    whether something is a place. Only GPE counts in a probe, and spans with an all-caps
    acronym (CBC, CTV) or a person's name ("David Muir") are skipped.
    """
    out: list[str] = []

    def have(p: str) -> bool:
        return p.casefold() in {x.casefold() for x in out}

    def add(p: str) -> None:
        p = tx.strip_entity(p)
        if p and not have(p) and p.casefold() not in NOT_PLACES:
            out.append(p)

    for seg in [*tx.title_segments(title, news=True), channel]:
        words = [w for w in seg.split() if re.sub(r"\W", "", w).casefold() not in tx.FILLER]
        text_ = tx.DATE_RE.sub(" ", " ".join(words))
        ents = tagger.entities(text_)
        for e in ents:
            if e.label in PLACE_LABELS:
                add(e.text)
        if seg != channel and not is_show_segment(seg, show, True):
            continue  # in a story segment every sentence-initial word is capitalized
        person = {w for e in ents if e.label == "PERSON" for w in e.text.split()}
        for run in _CAP_RUN_RE.findall(text_):
            used: set[str] = set()
            for span in _ngrams(run.split()):
                ws = span.split()
                if used & set(ws) or person & set(ws) or have(span) or not tx.tokens(span):
                    continue
                if any(len(w) > 1 and w.isupper() for w in ws):
                    continue
                probe = tagger.entities(f"in {span}")
                if len(probe) == 1 and probe[0].label == "GPE" and probe[0].text == span:
                    add(span)
                    used |= set(ws)
    return out


def _pick_lead(
    v: dict[str, Any], ch: ChannelInfo, is_news: bool
) -> tuple[str, str, str, list[str]]:
    """``(lead, source, headline part, secondary lines)``."""
    show = ch.template | ch.tokens
    title = clean_title(v.get("title") or "", show, is_news)
    desc = v.get("description") or ""
    chaps = tx.chapters(desc)
    lines = [
        ln for ln in tx.content_lines(desc, ch.boiler) if not (chaps and tx.is_chapter_line(ln))
    ]

    def specific(s: str, n: int = MIN_SPECIFIC) -> bool:
        return len([t for t in tx.tokens(s) if t not in show]) >= n

    def with_desc() -> tuple[str, str, str, list[str]]:
        for i, ln in enumerate(lines):
            if specific(ln, 3):
                rest = lines[:i] + lines[i + 1 :]
                return (
                    f"{title}. {tx.first_sentence(ln)}",
                    "title+desc",
                    title,
                    rest[:SECONDARY_LINES],
                )
        return title, "title", title, lines[:SECONDARY_LINES]

    if not is_news or specific(title):
        return with_desc()
    for _, ct in chaps:
        if not tx.is_generic_chapter(ct) and specific(ct):
            return ct, "chapter", ct, lines[:SECONDARY_LINES]
    for i, ln in enumerate(lines):
        if specific(ln, 3):
            first = tx.first_sentence(ln)
            return first, "description", first, lines[i + 1 : i + 1 + SECONDARY_LINES]
    # A template-only newscast title with nothing usable behind it: no topic to anchor on.
    return title, "none", "", []


def extract(ctx: TopicContext, video: dict[str, Any] | str) -> Topic | None:
    """The topic of one video (a ``videos`` row or its id); None if it isn't in the database."""
    v = ctx.store.get_video(video) if isinstance(video, str) else video
    if not v:
        return None
    channel = v.get("channel") or ""
    ch = ctx.channel(channel)
    reason = news_context(v)
    is_news = bool(reason)
    # In news the show's own words never count; outside news only the channel name doesn't.
    drop = ch.tokens | (ch.template if is_news else set())
    lead, source, head, secondary = _pick_lead(v, ch, is_news)
    rest = lead[len(head) :] if source == "title+desc" else ""
    tags = [k for k in _keywords(v) if not (is_news and k.strip().casefold() in ch.tags)]

    weights: dict[str, float] = {}
    for part, w in ((head, 1.0), (rest, LEAD_DESC_WEIGHT)):
        for t in tx.tokens(part):
            if t not in drop:
                weights[t] = max(weights.get(t, 0.0), w * ctx.idf(t))
    for t in tx.tokens(" ".join(tags + secondary)):
        if t not in drop and t not in weights:
            weights[t] = SECONDARY_WEIGHT * ctx.idf(t)
    if not is_news:
        for t in ch.template:
            if t not in drop and t not in weights:
                weights[t] = SECONDARY_WEIGHT * ctx.idf(t)

    ents: list[tuple[str, str]] = []
    ent_tokens: set[str] = set()
    seen: set[str] = set()
    for e in ctx.tagger.entities(lead) if source != "none" else []:
        if e.label in SKIP_LABELS:
            continue
        text_ = tx.strip_entity(e.text)
        toks = [t for t in tx.tokens(text_) if t not in drop]
        key = " ".join(toks)
        if not toks or key in seen:
            continue
        seen.add(key)
        ents.append((text_, e.label))
        ent_tokens.update(toks)

    places = (
        find_places(ctx.tagger, v.get("title") or "", channel, ch.template | ch.tokens)
        if is_news
        else []
    )

    phrases: dict[tuple[str, ...], float] = {}
    for p in ctx.tagger.phrases(lead) if source != "none" else []:
        toks = tuple(dict.fromkeys(t for t in tx.tokens(p) if t not in drop))
        if toks and toks not in phrases:  # mean weight × √length: multiword, not just long
            phrases[toks] = sum(weights.get(t, 0.0) for t in toks) / len(toks) ** 0.5
    keyphrases = sorted(((" ".join(k), round(s, 2)) for k, s in phrases.items()), key=_by_score)
    surface: dict[str, str] = {}
    for w in re.findall(r"[^\W_]+(?:['’.-][^\W_]+)*", tx.DATE_RE.sub(" ", lead)):
        surface.setdefault(tx.norm_token(w), w.strip(".'’-"))

    return Topic(
        video_id=v["video_id"],
        title=v.get("title") or "",
        channel=channel,
        lead=lead,
        lead_source=source,
        entities=ents[:8],
        keyphrases=keyphrases[:6],
        weights=weights if source != "none" else {},
        entity_tokens=ent_tokens,
        series=sorted(ch.template),
        places=places[:3],
        surface=surface,
        is_news=is_news,
        news_reason=reason,
        published_at=v.get("published_at"),
        tagger=getattr(ctx.tagger, "name", ""),
    )


def explain_text(ctx: TopicContext, video: dict[str, Any] | str) -> list[tuple[str, str]]:
    """How each title segment, description line and tag of a video was classified, for
    ``just topic --text``: ``(mark, text)`` with mark ``show`` / ``boiler`` / ``pattern`` /
    ``chapter`` / ``channel-tag`` / ``keep``."""
    v = ctx.store.get_video(video) if isinstance(video, str) else video
    if not v:
        return []
    ch = ctx.channel(v.get("channel") or "")
    show = ch.template | ch.tokens
    out: list[tuple[str, str]] = []
    is_news = bool(news_context(v))
    for seg in tx.title_segments(v.get("title") or "", news=is_news):
        is_show = is_show_segment(seg, show, is_news)
        out.append(("title:show" if is_show else "title:keep", seg))
    chaps = tx.chapters(v.get("description") or "")
    for line in (v.get("description") or "").splitlines():
        if not line.strip():
            continue
        if tx.norm_line(line) in ch.boiler:
            mark = "boiler"
        elif chaps and tx.is_chapter_line(line):
            mark = "chapter"
        elif tx.is_pattern_boilerplate(line):
            mark = "pattern"
        else:
            mark = "keep"
        out.append((mark, line.strip()[:150]))
    for k in _keywords(v):
        out.append(("tag:channel" if k.strip().casefold() in ch.tags else "tag:keep", k))
    return out


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
