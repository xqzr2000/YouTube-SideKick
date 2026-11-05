"""Text plumbing for topic extraction: token normalization, channel boilerplate, title
templates and chapters.

Why each piece exists (from the news seeds of 2026-11-05):

- **Channel boilerplate.** News descriptions are mostly the same lines on every video
  ("Subscribe to CTV News…", social links, "Watch more…"). Embedded whole, every newscast looks
  like every other newscast, and like the political Don't-show line. A description line that
  appears on two or more of a channel's videos in our database is boilerplate, plus a short
  list of patterns for channels we've seen only once.
- **Title templates.** "CTV News Vancouver at Six for Tuesday, November 4, 2026" says which
  *show* it is, not which story. Words that recur in most of a channel's titles are the
  template; they are kept as ``series`` (useful for lecture courses), but never as topic.
- **Chapters.** Full newscasts often list their stories as ``3:12 Housing vote…``. The lead
  story is the first chapter that isn't an intro, headlines, weather or sports.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words that never identify a story (on top of English stopwords).
FILLER = frozenset(
    """
    news breaking live update updates full episode episodes video videos watch today tonight
    report reports latest new says said clip clips official channel subscribe exclusive
    interview segment broadcast newscast highlights recap part watch special coverage
    """.split()
) | frozenset(
    # bare month and weekday names (dates with a day number are removed by DATE_RE)
    """
    january february march april june july august september october november december
    monday tuesday wednesday thursday friday saturday sunday
    """.split()
)
STOP = ENGLISH_STOP_WORDS | FILLER

_MONTHS = (
    "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|"
    "july|august|september|october|november|december"
)
_DAYS = "mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|"
_DAYS += "saturday|sunday"
DATE_RE = re.compile(
    rf"\b(?:(?:{_DAYS})\.?,?\s*)?(?:(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?(?:\s+\d{{4}})?"
    rf"|\d{{1,2}}\s+(?:{_MONTHS})\.?(?:\s+\d{{4}})?|\d{{1,4}}[/-]\d{{1,2}}[/-]\d{{1,4}}"
    rf"|(?:{_DAYS})\b|(?:19|20)\d{{2}})\b",
    re.I,
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+|\b\S+\.(?:com|ca|org|net|tv|ly|news)/\S*", re.I)
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’.-][^\W_]+)*")
_WS_RE = re.compile(r"\s+")

# Lines that are boilerplate wherever they appear (for channels seen only once).
_BOILER_RES = [
    re.compile(p, re.I)
    for p in (
        r"\bsubscribe\b",
        r"\bfollow us\b",
        r"\bfollow .{0,30}\bon (?:x|twitter|facebook|instagram|tiktok|threads|bluesky)\b",
        r"\b(?:facebook|instagram|twitter|tiktok|threads|bluesky|linkedin|x)\s*[:|]",
        r"\blike (?:and|&) (?:share|subscribe)\b",
        r"\bdownload the .{0,30}app\b",
        r"\bsign up for\b",
        r"\bnewsletter\b",
        r"\bwatch (?:more|the full|live|all)\b",
        r"\bfor more (?:news|stories|videos|on this story)\b",
        r"\bvisit (?:us|our)\b",
        r"\bstream .{0,30}\b(?:live|free)\b",
        r"©|\bcopyright\b|\ball rights reserved\b",
        r"\bsupport (?:us|our|the channel)\b",
        r"\bpatreon\b",
        r"\bbecome a member\b",
        r"\bhave a (?:story|tip|news tip)\b",
    )
]
_HASHTAG_LINE_RE = re.compile(r"^(?:\s*#\w+[\s,]*)+$")

# Chapters: "3:12 Title", "(03:12) - Title", "[1:02:03] Title", or "Title - 3:12".
_TS = r"(?:\d{1,2}:)?\d{1,2}:\d{2}"
_CHAPTER_LEAD_RE = re.compile(rf"^\s*[\(\[]?(?P<ts>{_TS})[\)\]]?\s*[-–—:|.]?\s*(?P<title>.+?)\s*$")
_CHAPTER_TRAIL_RE = re.compile(rf"^\s*(?P<title>.+?)\s*[-–—:|]?\s*[\(\[]?(?P<ts>{_TS})[\)\]]?\s*$")
GENERIC_CHAPTERS = frozenset(
    """
    intro introduction opening open cold start welcome headlines headline top stories preview
    coming up teaser weather forecast sports traffic outro end ending credits recap summary
    good evening good morning tonight on the show programming note sign off closing
    """.split()
)


def norm_token(w: str) -> str:
    """Casefold, drop possessives, fold simple plurals ("chemometrics" ~ "chemometric")."""
    t = w.casefold().replace("’", "'").strip(".'-")
    t = t.removesuffix("'s")
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def tokens(text: str) -> list[str]:
    """Normalized content tokens: no stopwords, no filler, no dates, no bare numbers."""
    text = DATE_RE.sub(" ", _URL_RE.sub(" ", text or ""))
    out = []
    for w in _TOKEN_RE.findall(text):
        if w.casefold() in STOP:
            continue
        t = norm_token(w)
        if len(t) < 2 or t.isdigit() or t in STOP:
            continue
        out.append(t)
    return out


def norm_line(line: str) -> str:
    """A description line in comparable form: no URLs, digits folded, whitespace collapsed."""
    t = _URL_RE.sub(" ", line).casefold()
    t = re.sub(r"\d", "0", t)
    return _WS_RE.sub(" ", t).strip(" -–—|•·*")


def is_pattern_boilerplate(line: str) -> bool:
    raw = line.strip()
    if not raw:
        return True
    if _HASHTAG_LINE_RE.match(raw):
        return True
    if any(rx.search(raw) for rx in _BOILER_RES):
        return True
    return len(tokens(raw)) < 2  # link-only lines, "—", "Credits:" and the like


def channel_boilerplate(descriptions: Iterable[str], min_videos: int = 2) -> set[str]:
    """Normalized lines that appear in at least ``min_videos`` of a channel's descriptions."""
    counts: Counter[str] = Counter()
    for d in descriptions:
        counts.update({n for n in (norm_line(x) for x in (d or "").splitlines()) if n})
    return {line for line, c in counts.items() if c >= min_videos}


def content_lines(description: str, boiler: set[str]) -> list[str]:
    """Description lines that are neither channel boilerplate nor pattern boilerplate."""
    out = []
    for line in (description or "").splitlines():
        n = norm_line(line)
        if not n or n in boiler or is_pattern_boilerplate(line):
            continue
        out.append(_WS_RE.sub(" ", _URL_RE.sub(" ", line)).strip())
    return out


def title_template(titles: Sequence[str], min_share: float = 0.5) -> set[str]:
    """Tokens in at least ``min_share`` of a channel's titles (needs 3+ titles, 2+ hits)."""
    if len(titles) < 3:
        return set()
    df: Counter[str] = Counter()
    for t in titles:
        df.update(set(tokens(t)))
    need = max(2, int(min_share * len(titles) + 0.999))
    return {w for w, c in df.items() if c >= need}


def chapters(description: str) -> list[tuple[str, str]]:
    """``[(timestamp, title)]`` from description lines; empty unless there are 2+ chapters."""
    out = []
    for line in (description or "").splitlines():
        s = line.strip()
        if not s or len(s) > 160:
            continue
        m = _CHAPTER_LEAD_RE.match(s) or _CHAPTER_TRAIL_RE.match(s)
        if m and m.group("title") and not re.fullmatch(rf"[\s\W]*{_TS}[\s\W]*", s):
            out.append((m.group("ts"), m.group("title").strip(" -–—|:")))
    return out if len(out) >= 2 else []


def is_chapter_line(line: str) -> bool:
    s = line.strip()
    return (
        bool(s) and len(s) <= 160 and bool(_CHAPTER_LEAD_RE.match(s) or _CHAPTER_TRAIL_RE.match(s))
    )


def is_generic_chapter(title: str) -> bool:
    words = [w.casefold() for w in _TOKEN_RE.findall(DATE_RE.sub(" ", title))]
    content = [w for w in words if w not in GENERIC_CHAPTERS and w not in STOP]
    return len(content) < 2


def first_sentence(text: str, max_chars: int = 300) -> str:
    """The first sentence of ``text`` (or its first ``max_chars``)."""
    t = _WS_RE.sub(" ", text or "").strip()
    m = re.search(r"(?<=[.!?])\s+(?=[A-Z\"“])", t)
    s = t[: m.start()] if m else t
    return s[:max_chars].strip()
