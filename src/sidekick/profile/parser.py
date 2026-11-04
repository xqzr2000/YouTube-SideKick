"""Parse ``profile.md``: the private, hand-edited file that says what you want and don't want.

The format is light-structure markdown with three sections::

    ## Interests
    - reinforcement learning, Bayesian statistics, chemometrics

    ## Rules
    - language: en
    - min_duration: 3m
    - max_duration: 90m
    - exclude_channels: Some Channel, Another Channel
    - exclude_keywords: reaction, prank

    ## Don't show me
    - reaction videos
    - 2026-11-02 — "clickbait title, no content" (video: <title> · <channel>)

The parser is lenient: a hand-edited file should never crash a compute. Anything it can't
read becomes a warning (shown by ``just profile`` and kept with the parsed result) instead of
an exception.

Every distinct profile text gets a hash (``Profile.hash``). The hash ignores line-ending and
trailing-whitespace changes, so only real edits create a new profile version.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from sidekick.parse.util import parse_duration

SECTION_INTERESTS = "interests"
SECTION_RULES = "rules"
SECTION_DONT_SHOW = "don't show me"
_SECTION_ALIASES = {
    "interests": SECTION_INTERESTS,
    "rules": SECTION_RULES,
    "don't show me": SECTION_DONT_SHOW,
    "dont show me": SECTION_DONT_SHOW,
    "do not show me": SECTION_DONT_SHOW,
}
RULE_KEYS = ("language", "min_duration", "max_duration", "exclude_channels", "exclude_keywords")

_HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*?)\s*$")
_RULE_RE = re.compile(r"^([A-Za-z_ ]+?)\s*:\s*(.*)$")
# - 2026-11-02 — "note" (video: <title> · <channel>)   (dash may be — – or -)
_DATED_RE = re.compile(
    r'^(\d{4}-\d{2}-\d{2})\s*[—–-]\s*"(.*)"\s*(?:\(video:\s*(.*)\))?\s*$', re.DOTALL
)
_DURATION_PART_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hours?|m|min|mins|minutes?|s|sec|secs|seconds?)", re.I
)
_DURATION_FULL_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)?\s*(?:h|hr|hrs|hours?|m|min|mins|minutes?|s|sec|secs|seconds?)\s*)+$",
    re.I,
)
_UNIT_SECONDS = {"h": 3600, "m": 60, "s": 1}


@dataclass
class Rules:
    """Hard filters. Empty lists / None mean "no rule"."""

    languages: list[str] = field(default_factory=list)  # ISO 639-1 codes, lowercase
    min_duration_s: int | None = None
    max_duration_s: int | None = None
    exclude_channels: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)


@dataclass
class DontShow:
    """One "Don't show me" line. ``text`` is what gets compared against videos."""

    text: str
    date: str = ""  # ISO date, set on lines appended by a 👎
    video: str = ""  # "<title> · <channel>" context, set on lines appended by a 👎
    line: str = ""  # the bullet as written


@dataclass
class Profile:
    interests: list[str] = field(default_factory=list)
    rules: Rules = field(default_factory=Rules)
    dont_show: list[DontShow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    text: str = ""
    hash: str = ""


def normalize(text: str) -> str:
    """Canonical form used for hashing: LF line endings, no trailing whitespace, one final LF."""
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip("\n") + "\n"


def profile_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def parse_duration_rule(value: str) -> int | None:
    """``"3m"`` -> 180, ``"1h30m"`` -> 5400, ``"45s"`` -> 45, ``"1:30:00"`` -> 5400,
    bare ``"90"`` -> 5400 (minutes). None if unreadable."""
    v = value.strip()
    if not v:
        return None
    if v.replace(".", "", 1).isdigit():
        return int(float(v) * 60)
    if ":" in v:
        return parse_duration(v)
    if not _DURATION_FULL_RE.match(v):
        return None
    total = 0.0
    for num, unit in _DURATION_PART_RE.findall(v):
        total += float(num) * _UNIT_SECONDS[unit[0].lower()]
    return int(total)


def _split_list(value: str) -> list[str]:
    return [p.strip() for p in value.split(",") if p.strip()]


def _parse_rule(body: str, rules: Rules, warnings: list[str], lineno: int) -> None:
    m = _RULE_RE.match(body)
    if not m:
        warnings.append(f"line {lineno}: rule is not 'key: value': {body!r}")
        return
    key = m.group(1).strip().lower().replace(" ", "_")
    value = m.group(2).strip()
    if key in ("language", "languages"):
        rules.languages = [p.lower() for p in _split_list(value)]
    elif key in ("min_duration", "max_duration"):
        if not value:
            setattr(rules, f"{key}_s", None)
            return
        seconds = parse_duration_rule(value)
        if seconds is None:
            warnings.append(f"line {lineno}: can't read {key} {value!r} (try 3m, 1h30m, 45s)")
        else:
            setattr(rules, f"{key}_s", seconds)
    elif key in ("exclude_channels", "exclude_channel"):
        rules.exclude_channels = _split_list(value)
    elif key in ("exclude_keywords", "exclude_keyword"):
        rules.exclude_keywords = _split_list(value)
    else:
        warnings.append(f"line {lineno}: unknown rule {key!r} (known: {', '.join(RULE_KEYS)})")


def _parse_dont_show(body: str) -> DontShow:
    m = _DATED_RE.match(body)
    if m:
        return DontShow(
            text=m.group(2).strip(), date=m.group(1), video=(m.group(3) or "").strip(), line=body
        )
    return DontShow(text=body, line=body)


def parse_profile(text: str) -> Profile:
    prof = Profile(text=text, hash=profile_hash(text))
    section: str | None = None
    for lineno, raw in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
        line = raw.strip()
        if not line or line.startswith("<!--"):
            continue
        h = _HEADING_RE.match(line)
        if h:
            name = h.group(1).replace("’", "'").strip().lower()
            section = _SECTION_ALIASES.get(name)
            if section is None:
                prof.warnings.append(f"line {lineno}: unknown section {h.group(1)!r} (ignored)")
            continue
        b = _BULLET_RE.match(raw)
        if not b:
            if section is not None:
                prof.warnings.append(f"line {lineno}: not a '- ' bullet (ignored): {line!r}")
            continue
        body = b.group(1)
        if not body or section is None:
            continue
        if section == SECTION_INTERESTS:
            prof.interests.extend(_split_list(body))
        elif section == SECTION_RULES:
            _parse_rule(body, prof.rules, prof.warnings, lineno)
        else:
            prof.dont_show.append(_parse_dont_show(body))
    r = prof.rules
    if r.min_duration_s and r.max_duration_s and r.min_duration_s > r.max_duration_s:
        prof.warnings.append(
            "min_duration is longer than max_duration: every video will be dropped"
        )
    return prof
