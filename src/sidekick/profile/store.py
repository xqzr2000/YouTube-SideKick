"""Read ``profile.md`` from the private folder and append 👎 notes to it."""

from __future__ import annotations

import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from sidekick.config import Paths, get_paths
from sidekick.profile.parser import Profile, parse_profile

_DONT_SHOW_HEADING_RE = re.compile(r"^#{1,6}\s+(don'?t|do not)\s+show\s+me\s*#*\s*$", re.I)
_HEADING_RE = re.compile(r"^#{1,6}\s+")


def load_profile(paths: Paths | None = None) -> Profile:
    """Parse the profile. A missing file gives an empty profile with a warning."""
    paths = paths or get_paths()
    if not paths.profile.exists():
        prof = parse_profile("")
        prof.warnings.append(f"no profile at {paths.profile} (run `just init`)")
        return prof
    return parse_profile(paths.profile.read_text(encoding="utf-8"))


def _clean(s: str) -> str:
    return " ".join(s.split())


def dont_show_line(note: str, *, title: str = "", channel: str = "", date: str = "") -> str:
    """The bullet a 👎 appends, e.g. ``- 2026-11-02 — "note" (video: title · channel)``."""
    note = _clean(note)
    if not note:
        raise ValueError("a 👎 needs a note")
    date = date or datetime.now(UTC).date().isoformat()
    video = " · ".join(p for p in (_clean(title), _clean(channel)) if p)
    return f'- {date} — "{note}"' + (f" (video: {video})" if video else "")


def insert_dont_show(text: str, line: str) -> str:
    """Return ``text`` with ``line`` added at the end of the "Don't show me" section
    (creating the section at the end of the file if it's missing)."""
    lines = text.replace("\r\n", "\n").split("\n")
    start = next((i for i, ln in enumerate(lines) if _DONT_SHOW_HEADING_RE.match(ln.strip())), None)
    if start is None:
        body = text.rstrip("\n")
        return (body + "\n\n" if body else "") + "## Don't show me\n" + line + "\n"
    end = next(
        (i for i in range(start + 1, len(lines)) if _HEADING_RE.match(lines[i].strip())), len(lines)
    )
    insert_at = end
    while insert_at > start + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, line)
    out = "\n".join(lines)
    return out if out.endswith("\n") else out + "\n"


def append_dont_show(
    note: str,
    *,
    title: str = "",
    channel: str = "",
    date: str = "",
    paths: Paths | None = None,
) -> str:
    """Append a 👎 note to ``profile.md`` (atomic write). Returns the line written."""
    paths = paths or get_paths()
    line = dont_show_line(note, title=title, channel=channel, date=date)
    path: Path = paths.profile
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    new = insert_dont_show(text, line)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".profile-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(new)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return line
