"""Run the parsers over saved HTML files and print what was (and wasn't) extracted.

Usage: ``just parse-check ~/.sidekick/fixtures/*.html``. Prints no page text beyond short
titles, so the output is safe to paste back when debugging the parser.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from sidekick.models import Candidate
from sidekick.parse.search import parse_search
from sidekick.parse.util import extract_json
from sidekick.parse.watch import parse_watch


def _fields(cands: list[Candidate]) -> str:
    n = len(cands) or 1
    attrs = ("title", "channel", "duration_s", "views", "published_text")
    cov = {a: sum(1 for c in cands if getattr(c, a) not in (None, "")) for a in attrs}
    flags = Counter()
    for c in cands:
        flags["live"] += c.is_live
        flags["upcoming"] += c.is_upcoming
        flags["short"] += c.is_short
    cov_s = ", ".join(f"{a} {v}/{n}" for a, v in cov.items())
    flag_s = ", ".join(f"{k}={v}" for k, v in flags.items())
    return f"{cov_s} | flags: {flag_s}"


def check_file(path: Path) -> list[str]:
    html = path.read_text(encoding="utf-8", errors="replace")
    lines = [f"== {path.name} ({len(html):,} chars)"]
    has_data = "yes" if extract_json(html, "ytInitialData") else "NO"
    has_player = "yes" if extract_json(html, "ytInitialPlayerResponse") else "NO"
    lines.append(f"  json blobs: ytInitialData={has_data}, ytInitialPlayerResponse={has_player}")
    if "consent.youtube.com" in html[:5000] or "Before you continue" in html[:20000]:
        lines.append("  WARNING: looks like a consent page")

    watch = parse_watch(html)
    if watch.meta or watch.upnext or watch.lockups_skipped:
        m = watch.meta
        if m:
            lines.append(
                f"  watch meta: id={m.video_id} title={m.title[:50]!r} duration_s={m.duration_s} "
                f"views={m.views} publish_date={m.publish_date or 'MISSING'} "
                f"category={m.category or 'MISSING'} keywords={len(m.keywords)} "
                f"description_chars={len(m.description)} live={m.is_live} upcoming={m.is_upcoming}"
            )
        else:
            lines.append("  watch meta: MISSING")
        lines.append(
            f"  up-next: {len(watch.upnext)} parsed, {watch.lockups_skipped} lockupViewModel "
            f"items NOT parsed"
        )
        if watch.upnext:
            lines.append(f"  up-next fields: {_fields(watch.upnext)}")

    search = parse_search(html)
    if search.candidates or search.shorts_skipped:
        n, sh = len(search.candidates), search.shorts_skipped
        lines.append(f"  search: {n} videos parsed, {sh} shorts items")
        if search.candidates:
            lines.append(f"  search fields: {_fields(search.candidates)}")
    return lines


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m sidekick.parse.check FILE.html [...]")
        return 2
    for arg in argv:
        print("\n".join(check_file(Path(arg))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
