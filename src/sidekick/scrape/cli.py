"""``just fetch``: fetch a video's watch page (and optional searches) and print the candidates.

Examples::

    just fetch bFZ-0FH5hfs
    just fetch "https://www.youtube.com/watch?v=bFZ-0FH5hfs" --search "bayesian statistics"
    just fetch bFZ-0FH5hfs --check-thumbs

Every page's JSON goes to the raw log (``~/.sidekick/raw/``). No filtering or ranking yet.
"""

from __future__ import annotations

import argparse
import sys
import time

from sidekick.config import ensure_home
from sidekick.models import Candidate
from sidekick.parse.util import parse_age_days
from sidekick.scrape.fetcher import BlockedError, FetchConfig, Fetcher, PageCapReached
from sidekick.scrape.record import video_id_from


def _fmt_duration(s: int | None) -> str:
    if s is None:
        return "-"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _fmt_views(v: int | None) -> str:
    if v is None:
        return "-"
    for div, suf in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if v >= div:
            return f"{v / div:.1f}{suf}"
    return str(v)


def _flags(c: Candidate) -> str:
    return "".join(f for f, on in (("L", c.is_live), ("U", c.is_upcoming), ("S", c.is_short)) if on)


def print_table(title: str, cands: list[Candidate]) -> None:
    print(f"\n{title} ({len(cands)})")
    print(
        f"{'#':>3}  {'video_id':<11}  {'dur':>8}  {'views':>7}  {'age_d':>6}  fl  channel / title"
    )
    for i, c in enumerate(cands, 1):
        age = parse_age_days(c.published_text)
        age_s = f"{age:.0f}" if age is not None else "-"
        print(
            f"{i:>3}  {c.video_id:<11}  {_fmt_duration(c.duration_s):>8}  "
            f"{_fmt_views(c.views):>7}  {age_s:>6}  {_flags(c):<2}  "
            f"{c.channel[:20]} | {c.title[:45]}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch a watch page (+ searches) and list candidates")
    ap.add_argument("video", help="video id or watch URL")
    ap.add_argument("--search", action="append", default=[], help="search query (repeatable)")
    ap.add_argument(
        "--check-thumbs", action="store_true", help="check midpoint thumbnails (first 3 up-next)"
    )
    args = ap.parse_args(argv)

    paths = ensure_home()
    vid = video_id_from(args.video)
    t0 = time.monotonic()
    try:
        with Fetcher(FetchConfig(), paths=paths) as f:
            page = f.watch(vid)
            m = page.meta
            if m:
                print(
                    f"current: {m.video_id}  {m.title[:60]!r}  ({m.channel}, "
                    f"{_fmt_duration(m.duration_s)}, {_fmt_views(m.views)} views, "
                    f"{m.publish_date[:10] or '?'}, {m.category or '?'})"
                )
            else:
                print(f"current: {vid}  (no metadata found)")
            print_table("up-next", page.upnext)
            if page.nonvideo_skipped:
                print(f"     ({page.nonvideo_skipped} non-video items skipped)")
            for q in args.search:
                sp = f.search(q)
                print_table(f"search {q!r}", sp.candidates)
                if sp.shorts_skipped:
                    print(f"     ({sp.shorts_skipped} Shorts items skipped)")
            if args.check_thumbs:
                print("\nmidpoint thumbnails (open the URLs in Chrome to eyeball them):")
                for c in [vid, *[c.video_id for c in page.upnext[:3]]]:
                    t = f.check_thumbnail(c)
                    print(f"  {t.status} {t.content_type:<11} {t.size:>7,} B  {t.url}")
            pages = f.pages_fetched
    except BlockedError as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        return 3
    except PageCapReached as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 4
    print(f"\n{pages} page(s) in {time.monotonic() - t0:.1f}s; raw log in {paths.raw_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
