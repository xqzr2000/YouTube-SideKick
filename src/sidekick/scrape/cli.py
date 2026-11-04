"""``just fetch``: fetch a video's watch page (and optional searches) and print the candidates.

Examples::

    just fetch bFZ-0FH5hfs
    just fetch "https://www.youtube.com/watch?v=bFZ-0FH5hfs" --search "bayesian statistics"
    just fetch bFZ-0FH5hfs --check-thumbs

Every page's JSON goes to the raw log (``~/.sidekick/raw/``), and the run is logged to the
SQLite store as a compute with ``trigger=cli`` (videos, view snapshots, queries, candidates).
``--no-db`` skips the database. No filtering or ranking yet.
"""

from __future__ import annotations

import argparse
import sys
import time

from sidekick.config import ensure_home
from sidekick.models import Candidate
from sidekick.parse.util import parse_age_days
from sidekick.profile import load_profile
from sidekick.scrape.fetcher import BlockedError, FetchConfig, Fetcher, PageCapReached
from sidekick.scrape.record import video_id_from
from sidekick.store.db import Store


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
    ap.add_argument("--no-db", action="store_true", help="don't log this run to sidekick.db")
    args = ap.parse_args(argv)

    paths = ensure_home()
    vid = video_id_from(args.video)
    store = None if args.no_db else Store.open(paths)
    cid = None
    if store:
        profile = load_profile(paths)
        cid = store.start_compute(
            vid, trigger="cli", profile_hash=store.save_profile_version(profile)
        )
    t0 = time.monotonic()
    pages = 0

    def finish(status: str, error: str | None = None) -> None:
        if store and cid:
            store.finish_compute(
                cid,
                status=status,
                rounds_used=1 if args.search else 0,
                pages_fetched=pages,
                duration_ms=int((time.monotonic() - t0) * 1000),
                error=error,
            )
            store.close()

    f = None
    try:
        with Fetcher(FetchConfig(), paths=paths) as f:
            page = f.watch(vid)
            m = page.meta
            if store:
                if m:
                    store.upsert_meta(m)
                store.log_candidates(cid, page.upnext, source="upnext", round=0)
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
                if store:
                    store.log_query(
                        cid, round=1, query_text=q, method="manual", n_results=len(sp.candidates)
                    )
                    store.log_candidates(cid, sp.candidates, source="query", round=1, query=q)
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
        pages = f.pages_fetched if f else pages
        finish("blocked", str(e))
        print(f"BLOCKED: {e}", file=sys.stderr)
        return 3
    except PageCapReached as e:
        pages = f.pages_fetched if f else pages
        finish("error", str(e))
        print(f"STOPPED: {e}", file=sys.stderr)
        return 4
    except BaseException as e:
        pages = f.pages_fetched if f else pages
        finish("error", repr(e))
        raise
    finish("ok")
    print(f"\n{pages} page(s) in {time.monotonic() - t0:.1f}s; raw log in {paths.raw_dir}")
    if cid:
        print(f"logged as compute {cid} in {paths.db} (`just db` to inspect)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
