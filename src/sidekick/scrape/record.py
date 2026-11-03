"""Save raw YouTube pages to ``~/.sidekick/fixtures/`` for parser development.

Usage::

    just record dQw4w9WgXcQ <another id or watch URL> ...
    just record --search "bayesian statistics lecture"

Pages are saved only to the private folder (never the repo). One request at a time with a
1-2 s jittered delay. A proper polite fetcher with page caps comes in the next PR.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from urllib.parse import quote_plus

import httpx

from sidekick.config import ensure_home

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
ID_RE = re.compile(r"(?:v=|youtu\.be/|^)([A-Za-z0-9_-]{11})(?:[&?#].*)?$")


def video_id_from(arg: str) -> str:
    m = ID_RE.search(arg.strip())
    if not m:
        raise SystemExit(f"cannot find an 11-character video id in {arg!r}")
    return m.group(1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="*", help="video ids or watch URLs")
    ap.add_argument("--search", action="append", default=[], help="search query (repeatable)")
    args = ap.parse_args(argv)
    if not args.videos and not args.search:
        ap.error("give at least one video or --search")

    paths = ensure_home()
    out_dir = paths.home / "fixtures"
    out_dir.mkdir(exist_ok=True)

    jobs: list[tuple[str, str]] = []
    for v in args.videos:
        vid = video_id_from(v)
        jobs.append((f"https://www.youtube.com/watch?v={vid}", f"watch_{vid}.html"))
    for q in args.search:
        slug = re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")[:40]
        jobs.append(
            (f"https://www.youtube.com/results?search_query={quote_plus(q)}", f"search_{slug}.html")
        )

    headers = {
        "User-Agent": os.environ.get("SIDEKICK_USER_AGENT", DEFAULT_UA),
        "Accept-Language": "en-US,en;q=0.9",
    }
    with httpx.Client(headers=headers, follow_redirects=True, timeout=20) as client:
        for i, (url, name) in enumerate(jobs):
            if i:
                time.sleep(random.uniform(1.0, 2.0))
            r = client.get(url)
            consent = "consent." in str(r.url)
            (out_dir / name).write_text(r.text, encoding="utf-8")
            note = "  <-- CONSENT PAGE" if consent else ""
            print(f"{r.status_code} {len(r.text):>9,} chars  {name}{note}")
    print(f"saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
