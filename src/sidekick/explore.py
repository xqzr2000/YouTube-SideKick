"""Load candidate pools for exploration (NB01) from whatever data is available.

Sources, in ``auto`` order:

1. ``raw``: the raw log (``~/.sidekick/raw/*.jsonl.gz``) written by the fetcher, replayed
   through the current parser.
2. ``fixtures``: pages saved with ``just record`` (``~/.sidekick/fixtures/*.html``).
3. ``sample``: the synthetic sample in ``data/sample/pages.jsonl.gz`` (same format as the raw
   log), so the notebook and CI run without anyone's private data.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sidekick.config import Paths, get_paths
from sidekick.parse.search import parse_search, parse_search_data
from sidekick.parse.util import parse_age_days
from sidekick.parse.watch import parse_watch, parse_watch_data
from sidekick.store import rawlog

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample" / "pages.jsonl.gz"
SOURCES = ("auto", "raw", "fixtures", "sample")


def _rows_for_page(kind: str, key: str, ts: str, page: Any) -> tuple[list[dict], dict | None]:
    cands = page.upnext if kind == "watch" else page.candidates
    rows = []
    for rank, c in enumerate(cands, 1):
        row = asdict(c)
        row.update(
            page_kind=kind,
            page_key=key,
            page_ts=ts,
            rank=rank,
            age_days=parse_age_days(c.published_text),
        )
        rows.append(row)
    seed = None
    if kind == "watch" and page.meta:
        seed = asdict(page.meta) | {"page_ts": ts}
    return rows, seed


def _records_from_rawlog(files: list[Path]) -> Iterator[dict]:
    for f in files:
        yield from rawlog.read(f)


def _load_records(records: Iterator[dict]) -> tuple[list[dict], list[dict], int]:
    rows, seeds, pages = [], [], 0
    for rec in records:
        kind, data = rec.get("kind"), rec.get("data") or {}
        if kind == "watch":
            page = parse_watch_data(data.get("player"), data.get("data"))
        elif kind == "search":
            page = parse_search_data(data.get("data"))
        else:
            continue
        pages += 1
        r, s = _rows_for_page(kind, rec.get("key", ""), rec.get("ts", ""), page)
        rows += r
        if s:
            seeds.append(s)
    return rows, seeds, pages


def _load_fixtures(files: list[Path]) -> tuple[list[dict], list[dict], int]:
    rows, seeds = [], []
    for f in files:
        html = f.read_text(encoding="utf-8", errors="replace")
        kind = "search" if f.name.startswith("search_") else "watch"
        page = parse_search(html) if kind == "search" else parse_watch(html)
        key = f.stem.split("_", 1)[-1]
        r, s = _rows_for_page(kind, key, "", page)
        rows += r
        if s:
            seeds.append(s)
    return rows, seeds, len(files)


def load_candidates(
    source: str = "auto", *, paths: Paths | None = None, sample_path: Path = SAMPLE_PATH
) -> dict[str, Any]:
    """Return ``{"source", "pages", "candidates": [row...], "seeds": [meta...]}``.

    Each candidate row has the ``Candidate`` fields plus ``page_kind`` (watch/search),
    ``page_key`` (video id or query), ``page_ts``, ``rank`` and ``age_days``.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    paths = paths or get_paths()
    raw_files = sorted(paths.raw_dir.glob("*.jsonl.gz")) if paths.raw_dir.is_dir() else []
    fixture_dir = paths.home / "fixtures"
    fixture_files = sorted(fixture_dir.glob("*.html")) if fixture_dir.is_dir() else []

    if source == "auto":
        source = "raw" if raw_files else "fixtures" if fixture_files else "sample"

    if source == "raw":
        rows, seeds, pages = _load_records(_records_from_rawlog(raw_files))
    elif source == "fixtures":
        rows, seeds, pages = _load_fixtures(fixture_files)
    else:
        rows, seeds, pages = _load_records(rawlog.read(sample_path))
    return {"source": source, "pages": pages, "candidates": rows, "seeds": seeds}
