"""Raw scrape log: ``raw/YYYY-MM-DD.jsonl.gz``, one JSON object per fetched page.

Each line stores the parsed-from-HTML JSON blobs (not the whole HTML) so the pipeline can be
replayed when the parser changes. Kept 90 days by default.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sidekick.config import Paths, get_paths

_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.jsonl\.gz$")


def append(
    kind: str,
    key: str,
    data: dict[str, Any],
    *,
    paths: Paths | None = None,
    ts: datetime | None = None,
) -> Path:
    """Append one record. ``kind`` is e.g. ``watch`` or ``search``; ``key`` is the id or query."""
    paths = paths or get_paths()
    ts = ts or datetime.now(UTC)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    path = paths.raw_dir / f"{ts.date().isoformat()}.jsonl.gz"
    record = {"ts": ts.isoformat(), "kind": kind, "key": key, "data": data}
    # Each append is its own gzip member; gzip readers concatenate members transparently.
    with gzip.open(path, "at", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prune(
    *, days: int = 90, paths: Paths | None = None, today: datetime | None = None
) -> list[Path]:
    """Delete raw log files older than ``days``. Returns the deleted paths."""
    paths = paths or get_paths()
    if not paths.raw_dir.is_dir():
        return []
    cutoff = (today or datetime.now(UTC)).date() - timedelta(days=days)
    deleted = []
    for p in sorted(paths.raw_dir.iterdir()):
        m = _FILE_RE.match(p.name)
        if m and datetime.fromisoformat(m.group(1)).date() < cutoff:
            p.unlink()
            deleted.append(p)
    return deleted
