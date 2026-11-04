"""SQLite store: ``~/.sidekick/sidekick.db``. Everything a compute sees and decides is logged here.

Tables (spec §7): ``videos``, ``video_snapshots``, ``embeddings``, ``computes``, ``queries``,
``candidates``, ``decisions``, ``impressions``, ``feedback``, ``labels``, ``profile_versions``.

Conventions:

- Timestamps are ISO-8601 UTC strings; lists and feature dicts are JSON text.
- ``compute_id`` is text like ``20261103T203000Z-a1b2c3``: sortable by time and unique across
  machines, so the WSL2 and Crostini databases can be combined later without id clashes.
- The schema version lives in ``PRAGMA user_version``; ``Store`` creates or migrates on open.
- Analysis reads this file directly (DuckDB's sqlite scanner, or pandas); there is no ORM.

``python -m sidekick.store.db`` (``just db``) prints row counts and the latest computes.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sidekick.config import Paths, get_paths
from sidekick.models import Candidate, VideoMeta
from sidekick.profile.parser import Profile, normalize

SCHEMA_VERSION = 3

TABLES = (
    "videos",
    "video_snapshots",
    "embeddings",
    "computes",
    "queries",
    "candidates",
    "decisions",
    "impressions",
    "feedback",
    "labels",
    "profile_versions",
)

_SCHEMA_V1 = """
CREATE TABLE videos (
    video_id        TEXT PRIMARY KEY,
    title           TEXT,
    channel         TEXT,
    channel_id      TEXT,
    duration_s      INTEGER,
    published_at    TEXT,            -- exact, from the watch page (enrichment)
    published_text  TEXT,            -- relative, e.g. "3 years ago", from up-next / search
    description     TEXT,
    keywords        TEXT,            -- JSON list
    category        TEXT,
    lang            TEXT,
    lang_conf       REAL,
    is_live         INTEGER NOT NULL DEFAULT 0,
    is_upcoming     INTEGER NOT NULL DEFAULT 0,
    is_short        INTEGER NOT NULL DEFAULT 0,
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    last_enriched   TEXT
);
CREATE INDEX idx_videos_channel ON videos(channel);

CREATE TABLE video_snapshots (
    video_id  TEXT NOT NULL,
    ts        TEXT NOT NULL,
    views     INTEGER,
    likes     INTEGER,
    source    TEXT NOT NULL            -- watch (exact) | upnext | query | snowball (rounded)
);
CREATE INDEX idx_snapshots_video ON video_snapshots(video_id, ts);

CREATE TABLE embeddings (
    video_id    TEXT NOT NULL,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,         -- float32 little-endian
    created_at  TEXT NOT NULL,
    PRIMARY KEY (video_id, model)
);

CREATE TABLE computes (
    compute_id        TEXT PRIMARY KEY,
    ts                TEXT NOT NULL,
    current_video_id  TEXT,
    trigger           TEXT NOT NULL,   -- open | refresh | cli
    mode              TEXT,            -- interest | filter-only (set from M3)
    rounds_used       INTEGER,
    pages_fetched     INTEGER,
    duration_ms       INTEGER,
    profile_hash      TEXT,
    status            TEXT NOT NULL DEFAULT 'running',  -- running | ok | blocked | error
    error             TEXT
);
CREATE INDEX idx_computes_ts ON computes(ts);

CREATE TABLE queries (
    compute_id  TEXT NOT NULL REFERENCES computes(compute_id),
    round       INTEGER NOT NULL,
    query_text  TEXT NOT NULL,
    method      TEXT NOT NULL,         -- manual | keyphrase | interest | rocchio
    n_results   INTEGER
);
CREATE INDEX idx_queries_compute ON queries(compute_id);

CREATE TABLE candidates (
    compute_id  TEXT NOT NULL REFERENCES computes(compute_id),
    video_id    TEXT NOT NULL,
    source      TEXT NOT NULL,         -- upnext | query | snowball
    round       INTEGER NOT NULL,      -- 0 = up-next of the current video
    query       TEXT,
    position    INTEGER NOT NULL       -- 1-based position in its list
);
CREATE INDEX idx_candidates_compute ON candidates(compute_id);
CREATE INDEX idx_candidates_video ON candidates(video_id);

CREATE TABLE decisions (
    compute_id  TEXT NOT NULL REFERENCES computes(compute_id),
    video_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,         -- A | B | C | rank
    kept        INTEGER NOT NULL,
    reason      TEXT,
    score       REAL,
    features    TEXT                   -- JSON object
);
CREATE INDEX idx_decisions_compute ON decisions(compute_id);
CREATE INDEX idx_decisions_video ON decisions(video_id);

CREATE TABLE impressions (
    compute_id  TEXT NOT NULL REFERENCES computes(compute_id),
    video_id    TEXT NOT NULL,
    rank        INTEGER NOT NULL,
    PRIMARY KEY (compute_id, rank)
);

CREATE TABLE feedback (
    feedback_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    compute_id   TEXT REFERENCES computes(compute_id),
    video_id     TEXT NOT NULL,
    type         TEXT NOT NULL CHECK (type IN ('click', 'downvote')),
    note         TEXT
);
CREATE INDEX idx_feedback_video ON feedback(video_id);

CREATE TABLE labels (
    label_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    video_id    TEXT NOT NULL,
    compute_id  TEXT REFERENCES computes(compute_id),
    label       TEXT NOT NULL CHECK (label IN ('good', 'meh', 'bad')),
    reason      TEXT,
    session     TEXT
);
CREATE INDEX idx_labels_video ON labels(video_id);

CREATE TABLE profile_versions (
    hash     TEXT PRIMARY KEY,
    ts       TEXT NOT NULL,           -- first time this version was seen
    content  TEXT NOT NULL
);
"""

# v2 (M2c): SponsorBlock results cached on the video row.
_SCHEMA_V2 = """
ALTER TABLE videos ADD COLUMN sponsor_frac REAL;
ALTER TABLE videos ADD COLUMN selfpromo_frac REAL;
ALTER TABLE videos ADD COLUMN sponsor_segments INTEGER;
ALTER TABLE videos ADD COLUMN sponsor_checked_at TEXT;
"""

# v3 (M3a): labels say what they label. ``candidate`` = "is this a good video to show next
# while watching the compute's current video?" (good/meh/bad). ``seed`` = "does this current
# video match my Interests?" (good = yes, meh = partly, bad = no), the ground truth for the
# interest gate (experiment E2).
_SCHEMA_V3 = """
ALTER TABLE labels ADD COLUMN kind TEXT NOT NULL DEFAULT 'candidate'
    CHECK (kind IN ('candidate', 'seed'));
"""

_MIGRATIONS = {1: _SCHEMA_V1, 2: _SCHEMA_V2, 3: _SCHEMA_V3}


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_compute_id(ts: datetime | None = None) -> str:
    ts = ts or datetime.now(UTC)
    return f"{ts.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


class SchemaTooNew(RuntimeError):
    """The database was written by a newer SideKick than this one."""


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    @classmethod
    def open(cls, paths: Paths | None = None) -> Store:
        return cls((paths or get_paths()).db)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- schema ------------------------------------------------------------------------------

    @property
    def schema_version(self) -> int:
        return self.conn.execute("PRAGMA user_version").fetchone()[0]

    def _migrate(self) -> None:
        current = self.schema_version
        if current > SCHEMA_VERSION:
            raise SchemaTooNew(
                f"{self.path} has schema v{current}; this code knows v{SCHEMA_VERSION}. git pull?"
            )
        for v in range(current + 1, SCHEMA_VERSION + 1):
            with self.conn:
                self.conn.executescript(_MIGRATIONS[v])
                self.conn.execute(f"PRAGMA user_version = {v}")

    # -- videos ------------------------------------------------------------------------------

    def upsert_candidate(self, c: Candidate, *, ts: str | None = None) -> None:
        """Cheap fields from up-next / search. Never overwrites enriched exact fields."""
        ts = ts or now_iso()
        self.conn.execute(
            """
            INSERT INTO videos (video_id, title, channel, duration_s, published_text,
                                is_live, is_upcoming, is_short, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                title          = COALESCE(NULLIF(excluded.title, ''), title),
                channel        = COALESCE(NULLIF(excluded.channel, ''), channel),
                duration_s     = COALESCE(duration_s, excluded.duration_s),
                published_text = COALESCE(NULLIF(excluded.published_text, ''), published_text),
                is_live        = excluded.is_live,
                is_upcoming    = excluded.is_upcoming,
                is_short       = MAX(is_short, excluded.is_short),
                last_seen      = excluded.last_seen
            """,
            (
                c.video_id,
                c.title,
                c.channel,
                c.duration_s,
                c.published_text,
                int(c.is_live),
                int(c.is_upcoming),
                int(c.is_short),
                ts,
                ts,
            ),
        )

    def upsert_meta(self, m: VideoMeta, *, ts: str | None = None) -> None:
        """Exact fields from a watch page (enrichment). Non-empty values win."""
        ts = ts or now_iso()
        self.conn.execute(
            """
            INSERT INTO videos (video_id, title, channel, channel_id, duration_s, published_at,
                                description, keywords, category, is_live, is_upcoming,
                                first_seen, last_seen, last_enriched)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                title         = COALESCE(NULLIF(excluded.title, ''), title),
                channel       = COALESCE(NULLIF(excluded.channel, ''), channel),
                channel_id    = COALESCE(NULLIF(excluded.channel_id, ''), channel_id),
                duration_s    = COALESCE(excluded.duration_s, duration_s),
                published_at  = COALESCE(NULLIF(excluded.published_at, ''), published_at),
                description   = COALESCE(NULLIF(excluded.description, ''), description),
                keywords      = COALESCE(NULLIF(excluded.keywords, '[]'), keywords),
                category      = COALESCE(NULLIF(excluded.category, ''), category),
                is_live       = excluded.is_live,
                is_upcoming   = excluded.is_upcoming,
                last_seen     = excluded.last_seen,
                last_enriched = excluded.last_enriched
            """,
            (
                m.video_id,
                m.title,
                m.channel,
                m.channel_id,
                m.duration_s,
                m.publish_date,
                m.description,
                json.dumps(m.keywords, ensure_ascii=False),
                m.category,
                int(m.is_live),
                int(m.is_upcoming),
                ts,
                ts,
                ts,
            ),
        )
        if m.views is not None:
            self.add_snapshot(m.video_id, m.views, source="watch", ts=ts)

    def set_language(self, video_id: str, lang: str, conf: float | None) -> None:
        self.conn.execute(
            "UPDATE videos SET lang = ?, lang_conf = ? WHERE video_id = ?", (lang, conf, video_id)
        )

    def set_sponsor(
        self,
        video_id: str,
        *,
        sponsor_frac: float,
        selfpromo_frac: float,
        n_segments: int,
        ts: str | None = None,
    ) -> None:
        self.conn.execute(
            """UPDATE videos SET sponsor_frac = ?, selfpromo_frac = ?, sponsor_segments = ?,
                   sponsor_checked_at = ? WHERE video_id = ?""",
            (sponsor_frac, selfpromo_frac, n_segments, ts or now_iso(), video_id),
        )

    def get_meta(self, video_id: str, *, max_age: timedelta | None = None) -> VideoMeta | None:
        """The enriched metadata for a video, if it was enriched (within ``max_age``)."""
        v = self.get_video(video_id)
        if not v or not v["last_enriched"]:
            return None
        if max_age is not None:
            age = datetime.now(UTC) - datetime.fromisoformat(v["last_enriched"])
            if age > max_age:
                return None
        views = self.conn.execute(
            "SELECT views FROM video_snapshots WHERE video_id = ? AND source = 'watch' "
            "ORDER BY ts DESC LIMIT 1",
            (video_id,),
        ).fetchone()
        return VideoMeta(
            video_id=video_id,
            title=v["title"] or "",
            channel=v["channel"] or "",
            channel_id=v["channel_id"] or "",
            description=v["description"] or "",
            keywords=json.loads(v["keywords"] or "[]"),
            duration_s=v["duration_s"],
            views=views[0] if views else None,
            publish_date=v["published_at"] or "",
            category=v["category"] or "",
            is_live=bool(v["is_live"]),
            is_upcoming=bool(v["is_upcoming"]),
        )

    def add_snapshot(
        self,
        video_id: str,
        views: int | None,
        *,
        likes: int | None = None,
        source: str,
        ts: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO video_snapshots (video_id, ts, views, likes, source) VALUES (?,?,?,?,?)",
            (video_id, ts or now_iso(), views, likes, source),
        )

    def get_video(self, video_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
        return dict(row) if row else None

    # -- profile -----------------------------------------------------------------------------

    def save_profile_version(self, profile: Profile, *, ts: str | None = None) -> str:
        """Store this profile text if its hash is new. Returns the hash."""
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO profile_versions (hash, ts, content) VALUES (?, ?, ?)",
                (profile.hash, ts or now_iso(), normalize(profile.text)),
            )
        return profile.hash

    # -- computes ----------------------------------------------------------------------------

    def start_compute(
        self,
        current_video_id: str | None,
        *,
        trigger: str,
        profile_hash: str | None = None,
        mode: str | None = None,
        ts: datetime | None = None,
    ) -> str:
        ts = ts or datetime.now(UTC)
        cid = new_compute_id(ts)
        with self.conn:
            self.conn.execute(
                """INSERT INTO computes (compute_id, ts, current_video_id, trigger, mode,
                                         profile_hash) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    cid,
                    ts.isoformat(timespec="seconds"),
                    current_video_id,
                    trigger,
                    mode,
                    profile_hash,
                ),
            )
        return cid

    def finish_compute(
        self,
        compute_id: str,
        *,
        status: str = "ok",
        mode: str | None = None,
        rounds_used: int | None = None,
        pages_fetched: int | None = None,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """UPDATE computes SET status = ?, mode = COALESCE(?, mode), rounds_used = ?,
                       pages_fetched = ?, duration_ms = ?, error = ?
                   WHERE compute_id = ?""",
                (status, mode, rounds_used, pages_fetched, duration_ms, error, compute_id),
            )

    def log_query(
        self,
        compute_id: str,
        *,
        round: int,
        query_text: str,
        method: str,
        n_results: int | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO queries (compute_id, round, query_text, method, n_results) "
            "VALUES (?, ?, ?, ?, ?)",
            (compute_id, round, query_text, method, n_results),
        )

    def log_candidates(
        self,
        compute_id: str,
        cands: Sequence[Candidate],
        *,
        source: str,
        round: int = 0,
        query: str | None = None,
    ) -> None:
        """Log one list of candidates (positions 1..n) and upsert their videos + view snapshots."""
        ts = now_iso()
        with self.conn:
            for pos, c in enumerate(cands, 1):
                self.upsert_candidate(c, ts=ts)
                if c.views is not None:
                    self.add_snapshot(c.video_id, c.views, source=source, ts=ts)
                self.conn.execute(
                    "INSERT INTO candidates (compute_id, video_id, source, round, query, position) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (compute_id, c.video_id, source, round, query, pos),
                )

    def log_decision(
        self,
        compute_id: str,
        video_id: str,
        *,
        stage: str,
        kept: bool,
        reason: str | None = None,
        score: float | None = None,
        features: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO decisions (compute_id, video_id, stage, kept, reason, score, features) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                compute_id,
                video_id,
                stage,
                int(kept),
                reason,
                score,
                json.dumps(features, ensure_ascii=False) if features is not None else None,
            ),
        )

    def log_impressions(self, compute_id: str, video_ids: Iterable[str]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT INTO impressions (compute_id, video_id, rank) VALUES (?, ?, ?)",
                [(compute_id, vid, rank) for rank, vid in enumerate(video_ids, 1)],
            )

    def log_feedback(
        self,
        video_id: str,
        type: str,
        *,
        compute_id: str | None = None,
        note: str | None = None,
        ts: str | None = None,
    ) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO feedback (ts, compute_id, video_id, type, note) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts or now_iso(), compute_id, video_id, type, note),
            )
        return int(cur.lastrowid)

    def add_label(
        self,
        video_id: str,
        label: str,
        *,
        kind: str = "candidate",
        compute_id: str | None = None,
        reason: str | None = None,
        session: str | None = None,
        ts: str | None = None,
    ) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO labels (ts, video_id, compute_id, label, reason, session, kind) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts or now_iso(), video_id, compute_id, label, reason or None, session, kind),
            )
        return int(cur.lastrowid)

    def delete_label(self, label_id: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM labels WHERE label_id = ?", (label_id,))

    def downvoted_video_ids(self) -> set[str]:
        rows = self.conn.execute("SELECT DISTINCT video_id FROM feedback WHERE type = 'downvote'")
        return {r[0] for r in rows}

    def commit(self) -> None:
        self.conn.commit()

    # -- inspection --------------------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        return {t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}

    def recent_computes(self, n: int = 5) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT c.*, (SELECT COUNT(*) FROM candidates k WHERE k.compute_id = c.compute_id)
                      AS n_candidates
               FROM computes c ORDER BY ts DESC, compute_id DESC LIMIT ?""",
            (n,),
        ).fetchall()
        return [dict(r) for r in rows]


def main() -> int:
    paths = get_paths()
    if not paths.db.exists():
        print(f"no database yet at {paths.db} (`just fetch <id>` creates it)")
        return 0
    with Store.open(paths) as s:
        print(f"database: {s.path}  (schema v{s.schema_version})\n")
        for table, n in s.counts().items():
            print(f"  {table:<17} {n:>7,}")
        recent = s.recent_computes()
        if recent:
            print("\nlatest computes:")
            for c in recent:
                secs = f"{c['duration_ms'] / 1000:.1f}s" if c["duration_ms"] is not None else "-"
                print(
                    f"  {c['compute_id']}  {c['trigger']:<7} {c['status']:<7} "
                    f"video={c['current_video_id']}  cands={c['n_candidates']}  "
                    f"pages={c['pages_fetched']}  {secs}  profile={(c['profile_hash'] or '')[:8]}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
