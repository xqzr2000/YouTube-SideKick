import json
import sqlite3

import pytest

from sidekick.config import ENV_VAR
from sidekick.models import Candidate, VideoMeta
from sidekick.profile import parse_profile
from sidekick.store import db as dbmod
from sidekick.store.db import SCHEMA_VERSION, TABLES, SchemaTooNew, Store, new_compute_id


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sk.db")
    yield s
    s.close()


def test_schema_created_and_reopen_is_idempotent(tmp_path):
    path = tmp_path / "sk.db"
    with Store(path) as s:
        assert s.schema_version == SCHEMA_VERSION
        assert set(s.counts()) == set(TABLES)
        assert all(n == 0 for n in s.counts().values())
    with Store(path) as s:  # second open must not try to re-create tables
        assert s.schema_version == SCHEMA_VERSION


def test_newer_schema_is_refused(tmp_path):
    path = tmp_path / "sk.db"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.close()
    with pytest.raises(SchemaTooNew):
        Store(path)


def test_compute_id_sorts_by_time():
    from datetime import UTC, datetime

    a = new_compute_id(datetime(2026, 11, 3, 1, 2, 3, tzinfo=UTC))
    b = new_compute_id(datetime(2026, 11, 3, 1, 2, 4, tzinfo=UTC))
    assert a.startswith("20261103T010203Z-") and a < b


def test_candidate_then_meta_upsert(store):
    c = Candidate(
        "AAAAAAAAAAA",
        title="Short title",
        channel="Chan",
        duration_s=600,
        views=33000,
        published_text="9y ago",
    )
    store.upsert_candidate(c, ts="2026-11-03T01:00:00+00:00")
    v = store.get_video("AAAAAAAAAAA")
    assert v["title"] == "Short title" and v["last_enriched"] is None
    assert v["first_seen"] == "2026-11-03T01:00:00+00:00"

    m = VideoMeta(
        "AAAAAAAAAAA",
        title="Full title",
        channel="Chan",
        channel_id="UC1",
        description="desc",
        keywords=["bayes", "mit"],
        duration_s=601,
        views=33123,
        publish_date="2017-01-05T00:00:00-08:00",
        category="Education",
    )
    store.upsert_meta(m, ts="2026-11-03T02:00:00+00:00")
    v = store.get_video("AAAAAAAAAAA")
    assert v["title"] == "Full title" and v["duration_s"] == 601
    assert json.loads(v["keywords"]) == ["bayes", "mit"]
    assert v["published_text"] == "9y ago"  # kept from the candidate
    assert v["first_seen"] == "2026-11-03T01:00:00+00:00"
    assert v["last_enriched"] == "2026-11-03T02:00:00+00:00"

    # a later cheap sighting must not clobber exact fields with blanks or rounded values
    store.upsert_candidate(
        Candidate("AAAAAAAAAAA", title="", duration_s=600), ts="2026-11-03T03:00:00+00:00"
    )
    v = store.get_video("AAAAAAAAAAA")
    assert v["title"] == "Full title" and v["duration_s"] == 601 and v["description"] == "desc"
    assert v["last_seen"] == "2026-11-03T03:00:00+00:00"

    snaps = store.conn.execute("SELECT views, source FROM video_snapshots").fetchall()
    assert [tuple(r) for r in snaps] == [(33123, "watch")]


def test_full_compute_round_trip(store):
    prof = parse_profile("## Interests\n- rl\n")
    h = store.save_profile_version(prof)
    assert store.save_profile_version(prof) == h  # same text -> one version
    cid = store.start_compute("SEEDSEEDSEE", trigger="cli", profile_hash=h)

    upnext = [Candidate(f"UPNEXT{i:05d}", title=f"u{i}", views=1000 * i) for i in range(1, 4)]
    search = [Candidate("UPNEXT00001", title="u1"), Candidate("SEARCH00001", title="s1")]
    store.log_candidates(cid, upnext, source="upnext", round=0)
    store.log_query(cid, round=1, query_text="rl lecture", method="manual", n_results=2)
    store.log_candidates(cid, search, source="query", round=1, query="rl lecture")
    store.log_decision(
        cid,
        "UPNEXT00002",
        stage="A",
        kept=False,
        reason="duration<min",
        features={"duration_s": 30},
    )
    store.log_impressions(cid, ["UPNEXT00001", "SEARCH00001"])
    store.log_feedback("UPNEXT00001", "click", compute_id=cid)
    store.finish_compute(cid, rounds_used=1, pages_fetched=2, duration_ms=9600)

    counts = store.counts()
    assert counts["videos"] == 4  # UPNEXT00001 appears twice but is one video
    assert counts["candidates"] == 5
    assert counts["video_snapshots"] == 3  # only candidates with views
    assert counts["profile_versions"] == 1
    assert (counts["queries"], counts["decisions"], counts["impressions"]) == (1, 1, 2)
    assert counts["feedback"] == 1

    (c,) = store.recent_computes()
    assert c["compute_id"] == cid and c["status"] == "ok" and c["n_candidates"] == 5
    assert c["profile_hash"] == h and c["duration_ms"] == 9600

    row = store.conn.execute("SELECT features, kept FROM decisions").fetchone()
    assert json.loads(row["features"]) == {"duration_s": 30} and row["kept"] == 0

    # recurrence query the features module will run later: distinct sources per video
    rec = dict(
        store.conn.execute(
            "SELECT video_id, COUNT(DISTINCT source) FROM candidates GROUP BY video_id"
        ).fetchall()
    )
    assert rec["UPNEXT00001"] == 2 and rec["SEARCH00001"] == 1


def test_feedback_type_is_checked(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.log_feedback("AAAAAAAAAAA", "like")


def test_db_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert dbmod.main() == 0
    assert "no database yet" in capsys.readouterr().out
    with Store(tmp_path / "sidekick.db") as s:
        cid = s.start_compute("SEEDSEEDSEE", trigger="cli")
        s.finish_compute(cid, pages_fetched=1, duration_ms=1500)
    assert dbmod.main() == 0
    out = capsys.readouterr().out
    assert f"schema v{SCHEMA_VERSION}" in out and cid in out and "1.5s" in out
