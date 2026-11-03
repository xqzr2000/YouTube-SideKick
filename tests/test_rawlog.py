from datetime import UTC, datetime

from sidekick.config import ENV_VAR, get_paths
from sidekick.store import rawlog


def test_append_read_roundtrip_same_day(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    ts = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
    p1 = rawlog.append("watch", "AAAAAAAAAAA", {"x": 1}, ts=ts)
    p2 = rawlog.append("search", "bayes", {"y": "é"}, ts=ts)
    assert p1 == p2 and p1.name == "2026-11-02.jsonl.gz"
    recs = list(rawlog.read(p1))
    assert [r["kind"] for r in recs] == ["watch", "search"]
    assert recs[1]["data"] == {"y": "é"}


def test_prune_removes_only_old_files(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    paths = get_paths()
    for day in ("2026-07-01", "2026-09-01", "2026-11-01"):
        rawlog.append("watch", "k", {}, ts=datetime.fromisoformat(day).replace(tzinfo=UTC))
    (paths.raw_dir / "notes.txt").write_text("keep me")
    deleted = rawlog.prune(days=90, today=datetime(2026, 11, 2, tzinfo=UTC))
    assert [p.name for p in deleted] == ["2026-07-01.jsonl.gz"]
    assert (paths.raw_dir / "notes.txt").exists()
    assert (paths.raw_dir / "2026-09-01.jsonl.gz").exists()
