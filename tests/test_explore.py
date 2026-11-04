import shutil
from datetime import UTC, datetime
from pathlib import Path

from sidekick.config import ENV_VAR, get_paths
from sidekick.explore import load_candidates
from sidekick.parse.watch import extract_watch_blobs
from sidekick.store import rawlog

FIX = Path(__file__).parent / "fixtures"


def test_auto_falls_back_to_sample(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    d = load_candidates()
    assert d["source"] == "sample"
    assert d["pages"] == 8 and len(d["seeds"]) == 4
    kinds = {r["page_kind"] for r in d["candidates"]}
    assert kinds == {"watch", "search"}
    row = d["candidates"][0]
    assert {"video_id", "rank", "age_days", "page_key", "views"} <= set(row)


def test_fixtures_source(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    (tmp_path / "fixtures").mkdir()
    shutil.copy(FIX / "watch_synthetic.html", tmp_path / "fixtures" / "watch_AAAAAAAAAAA.html")
    shutil.copy(FIX / "search_synthetic.html", tmp_path / "fixtures" / "search_bayes.html")
    d = load_candidates()
    assert d["source"] == "fixtures" and d["pages"] == 2
    assert {r["page_key"] for r in d["candidates"]} == {"AAAAAAAAAAA", "bayes"}


def test_raw_source_replays_parser(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    blobs = extract_watch_blobs((FIX / "watch_synthetic.html").read_text())
    rawlog.append("watch", "AAAAAAAAAAA", blobs, ts=datetime(2026, 11, 3, tzinfo=UTC))
    assert list(get_paths().raw_dir.iterdir())
    d = load_candidates()
    assert d["source"] == "raw"
    assert len(d["candidates"]) == 10 and d["seeds"][0]["video_id"] == "AAAAAAAAAAA"
    m = next(r for r in d["candidates"] if r["video_id"] == "MMMMMMMMMMM")
    assert m["age_days"] == 4 * 365


def test_load_decisions_sample_and_db(tmp_path):
    from sidekick.config import Paths
    from sidekick.explore import load_decisions
    from sidekick.models import Candidate
    from sidekick.store.db import Store

    paths = Paths(tmp_path)
    sample = load_decisions(paths=paths)
    assert sample["source"] == "sample" and len(sample["computes"]) == 8
    reasons = {r["reason"] for r in sample["decisions"]}
    assert {"live", "short;duration<min"} <= reasons
    assert all(r["sources"] for r in sample["decisions"])

    with Store(paths.db) as s:
        cid = s.start_compute("SEEDSEEDSEE", trigger="cli")
        s.log_candidates(cid, [Candidate("AAAAAAAAAAA", title="t", channel="c")], source="upnext")
        s.log_decision(
            cid,
            "AAAAAAAAAAA",
            stage="A",
            kept=False,
            reason="duration<min",
            features={"duration_s": 30},
        )
        s.finish_compute(cid)
    db = load_decisions(paths=paths)
    assert db["source"] == "db"
    (row,) = db["decisions"]
    assert row["kept"] is False and row["f_duration_s"] == 30 and row["upnext_position"] == 1
    assert db["computes"][0]["kept_a"] == 0
