import random
import sqlite3

import pytest

from sidekick import label as lab
from sidekick.config import ENV_VAR
from sidekick.models import Candidate, VideoMeta
from sidekick.store import db as dbmod
from sidekick.store.db import Store

SEED = "SEED0000001"
SEED2 = "SEED0000002"


def _cand(vid, **kw):
    return Candidate(vid, title=f"title {vid}", channel="Chan", duration_s=600, **kw)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sk.db")
    s.upsert_meta(VideoMeta(SEED, title="Bayesian Statistics", channel="MIT", duration_s=4000))
    cid = s.start_compute(SEED, trigger="cli")
    cands = [
        _cand(SEED),
        _cand("GOOD0000001"),
        _cand("SHORT000001", is_short=True),
        _cand("LONG0000001"),
        _cand("POLI0000001"),
    ]
    s.log_candidates(cid, cands, source="upnext")
    s.log_decision(cid, SEED, stage="A", kept=False, reason="current_video")
    s.log_decision(cid, "SHORT000001", stage="A", kept=False, reason="short")
    s.log_decision(cid, "LONG0000001", stage="A", kept=False, reason="duration>max")
    s.log_decision(cid, "GOOD0000001", stage="A", kept=True)
    s.log_decision(cid, "POLI0000001", stage="A", kept=True)
    cid2 = s.start_compute(SEED2, trigger="cli")
    s.log_candidates(cid2, [_cand("COMEDY00001"), _cand("COMEDY00002")], source="upnext")
    s.commit()
    yield s
    s.close()


def _answers(*xs):
    it = iter(xs)
    return lambda prompt: next(it)


def test_pool_skips_never_shown_but_keeps_rule_drops(store):
    ids = {it.video_id for it in lab.pool(store, SEED)}
    assert ids == {"GOOD0000001", "LONG0000001", "POLI0000001"}


def test_plan_round_robins_across_seeds(store):
    items = lab.plan(store, 4, rng=random.Random(0))
    assert [it.seed_id for it in items[:2]] == [SEED, SEED2]
    assert len(items) == 4


def test_run_saves_seed_and_candidate_labels(store):
    items = lab.plan(store, 10, seed=SEED, rng=random.Random(0))
    # seed: yes; then three candidates: good (no reason), bad + reason, skip
    counts = lab.run(
        store,
        items,
        session="t",
        input_fn=_answers("g", "g", "", "b", "political", "s"),
        out=lambda s: None,
    )
    assert counts == {"seed": 1, "candidate": 2, "skipped": 1}
    rows = store.conn.execute("SELECT kind, label, reason, session FROM labels ORDER BY label_id")
    rows = [tuple(r) for r in rows]
    assert rows[0] == ("seed", "good", None, "t")
    assert rows[1][:3] == ("candidate", "good", None)
    assert rows[2][:3] == ("candidate", "bad", "political")
    # labeled pairs are not asked again; the skipped one is
    assert len(lab.pool(store, SEED)) == 1


def test_undo_reasks_and_quit_keeps_saved(store):
    items = lab.plan(store, 10, seed=SEED, rng=random.Random(0))
    first = items[0].video_id
    # seed g; item0 bad; undo -> item0 asked again -> meh; quit
    lab.run(
        store,
        items,
        session="t",
        input_fn=_answers("g", "b", "", "u", "m", "", "q"),
        out=lambda s: None,
    )
    rows = store.conn.execute(
        "SELECT video_id, label FROM labels WHERE kind = 'candidate'"
    ).fetchall()
    assert [tuple(r) for r in rows] == [(first, "meh")]
    assert lab.seed_is_labeled(store, SEED)


def test_eof_quits_cleanly(store):
    def eof(prompt):
        raise EOFError

    items = lab.plan(store, 10, rng=random.Random(0))
    assert lab.run(store, items, session="t", input_fn=eof, out=lambda s: None)["candidate"] == 0


def test_stats_and_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert lab.main(["--stats"]) == 1  # no db yet
    with Store(tmp_path / "sidekick.db") as s:
        cid = s.start_compute(SEED, trigger="cli")
        s.log_candidates(cid, [_cand("GOOD0000001")], source="upnext")
        s.add_label("GOOD0000001", "good", compute_id=cid, session="x")
    assert lab.main(["--stats"]) == 0
    out = capsys.readouterr().out
    assert "candidate  good     1" in out and SEED in out


def test_v2_database_gets_label_kind(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(dbmod._SCHEMA_V1)
    conn.executescript(dbmod._SCHEMA_V2)
    conn.execute("INSERT INTO labels (ts, video_id, label) VALUES ('t', 'X', 'good')")
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()
    with Store(path) as s:
        assert s.schema_version == 4
        assert s.conn.execute("SELECT kind FROM labels").fetchone()[0] == "candidate"
        with pytest.raises(sqlite3.IntegrityError):
            s.add_label("Y", "good", kind="nope")
