from datetime import UTC, datetime, timedelta

import pytest

from sidekick import e3
from sidekick.store.db import Store

T0 = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sk.db")
    yield s
    s.close()


def compute(store, pages, seconds, *, i, trigger="cli", seed="SEED0000001"):
    cid = store.start_compute(seed, trigger=trigger, ts=T0 + timedelta(minutes=i))
    store.finish_compute(cid, pages_fetched=pages, duration_ms=int(seconds * 1000))
    return cid


def test_fit_recovers_the_line_and_bounds_it(store):
    noise = [0.8, -0.6, 0.3, -0.9, 0.5, -0.1, 0.7, -0.7]
    for i, (p, e) in enumerate(zip([5, 9, 12, 16, 20, 24, 28, 31], noise, strict=True)):
        compute(store, p, 4 + 2.5 * p + e, i=i)
    compute(store, 3, 999, i=99, trigger="sample")  # not a CLI compute: ignored
    f = e3.fit(e3.timing_rows(store))
    assert f.n == 8 and abs(f.b - 2.5) < 0.1 and abs(f.a - 4) < 1.5 and f.r2 > 0.99
    p = f.max_pages(90)
    assert f.upper(p) <= 90 < f.upper(p + 1) and f.mean(p) < f.upper(p)
    assert 31 <= p <= 34  # (90 - 4) / 2.5 = 34.4, minus the prediction margin


def test_too_few_computes_is_reported(store):
    compute(store, 5, 15, i=0)
    out = "\n".join(e3.report(store))
    assert "need 3+" in out


def test_replay_counts_shown_videos_past_each_budget(store):
    for i, p in enumerate([4, 10, 20]):
        compute(store, p, 3 + 2.5 * p, i=i)
    cid = compute(store, 12, 33, i=10)
    vids = [f"VID{i:08d}" for i in range(1, 31)]
    for i, v in enumerate(vids, start=1):  # budget position = order of Stage B decisions
        feats = {"enriched": i <= 25, "cached": i % 2 == 0}
        store.log_decision(cid, v, stage="B", kept=True, features=feats)
    shown = [vids[0], vids[11], vids[16], vids[22], vids[27]]  # positions 1, 12, 17, 23, 28
    for r, v in enumerate(shown, start=1):
        store.log_decision(cid, v, stage="R", kept=True, features={"rank": r})
    store.log_decision(cid, vids[2], stage="R", kept=False, features={"rank": None})
    store.conn.execute(
        "INSERT INTO queries (compute_id, round, query_text, method) VALUES (?, 1, 'q', 'topic')",
        (cid,),
    )
    store.commit()
    store.add_label(vids[16], "good", compute_id=cid)  # position 17
    store.add_label(vids[0], "good", compute_id=cid)  # position 1
    rp = e3.replay(store)
    assert rp.computes == 1 and rp.shown == 5 and rp.good_shown == 2
    assert rp.at_risk[10] == 4 and rp.good_at_risk[10] == 1
    assert rp.at_risk[15] == 3 and rp.at_risk[20] == 2 and rp.at_risk[25] == 1
    assert rp.good_at_risk[20] == 0
    # fetched (enriched, not cached) = odd positions 1..25; past budget 20: 21, 23, 25
    assert rp.pages_saved[20] == 3 and rp.pages_saved[30] == 0
    out = "\n".join(e3.report(store, md=True))
    assert "## Timing fit" in out and "| 20 |" in out and "searches per logged recommend" in out


def test_report_marks_cold_runs(store):
    for i, p in enumerate([4, 10, 20, 26]):
        compute(store, p, 3 + 2.5 * p + (0.5 if i % 2 else -0.5), i=i)
    compute(store, 28, 78, i=9, trigger="cli-cold", seed="COLD0000001")
    store.commit()
    rows = e3.timing_rows(store)
    assert sum(r["trigger"] == "cli-cold" for r in rows) == 1
    out = "\n".join(e3.report(store))
    assert "cold checks: COLD0000001 28 pages 78 s" in out and "fetch cold 1" in out
