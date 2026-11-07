import random
from datetime import UTC, datetime, timedelta

import pytest

from sidekick import label as lab
from sidekick.models import Candidate, VideoMeta
from sidekick.recommend import evaluate as ev
from sidekick.store.db import Store

SEED = "SEED0000001"
T0 = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


def _c(vid, **kw):
    return Candidate(vid, title=f"title {vid}", channel="Chan", duration_s=600, **kw)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sk.db")
    s.upsert_meta(VideoMeta(SEED, title="Lecture 22", channel="Course", duration_s=4000))
    # an older compute of the same seed: its lists must not count
    old = s.start_compute(SEED, trigger="cli", ts=T0)
    s.log_candidates(old, [_c("OLD00000001")], source="upnext")
    s.log_decision(old, "OLD00000001", stage="R", kept=True, features={"rank": 1})
    cid = s.start_compute(SEED, trigger="cli", ts=T0 + timedelta(hours=1))
    up = [_c(SEED), _c("SHORT000001", is_short=True)] + [_c(f"UP{i:09d}") for i in range(1, 12)]
    s.log_candidates(cid, up, source="upnext")
    s.log_decision(cid, SEED, stage="A", kept=False, reason="current_video")
    s.log_decision(cid, "SHORT000001", stage="A", kept=False, reason="short")
    # SideKick shows 3 up-next videos and 2 search results, logged out of rank order
    shown = ["UP000000003", "SRCH0000001", "UP000000001", "SRCH0000002", "UP000000011"]
    for r, v in sorted(enumerate(shown, start=1), key=lambda x: x[1]):
        s.log_decision(cid, v, stage="R", kept=True, features={"rank": r})
    s.log_decision(cid, "UP000000002", stage="R", kept=False, reason="rank", features={})
    s.commit()
    s.cid = cid
    yield s
    s.close()


def test_lists(store):
    assert ev.latest_recommends(store) == [(store.cid, SEED)]
    assert ev.sidekick_list(store, store.cid) == [
        "UP000000003",
        "SRCH0000001",
        "UP000000001",
        "SRCH0000002",
        "UP000000011",
    ]
    yt = ev.youtube_list(store, store.cid, SEED)
    assert yt == [f"UP{i:09d}" for i in range(1, 11)]  # current video and Short skipped


def test_precision_against_youtube(store):
    for v, label in [
        ("UP000000003", "good"),
        ("SRCH0000001", "good"),
        ("UP000000001", "meh"),
        ("UP000000002", "bad"),
        ("UP000000004", "bad"),
    ]:
        store.add_label(v, label, compute_id=store.cid)
    [row] = ev.evaluate(store)
    assert (row.sidekick.labeled, row.sidekick.good, row.sidekick.meh) == (3, 2, 1)
    assert row.sidekick.precision == pytest.approx(2 / 3) and row.sidekick.bad_share == 0
    assert (row.youtube.labeled, row.youtube.good, row.youtube.bad) == (4, 1, 2)
    assert row.youtube.bad_share == pytest.approx(0.5) and row.overlap == 2  # UP11 is 11th
    out = "\n".join(lab.stats(store))
    assert "top-10 precision" in out and "67%" in out and "still unlabeled" in out


def test_top10_plan_covers_both_lists_once(store):
    store.add_label("UP000000005", "good", compute_id=store.cid)
    items = lab.plan_top10(store, None, rng=random.Random(0))
    ids = [it.video_id for it in items]
    expected = {"SRCH0000001", "SRCH0000002", "UP000000011"} | {f"UP{i:09d}" for i in range(1, 11)}
    assert set(ids) == expected - {"UP000000005"} and len(ids) == len(set(ids))
    assert all(it.compute_id == store.cid for it in items)
    assert ids != sorted(ids, key=lambda v: v.startswith("UP"))  # mixed, not list by list
    assert len(lab.plan_top10(store, 4, rng=random.Random(0))) == 4
