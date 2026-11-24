from datetime import UTC, datetime, timedelta

import pytest

from sidekick import e6
from sidekick.config import ENV_VAR
from sidekick.models import Candidate, VideoMeta
from sidekick.store.db import Store

T0 = datetime(2026, 11, 10, tzinfo=UTC)
LECTURE, COMEDY = "LECTURE0001", "COMEDY00001"


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sk.db")
    meta = {
        LECTURE: ("17. Bayesian Statistics", "MIT", 400),
        COMEDY: ("Stand-up set", "Dry Bar", 30),
        "KIMMEL00001": ("Kimmel roasts the president", "Jimmy Kimmel Live", 2),
        "KIMMEL00002": ("Kimmel on the midterms", "Jimmy Kimmel Live", 1),
        "GOODLEC0001": ("18. Bayesian Statistics (cont.)", "MIT", 400),
        "UNKNOWN0001": ("Something else", "Other", 100),
    }
    for vid, (title, ch, days) in meta.items():
        s.upsert_meta(
            VideoMeta(vid, title=title, channel=ch,
                      publish_date=(T0 - timedelta(days=days)).isoformat())
        )  # fmt: skip

    def compute(seed, hours, lists, trigger="cli"):
        cid = s.start_compute(seed, trigger=trigger, ts=T0 + timedelta(hours=hours))
        for source, vids in lists.items():
            s.log_candidates(cid, [Candidate(v) for v in vids], source=source)
        return cid

    # the lecture, computed twice from the CLI (signed-out list), then once from the panel
    c1 = compute(LECTURE, 0, {"upnext": ["GOODLEC0001", "KIMMEL00001", "UNKNOWN0001"]})
    c2 = compute(LECTURE, 1, {"upnext": ["KIMMEL00001", "GOODLEC0001"]})
    s.log_decision(c1, "KIMMEL00001", stage="C", kept=False, reason="dont_show:political")
    s.add_label("GOODLEC0001", "good", kind="candidate", compute_id=c1, session="t")
    # Kimmel also under the comedy video, labeled bad there; its channel-mate too
    c3 = compute(COMEDY, 2, {"upnext": ["KIMMEL00001", "KIMMEL00002"]})
    s.add_label(
        "KIMMEL00001", "bad", kind="candidate", compute_id=c3, reason="politics", session="t"
    )
    # a 👎 on Kimmel between c2 and c3
    s.conn.execute(
        "INSERT INTO feedback (ts, compute_id, video_id, type, note) VALUES (?, ?, ?, ?, ?)",
        ((T0 + timedelta(hours=1, minutes=30)).isoformat(), c2, "KIMMEL00001", "downvote", "no"),
    )
    c4 = compute(
        LECTURE, 3,
        {"upnext": ["GOODLEC0001"], "tab": ["KIMMEL00001", "KIMMEL00002", "GOODLEC0001"]},
        trigger="open",
    )  # fmt: skip
    s.commit()
    s.ids = (c1, c2, c3, c4)
    yield s
    s.close()


def test_classes(store):
    d = e6.load(store)
    by = {(s.compute_id, s.video_id, s.source): s for s in d.sightings}
    c1, c2, c3, c4 = store.ids
    assert by[(c1, "KIMMEL00001", "upnext")].why == "dropped: don't-show"
    assert by[(c3, "KIMMEL00001", "upnext")].why == "labeled bad (politics)"
    assert by[(c2, "KIMMEL00001", "upnext")].cls == "unwanted"  # 👎'd at some point
    assert by[(c1, "GOODLEC0001", "upnext")].cls == "wanted"
    assert by[(c1, "UNKNOWN0001", "upnext")].cls == "unknown"
    assert by[(c3, "KIMMEL00002", "upnext")].cls == "unknown"  # same channel isn't enough
    assert all(s.video_id not in (LECTURE, COMEDY) for s in d.sightings)


def test_stickiness_spread_channels(store):
    d = e6.load(store)
    st = e6.stickiness(d)
    # lecture chain c1 → c2 → c4: Kimmel in c1 (returns), c2 (returns in c4's tab list)
    assert st["unwanted"][LECTURE] == (2, 2)
    assert st["wanted"][LECTURE] == (2, 2)
    assert st["unknown"][LECTURE] == (0, 1)
    sp = e6.spread(d)
    assert sp["unwanted"][2] == 1  # Kimmel under both current videos
    assert sp["unknown"] == {1: 1, 2: 1}  # UNKNOWN0001 once; KIMMEL00002 under both
    ch = e6.channels(d)
    assert ch[0]["channel"] == "Jimmy Kimmel Live" and ch[0]["unwanted_contexts"] == 2
    top = e6.top_unwanted(d)
    assert top[0]["title"].startswith("Kimmel roasts") and top[0]["computes"] == 4


def test_sources_downvote_and_profile(store):
    d = e6.load(store)
    bs = e6.by_source(d)
    assert bs["tab"][LECTURE] == (1, 3) and bs["upnext"][COMEDY] == (1, 2)
    (pair,) = e6.paired_sources(d)
    assert pair[1] == 0.0 and pair[2] == pytest.approx(1 / 3)
    (dv,) = e6.after_downvote(d)
    assert dv["again"] == 2 and dv["again_computes"] == 2  # c3 and c4
    assert dv["channel_after"] == 2  # KIMMEL00002 in c3 and c4
    pr = e6.profile_by_class(d)
    assert pr["unwanted"]["age"] < pr["wanted"]["age"]  # fresh political clip vs old lecture


def test_cluster_rate():
    r, lo, hi, n = e6.cluster_rate({"a": (1, 2), "b": (3, 4)})
    assert r == pytest.approx(4 / 6) and lo <= r <= hi and n == 6
    assert e6.cluster_rate({})[3] == 0


def test_report_and_cli(store, tmp_path, monkeypatch, capsys):
    text = "\n".join(e6.report(store))
    for part in (
        "== 1. Stickiness",
        "== 2. Spread",
        "Jimmy Kimmel Live",
        "After a 👎",
        "persistent",
    ):
        assert part in text
    assert "| class |" in "\n".join(e6.report(store, md=True))
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "empty"))
    assert e6.main([]) == 0
    assert "No YouTube up-next lists" in capsys.readouterr().out
