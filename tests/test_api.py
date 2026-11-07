import threading

import pytest
from fastapi.testclient import TestClient

from sidekick.api import app as appmod
from sidekick.api.jobs import Card, JobResult, JobRunner
from sidekick.config import ENV_VAR
from sidekick.models import VideoMeta
from sidekick.store.db import Store

VID = "bFZ-0FH5hfs"
VID2 = "wQXdPZikag0"
UP = [
    {
        "video_id": "AAAAAAAAAA1",
        "title": "Lecture 18",
        "channel": "MIT",
        "duration": "1:12:05",
        "views": "120K views",
        "age": "9 years ago",
    },
    {"video_id": "AAAAAAAAAA2", "title": "a short", "badges": ["SHORTS"]},
    {"video_id": "AAAAAAAAAA3", "title": "live now", "badges": ["LIVE"]},
]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    (tmp_path / "profile.md").write_text("## Interests\n- statistics\n\n## Don't show me\n")
    return tmp_path


def fake_run(seen):
    def run(job, step):
        seen.append(job)
        step("video", "Reading the video")
        step("r0", "Up-next: filtering 3 videos")
        step("r0", "Up-next: 2 of 3 kept")
        with Store.open() as store:
            cid = store.start_compute(job.video_id, trigger=job.trigger)
            store.finish_compute(cid, pages_fetched=1, duration_ms=10)
        card = Card("AAAAAAAAAA1", "Lecture 18", "MIT", 4325, "https://i.ytimg.com/x", 1)
        return JobResult(
            compute_id=cid, current_title="17. Bayesian", mode="interest", cards=[card]
        )

    return run


@pytest.fixture
def client(home, monkeypatch):
    seen = []
    old = appmod.app.state.runner
    appmod.app.state.runner = JobRunner(fake_run(seen), threaded=False)
    c = TestClient(appmod.app)
    c.seen = seen
    yield c
    appmod.app.state.runner = old


def test_compute_runs_and_returns_cards(client):
    r = client.post("/computes", json={"video_id": VID, "upnext": UP, "trigger": "open"})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "done" and j["video_id"] == VID and j["mode"] == "interest"
    assert [s["key"] for s in j["steps"]] == ["video", "r0"]  # same key replaces the text
    assert j["steps"][1]["text"] == "Up-next: 2 of 3 kept"
    assert j["cards"][0]["video_id"] == "AAAAAAAAAA1" and j["compute_id"]
    tab = client.seen[0].tab_upnext
    assert tab[0].duration_s == 4325 and tab[0].views == 120_000 and tab[0].source == "tab"
    assert tab[1].is_short and tab[2].is_live
    assert client.get(f"/computes/{j['job_id']}").json()["status"] == "done"
    assert client.get("/computes/latest").json()["job_id"] == j["job_id"]
    assert client.get("/computes/nope").status_code == 404


def test_no_video_falls_back_to_last_then_empty(client, home):
    r = client.post("/computes", json={"trigger": "open"})
    assert r.json()["status"] == "empty"
    client.post("/computes", json={"video_id": VID2})
    r = client.post("/computes", json={"trigger": "open", "upnext": UP})
    assert r.json()["video_id"] == VID2 and client.seen[-1].tab_upnext == []


def test_bad_video_id_is_rejected(client):
    assert client.post("/computes", json={"video_id": "not-an-id"}).status_code == 422


def test_downvote_needs_a_note_and_lands_in_the_profile(client, home):
    with Store.open() as store:
        store.upsert_meta(VideoMeta("AAAAAAAAAA1", title="Lecture 18", channel="MIT"))
        store.commit()
    r = client.post("/feedback", json={"type": "downvote", "video_id": "AAAAAAAAAA1", "note": " "})
    assert r.status_code == 422
    r = client.post(
        "/feedback",
        json={"type": "downvote", "video_id": "AAAAAAAAAA1", "note": "too long", "compute_id": "x"},
    )
    assert r.status_code == 200 and "too long" in r.json()["profile_line"]
    text = (home / "profile.md").read_text()
    assert "too long" in text and "Lecture 18 · MIT" in text
    with Store.open() as store:
        assert store.downvoted_video_ids() == {"AAAAAAAAAA1"}


def test_click_is_logged(client):
    j = client.post("/computes", json={"video_id": VID}).json()
    r = client.post(
        "/feedback",
        json={"type": "click", "video_id": "AAAAAAAAAA1", "compute_id": j["compute_id"]},
    )
    assert r.status_code == 200 and r.json()["profile_line"] is None
    with Store.open() as store:
        row = store.conn.execute("SELECT type, compute_id FROM feedback").fetchone()
    assert tuple(row) == ("click", j["compute_id"])


def test_runner_joins_queues_and_supersedes():
    gate = threading.Event()
    ran = []

    def run(job, step):
        ran.append(job.video_id)
        if job.video_id == "A":
            gate.wait(5)
        return JobResult()

    r = JobRunner(run)
    a = r.submit("A", trigger="open", tab_upnext=[])
    assert r.submit("A", trigger="refresh", tab_upnext=[]) is a  # joins the running job
    b = r.submit("B", trigger="click", tab_upnext=[])
    c = r.submit("C", trigger="click", tab_upnext=[])
    assert b.status == "superseded" and c.status == "queued"
    assert r.latest() is a
    gate.set()
    for _ in range(200):
        if c.status == "done":
            break
        threading.Event().wait(0.01)
    assert ran == ["A", "C"] and a.status == "done" and c.status == "done"
    assert r.latest() is c


def test_runner_reports_errors():
    class BlockedError(Exception):
        pass

    def run(job, step):
        raise BlockedError("consent page; back off until 18:30")

    job = JobRunner(run, threaded=False).submit("A", trigger="open", tab_upnext=[])
    assert job.status == "blocked" and "18:30" in job.error
