import pytest

from sidekick.config import ENV_VAR, Paths
from sidekick.filters import thresholds as thmod
from sidekick.models import Candidate, SearchPage, VideoMeta, WatchPage
from sidekick.profile import parse_profile
from sidekick.recommend import Config, recommend
from sidekick.recommend import __main__ as cli
from sidekick.recommend import queries as qmod
from sidekick.recommend import score as sc
from sidekick.store.db import Store
from sidekick.topic import Topic
from sidekick.topic.nlp import HeuristicTagger

PROFILE = parse_profile(
    "## Interests\n- bayesian statistics\n\n## Rules\n- min_duration: 3m\n\n"
    "## Don't show me\n- political late night show\n"
)
CUR = "CURRENT0001"


def meta(vid, title, channel="Chan", desc="", cat="Education", views=10_000):
    return VideoMeta(
        vid,
        title=title,
        channel=channel,
        description=desc,
        duration_s=900,
        views=views,
        publish_date="2026-10-01T10:00:00-07:00",
        category=cat,
    )


UPNEXT = {
    "BAYES000001": ("18. Bayesian Statistics (cont.)", "Open Course"),
    "BAYES000002": ("Bayesian statistics posterior priors explained", "Stats Hub"),
    "BAYES000003": ("Bayesian inference and priors for beginners", "Stats Hub"),
    "COOK0000001": ("Making the best pizza at home", "Food Channel"),
    "CAR00000001": ("Fixing a car engine in ten steps", "Garage"),
    "POLI0000001": ("late night political show trump clip", "Late Show"),
}
SEARCH = {
    "BAYES000002": ("Bayesian statistics posterior priors explained", "Stats Hub"),
    "BAYES000004": ("Bayesian statistics lecture: priors and posteriors", "Uni Lectures"),
    "BAYES000005": ("Bayesian statistics with Stan", "Uni Lectures"),
}


class FakeFetcher:
    def __init__(self, upnext=UPNEXT, search=SEARCH):
        self.pages_fetched = 0
        self.watched, self.searched = [], []
        self.up, self.sr = upnext, search
        self.metas = {
            CUR: meta(
                CUR,
                "17. Bayesian Statistics",
                "Open Course",
                desc="In this lecture we cover Bayesian priors and posteriors.",
            )
        }
        for d in (upnext, search):
            for v, (t, ch) in d.items():
                self.metas[v] = meta(v, t, ch)

    def watch(self, vid):
        self.pages_fetched += 1
        self.watched.append(vid)
        up = (
            [
                Candidate(v, title=t, channel=ch, duration_s=900, views=10_000)
                for v, (t, ch) in self.up.items()
            ]
            if vid == CUR
            else []
        )
        return WatchPage(meta=self.metas.get(vid), upnext=up)

    def search(self, q):
        self.pages_fetched += 1
        self.searched.append(q)
        return SearchPage(
            candidates=[
                Candidate(v, title=t, channel=ch, duration_s=900) for v, (t, ch) in self.sr.items()
            ]
        )


class FakeSB:
    calls = 0

    def segments(self, vid):
        self.calls += 1
        return []


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sk.db")
    yield s
    s.close()


def run(store, tmp_path, **cfg):
    f = FakeFetcher()
    th = thmod.Thresholds()
    th.sim["tfidf"] = {"gate": 0.05, "dont_show": 0.30, "downvoted": 0.9}
    res = recommend(
        store,
        PROFILE,
        video_id=CUR,
        fetcher=f,
        sponsorblock=FakeSB(),
        tagger=HeuristicTagger(),
        thresholds=th,
        paths=Paths(tmp_path),
        config=Config(**cfg),
    )
    return res, f


# -- scores -----------------------------------------------------------------------------------


def test_percentiles_and_parts():
    assert sc.percentiles([0.9, 0.1, None, 0.5]) == [1.0, 0.0, 0.0, 0.5]
    assert sc.percentiles([0.3, 0.3]) == [0.5, 0.5] and sc.percentiles([0.2]) == [1.0]
    assert sc.recurrence(1) == 0 and sc.recurrence(2) == 0.5 and sc.recurrence(5) == 1
    assert sc.position(1, 20) == 1.0 and sc.position(None, None) == 0.0
    assert sc.views_part(10_000_000) == 1.0 and sc.views_part(None) == 0.0
    assert sc.prescore(1.0, 1, 1, 10) == 0.75
    assert sc.v0({"topic": 1, "seed": 1, "interest": 1}, "interest") == 0.8
    assert sc.v0({"topic": 1, "seed": 1, "interest": 1}, "filter-only") == 0.75


def test_round1_queries():
    t = Topic(
        "x",
        "t",
        "c",
        "lead",
        "title",
        keyphrases=[("partially observable", 3.0)],
        weights={"partially": 1},
        surface={"partially": "Partially"},
    )
    t.entities = [("Chi Jin", "PERSON")]
    assert qmod.round1(t, "filter-only", None) == [("Chi Jin Partially observable", "topic")]
    qs = qmod.round1(t, "interest", "reinforcement learning")
    assert qs[1] == ("reinforcement learning Partially observable", "interest_x")
    assert qmod.round1(t, "interest", "chi jin") == [("Chi Jin Partially observable", "topic")]
    none = Topic("x", "t", "c", "", "none")
    assert qmod.round1(none, "interest", "rl") == []


# -- pipeline ---------------------------------------------------------------------------------


def test_no_backfill_when_enough_survive(store, tmp_path):
    res, f = run(store, tmp_path, min_pool=3)
    assert res.status == "ok" and f.searched == [] and "min_pool 3" in res.backfill_skipped
    shown = [r.candidate.video_id for r in res.shown]
    assert "POLI0000001" not in shown  # Stage C, political line
    assert shown.index("BAYES000001") < shown.index("COOK0000001")
    assert shown[:3] == sorted(shown[:3], key=lambda v: v.startswith("BAYES"), reverse=True)
    rows = store.conn.execute(
        "SELECT video_id, kept, score, features FROM decisions WHERE stage = 'R'"
    ).fetchall()
    assert len(rows) == len(res.ranked) and sum(r["kept"] for r in rows) == len(res.shown)
    c = store.conn.execute("SELECT status, rounds_used, pages_fetched FROM computes").fetchone()
    assert tuple(c) == ("ok", 0, f.pages_fetched)


def test_backfill_round_searches_and_counts_recurrence(store, tmp_path):
    res, f = run(store, tmp_path, min_pool=50)
    assert len(f.searched) >= 1 and res.rounds[1].queries[0][1] == "topic"
    q = store.conn.execute("SELECT round, method FROM queries").fetchall()
    assert [tuple(x) for x in q][0] == (1, "topic")
    by = {r.candidate.video_id: r for r in res.ranked}
    assert by["BAYES000002"].parts["recurrence"] > 0  # up-next and search
    assert by["BAYES000004"].parts["lists"][0].startswith("query:")
    src = dict(
        store.conn.execute("SELECT source, COUNT(*) FROM candidates GROUP BY source").fetchall()
    )
    assert src["upnext"] == len(UPNEXT) and src["query"] >= len(SEARCH)
    assert store.conn.execute("SELECT rounds_used FROM computes").fetchone()[0] == 1


def test_enrich_budget_goes_to_best_prescore(store, tmp_path):
    res, f = run(store, tmp_path, min_pool=3, enrich_cap=2)
    enriched = [v for v in f.watched if v != CUR]
    assert len(enriched) == 2 and all(v.startswith("BAYES") for v in enriched)
    by = {r.candidate.video_id: r for r in res.ranked}
    assert by["COOK0000001"].parts["enriched"] is False
    pre = dict(
        store.conn.execute(
            "SELECT video_id, json_extract(features, '$.prescore') FROM decisions WHERE stage = 'B'"
        ).fetchall()
    )
    assert pre["BAYES000001"] > pre["CAR00000001"]


def test_channel_cap(store, tmp_path):
    res, _ = run(store, tmp_path, min_pool=50, per_channel=1, top_k=4)
    chans = [r.candidate.channel for r in res.shown]
    assert len(chans) == 4 and len(set(chans)) == 4
    assert any(r.reason == "channel_cap" for r in res.ranked if r.rank is None)


def test_cli_prints_top(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    (tmp_path / "profile.md").write_text(
        "## Interests\n- bayesian statistics\n\n## Don't show me\n- political late night show\n"
    )

    class Ctx(FakeFetcher):
        def __init__(self, *a, **k):
            super().__init__()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class SB(FakeSB):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(cli, "Fetcher", Ctx)
    monkeypatch.setattr(cli, "SponsorBlock", SB)
    assert cli.main([f"https://www.youtube.com/watch?v={CUR}", "--min-pool", "50"]) == 0
    out = capsys.readouterr().out
    assert "round 0  up-next 6" in out and "round 1  search" in out
    assert "top " in out and "18. Bayesian Statistics (cont.)" in out and "compute " in out
