from datetime import timedelta

import pytest

from sidekick.config import ENV_VAR, Paths
from sidekick.filters import thresholds as thmod
from sidekick.models import Candidate, SearchPage, VideoMeta, WatchPage
from sidekick.profile import parse_profile
from sidekick.recommend import Config, recommend
from sidekick.recommend import __main__ as cli
from sidekick.recommend import diversity as dv
from sidekick.recommend import queries as qmod
from sidekick.recommend import score as sc
from sidekick.store.db import Store
from sidekick.topic import Topic
from sidekick.topic.nlp import Entity, HeuristicTagger

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
        if vid.startswith("BAYES"):  # related videos of an on-topic page (snowball)
            title = "Bayesian priors in practice"
            up = [Candidate("SNOW0000001", title=title, channel="Snow Lab", duration_s=900)]
            self.metas.setdefault("SNOW0000001", meta("SNOW0000001", title, "Snow Lab"))
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


def test_up_next_only_when_search_is_off(store, tmp_path):
    res, f = run(store, tmp_path, min_pool=3, always_search=False)
    assert res.status == "ok" and f.searched == [] and "min_pool 3" in res.backfill_skipped
    shown = [r.candidate.video_id for r in res.shown]
    assert "POLI0000001" not in shown  # Stage C, political line
    assert shown.index("BAYES000001") < shown.index("COOK0000001")
    rows = store.conn.execute(
        "SELECT video_id, kept, score, features FROM decisions WHERE stage = 'R'"
    ).fetchall()
    assert len(rows) == len(res.ranked) and sum(r["kept"] for r in rows) == len(res.shown)
    c = store.conn.execute("SELECT status, rounds_used, pages_fetched FROM computes").fetchone()
    assert tuple(c) == ("ok", 0, f.pages_fetched)


def test_round1_always_searches_then_stops_when_enough(store, tmp_path):
    res, f = run(store, tmp_path, min_pool=3)
    assert len(f.searched) == 1 and [r.round for r in res.rounds] == [0, 1]
    assert res.backfill_skipped.startswith("round 2:")
    by = {r.candidate.video_id: r for r in res.ranked}
    assert by["BAYES000002"].parts["recurrence"] > 0  # up-next and search
    assert by["BAYES000004"].parts["lists"][0].startswith("query:")
    q = store.conn.execute("SELECT round, method FROM queries").fetchall()
    assert [tuple(x) for x in q] == [(1, "topic")]


def test_rounds_2_3_add_rocchio_and_snowball_when_short(store, tmp_path):
    res, f = run(store, tmp_path, min_pool=50)
    assert [r.round for r in res.rounds][:3] == [0, 1, 2]
    r2 = res.rounds[2]
    assert r2.snowball and r2.snowball[0][0].startswith("BAYES")
    src = dict(
        store.conn.execute("SELECT source, COUNT(*) FROM candidates GROUP BY source").fetchall()
    )
    assert src["upnext"] == len(UPNEXT) and src["query"] >= len(SEARCH) and src["snowball"] >= 1
    by = {r.candidate.video_id: r for r in res.ranked}
    assert any(x.startswith("snowball:") for x in by["SNOW0000001"].parts["lists"])
    assert store.conn.execute("SELECT rounds_used FROM computes").fetchone()[0] >= 2


def test_deadline_stops_later_rounds(store, tmp_path):
    res, f = run(store, tmp_path, min_pool=50, deadline_s=-1)
    assert [r.round for r in res.rounds] == [0, 1] and "deadline" in res.backfill_skipped


def test_enrich_budget_goes_to_best_prescore(store, tmp_path):
    res, f = run(store, tmp_path, min_pool=3, enrich_cap=2, always_search=False)
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


def test_group_caps_in_the_pipeline(store, tmp_path):
    res, _ = run(store, tmp_path, min_pool=3, caps={"lecture": 1, "other": 1}, top_k=4)
    assert res.kind == "lecture"  # current video is Education
    groups = [r.parts["group"] for r in res.shown]
    assert len(groups) == 4 and len(set(groups)) == 4
    assert any(r.reason == "group_cap" for r in res.ranked if r.rank is None)


class FakeTagger:
    name = "fake"

    def entities(self, text):
        return [Entity("Chad Thornsberry", "PERSON")] if "Chad Thornsberry" in text else []

    def phrases(self, text):
        return []


def test_context_kind_and_group_key():
    assert dv.context_kind("Education", "Bayes") == "lecture"
    assert dv.context_kind("Comedy", "Lecture 22: RL") == "lecture"
    assert dv.context_kind("Comedy", "I Tried To Run A 5K") == "other"
    t = FakeTagger()
    a = dv.group_key("It's Great | Chad Thornsberry | Dry Bar", "Dry Bar Clips", "other", t)
    b = dv.group_key("The BEST of Chad Thornsberry", "Dry Bar Comedy", "other", t)
    assert a == b == "person:chad thornsberry"
    assert dv.group_key("Some clip", "Dry Bar Clips", "other", t) == "channel:dry bar clips"
    assert dv.group_key("Chad Thornsberry", "MIT", "lecture", t) == "channel:mit"


def test_mmr_caps_redundancy_and_fill():
    import numpy as np

    vecs = np.array([[1, 0], [0.99, 0.14], [0, 1], [0.1, 0.99]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    order, info = dv.mmr_select([0.9, 0.85, 0.6, 0.5], vecs, ["a", "b", "c", "d"], k=2, cap=5)
    assert order == [0, 2]  # 1 is a near-duplicate of 0
    order, _ = dv.mmr_select([0.9, 0.85, 0.6], None, ["a", "a", "b"], k=2, cap=1)
    assert order == [0, 2]
    order, info = dv.mmr_select([0.9, 0.85], None, ["a", "a"], k=2, cap=1)
    assert order == [0, 1] and info[1]["filled_past_cap"]


def test_rocchio_adds_shared_terms():
    anchor = Topic(
        "x",
        "t",
        "c",
        "Will there be a tunnel turnaround?",
        "title",
        keyphrases=[("tunnel turnaround", 3.0)],
        weights={"tunnel": 2.0, "turnaround": 3.0},
    )
    rel = [
        Topic(
            "a",
            "",
            "",
            "",
            "title",
            weights={"massey": 3, "tunnel": 2, "replacement": 2},
            surface={"massey": "Massey"},
        ),
        Topic(
            "b",
            "",
            "",
            "",
            "title",
            weights={"massey": 3, "replacement": 1, "bridge": 2},
            surface={"massey": "Massey"},
        ),
        Topic("c", "", "", "", "title", weights={"ferry": 1}),
    ]
    assert qmod.rocchio(anchor, rel) == "tunnel turnaround Massey replacement"
    assert qmod.rocchio(anchor, rel[2:]) == ""


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
    assert "round 0  up-next 6" in out and "round 1  topic" in out and "snowball" in out
    assert "lecture context: same course ≤ 5" in out
    assert "top " in out and "18. Bayesian Statistics (cont.)" in out and "compute " in out
    assert cli.main([CUR, "--cold", "--no-search"]) == 0
    with Store.open(Paths(tmp_path)) as st:
        triggers = [r[0] for r in st.conn.execute("SELECT trigger FROM computes ORDER BY rowid")]
    assert triggers[-1] == "cli-cold" and triggers[0] == "cli"


def test_person_name_trims_tails():
    assert dv.person_name("Chad Thornsberry - Full Special") == "chad thornsberry"
    assert dv.person_name("Chad Thornsberry Stand") == "chad thornsberry"
    assert dv.person_name("Trump") == "" and dv.person_name("the band") == ""


def test_assign_groups_matches_names_across_the_pool():
    class OnlyFirst(FakeTagger):  # NER finds the name in one title only
        def entities(self, text):
            return [Entity("Chad Thornsberry - Full Special", "PERSON")] if "5K" in text else []

    titles = [
        "I Tried To Run A 5K | Chad Thornsberry - Full Special",
        "Fine Dining And Utter Nonsense. Chad Thornsberry",
        "Pulling Over A Cop",
        "Running A 5k",  # name only in the channel
    ]
    channels = ["Dry Bar Clips", "Dry Bar Comedy", "Dry Bar Comedy", "Chad Thornsberry - Topic"]
    g = dv.assign_groups(titles, channels, "other", OnlyFirst())
    assert g == ["person:chad thornsberry"] * 2 + [
        "channel:dry bar comedy",
        "person:chad thornsberry",
    ]


def test_lecture_context_same_course_not_redundant():
    import numpy as np

    vecs = np.array([[1, 0], [0.99, 0.14], [0.98, 0.2], [0, 1]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    scores, groups = [0.9, 0.8, 0.7, 0.75], ["c", "c", "c", "other"]
    order, _ = dv.mmr_select(scores, vecs, groups, k=3, cap=5)
    assert order[1] == 3  # redundancy pushes the same-course lectures down
    order, _ = dv.mmr_select(scores, vecs, groups, k=3, cap=5, same_group_redundant=False)
    assert order == [0, 1, 3]  # same course isn't redundant; other courses still are


def test_cold_skips_the_page_cache(store, tmp_path):
    _, warm1 = run(store, tmp_path, min_pool=3, always_search=False)
    _, warm2 = run(store, tmp_path, min_pool=3, always_search=False)
    _, cold = run(store, tmp_path, min_pool=3, always_search=False, cache_ttl=timedelta(0))
    assert warm2.pages_fetched == 1  # everything cached but the current page
    assert cold.pages_fetched == warm1.pages_fetched > 1


def test_tab_upnext_joins_round_0_and_steps_are_reported(store, tmp_path):
    f = FakeFetcher()
    th = thmod.Thresholds()
    th.sim["tfidf"] = {"gate": 0.05, "dont_show": 0.30, "downvoted": 0.9}
    steps = []
    tab = [
        Candidate(
            "BAYES000002", title=UPNEXT["BAYES000002"][0], channel="Stats Hub", duration_s=900
        ),
        Candidate(
            "TAB00000001", title="Bayesian priors, signed in", channel="Tab Ch", duration_s=900
        ),
        Candidate(CUR, title="the current video itself"),
    ]
    f.metas["TAB00000001"] = meta("TAB00000001", "Bayesian priors, signed in", "Tab Ch")
    res = recommend(
        store,
        PROFILE,
        video_id=CUR,
        fetcher=f,
        sponsorblock=FakeSB(),
        tagger=HeuristicTagger(),
        thresholds=th,
        paths=Paths(tmp_path),
        config=Config(min_pool=3, always_search=False),
        tab_upnext=tab,
        on_step=lambda k, t: steps.append((k, t)),
    )
    src = dict(
        store.conn.execute(
            "SELECT video_id, GROUP_CONCAT(source) FROM candidates GROUP BY video_id"
        ).fetchall()
    )
    assert src["TAB00000001"] == "tab" and set(src["BAYES000002"].split(",")) == {"upnext", "tab"}
    assert CUR not in src  # the current video is never a candidate of its own
    by = {r.candidate.video_id: r for r in res.ranked}
    assert by["BAYES000002"].parts["recurrence"] > by["BAYES000003"].parts["recurrence"]
    assert "TAB00000001" in by
    keys = [k for k, _ in steps]
    assert keys[0] == "video" and "r0" in keys and keys[-1] == "rank"
    assert any("kept" in t for k, t in steps if k == "r0")
