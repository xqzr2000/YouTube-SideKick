import json
from pathlib import Path

import numpy as np
import pytest

from sidekick import tune as tunemod
from sidekick.config import ENV_VAR, Paths
from sidekick.embed import e5 as e5mod
from sidekick.features.text_signals import clickbait, promo
from sidekick.filters import thresholds as thmod
from sidekick.filters.stage_c import build_space, interest_gate, run_stage_c
from sidekick.models import Candidate, VideoMeta
from sidekick.profile import parse_profile
from sidekick.store.db import Store

TINY = str(Path(__file__).parent / "fixtures" / "tiny_e5")
PROFILE = parse_profile(
    "## Interests\n- bayesian statistics\n- reinforcement learning\n\n"
    "## Don't show me\n- political late night show\n"
)


# -- text signals -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "lo", "hi"),
    [
        ("🚨Trump FREAKS OUT OVER BLUE WAVE!!!", 0.99, 1.0),
        ("Nobody Explained Higgs Boson Like THIS!", 0.6, 0.85),
        ("StatQuest: PCA main ideas in only 5 minutes!!!", 0.19, 0.21),
        ("17. Bayesian Statistics", 0.0, 0.0),
        ("MCMC and DQN explained", 0.0, 0.0),  # acronyms aren't shouting
        ("", 0.0, 0.0),
    ],
)
def test_clickbait(title, lo, hi):
    score, parts = clickbait(title)
    assert lo <= score <= hi
    assert set(parts) >= {"caps", "punct", "emoji", "bait"}


def test_promo():
    s, g = promo("Sponsored by Ground News - go to https://ground.news/x to get 40% off")
    assert s == pytest.approx(0.667) and g == ["sponsor", "discount"]
    assert promo("Watch hundreds of specials on the Angel Studios App!")[1] == ["app_pitch"]
    assert promo("Book Michael for your event in 2027!")[1] == ["booking"]
    assert promo("MIT 18.650 Statistics for Applications. License: CC BY-NC-SA") == (0.0, [])
    assert promo("x" * 2000 + " sponsored by y") == (0.0, [])  # only the first 1500 chars


# -- thresholds -----------------------------------------------------------------------------


def test_thresholds_defaults_and_roundtrip(tmp_path):
    paths = Paths(tmp_path)
    t = thmod.load(paths)
    assert t.source == "defaults" and t.for_rep("e5-small-qp")["gate"] == 0.10
    assert t.for_rep("unknown-rep") == thmod.DEFAULT_SIM["tfidf"]
    t.sim["e5-small-qp"]["dont_show"] = 0.22
    t.promo, t.source = 0.5, "2026-11-04T00:00:00+00:00"
    thmod.save(t, paths)
    t2 = thmod.load(paths)
    assert t2.for_rep("e5-small-qp")["dont_show"] == 0.22 and t2.promo == 0.5
    assert t2.for_rep("e5-small-qp")["gate"] == 0.10  # untouched keys keep defaults
    (tmp_path / thmod.FILE_NAME).write_text("{not json")
    assert thmod.load(paths).source.startswith("defaults")


# -- gate + Stage C ---------------------------------------------------------------------------

SEED, OFF = "SEEDBAYES01", "SEEDCOMEDY1"
VIDEOS = {
    SEED: "bayesian statistics lecture prior posterior",
    OFF: "standup comedy joke funny",
    "GOOD0000001": "bayesian inference lecture prior",
    "POLI0000001": "late night political show trump clip",
    "BAIT0000001": "🚨 SHOCKING TRUTH ABOUT PRIORS!!!",
    "PROMO000001": "probability lecture",
    "RL000000001": "reinforcement learning policy reward agent",
}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sk.db")
    for vid, title in VIDEOS.items():
        desc = (
            "Sponsored by X. Use code SAVE for 20% off. Download the X app"
            if vid.startswith("PROMO")
            else ""
        )
        s.upsert_meta(VideoMeta(vid, title=title, description=desc, channel="Chan"))
    s.commit()
    yield s
    s.close()


@pytest.fixture(params=["tfidf", "e5"])
def space(request, store):
    e5 = e5mod.E5Rep(specific_model_path=TINY) if request.param == "e5" else None
    if e5 is None:
        return build_space(store, PROFILE, paths=Paths(store.path.parent))
    return build_space(store, PROFILE, e5=e5, extra_ids=list(VIDEOS))


def test_space_is_centered_and_uses_qp(space, store):
    assert space.name in ("tfidf", e5mod.NAME_QP)
    raw, cen = space.videos(store, list(VIDEOS))
    assert np.allclose(np.linalg.norm(cen, axis=1), 1, atol=1e-4)
    assert space.interest_vecs.shape[0] == 2 and space.dont_show_vecs.shape[0] == 1


def test_gate_modes(space, store):
    on = interest_gate(store, space, SEED, tau=0.0)
    off = interest_gate(store, space, OFF, tau=0.0)
    assert on.score > off.score and on.best_interest == "bayesian statistics"
    assert interest_gate(store, space, SEED, tau=on.score).mode == "interest"
    assert interest_gate(store, space, SEED, tau=on.score + 0.01).mode == "filter-only"
    assert interest_gate(store, space, "NOTINDB0001", tau=0.1).score is None


def test_stage_c_drops_with_reasons_and_logs_scores(space, store):
    th = thmod.Thresholds()
    th.sim[space.name] = {"gate": 0.1, "dont_show": 0.45, "downvoted": 0.5}
    cands = [Candidate(v, title=t) for v, t in VIDEOS.items() if v not in (SEED, OFF)]
    d = {x.video_id: x for x in run_stage_c(store, space, cands, current_id=SEED, thresholds=th)}
    assert d["GOOD0000001"].kept and d["RL000000001"].kept
    assert d["POLI0000001"].reasons == ["dont_show:political late night show"]
    assert "clickbait" in d["BAIT0000001"].reasons
    assert d["PROMO000001"].reasons == ["promo"]
    f = d["GOOD0000001"].features
    assert f["rep"] == space.name and f["seed_sim"] > d["POLI0000001"].features["seed_sim"]
    assert f["interest_line"] == "bayesian statistics" and f["downvoted_sim"] is None
    json.dumps(f)  # features must be JSON-serializable for the decisions table


def test_downvoted_videos_become_a_filter(store):
    store.log_feedback("POLI0000001", "downvote", note="politics")
    sp = build_space(store, parse_profile("## Interests\n- x\n"), paths=Paths(store.path.parent))
    th = thmod.Thresholds()
    th.sim["tfidf"] = {"gate": 0.1, "dont_show": 0.9, "downvoted": 0.3}
    (d,) = run_stage_c(store, sp, [Candidate("POLI0000001")], current_id=None, thresholds=th)
    assert d.reasons == ["like_downvoted"] and d.features["downvoted_sim"] > 0.9


# -- tune -------------------------------------------------------------------------------------


def _labels(store):
    c1 = store.start_compute(SEED, trigger="cli")
    survivors = [Candidate(v, title=t) for v, t in VIDEOS.items() if v not in (SEED, OFF)]
    store.log_candidates(c1, survivors, source="upnext")
    for c in survivors:
        store.log_decision(c1, c.video_id, stage="B", kept=True)
    store.add_label(SEED, "good", kind="seed")
    store.add_label(OFF, "bad", kind="seed")
    for vid, lab in [
        ("GOOD0000001", "good"),
        ("RL000000001", "good"),
        ("POLI0000001", "bad"),
        ("BAIT0000001", "bad"),
        ("PROMO000001", "meh"),
    ]:
        store.add_label(vid, lab, compute_id=c1)
    store.commit()


def test_tune_is_precision_first_with_floors(store):
    _labels(store)
    new, rep, report, gate = tunemod.tune(
        store, PROFILE, thmod.Thresholds(), paths=Paths(store.path.parent)
    )
    assert rep == "tfidf"
    by = {r["signal"]: r for r in report}
    for sig in ("dont_show", "clickbait", "promo"):
        r = by[sig]
        assert r["caught"]["good"] == 0  # precision-first: no good video would be dropped
        assert r["new"] >= thmod.FLOORS[sig]
    assert by["dont_show"]["caught"]["bad"] >= 1 and by["clickbait"]["caught"]["bad"] == 1
    assert by["downvoted"]["note"].startswith("no scores")
    assert gate["yes"] and gate["no"] and gate["new"] != gate["old"]
    assert new.for_rep("tfidf")["gate"] == gate["new"]
    assert min(gate["yes"]) > gate["new"] > max(gate["no"])


def test_tune_cli_dry_run_and_write(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert tunemod.main([]) == 1  # no db
    with Store(tmp_path / "sidekick.db") as s:
        for vid, title in VIDEOS.items():
            s.upsert_meta(VideoMeta(vid, title=title))
        assert tunemod.main([]) == 1  # no labels
        _labels(s)
    (tmp_path / "profile.md").write_text(
        "## Interests\n- bayesian statistics\n\n## Don't show me\n- political late night show\n"
    )
    assert tunemod.main([]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "interest gate tau" in out
    assert not (tmp_path / thmod.FILE_NAME).exists()
    assert tunemod.main(["--write"]) == 0
    assert thmod.load(Paths(tmp_path)).source != "defaults"


# -- news context: news reports stay, pundits and late-night go (2026-11-05) ------------------

NEWS = {
    "NEWSSEED001": (
        "Will there be a tunnel turnaround? | Metro News at Six",
        "Metro News",
        N := "News & Politics",
    ),
    "NEWSREPORT1": ("late night political show trump clip report", "Metro News", N),
    "PUNDIT00001": ("🚨 late night political show TRUMP DOOMED!!!", "Pundit Now", N),
    "LATENIGHT01": ("late night political show trump clip", "Late Show", "Entertainment"),
}


def test_news_context_keeps_reports_drops_pundits_and_late_night(store):
    for vid, (title, chan, cat) in NEWS.items():
        store.upsert_meta(VideoMeta(vid, title=title, channel=chan, category=cat))
    store.commit()
    sp = build_space(store, PROFILE, paths=Paths(store.path.parent))
    th = thmod.Thresholds()
    th.sim["tfidf"] = {"gate": 0.1, "dont_show": 0.05, "downvoted": 0.9}
    th.clickbait = 1.01  # isolate the Don't-show / pundit rule
    cands = [
        Candidate(v, title=t, channel=c) for v, (t, c, _) in NEWS.items() if v != "NEWSSEED001"
    ]
    d = {
        x.video_id: x
        for x in run_stage_c(store, sp, cands, current_id="NEWSSEED001", thresholds=th)
    }
    assert d["NEWSREPORT1"].kept and d["NEWSREPORT1"].features["dont_show_spared"] == "news_report"
    assert d["PUNDIT00001"].reasons == ["pundit:political late night show"]
    assert d["LATENIGHT01"].reasons == ["dont_show:political late night show"]
    assert d["LATENIGHT01"].features["cand_news"] == ""
    # outside a news context the same report is dropped as before
    (r,) = run_stage_c(store, sp, [cands[0]], current_id=SEED, thresholds=th)
    assert r.reasons == ["dont_show:political late night show"] and "news_context" not in r.features


def test_tune_ignores_news_context_labels_for_dont_show(store):
    _labels(store)
    store.upsert_meta(
        VideoMeta(
            "NEWSSEED001",
            title="Story | Metro News at Six",
            channel="Metro News",
            category="News & Politics",
        )
    )
    store.upsert_meta(
        VideoMeta(
            "NEWSREPORT1",
            title="late night political show trump clip report",
            channel="Metro News",
            category="News & Politics",
        )
    )
    c2 = store.start_compute("NEWSSEED001", trigger="cli")
    store.log_candidates(c2, [Candidate("NEWSREPORT1")], source="upnext")
    store.add_label("NEWSREPORT1", "good", compute_id=c2)
    store.commit()
    _, _, report, _ = tunemod.tune(
        store, PROFILE, thmod.Thresholds(), paths=Paths(store.path.parent)
    )
    ds = {r["signal"]: r for r in report}["dont_show"]
    assert ds["news_skipped"] == 1 and ds["caught"]["bad"] >= 1  # the news "good" didn't raise it
    assert ds["totals"]["good"] == 2
