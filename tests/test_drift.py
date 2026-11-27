"""E5: relabel queue, label-stability statistics, the half-life switch and its rule."""

import math
import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from sidekick import drift
from sidekick import labeling as L
from sidekick.config import Paths
from sidekick.embed import e5 as e5mod
from sidekick.filters import thresholds as thmod
from sidekick.filters.stage_c import build_space, run_stage_c
from sidekick.models import Candidate, VideoMeta
from sidekick.profile import parse_profile
from sidekick.store import db as dbmod
from sidekick.store.db import Store

T0 = datetime(2026, 11, 4, 20, 0, tzinfo=UTC)


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sk.db")
    yield s
    s.close()


def _labeled_db(store, *, early=60, late=20, seeds=3):
    """``early`` labels in the first 3 days, ``late`` labels 20 days later, over ``seeds``
    current videos."""
    labs = ("good", "meh", "bad")
    cids = {}
    for k in range(seeds):
        seed = f"SEED{k:07d}"
        store.upsert_meta(VideoMeta(seed, title=f"seed {k}", channel="C"))
        cids[seed] = store.start_compute(seed, trigger="cli", ts=T0 - timedelta(hours=1 + k))
    for i in range(early + late):
        seed = f"SEED{i % seeds:07d}"
        vid = f"V{i:010d}"
        store.upsert_meta(VideoMeta(vid, title=f"video {i}", channel="C"))
        when = T0 + timedelta(hours=i) if i < early else T0 + timedelta(days=20, hours=i)
        store.add_label(vid, labs[i % 3], compute_id=cids[seed], ts=_iso(when))
    store.commit()
    return cids


# -- schema ------------------------------------------------------------------------------------


def test_v5_database_gets_relabels_table(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    for v in range(1, 6):
        conn.executescript(dbmod._MIGRATIONS[v])
    conn.execute("PRAGMA user_version = 5")
    conn.commit()
    conn.close()
    with Store(path) as s:
        assert s.schema_version == dbmod.SCHEMA_VERSION == 6
        assert s.counts()["relabels"] == 0


def test_relabels_follow_their_label(store):
    _labeled_db(store, early=3, late=0, seeds=1)
    lid = store.conn.execute("SELECT MIN(label_id) FROM labels").fetchone()[0]
    rid = store.add_relabel(lid, "bad", reason="  ", session="t")
    assert store.conn.execute("SELECT reason FROM relabels").fetchone()[0] is None
    store.delete_relabel(rid)
    store.add_relabel(lid, "good")
    store.delete_label(lid)  # ON DELETE CASCADE: no orphan relabels
    assert store.counts()["relabels"] == 0
    with pytest.raises(sqlite3.IntegrityError):
        store.add_relabel(lid + 999, "good")  # must point at a real label
    with pytest.raises(sqlite3.IntegrityError):
        store.add_relabel(lid + 1, "great")


# -- the relabel queue -------------------------------------------------------------------------


def test_relabel_sample_is_fixed_first_week_and_grouped(store):
    _labeled_db(store)
    a = L.relabel_sample(store)
    b = L.relabel_sample(store)
    assert a == b and len(a) == L.RELABEL_N
    ids = [it.label_id for it in a]
    assert len(set(ids)) == len(ids)
    early = {
        r[0]
        for r in store.conn.execute(
            "SELECT label_id FROM labels WHERE ts < ?", (_iso(T0 + timedelta(days=7)),)
        )
    }
    assert set(ids) <= early  # the late labels are never sampled
    seeds = [it.seed_id for it in a]
    runs = [s for i, s in enumerate(seeds) if i == 0 or seeds[i - 1] != s]
    assert len(runs) == len(set(seeds))  # one block per current video
    assert all(it.kind == "relabel" and it.compute_id for it in a)
    assert len(L.relabel_sample(store, n=1000)) == 60  # can't sample more than the window has


def test_relabel_queue_save_undo_and_labels_untouched(store):
    _labeled_db(store)
    before = L.labeled(store)
    n = L.remaining(store, "relabel", "ignored")
    it = L.next_item(store, "relabel")
    assert it.kind == "relabel" and n == L.RELABEL_N
    rid = L.save(store, it, "bad", reason="changed my mind", session="app-x")
    assert L.remaining(store, "relabel", it.seed_id) == n - 1
    assert L.next_item(store, "relabel") != it
    assert L.labeled(store) == before  # relabels never become labels
    L.undo(store, "relabel", rid)
    assert L.next_item(store, "relabel") == it
    skipped = L.next_item(store, "relabel", skip=frozenset({(it.seed_id, it.video_id)}))
    assert skipped != it
    other = L.next_item(store, "relabel", seed="SEED0000001")
    assert other.seed_id == "SEED0000001"
    # undo of an ordinary label still deletes from labels
    lid = L.save(
        store, L.Item("SEED0000000", "V0000000000", None, kind="seed"), "good", session="s"
    )
    L.undo(store, "seed", lid)
    assert store.conn.execute("SELECT 1 FROM labels WHERE label_id = ?", (lid,)).fetchone() is None


def test_relabel_queue_on_empty_db(store):
    assert L.relabel_sample(store) == [] and L.next_item(store, "relabel") is None
    assert drift.relabel_pairs(store) == []


# -- stability statistics ----------------------------------------------------------------------


def _pairs(spec):
    return [
        drift.Pair(
            f"S{i % 2}", f"V{i}", o, n, "2026-11-04T00:00:00+00:00", "2026-11-27T00:00:00+00:00"
        )
        for i, (o, n) in enumerate(spec)
    ]


def test_kappa_agreement_and_shift():
    same = _pairs([("good", "good"), ("meh", "meh"), ("bad", "bad")] * 3)
    assert drift.agreement(same) == 1 and drift.cohen_kappa(same) == pytest.approx(1)
    assert drift.mean_shift(same) == 0
    # textbook case: 2x2 with po = 0.7, pe = 0.5 → κ = 0.4
    tb = _pairs(
        [("good", "good")] * 35
        + [("bad", "bad")] * 35
        + [("good", "bad")] * 15
        + [("bad", "good")] * 15
    )
    assert drift.cohen_kappa(tb) == pytest.approx(0.4)
    # weighted: good↔meh disagreements cost half as much as good↔bad
    near = _pairs([("good", "meh"), ("meh", "good")] * 5 + [("bad", "bad")] * 10)
    far = _pairs([("good", "bad"), ("bad", "good")] * 5 + [("meh", "meh")] * 10)
    assert drift.cohen_kappa(near, weighted=True) > drift.cohen_kappa(far, weighted=True)
    stricter = _pairs([("good", "bad")] * 4 + [("meh", "meh")] * 4)
    assert drift.mean_shift(stricter) == pytest.approx(-1.0)
    assert math.isnan(drift.cohen_kappa(_pairs([("good", "good")] * 4)))  # no variety: undefined
    assert math.isnan(drift.agreement([]))
    cm = drift.confusion(stricter)
    assert cm["good"]["bad"] == 4 and cm["meh"]["meh"] == 4


def test_cluster_boot_interval_brackets_point():
    pairs = _pairs([("good", "good"), ("good", "bad"), ("bad", "bad"), ("meh", "good")] * 6)
    p, lo, hi = drift.cluster_boot(drift.by_seed(pairs), drift.agreement)
    assert lo <= p <= hi
    p1, lo1, _ = drift.cluster_boot({"only": pairs}, drift.agreement)
    assert p1 == p and math.isnan(lo1)  # one current video: no interval


def test_pairs_and_report_after_relabeling(store):
    _labeled_db(store)
    for it in L.relabel_sample(store)[:10]:
        store.add_relabel(it.label_id, "bad", ts=_iso(T0 + timedelta(days=23)))
    pairs = drift.relabel_pairs(store)
    assert len(pairs) == 10 and all(p.new == "bad" for p in pairs)
    assert all(15 < p.gap_days < 24 for p in pairs)
    out = "\n".join(drift.report(store))
    assert "relabeled 10" in out and "Cohen's κ" in out and "stricter" in out
    md = "\n".join(drift.report(store, md=True))
    assert "| measure | value [95 % CI] |" in md


# -- the half-life switch -----------------------------------------------------------------------


def test_decay_weight_and_age():
    now = datetime(2026, 11, 27, tzinfo=UTC)
    assert drift.decay_weight(None, 14) == 1.0 and drift.decay_weight(30, None) == 1.0
    assert drift.decay_weight(14, 14) == pytest.approx(0.5)
    assert drift.decay_weight(0, 7) == 1.0
    assert drift.age_days("2026-11-20", now) == pytest.approx(7)
    assert drift.age_days("2026-11-26T12:00:00+00:00", now) == pytest.approx(0.5)
    assert drift.age_days("", now) is None and drift.age_days("not a date", now) is None
    assert drift.age_days("2026-12-01", now) == 0.0  # future dates don't amplify


def test_thresholds_half_life_roundtrip(tmp_path):
    paths = Paths(tmp_path)
    assert thmod.load(paths).half_life_days is None
    t = thmod.Thresholds(half_life_days=30.0)
    thmod.save(t, paths)
    assert thmod.load(paths).half_life_days == 30.0
    t.half_life_days = None
    thmod.save(t, paths)
    assert thmod.load(paths).half_life_days is None


POLI = "POLI0000001"
PROFILE = parse_profile(
    "## Interests\n- statistics\n\n## Don't show me\n"
    '- 2026-10-01 — "late night political show" (video: x · y)\n'
)


def _space_store(store):
    for vid, title in {
        POLI: "late night political show trump clip",
        "POLI0000002": "late night political show trump clip again",
        "GOOD0000001": "bayesian statistics lecture",
    }.items():
        store.upsert_meta(VideoMeta(vid, title=title, channel="C"))
    store.commit()


def test_stage_c_without_half_life_is_unchanged(store):
    _space_store(store)
    store.log_feedback(POLI, "downvote", note="politics", ts="2026-10-01T00:00:00+00:00")
    th = thmod.Thresholds()
    th.sim["tfidf"] = {"gate": 0.1, "dont_show": 0.3, "downvoted": 0.3}
    sp = build_space(store, PROFILE, paths=Paths(store.path.parent))
    assert sp.half_life_days is None and sp.downvoted_w is None
    (d,) = run_stage_c(store, sp, [Candidate("POLI0000002")], current_id=None, thresholds=th)
    assert set(d.reasons) == {"like_downvoted", "dont_show:late night political show"}
    assert "downvoted_sim_decayed" not in d.features and "half_life_days" not in d.features


def test_stage_c_half_life_lets_old_feedback_fade(store):
    _space_store(store)
    now = datetime(2026, 11, 27, tzinfo=UTC)
    store.log_feedback(POLI, "downvote", note="politics", ts="2026-10-01T00:00:00+00:00")
    th = thmod.Thresholds()
    th.sim["tfidf"] = {"gate": 0.1, "dont_show": 0.3, "downvoted": 0.3}
    sp = build_space(store, PROFILE, paths=Paths(store.path.parent), half_life_days=7, now=now)
    assert sp.dont_show_w[0] < 0.1 and sp.downvoted_w[0] < 0.1  # 57 days old at h = 7
    (d,) = run_stage_c(store, sp, [Candidate("POLI0000002")], current_id=None, thresholds=th)
    assert d.kept  # both signals have faded below their thresholds
    f = d.features
    assert f["half_life_days"] == 7 and f["downvoted_sim"] > 0.3 > f["downvoted_sim_decayed"]
    assert f["dont_show_sim"] > f["dont_show_sim_decayed"]
    # a fresh 👎 counts in full again
    store.log_feedback(POLI, "downvote", note="still politics", ts=_iso(now - timedelta(hours=1)))
    sp = build_space(store, PROFILE, paths=Paths(store.path.parent), half_life_days=7, now=now)
    (d,) = run_stage_c(store, sp, [Candidate("POLI0000002")], current_id=None, thresholds=th)
    assert "like_downvoted" in d.reasons
    # an undated, hand-written line never decays
    hand = parse_profile("## Don't show me\n- late night political show\n")
    sp = build_space(store, hand, paths=Paths(store.path.parent), half_life_days=7, now=now)
    assert sp.dont_show_w.tolist() == [1.0]


# -- the rule ----------------------------------------------------------------------------------


def _vec(store, vid, v):
    v = np.asarray(v, np.float32)
    store.conn.execute(
        "INSERT INTO embeddings (video_id, model, dim, vector, created_at) VALUES (?, ?, ?, ?, ?)",
        (vid, e5mod.NAME_QP, len(v), v.astype("<f4").tobytes(), "x"),
    )


def _rule_db(store, *, fresh_matters: bool):
    """Two 👎: an old one (day 0) and a new one (day 30). Labels at day 31. If
    ``fresh_matters``, bad videos look like the new 👎 and good ones like the old one; so decay
    helps. Otherwise bad videos look like the old 👎."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(40, 8)).astype(np.float32)  # background so centering is sane
    for i, v in enumerate(base):
        _vec(store, f"BG{i:09d}", v)
    old_d, new_d = np.eye(8)[0] * 5, np.eye(8)[1] * 5
    _vec(store, "DOWNOLD0001", old_d)
    _vec(store, "DOWNNEW0001", new_d)
    store.log_feedback("DOWNOLD0001", "downvote", ts=_iso(T0))
    store.log_feedback("DOWNNEW0001", "downvote", ts=_iso(T0 + timedelta(days=30)))
    for k in range(6):
        seed = f"SEED{k:07d}"
        cid = store.start_compute(seed, trigger="cli", ts=T0 + timedelta(days=31, hours=k))
        for j in range(4):
            bad = j % 2 == 0
            near = new_d if (bad == fresh_matters) else old_d
            vid = f"L{k}{j}{'B' if bad else 'G'}00000"[:11]
            _vec(store, vid, near * (0.9 + 0.02 * j) + rng.normal(size=8) * 0.3)
            store.add_label(vid, "bad" if bad else "good", compute_id=cid)
    store.commit()


def test_halflife_auc_prefers_decay_when_fresh_feedback_matters(store):
    _rule_db(store, fresh_matters=True)
    items = drift.downvote_dataset(store)
    assert len(items) == 24 and all(len(x.sims) == 2 for x in items)
    assert drift.auc(items, 7.0) > drift.auc(items, None)
    table = drift.halflife_table(items)
    assert [r["h"] for r in table] == list(drift.HALF_LIVES)
    assert drift.decide(table) is not None


def test_halflife_rule_stays_off_when_decay_does_not_help(store):
    _rule_db(store, fresh_matters=False)
    table = drift.halflife_table(drift.downvote_dataset(store))
    assert drift.decide(table) is None


def test_rule_needs_enough_downvotes_and_span(store):
    _rule_db(store, fresh_matters=True)
    met, n, span = drift.rule_met(store)
    assert not met and n == 2 and span == pytest.approx(30)
    out = "\n".join(drift.report(store))
    assert "not met yet" in out and "PREVIEW" not in out and "AUC" not in out
    prev = "\n".join(drift.report(store, preview=True))
    assert "PREVIEW, not for deciding" in prev and "AUC bad > good" in prev
    for i in range(drift.RULE_MIN_DOWNVOTES):
        store.log_feedback(f"X{i:010d}", "downvote", ts=_iso(T0 + timedelta(days=i % 29)))
    assert drift.rule_met(store)[0]
    assert "decision:" in "\n".join(drift.report(store))


def test_e5_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SIDEKICK_HOME", str(tmp_path))
    assert drift.main([]) == 1
    Store(tmp_path / "sidekick.db").close()
    assert drift.main(["--md"]) == 0
    assert "sample 0 labels" in capsys.readouterr().out
