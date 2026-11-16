import math
from datetime import UTC, datetime, timedelta

import pytest

from sidekick.config import ENV_VAR
from sidekick.models import Candidate, VideoMeta
from sidekick.rank import metrics as M
from sidekick.rank.__main__ import main as eval_main
from sidekick.rank.__main__ import report
from sidekick.rank.dataset import FEATURES, Pool, Row, load_pools, to_records
from sidekick.store.db import Store

T0 = datetime(2026, 11, 10, tzinfo=UTC)
SEED = "SEEDAAAAAAA"
SEED_JP = "SEEDJAPAN01"


def _r(**kw):
    base = {"topic": 0.0, "seed": 0.5, "interest": 0.0, "recurrence": 0.0, "same_channel": 0.0}
    base.update(persistence=0.0, views=0.5, mode="interest", kind="lecture", enriched=True)
    base.update(kw)
    return base


def _rank(s, cid, items):
    """items: (video, v0 score, shown rank or None, extra stage-R features)."""
    for vid, score, rank, extra in items:
        s.log_decision(
            cid, vid, stage="R", kept=rank is not None, score=score,
            reason=None if rank else "rank", features=_r(rank=rank, **extra),
        )  # fmt: skip


def _label(s, cid, vid, lab):
    s.add_label(vid, lab, kind="candidate", compute_id=cid, session="t")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sk.db")
    s.upsert_meta(VideoMeta(SEED, title="17. Bayesian Statistics", channel="MIT", duration_s=4000))
    s.upsert_meta(VideoMeta(SEED_JP, title="ベイズ統計と統計検定1級 何をどこまで勉強すべきか"))
    for vid, dur in (("GOOD1", 3000), ("GOOD2", 1200), ("MEH01", 600), ("BAD01", 300)):
        s.upsert_meta(
            VideoMeta(vid, title=f"t {vid}", channel="C", duration_s=dur,
                      publish_date=(T0 - timedelta(days=30)).isoformat())
        )  # fmt: skip
    # older compute for SEED (should be ignored by which="newest")
    old = s.start_compute(SEED, trigger="cli", ts=T0)
    _rank(s, old, [("GOOD1", 0.1, 1, {})])
    # newest compute for SEED: v0 put BAD01 first
    c = s.start_compute(SEED, trigger="open", ts=T0 + timedelta(hours=2))
    s.log_candidates(
        c, [Candidate("BAD01"), Candidate("GOOD1"), Candidate("MEH01")], source="upnext"
    )
    s.log_decision(c, "BAD01", stage="C", kept=True, features={"clickbait": 0.6, "promo": 0.0})
    s.log_decision(c, "GOOD1", stage="B", kept=True, features={"sponsor_frac": 0.1})
    _rank(
        s,
        c,
        [
            ("BAD01", 0.9, 1, {"lists": ["upnext"], "topic": 0.1}),
            ("GOOD1", 0.8, 2, {"lists": ["upnext", "query:bayes"], "topic": 0.9,
                               "topic_parts": {"tok": 0.7, "ent": 0.0, "series": 1.0}}),
            ("MEH01", 0.5, None, {"lists": ["upnext"], "topic": 0.3}),
            ("GOOD2", 0.4, None, {"lists": ["snowball:X"], "topic": 0.8, "enriched": False}),
            (SEED, 0.0, None, {}),  # the current video itself never counts
        ],
    )  # fmt: skip
    for vid, lab in (("BAD01", "bad"), ("GOOD1", "good"), ("MEH01", "meh"), ("GOOD2", "good")):
        _label(s, c, vid, lab)
    # a second current video, newer, non-English title, one label missing
    j = s.start_compute(SEED_JP, trigger="cli", ts=T0 + timedelta(days=1))
    _rank(s, j, [("GOOD2", 0.7, 1, {"topic": 0.0}), ("BAD01", 0.6, 2, {"topic": 0.0})])
    _label(s, j, "GOOD2", "good")
    s.commit()
    s.ids = (old, c, j)
    yield s
    s.close()


def test_load_pools_newest_and_features(store):
    old, c, j = store.ids
    pools = load_pools(store)
    assert [p.compute_id for p in pools] == [c, j]  # oldest first, newest compute per seed
    p = pools[0]
    assert [r.video_id for r in p.rows] == ["BAD01", "GOOD1", "MEH01", "GOOD2"]
    by = {r.video_id: r for r in p.rows}
    assert by["GOOD1"].gain == 2 and by["MEH01"].gain == 1 and by["BAD01"].gain == 0
    f = by["GOOD1"].features
    assert set(f) == set(FEATURES)
    assert f["topic"] == 0.9 and f["topic_series"] == 1.0 and f["sponsor_frac"] == 0.1
    assert (f["n_lists"], f["from_upnext"], f["from_search"], f["from_snowball"]) == (2, 1, 1, 0)
    assert f["upnext_pos"] == pytest.approx(1 - 1 / 3, abs=1e-3)  # 2nd of 3
    assert f["interest_mode"] == 1 and f["ctx_lecture"] == 1 and f["seed_foreign"] == 0
    assert f["duration_log"] == pytest.approx(math.log10(3000), abs=1e-3)
    assert by["BAD01"].features["clickbait"] == 0.6
    assert math.isnan(by["MEH01"].features["sponsor_frac"])  # not enriched-with-sponsor
    assert by["GOOD2"].features["enriched"] == 0 and by["GOOD2"].features["from_snowball"] == 1
    assert pools[1].rows[0].features["seed_foreign"] == 1
    assert pools[1].coverage == 0.5
    assert len(load_pools(store, which="all")) == 3


def test_metrics_by_hand():
    assert M.dcg([2, 1, 0]) == pytest.approx(2 + 1 / math.log2(3))
    assert M.ndcg([2, 1, 0]) == 1.0
    assert M.ndcg([0, 1, 2]) == pytest.approx(
        (1 / math.log2(3) + 2 / math.log2(4)) / (2 + 1 / math.log2(3))
    )
    assert M.ndcg([0, 0]) is None  # nothing good or meh: no ideal order
    assert M.precision([2, 0, 2, 1]) == 0.5 and M.bad_share([2, 0, 2, 1]) == 0.25
    rows = [Row("a", None, 0, None, {}), Row("b", "good", 0, None, {})]
    assert M.condensed_gains(rows) == [2]  # unlabeled dropped after ranking


def test_baselines_and_report(store):
    pools = load_pools(store)
    ev = M.evaluable(pools)
    assert [p.seed_id for p in ev] == [SEED]  # the half-labeled pool is skipped
    assert len(M.evaluable(pools, min_coverage=0.5)) == 2
    p = ev[0]
    assert [r.video_id for r in M.v0_shipped(p)] == ["BAD01", "GOOD1", "MEH01", "GOOD2"]
    assert [r.video_id for r in M.by_feature("topic")(p)] == ["GOOD1", "GOOD2", "MEH01", "BAD01"]
    shipped = M.per_pool(ev, M.v0_shipped)[0]
    topic = M.per_pool(ev, M.by_feature("topic"))[0]
    assert topic["ndcg"] == 1.0 and shipped["ndcg"] < 1.0
    rnd = M.random_per_pool(ev, n=50)[0]
    assert 0 < rnd["ndcg"] < 1
    lines = report(store)
    text = "\n".join(lines)
    assert "1 pools evaluable" in text and "skipped:" in text and "v0 shipped" in text
    assert any(line.startswith("topic ") for line in lines)  # feature screening row
    md = "\n".join(report(store, md=True, min_coverage=0.5))
    assert "| ranker | NDCG@10 |" in md


def test_feature_auc_direction(store):
    ev = M.evaluable(load_pools(store))
    auc = {n: a for n, a, _ in M.feature_auc(ev)}
    assert auc["topic"] == 1.0  # both good videos have the highest topic scores
    assert auc["clickbait"] != auc["clickbait"] or auc["clickbait"] <= 0.5  # nan or low


def test_bootstrap_and_paired_diff():
    m, lo, hi = M.bootstrap([0.5, 0.7, 0.9])
    assert m == pytest.approx(0.7) and lo <= m <= hi
    assert M.bootstrap([])[0] != M.bootstrap([])[0]  # nan
    d = M.paired_diff([0.9, 0.8, None], [0.5, 0.6, 0.7])
    assert d[0] == pytest.approx(0.3)


def _pool(seed, ts):
    return Pool(f"c-{seed}", seed, seed, ts, "cli", [Row("v", "good", 0, None, {})])


def test_splits_by_time_and_seed():
    pools = [_pool(s, f"2026-11-0{i}") for i, s in enumerate("abcde", 1)]
    assert M.seeds_in_time_order(pools[::-1]) == list("abcde")
    train, test = M.time_split(pools, test_frac=0.4)
    assert {p.seed_id for p in train} == {"a", "b", "c"} and {p.seed_id for p in test} == {"d", "e"}
    folds = list(M.leave_one_seed_out(pools))
    assert [s for s, _, _ in folds] == list("abcde")
    assert all(len(tr) == 4 and len(te) == 1 for _, tr, te in folds)


def test_cli_csv_and_empty(tmp_path, monkeypatch, capsys, store):
    recs = to_records(load_pools(store))
    assert {"label", "gain", "v0_score", "topic", "seed_foreign"} <= set(recs[0])
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "home"))
    assert eval_main([]) == 0
    assert "Nothing to evaluate yet" in capsys.readouterr().out
    out = tmp_path / "d.csv"
    assert eval_main(["--csv", str(out)]) == 0 and out.exists()
