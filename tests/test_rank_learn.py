"""M6c: replaying the pick step and the learned rankers, on a synthetic database where labels
follow two logged features (topic and clickbait) plus noise."""

import math
import random
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from sidekick.rank import learn as Ln
from sidekick.rank import metrics as M
from sidekick.rank import rerank as R
from sidekick.rank.__main__ import learn_report
from sidekick.rank.dataset import Pool, Row, load_pools
from sidekick.recommend import diversity as dv
from sidekick.store.db import Store

T0 = datetime(2026, 11, 1, tzinfo=UTC)
MODEL = "e5-small-qp"


def _vec(rng, topic_dir, dim=8):
    v = topic_dir + 0.3 * rng.standard_normal(dim)
    return (v / np.linalg.norm(v)).astype("<f4")


def make_store(path, n_seeds=8, pool=18, seed=0):
    rng = np.random.default_rng(seed)
    rnd = random.Random(seed)
    s = Store(path)
    dirs = [rng.standard_normal(8) for _ in range(4)]
    for k in range(n_seeds):
        cur = f"SEED{k:07d}"
        cid = s.start_compute(cur, trigger="cli", ts=T0 + timedelta(days=k))
        rows = []
        for i in range(pool):
            vid = f"V{k:02d}{i:08d}"
            topic = rnd.random()
            cb = rnd.random() * 0.5
            # the truth: on-topic and calm titles are good
            u = 2.2 * topic - 2.0 * cb + rnd.gauss(0, 0.35)
            label = "good" if u > 1.0 else "meh" if u > 0.4 else "bad"
            # v0 knows topic but not clickbait, and is noisier
            v0 = round(0.6 * topic + 0.4 * rnd.random(), 4)
            group = f"chan{i % 5}"
            rows.append((vid, v0, label, topic, cb, group))
            vec = _vec(rng, dirs[i % 4]) if label != "bad" else _vec(rng, dirs[3] * -1)
            s.conn.execute(
                "INSERT INTO embeddings (video_id, model, dim, vector, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (vid, MODEL, 8, vec.tobytes(), T0.isoformat()),
            )
        shown = {r[0] for r in sorted(rows, key=lambda r: -r[1])[:10]}
        rank = {v: i + 1 for i, v in enumerate(sorted(shown, key=lambda v: -dict(
            (r[0], r[1]) for r in rows)[v]))}  # fmt: skip
        for vid, v0, label, topic, cb, group in rows:
            s.log_decision(
                cid, vid, stage="C", kept=True,
                features={"rep": MODEL, "clickbait": cb, "seed_sim": 0.8, "promo": 0.0},
            )  # fmt: skip
            s.log_decision(
                cid, vid, stage="R", kept=vid in shown, score=v0,
                reason=None if vid in shown else "rank",
                features={"topic": topic, "seed": topic, "mode": "filter-only", "kind": "other",
                          "group": group, "lists": ["upnext"], "rank": rank.get(vid)},
            )  # fmt: skip
            s.add_label(vid, label, kind="candidate", compute_id=cid, session="t")
    s.commit()
    return s


@pytest.fixture
def store(tmp_path):
    s = make_store(tmp_path / "sk.db")
    yield s
    s.close()


def test_mmr_window_limits_outliers():
    scores = [0.9, 0.85, 0.3]
    vecs = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)  # 0 and 1 duplicates; 2 an outlier
    plain, _ = dv.mmr_select(scores, vecs, ["a", "b", "c"], k=2, cap=5, lam=0.5)
    assert plain == [0, 2]  # plain MMR promotes the outlier
    win, _ = dv.mmr_select(scores, vecs, ["a", "b", "c"], k=2, cap=5, lam=0.5, window=0.1)
    assert win == [0, 1]  # within 0.1 of the best only
    zero, _ = dv.mmr_select(scores, vecs, ["a", "b", "c"], k=3, cap=5, lam=0.5, window=0.0)
    assert zero == [0, 1, 2]  # window 0 = score order


def test_pick_variants_and_caps():
    rows = [Row(f"v{i}", "good", 1 - i / 10, None, {}, group="g" if i < 4 else f"h{i}")
            for i in range(6)]  # fmt: skip
    pool = Pool("c", "s", "s", "t", "cli", rows, kind="other")
    scores = R.v0_scores(pool)
    assert [r.video_id for r in R.pick(pool, scores, None, caps=False, mmr=False)][:3] == [
        "v0",
        "v1",
        "v2",
    ]
    capped = R.pick(pool, scores, None, mmr=False, k=4)
    assert [r.video_id for r in capped][:4] == ["v0", "v1", "v4", "v5"]  # group g capped at 2
    assert len(capped) == 6  # the rest follows by score


def test_vectors_and_replay(store):
    pools = M.evaluable(load_pools(store))
    vec = R.Vectors(store).for_pool(pools[0])
    assert vec.shape == (18, 8)
    assert np.allclose(np.linalg.norm(vec, axis=1), 1, atol=1e-5)
    assert R.Vectors(store).for_pool(Pool("c", "s", "s", "t", "cli", [], rep="")) is None
    assert 0 <= R.agreement_with_log(pools, R.Vectors(store)) <= 1


def test_models_learn_the_hidden_rule(store):
    pools = M.evaluable(load_pools(store))
    train, test = pools[:-2], pools[-2:]
    lr = Ln.fit_logreg(train)
    assert lr.importance["topic"] > 0 > lr.importance["clickbait"]
    gb = Ln.fit_lgbm(train)
    assert gb.importance["topic"] + gb.importance["clickbait"] > 0.5
    v0 = M.per_pool(test, M.v0_score)
    learned = M.per_pool(test, R.picker(lr.scorer(), None, caps=False, mmr=False))
    assert np.mean([r["ndcg"] for r in learned]) > np.mean([r["ndcg"] for r in v0])


def test_leave_one_out_and_time_split(store):
    pools = M.evaluable(load_pools(store))
    vectors = R.Vectors(store)
    ev = Ln.leave_one_out(pools, vectors)
    assert len(ev.seeds) == len(pools) == 8 and len(set(ev.seeds)) == 8
    assert all(len(v) == 8 for v in ev.results.values())
    assert len(ev.windows) == 8 and all(w in Ln.WINDOWS for w in ev.windows)
    nd = {k: np.mean([r["ndcg"] for r in v]) for k, v in ev.results.items()}
    assert nd["logistic regression"] > nd["v0 score only"]  # it sees clickbait, v0 doesn't
    ts = Ln.time_split_eval(pools, vectors)
    assert len(ts.seeds) == 2 and ts.seeds == [p.seed_id for p in pools[-2:]]
    imp = Ln.mean_importance(ev.importance["logistic regression"])
    assert {imp[0][0], imp[1][0]} >= {"clickbait"} or imp[0][0] in ("topic", "seed")
    assert not math.isnan(imp[0][1])


def test_learn_report(store):
    lines = learn_report(store)
    text = "\n".join(lines)
    assert "Leave one current video out (8 folds)" in text
    assert "LightGBM lambdarank + caps − v0 + caps" in text
    assert "Time split check: 6 older train, 2 newer test" in text
    assert "coefficient (per SD)" in text
    assert "| ranker |" in "\n".join(learn_report(store, md=True))


def test_learn_report_needs_pools(tmp_path):
    with Store(tmp_path / "empty.db") as s:
        assert "at least 3" in learn_report(s)[-1]


def test_cap_grid_order_and_choice(store):
    grid = Ln.cap_grid()
    assert grid[0] == {"other": 2, "lecture": 5} and grid[-1] == {"other": None, "lecture": None}
    assert len(grid) == len(Ln.PERFORMER_CAPS) * len(Ln.LECTURE_CAPS)
    pools = M.evaluable(load_pools(store))
    # five groups of ~4 videos each: performer 2 binds, so looser caps can only help or tie
    tight = Ln.score_caps(pools, grid[0])
    loose = Ln.score_caps(pools, grid[-1])
    assert loose["ndcg"] >= tight["ndcg"] - 1e-9 and loose["largest"] >= tight["largest"]
    assert tight["largest"] <= 2
    c = Ln.choose_caps(pools, tolerance=1.0)  # everything is "close enough": the tightest wins
    assert c == grid[0]
    ev = Ln.caps_leave_one_out(pools)
    assert len(ev.chosen) == 8 and all(len(v) == 8 for v in ev.results.values())


def test_variety_and_caps_report(store):
    rows = [Row(f"v{i}", "good", 0, None, {}, group="g" if i < 3 else "") for i in range(5)]
    assert R.variety(rows) == (3, 3)  # g, v3, v4
    from sidekick.rank.__main__ import caps_report

    text = "\n".join(caps_report(store))
    assert "Cap grid on all 8 pools" in text and "chosen − M5 caps" in text
    assert "performer 2, course 5" in text


def _lecture_pool(k: int) -> Pool:
    """8 same-course lectures (good, high score) and 4 others (bad, low score)."""
    rows = [Row(f"L{k:02d}{i:02d}", "good", 0.9 - i / 100, None, {}, group="course")
            for i in range(8)]  # fmt: skip
    rows += [Row(f"O{k:02d}{i:02d}", "bad", 0.2 - i / 100, None, {}, group=f"o{i}")
             for i in range(4)]  # fmt: skip
    return Pool(f"c{k}", f"seed{k}", f"lecture {k}", f"2026-11-{k + 1:02d}", "cli", rows,
                kind="lecture")  # fmt: skip


def test_course_cap_preregistered_rule():
    from sidekick.rank.__main__ import course_cap_section

    few = [_lecture_pool(k) for k in range(Ln.MIN_LECTURE_POOLS - 1)]
    ev = Ln.course_cap_leave_one_out(few)
    assert ev.n_pools == 7 and not ev.ready and len(ev.chosen) == 7
    assert all(c == 8 for c in ev.chosen)  # 8 fits all 8 lectures; tightest within tolerance
    text = "\n".join(course_cap_section(few, md=False))
    assert "No decision yet: 1 more labeled lecture pool(s)" in text
    many = [_lecture_pool(k) for k in range(Ln.MIN_LECTURE_POOLS)]
    text = "\n".join(course_cap_section(many, md=False))
    assert "DECISION (rule met, 8 lecture pools): course 8" in text
    assert "No lecture-context pools yet." in course_cap_section([], md=False)[-1]
