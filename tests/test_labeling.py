import importlib.util
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sidekick import labeling as L
from sidekick.config import ENV_VAR
from sidekick.explore import load_labeling
from sidekick.models import Candidate, VideoMeta
from sidekick.store.db import Store

ROOT = Path(__file__).resolve().parents[1]
SEED = "SEED0000001"
SEED2 = "SEED0000002"
T0 = datetime(2026, 11, 10, tzinfo=UTC)


def _cand(vid, **kw):
    return Candidate(vid, title=f"title {vid}", channel="Chan", duration_s=600, **kw)


def _ranked(s, cid, ranked):
    """Stage R decisions: ``ranked`` = [(video, shown)]."""
    for i, (vid, shown) in enumerate(ranked, 1):
        s.log_decision(
            cid, vid, stage="R", kept=shown, reason=None if shown else "rank", score=1 / i
        )


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sk.db")
    s.upsert_meta(VideoMeta(SEED, title="Bayesian Statistics", channel="MIT", duration_s=4000))
    s.upsert_meta(VideoMeta(SEED2, title="Stand-up set", channel="Dry Bar", duration_s=600))
    # seed 1, two computes (CLI, then the panel); pools overlap
    c1 = s.start_compute(SEED, trigger="cli", ts=T0)
    s.log_candidates(
        c1,
        [_cand(v) for v in ("V1", "V2", "V3", "LONG", "SHRT", "LIVE")]
        + [_cand("SHORT2", is_short=True)],
        source="upnext",
    )
    s.log_decision(c1, "LONG", stage="A", kept=False, reason="duration>max")
    s.log_decision(c1, "LIVE", stage="B", kept=False, reason="live")
    s.log_decision(c1, "SHORT2", stage="A", kept=False, reason="duration<min")
    s.log_decision(c1, "POLI", stage="C", kept=False, reason="dont_show:politics")
    s.log_decision(c1, "V3", stage="C", kept=False, reason="clickbait")  # ranked in c2
    _ranked(s, c1, [("V1", True), ("V2", False)])
    c2 = s.start_compute(SEED, trigger="open", ts=T0 + timedelta(hours=1))
    _ranked(s, c2, [("V2", True), ("V3", True), ("V4", False), (SEED, False)])
    # seed 2, older
    c3 = s.start_compute(SEED2, trigger="cli", ts=T0 - timedelta(days=1))
    _ranked(s, c3, [("C1", True), ("C2", True)])
    s.commit()
    s.ids = (c1, c2, c3)
    yield s
    s.close()


def test_ranked_pools_union_newest_compute_and_shown(store):
    c1, c2, c3 = store.ids
    pools = L.ranked_pools(store)
    assert set(pools) == {SEED, SEED2}
    assert pools[SEED] == {"V1": (c1, True), "V2": (c2, True), "V3": (c2, True), "V4": (c2, False)}
    assert SEED not in pools[SEED]  # the current video itself is never a candidate


def test_filtered_pool_skips_never_shown_shorts_and_ranked_elsewhere(store):
    pools = L.filtered_pools(store)
    # LIVE: never shown; SHORT2: a Short; V3: reached ranking in another compute
    assert set(pools) == {SEED} and set(pools[SEED]) == {"LONG", "POLI"}


def test_next_item_asks_seed_question_first_then_whole_pool(store):
    session = "t"
    it = L.next_item(store)
    assert it.kind == "seed" and it.seed_id == SEED  # newest seed first
    L.save(store, it, "good", session=session)
    seen = []
    while (it := L.next_item(store)) is not None and it.seed_id == SEED:
        assert it.kind == "candidate"
        seen.append(it.video_id)
        L.save(store, it, "meh", reason=" drifted ", session=session)
    assert sorted(seen) == ["V1", "V2", "V3", "V4"]
    # seed 1 finished: the app moves on to seed 2, seed question first
    assert it == L.Item(SEED2, SEED2, None, kind="seed")
    (p1,) = [p for p in L.progress(store) if p.seed_id == SEED]
    assert p1.complete and p1.panel and p1.seed_label == "good"
    assert (p1.ranked, p1.shown, p1.ranked_labeled, p1.counts["meh"]) == (4, 3, 4, 4)
    assert L.recent_reasons(store) == ["drifted"]


def test_order_is_stable_and_skip_moves_on(store):
    L.save(store, L.Item(SEED, SEED, None, kind="seed"), "good", session="t")
    first = L.next_item(store)
    assert L.next_item(store) == first  # reruns don't reshuffle
    nxt = L.next_item(store, skip=frozenset({(SEED, first.video_id)}))
    assert nxt.video_id != first.video_id and nxt.seed_id == SEED


def test_labels_count_per_seed_and_undo(store):
    c1, _, _ = store.ids
    lid = L.save(store, L.Item(SEED, "V1", c1), "bad", session="t")
    assert L.remaining(store, "ranked", SEED) == 3
    assert L.labeled(store) == {(SEED, "V1"): "bad"}
    store.delete_label(lid)
    assert L.remaining(store, "ranked", SEED) == 4


def test_filtered_queue_and_explicit_seed(store):
    L.save(store, L.Item(SEED2, SEED2, None, kind="seed"), "bad", session="t")
    it = L.next_item(store, "ranked", seed=SEED2)
    assert it.seed_id == SEED2 and it.video_id in {"C1", "C2"}
    L.save(store, L.Item(SEED, SEED, None, kind="seed"), "good", session="t")
    got = {L.next_item(store, "filtered").video_id}
    assert got <= {"LONG", "POLI"}
    with pytest.raises(ValueError):
        L.next_item(store, "nope")
    with pytest.raises(ValueError):
        L.save(store, it, "great", session="t")


def test_card_hides_scores(store):
    c = L.card(store, SEED)
    assert c["title"] == "Bayesian Statistics" and c["thumbnail"].endswith(f"/{SEED}/hq2.jpg")
    assert not {"score", "rank", "reason"} & set(c)


def test_load_labeling_picks_db_only_with_ranked_computes(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert load_labeling()["source"] == "sample"
    s = load_labeling("sample")["store"]
    assert L.ranked_pools(s) and L.next_item(s).kind == "seed"
    db = sqlite3.connect(tmp_path / "sidekick.db")
    s.conn.backup(db)
    db.close()
    d = load_labeling()
    assert d["source"] == "db" and d["store"] is None and d["db_path"] == tmp_path / "sidekick.db"


def test_nb05_runs_on_sample(tmp_path, monkeypatch):
    pytest.importorskip("marimo")
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    spec = importlib.util.spec_from_file_location("nb05", ROOT / "notebooks" / "05_labeling.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _, defs = mod.app.run()
    assert defs["data"]["source"] == "sample"
    assert defs["item"].kind == "seed" and defs["left"] > 0
