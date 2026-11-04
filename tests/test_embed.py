import sqlite3
from pathlib import Path

import numpy as np
import pytest

from sidekick import embed
from sidekick.config import ENV_VAR
from sidekick.embed import __main__ as simcli
from sidekick.embed import e5 as e5mod
from sidekick.embed.text import clean, text_hash, video_text
from sidekick.embed.tfidf import LsaRep, TfidfRep
from sidekick.models import Candidate, VideoMeta
from sidekick.store import db as dbmod
from sidekick.store.db import Store

TINY = str(Path(__file__).parent / "fixtures" / "tiny_e5")

DOCS = [
    "Bayesian statistics lecture: priors, posteriors and inference",
    "Probability and Bayesian inference for beginners",
    "Reinforcement learning: policy gradients and reward",
    "Late night show: political jokes about the election",
    "Stand-up comedy special, funny jokes",
]


def test_clean_and_video_text():
    assert clean("see https://x.y/z at 1:02:03 and   www.a.b now") == "see at and now"
    t = video_text(
        {"title": "T", "channel": "C", "keywords": '["k1", "k2"]', "description": "d" * 2000}
    )
    assert t.startswith("T. C. k1 k2. d") and len(t) == len("T. C. k1 k2. ") + 1000
    assert video_text({"title": "Only title"}) == "Only title"
    assert text_hash("a") == text_hash("a") != text_hash("b")


@pytest.mark.parametrize("rep_cls", [TfidfRep, LsaRep])
def test_corpus_reps_rank_the_obvious_neighbour_first(rep_cls):
    rep = rep_cls().fit(DOCS)
    v = rep.encode(DOCS)
    assert v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    sims = v @ v.T
    np.fill_diagonal(sims, -1)
    assert sims[0].argmax() == 1  # the two Bayesian texts find each other
    assert sims[3].argmax() == 4  # late-night jokes ~ comedy jokes


def test_lsa_degenerate_corpus_falls_back_to_tfidf():
    rep = LsaRep().fit(["one lonely document"])
    assert rep.svd is None
    assert rep.encode(["lonely document"]).shape[0] == 1


def test_unfitted_reps_refuse():
    with pytest.raises(RuntimeError):
        TfidfRep().encode(["x"])
    with pytest.raises(RuntimeError):
        LsaRep().encode(["x"])


@pytest.fixture
def tiny_e5():
    return e5mod.E5Rep(specific_model_path=TINY)


def test_e5_through_fastembed_with_tiny_model(tiny_e5):
    v = tiny_e5.encode(
        ["bayesian statistics lecture", "bayesian inference prior posterior", "late night show"]
    )
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2]
    assert tiny_e5.encode([]).shape == (0, e5mod.DIM)


def test_e5_not_downloaded_means_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert not e5mod.is_downloaded() and e5mod.load() is None
    fake = e5mod.cache_dir() / "models--x" / "snapshots" / "abc" / e5mod.MODEL_FILE
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"x")
    assert e5mod.is_downloaded()


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sk.db")
    for i, d in enumerate(DOCS):
        s.upsert_candidate(Candidate(f"VID{i:08d}", title=d, channel="Chan"))
    s.commit()
    yield s
    s.close()


def test_encode_videos_caches_and_reembeds_on_text_change(store, tiny_e5):
    ids = [f"VID{i:08d}" for i in range(len(DOCS))]
    v1, st1 = embed.encode_videos(store, tiny_e5, ids)
    assert st1 == {"cached": 0, "computed": 5} and v1.shape == (5, 8)
    v2, st2 = embed.encode_videos(store, tiny_e5, ids)
    assert st2 == {"cached": 5, "computed": 0} and np.allclose(v1, v2)
    # enrichment adds a description -> that one video is embedded again
    store.upsert_meta(VideoMeta("VID00000003", description="political late night clip"))
    _, st3 = embed.encode_videos(store, tiny_e5, ids)
    assert st3 == {"cached": 4, "computed": 1}
    assert store.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 5


def test_noncacheable_reps_never_touch_the_table(store):
    rep = TfidfRep().fit(embed.corpus(store))
    v, st = embed.encode_videos(store, rep, ["VID00000000", "MISSING0000"])
    assert st["cached"] == 0 and v.shape[0] == 2 and not v[1].any()
    assert store.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0


def test_available_and_max_sim(store, tiny_e5):
    reps = embed.available(store, ["political comedy"], e5=tiny_e5)
    assert [r.name for r in reps] == ["tfidf", "lsa", e5mod.NAME]
    a = reps[0].encode(["late night political jokes"])
    assert embed.max_sim(a, reps[0].encode(["political", "bayesian"]))[0] > 0
    assert np.isnan(embed.max_sim(a, np.zeros((0, a.shape[1])))[0])


def test_v3_database_gets_text_hash(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    for script in (dbmod._SCHEMA_V1, dbmod._SCHEMA_V2, dbmod._SCHEMA_V3):
        conn.executescript(script)
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    conn.close()
    with Store(path) as s:
        assert s.schema_version == 4
        cols = {r[1] for r in s.conn.execute("PRAGMA table_info(embeddings)")}
        assert "text_hash" in cols


def test_sim_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert simcli.main([]) == 1  # no db yet
    with Store(tmp_path / "sidekick.db") as s:
        cid = s.start_compute("VID00000000", trigger="cli")
        cands = [Candidate(f"VID{i:08d}", title=d, channel="Chan") for i, d in enumerate(DOCS)]
        s.log_candidates(cid, cands, source="upnext")
    (tmp_path / "profile.md").write_text(
        "## Interests\n- bayesian statistics\n\n## Don't show me\n- political late night show\n"
    )
    assert simcli.main([]) == 0
    out = capsys.readouterr().out
    assert "model not downloaded" in out and "Interest gate preview" in out
    assert "Late night show" in out.split("Don't show me lines")[1].split("\n")[2]
    assert simcli.main(["VID00000000", "--rep", "lsa", "--k", "2"]) == 0
    out = capsys.readouterr().out
    assert "[lsa]" in out and "[tfidf]" not in out and "Probability and Bayesian" in out
    assert simcli.main(["comedy jokes", "--rep", "nope"]) == 1
