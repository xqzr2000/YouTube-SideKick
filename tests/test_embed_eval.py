import math
from pathlib import Path

import numpy as np
import pytest

from sidekick import embed
from sidekick.config import ENV_VAR
from sidekick.embed import __main__ as simcli
from sidekick.embed import e5 as e5mod
from sidekick.embed.evaluate import auc, candidate_labels, evaluate, report, seed_labels
from sidekick.embed.tfidf import TfidfRep
from sidekick.models import Candidate, VideoMeta
from sidekick.store.db import Store

TINY = str(Path(__file__).parent / "fixtures" / "tiny_e5")


def test_auc():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1], [2]) == 0.0
    assert auc([1, 2], [2]) == 0.25
    assert math.isnan(auc([], [1]))


def test_center_removes_common_direction():
    x = np.array([[1.0, 0.1], [1.0, -0.1]], dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    assert x[0] @ x[1] > 0.9
    c = embed.center(x, x.mean(axis=0, keepdims=True))
    assert c[0] @ c[1] == pytest.approx(-1.0, abs=1e-5)


def test_passage_variant_shares_model_and_caches_separately(tmp_path):
    rep = e5mod.E5Rep(specific_model_path=TINY)
    qp = rep.with_passages()
    assert (rep.name, qp.name) == (e5mod.NAME, e5mod.NAME_QP) and qp._model is rep._model
    with Store(tmp_path / "sk.db") as s:
        s.upsert_candidate(Candidate("VID00000001", title="bayesian lecture"))
        embed.encode_videos(s, rep, ["VID00000001"])
        embed.encode_videos(s, qp, ["VID00000001"])
        models = {r[0] for r in s.conn.execute("SELECT model FROM embeddings")}
    assert models == {e5mod.NAME, e5mod.NAME_QP}


SEED_YES, SEED_NO = "SEEDYES0001", "SEEDNO00001"


def _build(s: Store) -> None:
    s.upsert_meta(VideoMeta(SEED_YES, title="bayesian statistics lecture prior posterior"))
    s.upsert_meta(VideoMeta(SEED_NO, title="standup comedy joke"))
    c1 = s.start_compute(SEED_YES, trigger="cli")
    s.log_candidates(
        c1,
        [
            Candidate("GOOD0000001", title="bayesian statistics lecture inference"),
            Candidate("BAD00000001", title="late night political show trump clip"),
        ],
        source="upnext",
    )
    c2 = s.start_compute(SEED_NO, trigger="cli")
    s.log_candidates(c2, [Candidate("MEH00000001", title="funny comedy clip")], source="upnext")
    s.add_label(SEED_YES, "good", kind="seed", ts="2026-11-04T01:00:00+00:00")
    s.add_label(SEED_NO, "good", kind="seed", ts="2026-11-04T01:00:00+00:00")
    s.add_label(SEED_NO, "bad", kind="seed", ts="2026-11-04T02:00:00+00:00")  # newest wins
    s.add_label("GOOD0000001", "good", compute_id=c1)
    s.add_label("BAD00000001", "bad", compute_id=c1)
    s.add_label("MEH00000001", "meh", compute_id=c2)


def test_evaluate_on_tiny_labeled_store(tmp_path):
    with Store(tmp_path / "sk.db") as s:
        _build(s)
        assert seed_labels(s) == {SEED_YES: "good", SEED_NO: "bad"}
        assert len(candidate_labels(s)) == 3
        tf = TfidfRep().fit(embed.corpus(s, ["bayesian statistics", "political late night"]))
        e5 = e5mod.E5Rep(specific_model_path=TINY)
        res = evaluate(
            s, [tf, e5, e5.with_passages()], ["bayesian statistics"], ["political late night"]
        )
    assert [(r.rep, r.variant) for r in res][:2] == [("tfidf", "raw"), ("tfidf", "centered")]
    raw = res[0]
    assert raw.gate_gap > 0  # tfidf: the bayesian seed matches, the comedy seed doesn't
    assert raw.auc_dont == 1.0 and raw.auc_seed == 1.0
    assert raw.n == {"seeds_yes": 1, "seeds_no": 1, "good": 1, "meh": 1, "bad": 1}
    text = "\n".join(report(res))
    assert "seeds 1 yes / 1 no" in text and "e5-small-qp" in text


def test_sim_eval_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    with Store(tmp_path / "sidekick.db") as s:
        _build(s)
    (tmp_path / "profile.md").write_text(
        "## Interests\n- bayesian statistics\n\n## Don't show me\n- political late night\n"
    )
    assert simcli.main(["--eval"]) == 0
    out = capsys.readouterr().out
    assert "AUC dont-show" in out and "tfidf" in out and "lsa" in out


def test_auc_ci_and_recall_at_zero_fp():
    from sidekick.embed.evaluate import auc_ci, recall_at_zero_fp

    lo, hi = auc_ci([3, 4, 5, 6], [1, 2, 3.5])
    assert 0 <= lo <= auc([3, 4, 5, 6], [1, 2, 3.5]) <= hi <= 1
    assert auc_ci([1], [0]) == (1.0, 1.0)
    assert math.isnan(auc_ci([], [1])[0])
    assert recall_at_zero_fp([0.5, 0.3, 0.1], [0.2, 0.25]) == pytest.approx(2 / 3)
    assert math.isnan(recall_at_zero_fp([], [1]))


def test_sim_eval_markdown(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    with Store(tmp_path / "sidekick.db") as s:
        _build(s)
    (tmp_path / "profile.md").write_text(
        "## Interests\n- bayesian statistics\n\n## Don't show me\n- political late night\n"
    )
    assert simcli.main(["--eval", "--md"]) == 0
    out = capsys.readouterr().out
    assert "| representation | variant |" in out and "| tfidf | raw |" in out


def test_load_representations_db_and_sample(tmp_path, monkeypatch):
    from sidekick.explore import load_representations

    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert load_representations("auto")["source"] == "sample"  # no db yet
    with Store(tmp_path / "sidekick.db") as s:
        _build(s)
    (tmp_path / "profile.md").write_text(
        "## Interests\n- bayesian statistics\n\n## Don't show me\n- political late night\n"
    )
    e5 = e5mod.E5Rep(specific_model_path=TINY)
    d = load_representations("auto", e5=e5)
    assert d["source"] == "db"
    names = {(r.rep, r.variant) for r in d["results"]}
    assert ("e5-small-qp", "centered") in names and ("tfidf", "raw") in names
    r = next(x for x in d["results"] if x.rep == "tfidf" and x.variant == "raw")
    assert {lab for *_, lab, _, _ in r.scores} == {"good", "bad", "meh"}
    assert d["titles"][SEED_YES].startswith("bayesian")
