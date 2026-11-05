import importlib.util
from pathlib import Path

import httpx
import pytest

from sidekick.config import ENV_VAR
from sidekick.scrape import cli
from sidekick.scrape.fetcher import Fetcher
from sidekick.scrape.sponsorblock import SponsorBlock

ROOT = Path(__file__).resolve().parents[1]
WATCH_HTML = (ROOT / "tests" / "fixtures" / "watch_synthetic.html").read_text()
SEARCH_HTML = (ROOT / "tests" / "fixtures" / "search_synthetic.html").read_text()


def test_fetch_cli_prints_table(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))

    def handler(request):
        if request.url.host == "i.ytimg.com":
            return httpx.Response(200, content=b"x" * 5000, headers={"content-type": "image/jpeg"})
        return httpx.Response(200, text=WATCH_HTML)

    def fake_fetcher(config, paths):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        return Fetcher(config, client=client, paths=paths, sleep=lambda s: None)

    monkeypatch.setattr(cli, "Fetcher", fake_fetcher)
    argv = ["https://www.youtube.com/watch?v=AAAAAAAAAAA", "--check-thumbs", "--no-enrich"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "current: AAAAAAAAAAA" in out
    assert "up-next (10)" in out
    assert "MMMMMMMMMMM" in out and "1:15:00" in out and "2.5M" in out and "1460" in out
    assert out.count("hq2.jpg") == 4
    assert "1 page(s)" in out


def test_fetch_cli_logs_compute_to_db(tmp_path, monkeypatch, capsys):
    from sidekick.store.db import Store

    monkeypatch.setenv(ENV_VAR, str(tmp_path))

    def handler(request):
        if "results" in request.url.path:
            return httpx.Response(200, text=SEARCH_HTML)
        # serve the synthetic page under whichever id was requested
        return httpx.Response(200, text=WATCH_HTML.replace("AAAAAAAAAAA", request.url.params["v"]))

    def fake_fetcher(config, paths):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        return Fetcher(config, client=client, paths=paths, sleep=lambda s: None)

    sb_requests = []

    def sb_handler(request):
        sb_requests.append(request)
        return httpx.Response(404)

    monkeypatch.setattr(cli, "Fetcher", fake_fetcher)
    monkeypatch.setattr(
        cli,
        "SponsorBlock",
        lambda: SponsorBlock(httpx.Client(transport=httpx.MockTransport(sb_handler))),
    )
    assert cli.main(["AAAAAAAAAAA", "--search", "bayes lecture", "--enrich-cap", "3"]) == 0
    out = capsys.readouterr().out
    assert "logged as compute" in out and "Stage A: kept" in out and "run `just models`" in out
    assert "Stage B: kept" in out and "over cap" in out
    assert len(sb_requests) == 3
    with Store(tmp_path / "sidekick.db") as s:
        (c,) = s.recent_computes()
        assert c["status"] == "ok" and c["trigger"] == "cli" and c["pages_fetched"] == 2 + 3
        assert c["profile_hash"] and s.counts()["profile_versions"] == 1
        sources = dict(
            s.conn.execute("SELECT source, COUNT(*) FROM candidates GROUP BY source").fetchall()
        )
        assert sources["upnext"] == 10 and sources["query"] > 0
        (q,) = s.conn.execute("SELECT query_text, method FROM queries").fetchall()
        assert tuple(q) == ("bayes lecture", "manual")
        assert s.get_video("AAAAAAAAAAA")["last_enriched"] is not None
        n_unique = s.conn.execute("SELECT COUNT(DISTINCT video_id) FROM candidates").fetchone()[0]
        stages = dict(
            s.conn.execute("SELECT stage, COUNT(*) FROM decisions GROUP BY stage").fetchall()
        )
        n_kept_a = s.conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE stage = 'A' AND kept = 1"
        ).fetchone()[0]
        n_kept_b = s.conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE stage = 'B' AND kept = 1"
        ).fetchone()[0]
        assert stages == {"A": n_unique, "B": n_kept_a, "C": n_kept_b}
        c = s.recent_computes()[0]
        assert c["mode"] in ("interest", "filter-only") and c["gate_rep"] == "tfidf"

    assert cli.main(["AAAAAAAAAAA", "--no-db", "--no-enrich"]) == 0
    with Store(tmp_path / "sidekick.db") as s:
        assert len(s.recent_computes()) == 1


def test_nb01_runs_on_sample(tmp_path, monkeypatch):
    pytest.importorskip("marimo")
    monkeypatch.setenv(ENV_VAR, str(tmp_path))  # empty home -> synthetic sample
    spec = importlib.util.spec_from_file_location("nb01", ROOT / "notebooks" / "01_candidates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _, defs = mod.app.run()
    assert len(defs["df"]) == 154


def _run_nb02():
    spec = importlib.util.spec_from_file_location("nb02", ROOT / "notebooks" / "02_filters.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _, defs = mod.app.run()
    return defs


def test_nb02_runs_on_sample(tmp_path, monkeypatch):
    pytest.importorskip("marimo")
    monkeypatch.setenv(ENV_VAR, str(tmp_path))  # no database -> sample replayed through Stage A
    defs = _run_nb02()
    assert len(defs["df"]) == 154 and set(defs["df"].stage) == {"A"}
    assert len(defs["computes"]) == 8


def test_nb02_runs_on_db_with_stage_b(tmp_path, monkeypatch, capsys):
    pytest.importorskip("marimo")
    test_fetch_cli_logs_compute_to_db(tmp_path, monkeypatch, capsys)  # builds a real-shaped db
    defs = _run_nb02()
    assert set(defs["df"].stage) == {"A", "B", "C"} and len(defs["computes"]) == 1


def _run_nb03():
    spec = importlib.util.spec_from_file_location(
        "nb03", ROOT / "notebooks" / "03_representations.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _, defs = mod.app.run()
    return defs


def test_nb03_runs_on_sample(tmp_path, monkeypatch):
    pytest.importorskip("marimo")
    monkeypatch.setenv(ENV_VAR, str(tmp_path))  # no database -> sample with synthetic labels
    defs = _run_nb03()
    reps = {(r.rep, r.variant) for r in defs["results"]}
    assert reps == {("tfidf", "raw"), ("tfidf", "centered"), ("lsa", "raw"), ("lsa", "centered")}
    assert all(r.n["good"] > 0 and r.n["bad"] > 0 for r in defs["results"])
    assert len(defs["gate_rows"]) == 16  # 4 seeds x 4 representation variants
