import importlib.util
from pathlib import Path

import httpx
import pytest

from sidekick.config import ENV_VAR
from sidekick.scrape import cli
from sidekick.scrape.fetcher import Fetcher

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
    assert cli.main(["https://www.youtube.com/watch?v=AAAAAAAAAAA", "--check-thumbs"]) == 0
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
        return httpx.Response(
            200, text=SEARCH_HTML if "results" in request.url.path else WATCH_HTML
        )

    def fake_fetcher(config, paths):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        return Fetcher(config, client=client, paths=paths, sleep=lambda s: None)

    monkeypatch.setattr(cli, "Fetcher", fake_fetcher)
    assert cli.main(["AAAAAAAAAAA", "--search", "bayes lecture"]) == 0
    assert "logged as compute" in capsys.readouterr().out
    with Store(tmp_path / "sidekick.db") as s:
        (c,) = s.recent_computes()
        assert c["status"] == "ok" and c["trigger"] == "cli" and c["pages_fetched"] == 2
        assert c["profile_hash"] and s.counts()["profile_versions"] == 1
        sources = dict(
            s.conn.execute("SELECT source, COUNT(*) FROM candidates GROUP BY source").fetchall()
        )
        assert sources["upnext"] == 10 and sources["query"] > 0
        (q,) = s.conn.execute("SELECT query_text, method FROM queries").fetchall()
        assert tuple(q) == ("bayes lecture", "manual")
        assert s.get_video("AAAAAAAAAAA")["last_enriched"] is not None

    assert cli.main(["AAAAAAAAAAA", "--no-db"]) == 0
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
