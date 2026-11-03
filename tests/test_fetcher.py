from pathlib import Path

import httpx
import pytest

from sidekick.config import ENV_VAR, get_paths
from sidekick.scrape.fetcher import (
    BlockedError,
    FetchConfig,
    Fetcher,
    PageCapReached,
)
from sidekick.store import rawlog

FIX = Path(__file__).parent / "fixtures"
WATCH_HTML = (FIX / "watch_synthetic.html").read_text()
SEARCH_HTML = (FIX / "search_synthetic.html").read_text()


def make_fetcher(tmp_path, monkeypatch, handler, **cfg):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    sleeps = []
    f = Fetcher(
        FetchConfig(**cfg),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
    )
    return f, sleeps


def ok_handler(request: httpx.Request) -> httpx.Response:
    if request.url.host == "i.ytimg.com":
        return httpx.Response(200, content=b"\xff\xd8jpeg", headers={"content-type": "image/jpeg"})
    if request.url.path == "/results":
        return httpx.Response(200, text=SEARCH_HTML)
    return httpx.Response(200, text=WATCH_HTML)


def test_watch_parses_and_logs_blobs(tmp_path, monkeypatch):
    f, sleeps = make_fetcher(tmp_path, monkeypatch, ok_handler)
    page = f.watch("AAAAAAAAAAA")
    assert page.meta.video_id == "AAAAAAAAAAA"
    assert len(page.upnext) == 10
    assert sleeps == []  # no delay before the first request
    files = list(get_paths().raw_dir.glob("*.jsonl.gz"))
    (rec,) = list(rawlog.read(files[0]))
    assert rec["kind"] == "watch" and rec["key"] == "AAAAAAAAAAA"
    assert set(rec["data"]) == {"player", "data"}  # JSON blobs, not HTML


def test_delay_between_requests_and_search(tmp_path, monkeypatch):
    f, sleeps = make_fetcher(tmp_path, monkeypatch, ok_handler)
    f.watch("AAAAAAAAAAA")
    page = f.search("bayes")
    assert len(page.candidates) == 3
    assert len(sleeps) == 1 and 1.0 <= sleeps[0] <= 2.0
    assert f.pages_fetched == 2


def test_page_cap(tmp_path, monkeypatch):
    f, _ = make_fetcher(tmp_path, monkeypatch, ok_handler, page_cap=2)
    f.watch("AAAAAAAAAAA")
    f.search("q")
    with pytest.raises(PageCapReached):
        f.search("again")


def test_thumbnail_check_not_capped(tmp_path, monkeypatch):
    f, _ = make_fetcher(tmp_path, monkeypatch, ok_handler, page_cap=1)
    f.watch("AAAAAAAAAAA")
    t = f.check_thumbnail("AAAAAAAAAAA")
    assert t.status == 200 and t.content_type == "image/jpeg"
    assert t.url.endswith("/vi/AAAAAAAAAAA/hq2.jpg")
    assert f.pages_fetched == 1


def test_429_starts_backoff_and_refuses_next(tmp_path, monkeypatch):
    f, _ = make_fetcher(tmp_path, monkeypatch, lambda r: httpx.Response(429))
    with pytest.raises(BlockedError, match="429"):
        f.watch("AAAAAAAAAAA")
    assert (get_paths().home / "blocked_until").exists()
    g, _ = make_fetcher(tmp_path, monkeypatch, ok_handler)
    with pytest.raises(BlockedError, match="backing off"):
        g.watch("AAAAAAAAAAA")


def test_consent_redirect_is_blocked(tmp_path, monkeypatch):
    def handler(request):
        if request.url.host == "consent.youtube.com":
            return httpx.Response(200, text="<html>Before you continue to YouTube</html>")
        return httpx.Response(302, headers={"location": "https://consent.youtube.com/m?x=1"})

    f, _ = make_fetcher(tmp_path, monkeypatch, handler)
    f.client.follow_redirects = True
    with pytest.raises(BlockedError, match="consent"):
        f.watch("AAAAAAAAAAA")


def test_expired_backoff_is_ignored(tmp_path, monkeypatch):
    f, _ = make_fetcher(tmp_path, monkeypatch, ok_handler)
    (get_paths().home / "blocked_until").write_text("2000-01-01T00:00:00+00:00")
    assert f.blocked_until() is None
    assert f.watch("AAAAAAAAAAA").meta is not None
