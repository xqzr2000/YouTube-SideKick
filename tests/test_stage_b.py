import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from sidekick.features.lang import LangGuess
from sidekick.filters.stage_b import age_days, lid_text, run_stage_b
from sidekick.models import Candidate, VideoMeta, WatchPage
from sidekick.profile import parse_profile
from sidekick.scrape.fetcher import BlockedError, PageCapReached
from sidekick.scrape.sponsorblock import (
    MAX_FAILURES,
    Segment,
    SponsorBlock,
    covered_fraction,
    hash_prefix,
)
from sidekick.store import db as dbmod
from sidekick.store.db import SCHEMA_VERSION, Store

PROFILE = parse_profile("## Rules\n- language: en\n- min_duration: 3m\n- max_duration: 90m\n")


def meta(vid, **kw):
    base = dict(
        title=f"title {vid}",
        channel="Chan",
        description="An English description https://x.y/z #tag",
        keywords=["a", "b"],
        duration_s=600,
        views=1000,
        publish_date="2026-10-01T10:00:00-07:00",
        category="Education",
    )
    base.update(kw)
    return VideoMeta(vid, **base)


class FakeFetcher:
    def __init__(self, metas, *, cap=None, fail=(), blocked=()):
        self.metas, self.cap, self.fail, self.blocked = metas, cap, set(fail), set(blocked)
        self.requested = []

    def watch(self, vid):
        if vid in self.blocked:
            raise BlockedError("bot page")
        if self.cap is not None and len(self.requested) >= self.cap:
            raise PageCapReached("cap")
        self.requested.append(vid)
        if vid in self.fail:
            raise httpx.ConnectError("boom")
        return WatchPage(meta=self.metas.get(vid), upnext=[])


class FakeSB:
    def __init__(self, segs=None, unknown=()):
        self.segs, self.unknown, self.calls = segs or {}, set(unknown), 0

    def segments(self, vid):
        self.calls += 1
        return None if vid in self.unknown else self.segs.get(vid, [])


class FakeDetector:
    def __init__(self, table):
        self.table = table

    def detect(self, text):
        for key, guess in self.table.items():
            if key in text:
                return guess
        return LangGuess("en", 0.95)


def cands(*ids):
    return [Candidate(i, title=f"title {i}") for i in ids]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sk.db")
    yield s
    s.close()


def test_enrich_decide_and_store(store):
    metas = {
        "EN000000001": meta("EN000000001"),
        "DE000000001": meta("DE000000001", description="Eine deutsche Beschreibung"),
        "LONG0000001": meta("LONG0000001", duration_s=3 * 3600),
        "LIVE0000001": meta("LIVE0000001", is_live=True),
        "SPONS000001": meta("SPONS000001"),
    }
    sb = FakeSB(
        {
            "SPONS000001": [
                Segment(0, 60, "sponsor"),
                Segment(30, 90, "sponsor"),
                Segment(500, 530, "selfpromo"),
            ]
        }
    )
    det = FakeDetector({"deutsche": LangGuess("de", 0.97)})
    decisions, stats = run_stage_b(
        cands(*metas),
        PROFILE,
        fetcher=FakeFetcher(metas),
        store=store,
        sponsorblock=sb,
        detector=det,
    )
    d = {x.video_id: x for x in decisions}
    assert {k: v.reason for k, v in d.items()} == {
        "EN000000001": None,
        "DE000000001": "lang:de",
        "LONG0000001": "duration>max",
        "LIVE0000001": "live",
        "SPONS000001": None,
    }
    sp = d["SPONS000001"].features
    assert (sp["sponsor_frac"], sp["selfpromo_frac"], sp["sponsor_segments"]) == (0.15, 0.05, 3)
    assert d["EN000000001"].features["sponsor_frac"] == 0.0
    assert d["EN000000001"].features["enriched"] and not d["EN000000001"].features["cached"]
    assert d["EN000000001"].features["category"] == "Education"
    assert stats.fetched == 5 and stats.cached == 0 and stats.sponsor_calls == 5

    row = store.get_video("DE000000001")
    assert row["lang"] == "de" and row["lang_conf"] == 0.97 and row["last_enriched"]
    assert store.get_video("SPONS000001")["sponsor_frac"] == 0.15


def test_second_run_uses_cache_for_pages_and_sponsorblock(store):
    metas = {"EN000000001": meta("EN000000001")}
    run_stage_b(
        cands("EN000000001"),
        PROFILE,
        fetcher=FakeFetcher(metas),
        store=store,
        sponsorblock=FakeSB(),
    )
    f2, sb2 = FakeFetcher(metas), FakeSB()
    decisions, stats = run_stage_b(
        cands("EN000000001"), PROFILE, fetcher=f2, store=store, sponsorblock=sb2
    )
    assert f2.requested == [] and sb2.calls == 0
    assert stats.cached == 1 and decisions[0].features["cached"]
    assert decisions[0].features["views"] == 1000 and decisions[0].features["sponsor_frac"] == 0

    # ttl=0 forces a refetch
    f3 = FakeFetcher(metas)
    run_stage_b(
        cands("EN000000001"),
        PROFILE,
        fetcher=f3,
        store=store,
        sponsorblock=FakeSB(),
        ttl=timedelta(0),
    )
    assert f3.requested == ["EN000000001"]


def test_cap_failures_and_mismatched_ids_are_kept_unenriched():
    metas = {f"V{i:010d}": meta(f"V{i:010d}") for i in range(6)}
    metas["V0000000002"] = meta("SOMEONEELSE")  # redirected page
    f = FakeFetcher(metas, fail={"V0000000001"})
    decisions, stats = run_stage_b(cands(*metas), PROFILE, fetcher=f, cap=4)
    enriched = [x.features["enriched"] for x in decisions]
    assert enriched == [True, False, False, True, False, False]
    assert all(x.kept for x in decisions)
    assert (stats.fetched, stats.failed, stats.over_cap) == (2, 2, 2)


def test_page_cap_stops_enrichment():
    metas = {f"V{i:010d}": meta(f"V{i:010d}") for i in range(4)}
    decisions, stats = run_stage_b(cands(*metas), PROFILE, fetcher=FakeFetcher(metas, cap=2))
    assert [x.features["enriched"] for x in decisions] == [True, True, False, False]
    assert stats.over_cap == 2


def test_blocked_propagates():
    with pytest.raises(BlockedError):
        run_stage_b(cands("V0000000000"), PROFILE, fetcher=FakeFetcher({}, blocked={"V0000000000"}))


def test_unknown_sponsorblock_is_none_not_zero():
    metas = {"EN000000001": meta("EN000000001")}
    decisions, stats = run_stage_b(
        cands("EN000000001"),
        PROFILE,
        fetcher=FakeFetcher(metas),
        sponsorblock=FakeSB(unknown={"EN000000001"}),
    )
    assert decisions[0].features["sponsor_frac"] is None and stats.sponsor_unknown == 1


def test_lid_text_strips_links_and_tags():
    t = lid_text(meta("X", title="Hi", description="see https://a.b/c and www.d.e #ml @me ok"))
    assert "http" not in t and "www" not in t and "#" not in t and "@" not in t
    assert t.split() == ["Hi", "see", "and", "ok"]


def test_age_days():
    now = datetime(2026, 11, 3, 17, 0, tzinfo=UTC)
    assert age_days("2026-11-01T17:00:00+00:00", now) == 2.0
    assert age_days("2026-11-02", now) == pytest.approx(1.71, abs=0.01)
    assert age_days("", now) is None and age_days("garbage", now) is None


# -- SponsorBlock client ------------------------------------------------------------------


def test_covered_fraction_merges_overlaps_and_clips():
    segs = [
        Segment(0, 60, "sponsor"),
        Segment(30, 90, "sponsor"),
        Segment(590, 700, "sponsor"),
        Segment(100, 200, "selfpromo"),
        Segment(50, 40, "sponsor"),
    ]
    assert covered_fraction(segs, 600, "sponsor") == 0.1667
    assert covered_fraction(segs, 600, "selfpromo") == 0.1667
    assert covered_fraction(segs, None, "sponsor") == 0.0
    assert covered_fraction([], 600, "sponsor") == 0.0


def _sb(handler):
    return SponsorBlock(httpx.Client(transport=httpx.MockTransport(handler)))


def test_sponsorblock_uses_hash_prefix_and_picks_our_video():
    vid = "dQw4w9WgXcQ"
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json=[
                {
                    "videoID": "otherVideo1",
                    "segments": [{"segment": [0, 5], "category": "sponsor"}],
                },
                {
                    "videoID": vid,
                    "segments": [
                        {"segment": [10.5, 40], "category": "sponsor"},
                        {"segment": "bad"},
                    ],
                },
            ],
        )

    segs = _sb(handler).segments(vid)
    assert segs == [Segment(10.5, 40.0, "sponsor")]
    url = seen[0].url
    assert url.path.endswith("/" + hashlib.sha256(vid.encode()).hexdigest()[:4])
    assert vid not in str(url)  # the exact id never leaves the machine
    assert json.loads(url.params["categories"]) == ["sponsor", "selfpromo"]
    assert hash_prefix(vid) == url.path.rsplit("/", 1)[1]


def test_sponsorblock_404_is_empty_and_errors_give_up():
    assert _sb(lambda r: httpx.Response(404)).segments("x") == []
    assert _sb(lambda r: httpx.Response(200, json=[{"videoID": "y"}])).segments("x") == []

    calls = []

    def down(request):
        calls.append(request)
        raise httpx.ConnectError("down")

    sb = _sb(down)
    results = [sb.segments("x") for _ in range(MAX_FAILURES + 2)]
    assert results == [None] * (MAX_FAILURES + 2)
    assert len(calls) == MAX_FAILURES and sb.gave_up
    assert _sb(lambda r: httpx.Response(500)).segments("x") is None


# -- schema migration ---------------------------------------------------------------------


def test_v1_database_is_migrated_in_place(tmp_path):
    path = tmp_path / "sk.db"
    conn = sqlite3.connect(path)
    conn.executescript(dbmod._SCHEMA_V1)
    conn.execute("PRAGMA user_version = 1")
    conn.execute(
        "INSERT INTO videos (video_id, title, first_seen, last_seen) VALUES ('OLD', 't', 'a', 'b')"
    )
    conn.commit()
    conn.close()
    with Store(path) as s:
        assert s.schema_version == SCHEMA_VERSION == 5
        v = s.get_video("OLD")
        assert v["title"] == "t" and v["sponsor_frac"] is None
        s.set_sponsor("OLD", sponsor_frac=0.1, selfpromo_frac=0.0, n_segments=1)
        assert s.get_video("OLD")["sponsor_segments"] == 1


def test_get_meta_respects_max_age(store):
    old = (datetime.now(UTC) - timedelta(days=3)).isoformat(timespec="seconds")
    store.upsert_meta(meta("EN000000001"), ts=old)
    assert store.get_meta("EN000000001").views == 1000
    assert store.get_meta("EN000000001", max_age=timedelta(days=1)) is None
    assert store.get_meta("NEVERSEEN01") is None
