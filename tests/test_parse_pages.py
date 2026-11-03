from pathlib import Path

from sidekick.parse.search import parse_search
from sidekick.parse.watch import parse_watch

FIX = Path(__file__).parent / "fixtures"


def _watch():
    return parse_watch((FIX / "watch_synthetic.html").read_text())


def test_watch_meta():
    m = _watch().meta
    assert m is not None
    assert m.video_id == "AAAAAAAAAAA"
    assert m.title == "Intro to Bayesian {Statistics}; part 1"
    assert m.channel == "Synthetic Stats Channel"
    assert m.duration_s == 1523
    assert m.views == 123456
    assert m.publish_date == "2024-03-05"
    assert m.category == "Education"
    assert m.keywords == ["bayes", "statistics", "tutorial"]
    assert "} brace" in m.description
    assert not m.is_live and not m.is_upcoming


def test_watch_upnext_fields_and_nesting():
    page = _watch()
    by_id = {c.video_id: c for c in page.upnext}
    assert list(by_id) == [
        "BBBBBBBBBBB",
        "CCCCCCCCCCC",
        "DDDDDDDDDDD",
        "EEEEEEEEEEE",
        "FFFFFFFFFFF",
        "GGGGGGGGGGG",
        "MMMMMMMMMMM",
        "NNNNNNNNNNN",
        "OOOOOOOOOOO",
        "PPPPPPPPPPP",
    ]
    b = by_id["BBBBBBBBBBB"]
    assert (b.title, b.channel, b.duration_s, b.views) == ("Normal video", "Chan One", 754, 1234)
    assert b.published_text == "2 years ago" and b.source == "upnext"
    assert by_id["CCCCCCCCCCC"].duration_s == 3723
    assert by_id["CCCCCCCCCCC"].views == 1_200_000
    assert by_id["DDDDDDDDDDD"].views == 0


def test_watch_flags_and_nonvideo_skipped():
    page = _watch()
    by_id = {c.video_id: c for c in page.upnext}
    assert by_id["EEEEEEEEEEE"].is_live and by_id["EEEEEEEEEEE"].views is None
    assert by_id["FFFFFFFFFFF"].is_upcoming and not by_id["FFFFFFFFFFF"].is_live
    assert by_id["GGGGGGGGGGG"].is_short
    assert not by_id["BBBBBBBBBBB"].is_live and not by_id["BBBBBBBBBBB"].is_short
    assert page.nonvideo_skipped == 1  # the playlist/mix lockup


def test_watch_lockup_view_model_items():
    by_id = {c.video_id: c for c in _watch().upnext}
    m = by_id["MMMMMMMMMMM"]
    assert (m.title, m.channel, m.duration_s, m.views) == (
        "Lockup lecture",
        "Lockup Chan",
        4500,
        2_500_000,
    )
    assert m.published_text == "4y ago" and m.source == "upnext"
    assert not (m.is_live or m.is_upcoming or m.is_short)
    n = by_id["NNNNNNNNNNN"]
    assert n.is_live and n.views is None and n.duration_s is None
    o = by_id["OOOOOOOOOOO"]
    assert o.is_upcoming and not o.is_live
    assert by_id["PPPPPPPPPPP"].is_short
    assert "QQQQQQQQQQQ" not in by_id  # playlist skipped
    # The duplicate in lockup form doesn't replace or repeat the first entry
    assert by_id["BBBBBBBBBBB"].title == "Normal video"


def test_search_dedupes_and_counts_shorts():
    page = parse_search((FIX / "search_synthetic.html").read_text())
    assert [c.video_id for c in page.candidates] == [
        "BBBBBBBBBBB",
        "IIIIIIIIIII",
        "LLLLLLLLLLL",
    ]
    assert page.candidates[0].source == "query"
    assert page.candidates[1].channel == "Chan Four"
    assert page.candidates[2].is_live
    assert page.shorts_skipped == 2


def test_garbage_html_is_safe():
    assert parse_watch("<html>nothing</html>").meta is None
    assert parse_watch("<html>nothing</html>").upnext == []
    assert parse_search("<html>nothing</html>").candidates == []
