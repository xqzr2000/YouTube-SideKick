from pathlib import Path

from sidekick.parse.check import check_file, skeleton_of_first_lockup
from sidekick.scrape.record import video_id_from

FIX = Path(__file__).parent / "fixtures"


def test_check_reports_watch_and_search():
    out = "\n".join(check_file(FIX / "watch_synthetic.html"))
    assert "10 videos parsed, 1 non-video items skipped" in out
    assert "publish_date=2024-03-05" in out
    out = "\n".join(check_file(FIX / "search_synthetic.html"))
    assert "3 videos parsed, 2 shorts items" in out


def test_video_id_from_urls_and_ids():
    assert video_id_from("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert video_id_from("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=5") == "dQw4w9WgXcQ"
    assert video_id_from("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_skeleton_shows_structure():
    out = "\n".join(skeleton_of_first_lockup(FIX / "watch_synthetic.html"))
    assert "lockupMetadataViewModel:" in out and "contentId: 'MMMMMMMMMMM'" in out
