import pytest

from sidekick.config import ENV_VAR, Paths
from sidekick.features import lang as langmod
from sidekick.features.lang import LangGuess, clean_for_lid, load_detector
from sidekick.filters.stage_a import log_decisions, run_stage_a, unique_candidates
from sidekick.models import Candidate
from sidekick.profile import parse_profile
from sidekick.store.db import Store

PROFILE = parse_profile(
    "## Rules\n"
    "- language: en\n"
    "- min_duration: 3m\n"
    "- max_duration: 90m\n"
    "- exclude_channels: Jimmy Kimmel Live\n"
    "- exclude_keywords: reaction, late show\n"
)


class FakeDetector:
    """Maps exact titles to guesses; everything else is confident English."""

    def __init__(self, table):
        self.table = table
        self.seen = []

    def detect(self, text):
        self.seen.append(text)
        return self.table.get(text, LangGuess("en", 0.95))


def c(vid, title="A fine lecture", channel="Chan", duration_s=600, **kw):
    return Candidate(vid, title=title, channel=channel, duration_s=duration_s, **kw)


def by_id(decisions):
    return {d.video_id: d for d in decisions}


def test_each_rule_and_its_reason():
    cands = [
        c("KEEP0000001"),
        c("SEED0000000"),
        c("SHORT000001", is_short=True, duration_s=40),
        c("LIVE0000001", is_live=True, duration_s=None),
        c("UPCOMING001", is_upcoming=True),
        c("DOWNVOTED01"),
        c("KIMMEL00001", channel="jimmy  KIMMEL live"),
        c("KEYWORD0001", title="My REACTION to the lecture"),
        c("KEYWORD0002", title="Reactions are fine"),  # whole words only
        c("KEYWORD0003", title="Best of the Late   Show"),
        c("TOOSHORT001", duration_s=120),
        c("TOOLONG0001", duration_s=2 * 3600),
        c("NODURATION1", duration_s=None),
        c("GERMAN00001", title="Statistik für Anfänger"),
        c("UNSURE00001", title="Bayes!!"),
    ]
    det = FakeDetector(
        {"Statistik für Anfänger": LangGuess("de", 0.97), "Bayes!!": LangGuess("fr", 0.41)}
    )
    d = by_id(
        run_stage_a(
            cands, PROFILE, current_video_id="SEED0000000", downvoted={"DOWNVOTED01"}, detector=det
        )
    )
    expected = {
        "KEEP0000001": None,
        "SEED0000000": "current_video",
        "SHORT000001": "short;duration<min",
        "LIVE0000001": "live",
        "UPCOMING001": "upcoming",
        "DOWNVOTED01": "downvoted",
        "KIMMEL00001": "channel:jimmy  KIMMEL live",
        "KEYWORD0001": "keyword:reaction",
        "KEYWORD0002": None,
        "KEYWORD0003": "keyword:late show",
        "TOOSHORT001": "duration<min",
        "TOOLONG0001": "duration>max",
        "NODURATION1": None,
        "GERMAN00001": "lang:de",
        "UNSURE00001": None,  # low confidence -> kept for Stage B to re-check
    }
    assert {k: v.reason for k, v in d.items()} == expected
    assert all(v.kept == (v.reason is None) for v in d.values())
    assert d["UNSURE00001"].features["title_lang"] == "fr"
    assert d["UNSURE00001"].features["title_lang_conf"] == 0.41
    assert d["KEEP0000001"].features["duration_s"] == 600


def test_duplicates_are_judged_once_in_first_seen_order():
    cands = [c("AAAAAAAAAAA", source="upnext"), c("BBBBBBBBBBB"), c("AAAAAAAAAAA", source="query")]
    assert [x.video_id for x in unique_candidates(cands)] == ["AAAAAAAAAAA", "BBBBBBBBBBB"]
    assert len(run_stage_a(cands, PROFILE)) == 2


def test_no_detector_skips_language_and_marks_it():
    (d,) = run_stage_a([c("GERMAN00001", title="Statistik für Anfänger")], PROFILE)
    assert d.kept and d.features["title_lang"] is None


def test_empty_rules_keep_everything_but_flags():
    empty = parse_profile("")
    det = FakeDetector({"x": LangGuess("de", 0.99)})
    ds = run_stage_a(
        [c("AAAAAAAAAAA", title="x", duration_s=5), c("SHORT000001", is_short=True)],
        empty,
        detector=det,
    )
    assert [d.reason for d in ds] == [None, "short"]


def test_decisions_are_logged(tmp_path):
    with Store(tmp_path / "sk.db") as s:
        cid = s.start_compute("SEED0000000", trigger="cli")
        ds = run_stage_a([c("KEEP0000001"), c("TOOSHORT001", duration_s=10)], PROFILE)
        log_decisions(s, cid, ds)
        rows = s.conn.execute(
            "SELECT video_id, stage, kept, reason FROM decisions ORDER BY video_id"
        ).fetchall()
        assert [tuple(r) for r in rows] == [
            ("KEEP0000001", "A", 1, None),
            ("TOOSHORT001", "A", 0, "duration<min"),
        ]


def test_downvoted_ids_come_from_feedback(tmp_path):
    with Store(tmp_path / "sk.db") as s:
        s.log_feedback("AAAAAAAAAAA", "downvote", note="meh")
        s.log_feedback("BBBBBBBBBBB", "click")
        assert s.downvoted_video_ids() == {"AAAAAAAAAAA"}


def test_clean_for_lid():
    assert clean_for_lid("  BAYESIAN\nStatistics\t101 ") == "bayesian statistics 101"


def test_missing_model_gives_no_detector(tmp_path, monkeypatch):
    assert load_detector(Paths(tmp_path)) is None
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert langmod.main(["hello"]) == 1


def test_download_rejects_tiny_file(tmp_path, monkeypatch):
    import contextlib

    class FakeResp:
        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield b"<html>nope</html>"

    @contextlib.contextmanager
    def fake_stream(*a, **kw):
        yield FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "stream", fake_stream)
    with pytest.raises(RuntimeError, match="looks wrong"):
        langmod.download_model(Paths(tmp_path))
    assert not list((tmp_path / "models").iterdir())
