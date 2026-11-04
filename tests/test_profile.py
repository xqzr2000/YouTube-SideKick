from importlib import resources

import pytest

from sidekick.config import ENV_VAR, ensure_home
from sidekick.profile import append_dont_show, load_profile, parse_profile, profile_hash
from sidekick.profile.parser import parse_duration_rule
from sidekick.profile.store import dont_show_line, insert_dont_show

FULL = """\
## Interests
- reinforcement learning, Bayesian statistics, chemometrics
- time series

## Rules
- language: en, FR
- min_duration: 3m
- max_duration: 1h30m
- exclude_channels: Jimmy Kimmel Live, The Daily Show
- exclude_keywords: reaction,  prank ,

## Don't show me
- reaction videos
- political late-night clips
- 2026-11-02 — "clickbait title, no content" (video: Top 10 Tricks · Some Channel)
"""


def test_parse_full_profile():
    p = parse_profile(FULL)
    assert p.interests == [
        "reinforcement learning",
        "Bayesian statistics",
        "chemometrics",
        "time series",
    ]
    r = p.rules
    assert r.languages == ["en", "fr"]
    assert (r.min_duration_s, r.max_duration_s) == (180, 5400)
    assert r.exclude_channels == ["Jimmy Kimmel Live", "The Daily Show"]
    assert r.exclude_keywords == ["reaction", "prank"]
    assert [d.text for d in p.dont_show] == [
        "reaction videos",
        "political late-night clips",
        "clickbait title, no content",
    ]
    dated = p.dont_show[2]
    assert dated.date == "2026-11-02" and dated.video == "Top 10 Tricks · Some Channel"
    assert p.warnings == []


def test_starter_template_parses_cleanly():
    text = resources.files("sidekick").joinpath("templates/profile.md").read_text("utf-8")
    p = parse_profile(text)
    assert p.warnings == []
    assert p.rules.languages == ["en"] and p.rules.exclude_channels == []
    assert p.dont_show[0].text == "reaction videos"


@pytest.mark.parametrize(
    "value,seconds",
    [
        ("3m", 180),
        ("90m", 5400),
        ("1h30m", 5400),
        ("1h 30 min", 5400),
        ("45s", 45),
        ("1:30:00", 5400),
        ("90", 5400),
        ("2 hours", 7200),
        ("", None),
        ("soon", None),
        ("3 parsecs", None),
    ],
)
def test_parse_duration_rule(value, seconds):
    assert parse_duration_rule(value) == seconds


def test_lenient_parsing_collects_warnings():
    p = parse_profile(
        "intro text outside any section\n"
        "## Interests\n- rl\nloose line\n"
        "## Rules\n- min_duration: soon\n- colour: blue\n- no colon here\n"
        "## Wishlist\n- ignored\n"
        "## Dont show me\n* mukbang\n"
    )
    assert p.interests == ["rl"]
    assert p.rules.min_duration_s is None
    assert [d.text for d in p.dont_show] == ["mukbang"]
    joined = "\n".join(p.warnings)
    for fragment in (
        "not a '- ' bullet",
        "can't read min_duration",
        "unknown rule 'colour'",
        "not 'key: value'",
        "unknown section 'Wishlist'",
    ):
        assert fragment in joined
    assert len(p.warnings) == 5


def test_min_longer_than_max_warns():
    p = parse_profile("## Rules\n- min_duration: 2h\n- max_duration: 10m\n")
    assert any("longer than max_duration" in w for w in p.warnings)


def test_curly_apostrophe_heading():
    p = parse_profile("## Don’t show me\n- unboxing\n")
    assert [d.text for d in p.dont_show] == ["unboxing"] and p.warnings == []


def test_hash_ignores_whitespace_only_changes():
    base = profile_hash(FULL)
    assert profile_hash(FULL.replace("\n", "\r\n")) == base
    assert profile_hash(FULL.replace("time series", "time series   ") + "\n\n") == base
    assert profile_hash(FULL.replace("time series", "time-series")) != base


def test_dont_show_line_requires_note():
    with pytest.raises(ValueError):
        dont_show_line("   ")
    line = dont_show_line("too\nmuch   hype", title="A  Title", channel="Chan", date="2026-11-03")
    assert line == '- 2026-11-03 — "too much hype" (video: A Title · Chan)'
    assert parse_profile("## Don't show me\n" + line + "\n").dont_show[0].text == "too much hype"


def test_insert_dont_show_goes_at_end_of_section():
    line = '- 2026-11-03 — "x"'
    out = insert_dont_show(FULL + "\n## Notes\n- keep me\n", line)
    lines = out.splitlines()
    i = lines.index(line)
    assert lines[i - 1].startswith("- 2026-11-02")
    assert lines[i + 1] == "" and lines[i + 2] == "## Notes"

    assert insert_dont_show("## Interests\n- rl\n", line) == (
        "## Interests\n- rl\n\n## Don't show me\n" + line + "\n"
    )
    assert insert_dont_show("", line) == "## Don't show me\n" + line + "\n"


def test_quotes_in_note_survive_round_trip():
    line = dont_show_line(
        'says "AI" every 5 seconds', title="T (part 2)", channel="C", date="2026-11-03"
    )
    d = parse_profile("## Don't show me\n" + line + "\n").dont_show[0]
    assert d.text == 'says "AI" every 5 seconds' and d.video == "T (part 2) · C"


def test_append_dont_show_and_load(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    paths = ensure_home()
    before = load_profile(paths)
    line = append_dont_show(
        "same-day politics", title="Monologue", channel="Late Show", date="2026-11-03", paths=paths
    )
    after = load_profile(paths)
    assert after.dont_show[-1].text == "same-day politics"
    assert after.dont_show[-1].date == "2026-11-03"
    assert after.hash != before.hash
    assert line in paths.profile.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".profile-*"))  # temp file cleaned up


def test_load_missing_profile_warns(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "nothing"))
    p = load_profile()
    assert p.interests == [] and any("no profile" in w for w in p.warnings)
