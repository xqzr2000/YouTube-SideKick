import pytest

from sidekick.parse.util import extract_json, find_all, parse_count, parse_duration, text_of


@pytest.mark.parametrize(
    ("text", "expected"),
    [("4:05", 245), ("1:02:03", 3723), ("0:30", 30), ("LIVE", None), ("", None)],
)
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,234 views", 1234),
        ("1.2M views", 1_200_000),
        ("15K views", 15_000),
        ("No views", 0),
        ("1,500 watching", 1500),
        ("", None),
    ],
)
def test_parse_count(text, expected):
    assert parse_count(text) == expected


def test_extract_json_handles_braces_in_strings_and_trailing_script():
    html = '<script>var x = {"a": "}{;</script>", "b": [1, 2]};var y = {};</script>'
    assert extract_json(html, "x") == {"a": "}{;</script>", "b": [1, 2]}


def test_extract_json_missing_returns_none():
    assert extract_json("<html></html>", "ytInitialData") is None


def test_text_of_and_find_all():
    assert text_of({"simpleText": "hi"}) == "hi"
    assert text_of({"runs": [{"text": "a"}, {"text": "b"}]}) == "ab"
    assert text_of(None) == ""
    assert list(find_all({"a": [{"k": 1}, {"b": {"k": 2}}]}, "k")) == [1, 2]
