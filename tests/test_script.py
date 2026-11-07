import numpy as np
import pytest

from sidekick.features.script import applies, non_latin
from sidekick.filters.stage_a import run_stage_a
from sidekick.models import Candidate, VideoMeta
from sidekick.profile import parse_profile
from sidekick.recommend import queries as q


@pytest.mark.parametrize(
    ("title", "script"),
    [
        ("東京の街を歩く 4K", "japanese"),
        ("【MV】夜に駆ける / YOASOBI", "japanese"),  # kana + romanized artist: still Japanese
        ("PCA 入門", "cjk"),
        ("Статистика для начинающих", "cyrillic"),
        ("한국어 강의 1", "korean"),
        ("Bayesian Statistics", None),
        ("Café crème brûlée", None),  # accented Latin is Latin
        ("Zaz - Je veux (Clip officiel)", None),  # French: left to fastText in Stage B
        ("The Great Guitars - 東京 1980", None),  # mostly Latin
        ("BTS (방탄소년단) Dynamite Official MV", None),
        ("🎸🎸 !!", None),  # no letters
        ("", None),
    ],
)
def test_non_latin(title, script):
    assert non_latin(title) == script


def test_applies_only_to_latin_script_profiles():
    assert applies(["en"]) and applies(["en", "fr"])
    assert not applies(["en", "ja"]) and not applies([])


def test_stage_a_drops_non_latin_titles_even_in_music():
    prof = parse_profile("## Rules\n- language: en\n")
    cands = [
        Candidate("JAPAN000001", title="東京の街を歩く 4K", duration_s=600),
        Candidate("JPSONG00001", title="【MV】夜に駆ける / YOASOBI", duration_s=260),
        Candidate("FRENCH00001", title="Zaz - Je veux (Clip officiel)", duration_s=220),
        Candidate("ENGLISH0001", title="Bayesian Statistics", duration_s=600),
    ]
    d = {x.video_id: x for x in run_stage_a(cands, prof)}
    assert d["JAPAN000001"].reasons == ["script:japanese"] and not d["JAPAN000001"].kept
    assert d["JPSONG00001"].features["title_script"] == "japanese"
    assert d["FRENCH00001"].kept and d["ENGLISH0001"].kept
    # a profile that allows Japanese keeps them
    both = parse_profile("## Rules\n- language: en, ja\n")
    assert all(x.kept for x in run_stage_a(cands, both))


def test_english_words_from_title_tags_description():
    m = VideoMeta(
        "JAPAN000001",
        title="【解説】PCA 主成分分析 入門 | Python",
        channel="Data Lab JP",
        description="https://example.com 今日はscikit-learnで主成分分析をします。Lab official",
        keywords=["統計", "machine learning", "MV"],
    )
    # acronyms count; channel words ("Lab"), URLs and video filler ("MV", "official") don't
    assert q.english_words(m) == ["PCA", "Python", "machine"]


def test_crosslingual_terms_vote_by_similarity():
    rng = np.random.default_rng(0)
    topic = rng.normal(size=8)
    other = rng.normal(size=8)
    vocab = [
        ("Principal component analysis explained", "Stats Hub", topic + 0.1 * rng.normal(size=8)),
        (
            "PCA for spectroscopy: principal components",
            "Chem Lab",
            topic + 0.1 * rng.normal(size=8),
        ),
        ("Principal component analysis in Python", "Stats Hub", topic + 0.2 * rng.normal(size=8)),
        ("Pizza dough at home", "Cook Hub", other),
        ("Pizza sauce secrets", "Cook Hub", other + 0.1),
    ]
    terms = q.crosslingual_terms(topic, vocab, k=3, exclude=["python"])
    low = [t.lower() for t in terms]
    assert low[:2] == ["principal", "component"]  # shared by the 3 nearest, channel words out
    assert "pizza" not in low and "hub" not in low


def test_english_round1_queries():
    m = VideoMeta("JAPAN000001", title="主成分分析 PCA 入門 Python", channel="JP")
    vocab = [
        ("Principal component analysis explained", "A", np.array([1.0, 0.0])),
        ("Principal component analysis tutorial", "B", np.array([0.9, 0.1])),
    ]
    qs = q.english_round1(m, np.array([1.0, 0.0]), vocab)
    assert qs[0] == ("PCA Python", "english")
    assert qs[1][1] == "crosslingual" and qs[1][0].startswith("PCA Principal")
    nothing = VideoMeta("JAPAN000002", title="東京の街を歩く", channel="JP")
    assert q.english_round1(nothing, None, []) == []
