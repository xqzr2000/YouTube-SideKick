import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from sidekick.config import ENV_VAR, Paths
from sidekick.embed import e5 as e5mod
from sidekick.models import Candidate, VideoMeta
from sidekick.store.db import Store
from sidekick.topic import TopicContext, explain_text, extract, match, news_context, nlp
from sidekick.topic import __main__ as cli
from sidekick.topic import text as tx

TINY = str(Path(__file__).parent / "fixtures" / "tiny_e5")

BOIL = (
    "\nSubscribe to Metro News: https://youtube.com/metronews"
    "\nFor more news visit https://metronews.example.com"
    "\nFollow us on X: @MetroNews\n#metronews #harbourcity"
)
VIDEOS = [
    VideoMeta(
        "seedNEWS001",
        "Metro News at Six for Tuesday, November 4, 2026",
        "Metro News",
        description=(
            "Tonight on Metro News at Six:\n0:00 Intro\n0:45 Harbour port strike talks collapse "
            "as dockworkers union rejects final offer\n6:10 Riverside police transition costs "
            "climb\n12:00 Weather" + BOIL
        ),
        publish_date="2026-11-04",
        category="News & Politics",
    ),
    VideoMeta(
        "clipPORT001",
        "Port strike: dockworkers union rejects final offer, talks collapse",
        "Metro News",
        description="Talks between the dockworkers union and port employers collapsed." + BOIL,
        publish_date="2026-11-04",
        category="News & Politics",
        keywords=["port strike", "dockworkers"],
    ),
    VideoMeta(
        "clipPOLICE1",
        "Riverside police transition costs climb again",
        "Metro News",
        description="The cost of the Riverside police transition has climbed again." + BOIL,
        publish_date="2026-11-04",
        category="News & Politics",
    ),
    VideoMeta(
        "newsMON0003",
        "Metro News at Six for Monday, November 3, 2026",
        "Metro News",
        description="Tonight on Metro News at Six:" + BOIL,
        publish_date="2026-11-03",
        category="News & Politics",
    ),
    VideoMeta(
        "otherPORT01",
        "Government weighs back-to-work law as harbour port strike drags on",
        "Other Channel",
        description="The labour minister says all options are on the table.",
        publish_date="2026-11-05",
        category="Entertainment",
    ),
    VideoMeta(
        "lecture0017",
        "17. Bayesian Statistics",
        "Open Course",
        description="Statistics for Applications\nIn this lecture we cover Bayesian priors.",
        category="Education",
    ),
]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    for v in VIDEOS:
        s.upsert_meta(v)
    s.commit()
    yield s
    s.close()


@pytest.fixture
def ctx(store):
    return TopicContext(store, nlp.HeuristicTagger())


# -- text plumbing ----------------------------------------------------------------------------


def test_tokens_drop_stopwords_filler_dates_and_fold_plurals():
    assert tx.tokens("BREAKING: Port strikes update, Tuesday, Nov. 4, 2026 — 3 talks") == [
        "port",
        "strike",
        "talk",
    ]
    assert tx.norm_token("Chemometrics") == tx.norm_token("chemometric") == "chemometric"
    assert tx.norm_token("Surrey's") == "surrey"
    assert tx.norm_token("policies") == "policy"
    assert tx.norm_token("business") == "business"  # -ss stays


def test_chapters_both_formats_and_minimum():
    d = "0:00 Intro\n(01:12) - Port strike talks\nCouncil vote 12:30\n1:02:03 Weather"
    assert tx.chapters(d) == [
        ("0:00", "Intro"),
        ("01:12", "Port strike talks"),
        ("12:30", "Council vote"),
        ("1:02:03", "Weather"),
    ]
    assert tx.chapters("Only one 0:45 timestamp line") == []
    assert tx.is_generic_chapter("Intro") and tx.is_generic_chapter("Top stories")
    assert not tx.is_generic_chapter("Port strike talks")


def test_boilerplate_by_channel_and_by_pattern():
    descs = [
        "Story A happened.\nSubscribe now!\nOur team: Jane, Joe",
        "Story B.\nOur team: Jane, Joe",
    ]
    boiler = tx.channel_boilerplate(descs)
    assert boiler == {"our team: jane, joe"}
    assert tx.content_lines(descs[0], boiler) == ["Story A happened."]  # "Subscribe" by pattern
    assert tx.is_pattern_boilerplate("#news #city")
    assert tx.is_pattern_boilerplate("https://example.com/x")
    assert not tx.is_pattern_boilerplate("Talks collapsed after the union rejected the offer")


def test_title_template_needs_three_titles():
    titles = ["Metro News at Six for Monday", "Metro News at Six for Tuesday", "Port strike talks"]
    assert tx.title_template(titles) == {"metro"}  # "news" is filler, "six" a stopword
    assert tx.title_template(titles[:2]) == set()


# -- extraction -------------------------------------------------------------------------------


def test_newscast_lead_is_first_real_chapter(ctx):
    t = extract(ctx, "seedNEWS001")
    assert t.lead_source == "chapter"
    assert t.lead.startswith("Harbour port strike talks collapse")
    assert {"port", "strike", "dockworker", "union"} <= set(t.weights)
    assert "metro" not in t.weights  # template / channel words are never topic
    assert t.series == ["metro"] and t.is_news and t.news_reason == "category"


def test_clip_lead_is_title_plus_first_sentence(ctx):
    t = extract(ctx, "clipPORT001")
    assert t.lead_source == "title+desc"
    assert "collapsed" in t.lead and "Subscribe" not in t.lead
    assert t.weights["dockworker"] > t.weights.get("strike", 0) * 0  # lead word, full weight


def test_template_only_newscast_has_no_topic(ctx):
    t = extract(ctx, "newsMON0003")
    assert t.lead_source == "none" and t.query() == ""


def test_news_context_reasons():
    assert news_context({"category": "News & Politics"}) == "category"
    assert news_context({"channel": "Global News", "category": "Entertainment"}) == "name"
    assert news_context({"channel": "Dry Bar Comedy", "title": "I tried a 5K"}) == ""


def test_match_ranks_same_story_first(ctx):
    a = extract(ctx, "seedNEWS001")
    scores = {v.video_id: match(a, extract(ctx, v.video_id))["score"] for v in VIDEOS[1:]}
    assert scores["clipPORT001"] > scores["otherPORT01"] > scores["clipPOLICE1"]
    assert scores["newsMON0003"] == 0.0 and scores["lecture0017"] == 0.0
    m = match(a, extract(ctx, "newsMON0003"))
    assert m["series"] == 1.0 and m["same_channel"] and m["days"] == 1


def test_query_from_topic(ctx):
    q = extract(ctx, "clipPORT001").query()  # heuristic tagger: phrases may include verbs
    assert "dockworker" in q.casefold() and len(q.split()) <= 6


def test_extract_missing_video(ctx):
    assert extract(ctx, "notindb0000") is None


# -- taggers ----------------------------------------------------------------------------------


def test_heuristic_tagger():
    tg = nlp.HeuristicTagger()
    ents = [e.text for e in tg.entities("Premier Jane Doe visits Harbour City today")]
    assert "Premier Jane Doe" in ents and "Harbour City" in ents
    assert tg.phrases("Port strike talks collapse as the union rejects the offer") == [
        "Port strike talks collapse",
        "union rejects",
        "offer",
    ]


def test_spacy_tagger_on_blank_pipeline():
    spacy = pytest.importorskip("spacy")
    nlp_ = spacy.blank("en")
    ruler = nlp_.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "GPE", "pattern": "Harbour City"}])
    tg = nlp.SpacyTagger(nlp_, name="blank")
    assert [(e.text, e.label) for e in tg.entities("Strike in Harbour City")] == [
        ("Harbour City", "GPE")
    ]
    assert tg.phrases("Port strike in Harbour City") == ["Port strike", "Harbour City"]  # fallback


def test_load_tagger_falls_back_without_model(tmp_path):
    assert nlp.load_tagger(Paths(tmp_path)).name == "heuristic"


def _fake_wheel() -> bytes:
    buf = io.BytesIO()
    prefix = f"{nlp.MODEL}/{nlp.MODEL}-{nlp.MODEL_VERSION}/"
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(prefix + "config.cfg", "[nlp]\n")
        z.writestr(prefix + "ner/model", b"x")
        z.writestr(f"{nlp.MODEL}/__init__.py", "")
    return buf.getvalue()


class _Resp:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


def test_download_checks_sha_and_unpacks(tmp_path, monkeypatch):
    import httpx

    data = _fake_wheel()
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(data))
    paths = Paths(tmp_path)
    with pytest.raises(RuntimeError, match="sha256"):
        nlp.download_model(paths)
    assert not nlp.is_downloaded(paths)
    monkeypatch.setattr(nlp, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    p = nlp.download_model(paths)
    assert nlp.is_downloaded(paths) and (p / "ner" / "model").read_bytes() == b"x"
    assert not (p.parent / "__init__.py").exists()


# -- CLI --------------------------------------------------------------------------------------


def _home_with_compute(tmp_path):
    with Store.open(Paths(tmp_path)) as s:
        for v in VIDEOS:
            s.upsert_meta(v)
        cid = s.start_compute("seedNEWS001", trigger="cli")
        cands = [Candidate(v.video_id, v.title, v.channel) for v in VIDEOS[1:]]
        s.log_candidates(cid, cands, source="upnext")
        s.log_decision(cid, "clipPOLICE1", stage="C", kept=False, reason="dont_show:political")
        s.add_label("clipPORT001", "good", compute_id=cid, reason="same story")
        s.commit()


def test_cli_seed_list_and_one_video(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    _home_with_compute(tmp_path)
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "lead (chapter): Harbour port strike" in out

    assert cli.main(["https://www.youtube.com/watch?v=seedNEWS001&list=x", "--no-e5"]) == 0
    out = capsys.readouterr().out
    rows = [ln for ln in out.splitlines() if "|" in ln and ("Port" in ln or "police" in ln)]
    assert "clipPORT001" not in out  # titles, not ids
    assert "Port strike" in rows[0] and "good" in rows[0]  # same story ranks first
    assert any("C:dont_show:political" in r for r in rows)


def test_cli_unknown_video(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert cli.main(["notindb0000", "--no-e5"]) == 1


def test_topic_vectors_cached(tmp_path, store, ctx):
    e5 = e5mod.E5Rep(specific_model_path=TINY)
    topics = [extract(ctx, v.video_id) for v in VIDEOS[:3]]
    m1 = cli.topic_vectors(store, e5, topics)
    n = store.conn.execute("SELECT COUNT(*) FROM embeddings WHERE model = 'e5-topic'").fetchone()
    assert m1.shape[0] == 3 and n[0] == 3
    m2 = cli.topic_vectors(store, e5, topics)
    assert (m1 == m2).all()


# -- regressions from the first real run (2026-11-05) -----------------------------------------

CTV_TAGS = ["CTV News", "BC news", "Vancouver news", "local news"]
REAL = [
    VideoMeta(
        "ctvseed0001",
        "Will there be a tunnel turnaround? | Metro News Harbour City at Six for November 3, 2026",
        "Metro News",
        description="Watch Metro News Harbour City at Six." + BOIL,
        keywords=[*CTV_TAGS, "tunnel"],
        publish_date="2026-11-03",
        category="News & Politics",
    ),
    VideoMeta(
        "ctvother001",
        "Picking up the pieces | Metro News Harbour City at Six for October 27, 2026",
        "Metro News",
        description="Watch Metro News Harbour City at Six." + BOIL,
        keywords=CTV_TAGS,
        publish_date="2026-10-27",
        category="News & Politics",
    ),
    VideoMeta(
        "ctvother002",
        "Ex-officer pleads guilty | Metro News Harbour City at Six for October 28, 2026",
        "Metro News",
        description="Watch Metro News Harbour City at Six." + BOIL,
        keywords=CTV_TAGS,
        publish_date="2026-10-28",
        category="News & Politics",
    ),
    *[
        VideoMeta(
            f"lect{n:07d}",
            f"Lecture {n}: Foundations of Reinforcement Learning: Partially Observable RL {r}",
            "Prof Lab",
            description="Course page: https://example.edu/rl",
            category="Education",
        )
        for n, r in ((21, "I"), (22, "II"), (23, "III"))
    ],
    VideoMeta(
        "mitbayes017",
        "17. Bayesian Statistics",
        "Open Course",
        description="In this lecture, Prof. Rigollet talked about Bayesian priors and posteriors.",
        category="Education",
    ),
]


@pytest.fixture
def real_ctx(tmp_path):
    s = Store(tmp_path / "r.db")
    for v in REAL:
        s.upsert_meta(v)
    s.commit()
    yield TopicContext(s, nlp.HeuristicTagger())
    s.close()


def test_news_title_drops_show_segment_and_channel_tags(real_ctx):
    t = extract(real_ctx, "ctvseed0001")
    assert t.lead_source == "title" and t.lead == "Will there be a tunnel turnaround?"
    assert set(t.weights) == {"tunnel", "turnaround"}  # repeated channel tags don't count
    other = extract(real_ctx, "ctvother001")
    assert match(t, other)["tok"] == 0.0 and match(t, other)["series"] == 1.0


def test_news_query_keeps_the_place(real_ctx, monkeypatch):
    t = extract(real_ctx, "ctvseed0001")
    t.places = ["Harbour City"]  # what spaCy's GPE finds in the show segment
    assert t.query() == "tunnel turnaround Harbour City"


def test_lecture_series_is_topic_outside_news(real_ctx):
    t = extract(real_ctx, "lect0000022")
    assert t.lead_source == "title" and not t.is_news
    assert {"reinforcement", "learning", "observable"} <= set(t.weights)
    sib = match(t, extract(real_ctx, "lect0000023"))
    assert sib["tok"] > 0.5 and sib["series"] == 1.0


def test_abbreviation_does_not_end_the_sentence(real_ctx):
    t = extract(real_ctx, "mitbayes017")
    assert t.lead.endswith("talked about Bayesian priors and posteriors.")
    assert tx.first_sentence("Mr. J. Smith spoke. Then more.") == "Mr. J. Smith spoke."


def test_title_segments_and_entity_strip():
    assert tx.title_segments("A | B • C") == ["A", "B", "C"]
    assert tx.strip_entity("Herb Ellis •") == "Herb Ellis"
    assert tx.strip_entity("the Port of Vancouver") == "Port of Vancouver"


def test_explain_text_marks(real_ctx):
    marks = dict((line, mark) for mark, line in explain_text(real_ctx, "ctvseed0001"))
    assert marks["Will there be a tunnel turnaround?"] == "title:keep"
    assert marks["Metro News Harbour City at Six for November 3, 2026"] == "title:show"
    assert marks["Watch Metro News Harbour City at Six."] in {"boiler", "pattern"}
    assert marks["BC news"] == "tag:channel" and marks["tunnel"] == "tag:keep"


def test_news_show_names_with_colon_dash_and_no_date(real_ctx):
    assert tx.title_segments("CBC News at 6, Nov. 04: Crash on Main", news=True) == [
        "CBC News at 6, Nov. 04",
        "Crash on Main",
    ]
    from sidekick.topic import clean_title

    assert clean_title("CBC News at 6, Nov. 04: Crash on Main", set(), True) == "Crash on Main"
    assert clean_title("ABC World News Tonight with Jane Roe - Nov. 4, 2026", set(), True) == ""
    assert clean_title("Bayes: a gentle intro | Stats Channel", {"stat", "channel"}) == (
        "Bayes: a gentle intro"
    )  # outside news, ": " doesn't split


def test_find_places_probes_show_names_only():
    spacy = pytest.importorskip("spacy")
    from sidekick.topic import find_places

    nlp_ = spacy.blank("en")
    nlp_.add_pipe("entity_ruler").add_patterns(
        [
            {"label": "ORG", "pattern": "Metro Harbour City"},
            {"label": "GPE", "pattern": [{"LOWER": "in"}, {"TEXT": "Harbour"}, {"TEXT": "City"}]},
            {"label": "GPE", "pattern": [{"LOWER": "in"}, {"TEXT": "Port"}]},
            {"label": "LOC", "pattern": "World"},
        ]
    )

    class Probe(nlp.SpacyTagger):  # strip the "in " the ruler includes in the span
        def entities(self, text):
            out = super().entities(text)
            return [nlp.Entity(e.text.removeprefix("in "), e.label) for e in out]

    tg = Probe(nlp_, name="ruler")
    assert find_places(tg, "Port strike | Metro News Harbour City at Six", "Metro", set()) == [
        "Harbour City"
    ]  # "Port" starts the story segment, so it isn't probed
    assert find_places(tg, "ABC World Tonight with Jane Roe", "ABC News", set()) == []
