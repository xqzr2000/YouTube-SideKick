"""Load data for the exploration notebooks from whatever is available.

``load_candidates`` (NB01), ``load_decisions`` (NB02) and ``load_representations`` (NB03),
see each function.

Candidate sources (NB01), in ``auto`` order:

Sources, in ``auto`` order:

1. ``raw``: the raw log (``~/.sidekick/raw/*.jsonl.gz``) written by the fetcher, replayed
   through the current parser.
2. ``fixtures``: pages saved with ``just record`` (``~/.sidekick/fixtures/*.html``).
3. ``sample``: the synthetic sample in ``data/sample/pages.jsonl.gz`` (same format as the raw
   log), so the notebook and CI run without anyone's private data.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any

from sidekick.config import Paths, get_paths
from sidekick.filters.stage_a import log_decisions, run_stage_a
from sidekick.parse.search import parse_search, parse_search_data
from sidekick.parse.util import parse_age_days
from sidekick.parse.watch import parse_watch, parse_watch_data
from sidekick.profile import Profile, load_profile, parse_profile
from sidekick.store import rawlog
from sidekick.store.db import Store

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample" / "pages.jsonl.gz"
SOURCES = ("auto", "raw", "fixtures", "sample")


def _rows_for_page(kind: str, key: str, ts: str, page: Any) -> tuple[list[dict], dict | None]:
    cands = page.upnext if kind == "watch" else page.candidates
    rows = []
    for rank, c in enumerate(cands, 1):
        row = asdict(c)
        row.update(
            page_kind=kind,
            page_key=key,
            page_ts=ts,
            rank=rank,
            age_days=parse_age_days(c.published_text),
        )
        rows.append(row)
    seed = None
    if kind == "watch" and page.meta:
        seed = asdict(page.meta) | {"page_ts": ts}
    return rows, seed


def _records_from_rawlog(files: list[Path]) -> Iterator[dict]:
    for f in files:
        yield from rawlog.read(f)


def _load_records(records: Iterator[dict]) -> tuple[list[dict], list[dict], int]:
    rows, seeds, pages = [], [], 0
    for rec in records:
        kind, data = rec.get("kind"), rec.get("data") or {}
        if kind == "watch":
            page = parse_watch_data(data.get("player"), data.get("data"))
        elif kind == "search":
            page = parse_search_data(data.get("data"))
        else:
            continue
        pages += 1
        r, s = _rows_for_page(kind, rec.get("key", ""), rec.get("ts", ""), page)
        rows += r
        if s:
            seeds.append(s)
    return rows, seeds, pages


def _load_fixtures(files: list[Path]) -> tuple[list[dict], list[dict], int]:
    rows, seeds = [], []
    for f in files:
        html = f.read_text(encoding="utf-8", errors="replace")
        kind = "search" if f.name.startswith("search_") else "watch"
        page = parse_search(html) if kind == "search" else parse_watch(html)
        key = f.stem.split("_", 1)[-1]
        r, s = _rows_for_page(kind, key, "", page)
        rows += r
        if s:
            seeds.append(s)
    return rows, seeds, len(files)


def load_candidates(
    source: str = "auto", *, paths: Paths | None = None, sample_path: Path = SAMPLE_PATH
) -> dict[str, Any]:
    """Return ``{"source", "pages", "candidates": [row...], "seeds": [meta...]}``.

    Each candidate row has the ``Candidate`` fields plus ``page_kind`` (watch/search),
    ``page_key`` (video id or query), ``page_ts``, ``rank`` and ``age_days``.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    paths = paths or get_paths()
    raw_files = sorted(paths.raw_dir.glob("*.jsonl.gz")) if paths.raw_dir.is_dir() else []
    fixture_dir = paths.home / "fixtures"
    fixture_files = sorted(fixture_dir.glob("*.html")) if fixture_dir.is_dir() else []

    if source == "auto":
        source = "raw" if raw_files else "fixtures" if fixture_files else "sample"

    if source == "raw":
        rows, seeds, pages = _load_records(_records_from_rawlog(raw_files))
    elif source == "fixtures":
        rows, seeds, pages = _load_fixtures(fixture_files)
    else:
        rows, seeds, pages = _load_records(rawlog.read(sample_path))
    return {"source": source, "pages": pages, "candidates": rows, "seeds": seeds}


# -- NB02: filter decisions -----------------------------------------------------------------

DECISION_SOURCES = ("auto", "db", "sample")

_DECISIONS_SQL = """
SELECT d.compute_id, c.ts AS compute_ts, c.current_video_id, d.video_id, d.stage, d.kept,
       d.reason, d.features,
       v.title, v.channel, v.duration_s AS video_duration_s, v.lang, v.lang_conf,
       v.sponsor_frac, v.selfpromo_frac, v.category,
       (SELECT GROUP_CONCAT(DISTINCT k.source) FROM candidates k
         WHERE k.compute_id = d.compute_id AND k.video_id = d.video_id) AS sources,
       (SELECT MIN(k.position) FROM candidates k
         WHERE k.compute_id = d.compute_id AND k.video_id = d.video_id
           AND k.source = 'upnext') AS upnext_position
FROM decisions d
JOIN computes c USING (compute_id)
LEFT JOIN videos v USING (video_id)
ORDER BY c.ts, d.compute_id, d.stage, d.rowid
"""

_COMPUTES_SQL = """
SELECT c.compute_id, c.ts, c.current_video_id, c.trigger, c.status, c.pages_fetched,
       c.duration_ms, c.profile_hash,
       (SELECT COUNT(DISTINCT video_id) FROM candidates k WHERE k.compute_id = c.compute_id)
           AS unique_candidates,
       (SELECT SUM(kept) FROM decisions d WHERE d.compute_id = c.compute_id AND stage = 'A')
           AS kept_a,
       (SELECT SUM(kept) FROM decisions d WHERE d.compute_id = c.compute_id AND stage = 'B')
           AS kept_b
FROM computes c
WHERE EXISTS (SELECT 1 FROM decisions d WHERE d.compute_id = c.compute_id)
ORDER BY c.ts
"""


def _decision_rows(store) -> tuple[list[dict], list[dict]]:
    decisions = []
    for r in store.conn.execute(_DECISIONS_SQL):
        row = dict(r)
        feats = json.loads(row.pop("features") or "{}")
        row["kept"] = bool(row["kept"])
        row.update({f"f_{k}": v for k, v in feats.items()})
        decisions.append(row)
    computes = [dict(r) for r in store.conn.execute(_COMPUTES_SQL)]
    return decisions, computes


def _sample_store(sample_path: Path, profile: Profile) -> Store:
    """Replay the synthetic sample through Stage A into an in-memory database."""
    store = Store(":memory:")
    for i, rec in enumerate(rawlog.read(sample_path)):
        kind, data = rec.get("kind"), rec.get("data") or {}
        if kind == "watch":
            page = parse_watch_data(data.get("player"), data.get("data"))
            cands, source, seed = page.upnext, "upnext", rec.get("key")
            if page.meta:
                store.upsert_meta(page.meta)
        elif kind == "search":
            cands, source, seed = parse_search_data(data.get("data")).candidates, "query", None
        else:
            continue
        ts = datetime.fromisoformat(rec["ts"]) + timedelta(seconds=i)
        cid = store.start_compute(seed, trigger="sample", profile_hash=profile.hash, ts=ts)
        store.log_candidates(cid, cands, source=source, round=0 if source == "upnext" else 1)
        log_decisions(store, cid, run_stage_a(cands, profile, current_video_id=seed))
        store.finish_compute(cid, pages_fetched=1)
    return store


def load_decisions(
    source: str = "auto", *, paths: Paths | None = None, sample_path: Path = SAMPLE_PATH
) -> dict[str, Any]:
    """Return ``{"source", "profile", "decisions": [row...], "computes": [row...]}``.

    ``db`` reads ``~/.sidekick/sidekick.db``; ``sample`` replays the synthetic sample through
    Stage A with the starter profile (no enrichment). ``auto`` uses the database if it has any
    decisions. Feature values from each decision are flattened into ``f_<name>`` columns.
    """
    if source not in DECISION_SOURCES:
        raise ValueError(f"source must be one of {DECISION_SOURCES}")
    paths = paths or get_paths()
    if source == "auto":
        source = "sample"
        if paths.db.exists():
            with Store(paths.db) as s:
                if s.conn.execute("SELECT 1 FROM decisions LIMIT 1").fetchone():
                    source = "db"
    if source == "db":
        profile = load_profile(paths)
        with Store(paths.db) as s:
            decisions, computes = _decision_rows(s)
    else:
        template = resources.files("sidekick").joinpath("templates/profile.md")
        profile = parse_profile(template.read_text(encoding="utf-8"))
        with _sample_store(sample_path, profile) as s:
            decisions, computes = _decision_rows(s)
    return {"source": source, "profile": profile, "decisions": decisions, "computes": computes}


# -- NB03: representations vs labels --------------------------------------------------------

REPRESENTATION_SOURCES = ("auto", "db", "sample")

# Topic phrases of the synthetic sample (scripts/make_sample.py), used to give the sample
# synthetic labels: same topic as the current video = good, other topic = bad.
SAMPLE_TOPICS = {
    "bayes": ("Bayesian inference", "Priors explained", "MCMC by hand", "Credible intervals"),
    "rl": ("Q-learning", "Policy gradients", "Double DQN", "Reward shaping"),
    "chem": ("PCA for spectra", "PLS regression", "GC-MS peak picking", "Baseline correction"),
    "comedy": ("Stand-up set", "Crowd work", "Road stories", "Family bits"),
}
SAMPLE_PROFILE = """## Interests
- reinforcement learning
- Bayesian statistics
- chemometrics

## Don't show me
- stand-up comedy and crowd work
"""


def _sample_topic(title: str | None) -> str | None:
    for topic, phrases in SAMPLE_TOPICS.items():
        if any((title or "").startswith(p) for p in phrases):
            return topic
    return None


def _label_sample(store: Store) -> None:
    """Synthetic labels for the sample: comedy seeds are off-interest, the others on;
    a candidate is good when it shares the current video's topic, else bad."""
    seeds = [
        r[0]
        for r in store.conn.execute(
            "SELECT DISTINCT current_video_id FROM computes WHERE current_video_id IS NOT NULL"
        )
    ]
    for seed in seeds:
        topic = _sample_topic((store.get_video(seed) or {}).get("title"))
        if topic is None:
            continue
        store.add_label(seed, "bad" if topic == "comedy" else "good", kind="seed", session="sample")
        rows = store.conn.execute(
            """SELECT k.video_id, MAX(k.compute_id) FROM candidates k
               JOIN computes c ON c.compute_id = k.compute_id
               WHERE c.current_video_id = ? AND k.video_id != ? GROUP BY k.video_id""",
            (seed, seed),
        ).fetchall()
        for vid, cid in rows:
            t = _sample_topic((store.get_video(vid) or {}).get("title"))
            if t is not None:
                label = "good" if t == topic else "bad"
                store.add_label(vid, label, compute_id=cid, session="sample")


def load_representations(
    source: str = "auto",
    *,
    paths: Paths | None = None,
    sample_path: Path = SAMPLE_PATH,
    e5=None,
) -> dict[str, Any]:
    """Every representation scored against the labels (``embed.evaluate``), for NB03.

    Returns ``{"source", "profile", "results": [Result...], "thresholds", "n_labels"}``.
    ``db`` uses ``~/.sidekick/sidekick.db`` and its hand labels, with e5 (both prefix variants)
    if the model is downloaded. ``sample`` replays the synthetic sample with synthetic labels
    and a sample profile, TF-IDF and LSA only (no model download, so CI can run it). ``auto``
    picks ``db`` when the database has candidate labels.
    """
    from sidekick.embed import available
    from sidekick.embed import e5 as e5mod
    from sidekick.embed.evaluate import evaluate
    from sidekick.filters import thresholds as thmod

    if source not in REPRESENTATION_SOURCES:
        raise ValueError(f"source must be one of {REPRESENTATION_SOURCES}")
    paths = paths or get_paths()
    if source == "auto":
        source = "sample"
        if paths.db.exists():
            with Store(paths.db) as s:
                q = "SELECT 1 FROM labels WHERE kind = 'candidate' LIMIT 1"
                if s.conn.execute(q).fetchone():
                    source = "db"
    if source == "db":
        profile = load_profile(paths)
        th = thmod.load(paths)
        store = Store(paths.db)
        lines = profile.interests + [d.text for d in profile.dont_show]
        reps = available(store, lines, paths=paths, e5=e5)
        e5_rep = next((r for r in reps if r.name == e5mod.NAME), None)
        if e5_rep is not None:
            reps.append(e5_rep.with_passages())
    else:
        profile = parse_profile(SAMPLE_PROFILE)
        th = thmod.Thresholds()
        store = _sample_store(sample_path, profile)
        _label_sample(store)
        lines = profile.interests + [d.text for d in profile.dont_show]
        reps = available(store, lines, e5=e5) if e5 is not None else _corpus_reps(store, lines)
    with store:
        results = evaluate(store, reps, profile.interests, [d.text for d in profile.dont_show])
        n_labels = store.conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        titles = {r[0]: r[1] for r in store.conn.execute("SELECT video_id, title FROM videos")}
    return {
        "source": source,
        "profile": profile,
        "results": results,
        "thresholds": th,
        "n_labels": n_labels,
        "titles": titles,
    }


def _corpus_reps(store: Store, lines: list[str]) -> list:
    from sidekick.embed import corpus
    from sidekick.embed.tfidf import LsaRep, TfidfRep

    docs = corpus(store, lines)
    return [TfidfRep().fit(docs), LsaRep().fit(docs)]


# -- NB05: the labeling app --------------------------------------------------------------------

LABELING_SOURCES = ("auto", "db", "sample")


def _rank_sample(store: Store) -> None:
    """Synthetic stage R decisions for the sample, so the labeling app has pools to show: Stage
    A survivors of each watch compute, scored 1 for the current video's topic, 0.5 for no
    known topic and 0 otherwise, minus a little for position; the top 10 count as shown.
    Not SideKick's ranker; only a stand-in so the app and its tests run without a database."""
    computes = store.conn.execute(
        "SELECT compute_id, current_video_id FROM computes WHERE current_video_id IS NOT NULL"
    ).fetchall()
    with store.conn:
        for cid, seed in computes:
            topic = _sample_topic((store.get_video(seed) or {}).get("title"))
            rows = store.conn.execute(
                """SELECT d.video_id, MIN(k.position) FROM decisions d
                   JOIN candidates k ON k.compute_id = d.compute_id AND k.video_id = d.video_id
                   WHERE d.compute_id = ? AND d.stage = 'A' AND d.kept = 1
                   GROUP BY d.video_id""",
                (cid,),
            ).fetchall()
            scored = []
            for vid, pos in rows:
                t = _sample_topic((store.get_video(vid) or {}).get("title"))
                s = 0.5 if t is None else float(t == topic)
                scored.append((s - 0.01 * (pos or 0), vid))
            scored.sort(reverse=True)
            for rank, (score, vid) in enumerate(scored, 1):
                shown = rank <= 10
                store.log_decision(
                    cid,
                    vid,
                    stage="R",
                    kept=shown,
                    reason=None if shown else "rank",
                    score=score,
                    features={"rank": rank if shown else None, "sample": True},
                )


def load_labeling(
    source: str = "auto", *, paths: Paths | None = None, sample_path: Path = SAMPLE_PATH
) -> dict[str, Any]:
    """Return ``{"source", "db_path", "store", "interests"}`` for the labeling app.

    ``db`` labels into ``~/.sidekick/sidekick.db`` (``store`` is None: the app opens the file
    for each read or write, so it never holds a connection the backend might be waiting on).
    ``sample`` is an in-memory replay of the synthetic sample with stand-in rankings; its labels
    vanish when the app stops. ``auto`` uses the database if it has any ranked computes."""
    if source not in LABELING_SOURCES:
        raise ValueError(f"source must be one of {LABELING_SOURCES}")
    paths = paths or get_paths()
    if source == "auto":
        source = "sample"
        if paths.db.exists():
            with Store(paths.db) as s:
                if s.conn.execute("SELECT 1 FROM decisions WHERE stage = 'R' LIMIT 1").fetchone():
                    source = "db"
    if source == "db":
        return {
            "source": "db",
            "db_path": paths.db,
            "store": None,
            "interests": load_profile(paths).interests,
        }
    profile = parse_profile(SAMPLE_PROFILE)
    store = _sample_store(sample_path, profile)
    _rank_sample(store)
    return {"source": "sample", "db_path": None, "store": store, "interests": profile.interests}
