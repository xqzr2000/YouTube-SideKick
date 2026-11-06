"""One compute, end to end (spec §5): ``just recommend <video>`` → 10 videos.

1. **Round 0:** the current video's watch page (signed out) gives its metadata and the up-next
   list. The interest gate picks the mode and the topic anchor is extracted.
2. Every round runs the same stages on the candidates it hasn't seen before:
   **Stage A** (hard filters) → **pre-score** on cheap fields (:mod:`.score`) → **Stage B** on the
   best-scoring ones until the compute's enrichment budget is spent (the rest pass through
   unenriched) → **Stage C** (soft filters).
3. **Round 1 always searches** (Russell, 2026-11-05: on the Chi Jin seed it added several good
   picks for +17 s) with the topic query, plus an Interests-crossed query in interest mode.
4. **Rounds 2–3** only while fewer than ``min_pool`` videos survive and the compute is still
   under ``deadline_s``: a Rocchio-expanded query and snowball lists (up-next of the best
   survivors, taken from watch pages already fetched for enrichment).
5. **Pick:** the v0 score over all survivors, then MMR with group caps
   (:mod:`.diversity`: same course ≤ 5 in a lecture context, same performer/show ≤ 2 elsewhere).

Everything is logged: queries, candidates (source, round, query, position), decisions for
stages A/B/C, and the pick as stage ``R`` (``kept`` = shown, ``score`` = v0, features = all
parts, group, MMR value and the rank). Pre-scores are stored on the Stage B decisions.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from sidekick.filters import thresholds as thmod
from sidekick.filters.stage_a import log_decisions, run_stage_a
from sidekick.filters.stage_b import run_stage_b
from sidekick.filters.stage_c import GateResult, build_space, interest_gate, run_stage_c
from sidekick.models import Candidate, VideoMeta
from sidekick.recommend import diversity as dv
from sidekick.recommend import queries as qmod
from sidekick.recommend import score as sc
from sidekick.store.db import Store
from sidekick.topic import Topic, TopicContext, extract, match
from sidekick.topic.nlp import Tagger, load_tagger

STAGE_RANK = "R"


@dataclass
class Config:
    min_pool: int = 10  # rounds 2-3 run while fewer survivors than this
    max_rounds: int = 3  # search rounds after the up-next list (spec: at most 3)
    always_search: bool = True  # round 1 runs even when up-next alone is enough
    deadline_s: float = 60.0  # don't start round 2 or 3 after this many seconds
    enrich_cap: int = 25  # candidates considered for enrichment per compute (cache hits count)
    top_k: int = 10
    lam: float = dv.LAMBDA  # MMR: relevance vs. variety
    caps: dict[str, int] = field(default_factory=lambda: dict(dv.CAPS))
    snowball_from: int = 2  # best survivors whose up-next is added in rounds 2-3
    cache_ttl: timedelta | None = None  # enrichment cache age; timedelta(0) = cold (E3)


@dataclass
class RoundLog:
    round: int
    queries: list[tuple[str, str, int]] = field(default_factory=list)  # (query, method, n)
    snowball: list[tuple[str, int]] = field(default_factory=list)  # (from video, n)
    n_candidates: int = 0
    n_new: int = 0
    a_kept: int = 0
    b_fetched: int = 0
    b_cached: int = 0
    b_over_cap: int = 0
    b_dropped: int = 0
    c_kept: int = 0
    seconds: float = 0.0


@dataclass
class Ranked:
    candidate: Candidate
    score: float
    parts: dict[str, Any]
    rank: int | None  # 1..top_k when shown, else None
    reason: str | None = None  # why not shown (rank / group_cap)


@dataclass
class Result:
    compute_id: str | None
    current: VideoMeta | None
    gate: GateResult | None = None
    anchor: Topic | None = None
    rounds: list[RoundLog] = field(default_factory=list)
    ranked: list[Ranked] = field(default_factory=list)
    backfill_skipped: str = ""
    kind: str = "other"  # lecture | other (diversity caps)
    pages: int = 0
    seconds: float = 0.0
    status: str = "ok"

    @property
    def shown(self) -> list[Ranked]:
        return [r for r in self.ranked if r.rank is not None]


class _Pool:
    """What one compute has seen so far: first-seen candidates, lists, positions."""

    def __init__(self) -> None:
        self.cands: dict[str, Candidate] = {}
        self.lists: dict[str, set[str]] = defaultdict(set)  # vid -> list names
        self.best_pos: dict[str, tuple[int, int]] = {}  # vid -> (position, list length)
        self.c_feats: dict[str, dict[str, Any]] = {}  # Stage C features of survivors
        self.enriched: dict[str, bool] = {}
        self.upnext: dict[str, list[Candidate]] = {}  # up-next of pages fetched in Stage B
        self.snowballed: set[str] = set()

    def add_list(self, name: str, items: list[Candidate]) -> list[Candidate]:
        new = []
        for pos, c in enumerate(items, 1):
            if c.video_id not in self.cands:
                self.cands[c.video_id] = c
                new.append(c)
            self.lists[c.video_id].add(name)
            old = self.best_pos.get(c.video_id)
            if old is None or pos < old[0]:
                self.best_pos[c.video_id] = (pos, len(items))
        return new


def _earlier_counts(store: Store, current: str, ids: list[str], cid: str | None) -> dict[str, int]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = store.conn.execute(
        f"""SELECT k.video_id, COUNT(DISTINCT k.compute_id) FROM candidates k
            JOIN computes c ON c.compute_id = k.compute_id
            WHERE c.current_video_id = ? AND k.compute_id != ? AND k.video_id IN ({marks})
            GROUP BY k.video_id""",
        (current, cid or "", *ids),
    )
    return {r[0]: r[1] for r in rows}


def recommend(
    store: Store,
    profile,
    *,
    video_id: str,
    fetcher,
    sponsorblock=None,
    detector=None,
    e5=None,
    tagger: Tagger | None = None,
    thresholds: thmod.Thresholds | None = None,
    paths=None,
    config: Config | None = None,
    trigger: str = "cli",
    on_round=None,
) -> Result:
    """Run one compute. ``fetcher`` needs ``watch(id)``, ``search(q)`` and ``pages_fetched``.
    ``BlockedError`` / ``PageCapReached`` from the current page or a search propagate after the
    compute is logged with that status."""
    cfg = config or Config()
    th = thresholds or thmod.load(paths)
    tagger = tagger or load_tagger(paths)
    t0 = time.monotonic()
    cid = store.start_compute(
        video_id, trigger=trigger, profile_hash=store.save_profile_version(profile)
    )
    res = Result(compute_id=cid, current=None)
    pool = _Pool()
    budget = {"left": cfg.enrich_cap}
    space = None
    anchor = None
    ctx = None

    def run_round(rnd: int, new: list[Candidate], log: RoundLog) -> None:
        nonlocal space
        t = time.monotonic()
        log.n_new = len(new)
        a = run_stage_a(
            new,
            profile,
            current_video_id=video_id,
            downvoted=store.downvoted_video_ids(),
            detector=detector,
        )
        log_decisions(store, cid, a)
        survivors = [d.candidate for d in a if d.kept]
        log.a_kept = len(survivors)
        # pre-score on cheap fields; the enrichment budget goes to the best first
        pre: dict[str, float] = {}
        for c in survivors:
            topic = 0.0
            if anchor is not None:
                ct = extract(ctx, c.video_id)
                topic = match(anchor, ct)["score"] if ct is not None else 0.0
            pos, n = pool.best_pos.get(c.video_id, (None, None))
            pre[c.video_id] = sc.prescore(topic, len(pool.lists[c.video_id]), pos, n)
        survivors.sort(key=lambda c: -pre[c.video_id])
        cap = max(0, budget["left"])
        b, stats = run_stage_b(
            survivors,
            profile,
            fetcher=fetcher,
            store=store,
            sponsorblock=sponsorblock,
            detector=detector,
            cap=cap,
            **({"ttl": cfg.cache_ttl} if cfg.cache_ttl is not None else {}),
            current_category=res.current.category if res.current else None,
            on_page=lambda vid, page: pool.upnext.__setitem__(vid, list(page.upnext)),
        )
        budget["left"] -= min(cap, len(survivors))
        for d in b:
            d.features["prescore"] = pre[d.video_id]
            pool.enriched[d.video_id] = bool(d.features.get("enriched", True))
        log_decisions(store, cid, b, stage="B")
        log.b_fetched, log.b_cached, log.b_over_cap = stats.fetched, stats.cached, stats.over_cap
        log.b_dropped = sum(not d.kept for d in b)
        kept_b = [d.candidate for d in b if d.kept]
        if space is None:
            space = build_space(
                store, profile, e5=e5, paths=paths, extra_ids=[video_id, *pool.cands]
            )
        c = run_stage_c(store, space, kept_b, current_id=video_id, thresholds=th)
        log_decisions(store, cid, c, stage="C")
        for d in c:
            if d.kept:
                pool.c_feats[d.video_id] = d.features
        log.c_kept = sum(d.kept for d in c)
        log.seconds = round(time.monotonic() - t, 1)
        res.rounds.append(log)
        if on_round:
            on_round(log)

    try:
        page = fetcher.watch(video_id)
        res.current = page.meta
        if page.meta:
            store.upsert_meta(page.meta)
        store.log_candidates(cid, page.upnext, source="upnext", round=0)
        store.commit()
        ctx = TopicContext(store, tagger)
        anchor = extract(ctx, video_id)
        res.anchor = anchor

        new = pool.add_list("upnext", page.upnext)
        run_round(0, new, RoundLog(round=0, n_candidates=len(page.upnext)))
        res.gate = interest_gate(store, space, video_id, th.for_rep(space.name)["gate"])
        store.set_gate(
            cid,
            mode=res.gate.mode,
            score=res.gate.score,
            interest=res.gate.best_interest,
            rep=res.gate.rep,
        )

        res.kind = dv.context_kind(
            res.current.category if res.current else None, res.current.title if res.current else ""
        )
        for rnd in range(1, cfg.max_rounds + 1):
            have = len(pool.c_feats)
            elapsed = time.monotonic() - t0
            if rnd == 1 and have >= cfg.min_pool and not cfg.always_search:
                res.backfill_skipped = f"{have} survivors ≥ min_pool {cfg.min_pool}"
                break
            if rnd > 1 and have >= cfg.min_pool:
                res.backfill_skipped = f"round {rnd}: {have} survivors ≥ min_pool {cfg.min_pool}"
                break
            if rnd > 1 and elapsed > cfg.deadline_s:
                res.backfill_skipped = (
                    f"round {rnd}: {elapsed:.0f} s > deadline {cfg.deadline_s:.0f} s"
                )
                break
            log = RoundLog(round=rnd)
            if rnd == 1:
                qs = qmod.round1(anchor, res.gate.mode, res.gate.best_interest)
                lists: list[tuple[str, str, list[Candidate]]] = []
            else:
                best = _score(store, pool, anchor, res, tagger, cid, video_id)
                best.sort(key=lambda r: -r.score)
                rel_ctx = TopicContext(store, tagger)
                rel = [extract(rel_ctx, r.candidate.video_id) for r in best[:3]]
                q = qmod.rocchio(anchor, [t for t in rel if t is not None])
                done = {x for lg in res.rounds for x, _, _ in lg.queries}
                qs = [(q, "rocchio")] if q and q not in done else []
                lists = []
                for r in best[: cfg.snowball_from]:
                    v = r.candidate.video_id
                    if v in pool.upnext and v not in pool.snowballed:
                        pool.snowballed.add(v)
                        lists.append((v, "snowball", pool.upnext[v]))
            if not qs and not lists:
                res.backfill_skipped = (
                    "no usable topic for a query" if rnd == 1 else f"round {rnd}: nothing new"
                )
                break
            new = []
            for q, method in qs:
                sp = fetcher.search(q)
                store.log_query(
                    cid, round=rnd, query_text=q, method=method, n_results=len(sp.candidates)
                )
                store.log_candidates(cid, sp.candidates, source="query", round=rnd, query=q)
                log.queries.append((q, method, len(sp.candidates)))
                log.n_candidates += len(sp.candidates)
                new += pool.add_list(f"query:{q}", sp.candidates)
            for v, _, items in lists:
                store.log_candidates(cid, items, source="snowball", round=rnd, query=v)
                log.snowball.append((v, len(items)))
                log.n_candidates += len(items)
                new += pool.add_list(f"snowball:{v}", items)
            store.commit()
            ctx = TopicContext(store, tagger)  # IDF and channel stats with the new sightings
            run_round(rnd, new, log)

        res.ranked = _rank(store, pool, anchor, res, cfg, tagger, cid, video_id, space)
    except BaseException as e:
        res.status = "blocked" if type(e).__name__ == "BlockedError" else "error"
        res.pages = getattr(fetcher, "pages_fetched", 0)
        store.finish_compute(
            cid,
            status=res.status,
            rounds_used=max(0, len(res.rounds) - 1),
            pages_fetched=res.pages,
            duration_ms=int((time.monotonic() - t0) * 1000),
            error=repr(e),
        )
        raise
    res.pages = getattr(fetcher, "pages_fetched", 0)
    res.seconds = round(time.monotonic() - t0, 1)
    store.finish_compute(
        cid,
        status="ok",
        rounds_used=max(0, len(res.rounds) - 1),
        pages_fetched=res.pages,
        duration_ms=int(res.seconds * 1000),
    )
    return res


def _score(store, pool: _Pool, anchor, res: Result, tagger, cid, video_id) -> list[Ranked]:
    """v0 scores (unsorted, unpicked) for every survivor so far."""
    ids = list(pool.c_feats)
    if not ids:
        return []
    mode = res.gate.mode if res.gate else "filter-only"
    ctx = TopicContext(store, tagger)  # after enrichment: descriptions and channel stats
    anchor = extract(ctx, video_id) or anchor
    seed_p = sc.percentiles([pool.c_feats[v].get("seed_sim") for v in ids])
    int_p = sc.percentiles([pool.c_feats[v].get("interest_sim") for v in ids])
    earlier = _earlier_counts(store, video_id, ids, cid)
    cur_channel = res.current.channel if res.current else ""
    rows = []
    for i, v in enumerate(ids):
        meta = store.get_video(v) or {}
        cand = pool.cands[v]
        m = match(anchor, extract(ctx, v)) if anchor is not None else {"score": 0.0}
        parts = {
            "topic": m["score"],
            "seed": seed_p[i],
            "interest": int_p[i] if mode == "interest" else 0.0,
            "recurrence": sc.recurrence(len(pool.lists[v])),
            "same_channel": 1.0 if cur_channel and meta.get("channel") == cur_channel else 0.0,
            "persistence": sc.persistence(earlier.get(v, 0)),
            "views": sc.views_part(meta.get("views") or cand.views),
        }
        feats = {
            **parts,
            "mode": mode,
            "lists": sorted(pool.lists[v]),
            "enriched": pool.enriched.get(v, False),
            "seed_sim": pool.c_feats[v].get("seed_sim"),
            "interest_sim": pool.c_feats[v].get("interest_sim"),
            "topic_parts": {k: m.get(k) for k in ("tok", "ent", "series", "days")},
            "earlier_computes": earlier.get(v, 0),
        }
        rows.append(Ranked(cand, sc.v0(parts, mode), feats, None))
    return rows


def _rank(store, pool: _Pool, anchor, res: Result, cfg: Config, tagger, cid, video_id, space):
    rows = _score(store, pool, anchor, res, tagger, cid, video_id)
    if not rows:
        return []
    rows.sort(key=lambda r: (-r.score, r.candidate.video_id))
    ids = [r.candidate.video_id for r in rows]
    vecs = space.videos(store, ids)[1] if space is not None else None
    titles, channels = [], []
    for r in rows:
        meta = store.get_video(r.candidate.video_id) or {}
        titles.append(meta.get("title") or r.candidate.title)
        channels.append(meta.get("channel") or r.candidate.channel)
    groups = dv.assign_groups(titles, channels, res.kind, tagger)
    cap = cfg.caps.get(res.kind, cfg.caps.get("other", 2))
    order, info = dv.mmr_select(
        [r.score for r in rows],
        vecs,
        groups,
        k=cfg.top_k,
        cap=cap,
        lam=cfg.lam,
        same_group_redundant=res.kind != "lecture",
    )
    for k, i in enumerate(order, 1):
        rows[i].rank = k
    for i, r in enumerate(rows):
        r.parts.update(group=groups[i], kind=res.kind, **{k: v for k, v in info.get(i, {}).items()})
        if r.rank is None:
            r.reason = "group_cap" if info.get(i, {}).get("capped") else "rank"
    ordered = [rows[i] for i in order] + [r for r in rows if r.rank is None]
    with store.conn:
        for r in ordered:
            store.log_decision(
                cid,
                r.candidate.video_id,
                stage=STAGE_RANK,
                kept=r.rank is not None,
                reason=r.reason if r.rank is None else None,
                score=r.score,
                features={**r.parts, "rank": r.rank},
            )
    return ordered
