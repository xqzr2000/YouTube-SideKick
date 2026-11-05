"""``just fetch``: fetch a video's watch page (and optional searches) and print the candidates.

Examples::

    just fetch bFZ-0FH5hfs
    just fetch "https://www.youtube.com/watch?v=bFZ-0FH5hfs" --search "bayesian statistics"
    just fetch bFZ-0FH5hfs --check-thumbs

Every page's JSON goes to the raw log (``~/.sidekick/raw/``), and the run is logged to the
SQLite store as a compute with ``trigger=cli`` (videos, view snapshots, queries, candidates).
Stage A hard filters then run over all unique candidates, Stage B enriches the survivors
(watch pages, SponsorBlock, language on title + description), the interest gate picks the mode
and Stage C applies the soft filters (Don't-show-me / downvoted similarity, clickbait, promo).
Dropped videos are listed with their reasons and every decision is logged. ``--no-enrich``
skips Stage B, ``--no-stage-c`` the gate and Stage C, ``--no-db`` the database (and with it
Stage C, which needs the database for vectors). No ranking yet.
"""

from __future__ import annotations

import argparse
import sys
import time

from sidekick.config import ensure_home
from sidekick.features.lang import load_detector
from sidekick.filters import thresholds as thresholds_mod
from sidekick.filters.stage_a import Decision, log_decisions, run_stage_a
from sidekick.filters.stage_b import ENRICH_CAP, EnrichStats, run_stage_b
from sidekick.filters.stage_c import GateResult, build_space, interest_gate, run_stage_c
from sidekick.models import Candidate
from sidekick.parse.util import parse_age_days
from sidekick.profile import load_profile
from sidekick.scrape.fetcher import BlockedError, FetchConfig, Fetcher, PageCapReached
from sidekick.scrape.record import video_id_from
from sidekick.scrape.sponsorblock import SponsorBlock
from sidekick.store.db import Store


def _fmt_duration(s: int | None) -> str:
    if s is None:
        return "-"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _fmt_views(v: int | None) -> str:
    if v is None:
        return "-"
    for div, suf in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if v >= div:
            return f"{v / div:.1f}{suf}"
    return str(v)


def _flags(c: Candidate) -> str:
    return "".join(f for f, on in (("L", c.is_live), ("U", c.is_upcoming), ("S", c.is_short)) if on)


def print_table(title: str, cands: list[Candidate]) -> None:
    print(f"\n{title} ({len(cands)})")
    print(
        f"{'#':>3}  {'video_id':<11}  {'dur':>8}  {'views':>7}  {'age_d':>6}  fl  channel / title"
    )
    for i, c in enumerate(cands, 1):
        age = parse_age_days(c.published_text)
        age_s = f"{age:.0f}" if age is not None else "-"
        print(
            f"{i:>3}  {c.video_id:<11}  {_fmt_duration(c.duration_s):>8}  "
            f"{_fmt_views(c.views):>7}  {age_s:>6}  {_flags(c):<2}  "
            f"{c.channel[:20]} | {c.title[:45]}"
        )


def print_stage_a(decisions: list[Decision], lang_checked: bool) -> None:
    kept = sum(d.kept for d in decisions)
    print(f"\nStage A: kept {kept} of {len(decisions)} unique videos", end="")
    print("" if lang_checked else "  (language not checked: run `just models`)")
    dropped = [d for d in decisions if not d.kept]
    if dropped:
        print(f"  {'video_id':<11}  {'reason':<24}  channel / title")
        for d in dropped:
            c = d.candidate
            print(f"  {c.video_id:<11}  {d.reason[:24]:<24}  {c.channel[:20]} | {c.title[:40]}")
    langs = [
        (d.features.get("title_lang"), d.features.get("title_lang_conf"))
        for d in decisions
        if d.features.get("title_lang")
    ]
    if langs:
        low = sum(1 for _, conf in langs if conf < 0.8)
        top = sorted({lang for lang, _ in langs})
        print(f"  title languages: {', '.join(top)}  ({low} below 0.80; Stage B decides)")


def print_stage_b(decisions: list[Decision], stats: EnrichStats) -> None:
    kept = sum(d.kept for d in decisions)
    print(
        f"\nStage B: kept {kept} of {len(decisions)}  (pages fetched {stats.fetched}, "
        f"from cache {stats.cached}, failed {stats.failed}, over cap {stats.over_cap}; "
        f"SponsorBlock calls {stats.sponsor_calls}, unknown {stats.sponsor_unknown})"
    )
    print(f"  {'video_id':<11}  {'kept':<20}  {'age_d':>6}  {'lang':<8}  {'spons':>5}  title")
    for d in decisions:
        f = d.features
        if not f.get("enriched"):
            print(f"  {d.video_id:<11}  {'(not enriched)':<20}")
            continue
        lang = f"{f.get('lang', '-')}" + (
            f" {f['lang_conf']:.2f}" if f.get("lang_conf") is not None else ""
        )
        sp = f.get("sponsor_frac")
        sp_s = "-" if sp is None else f"{100 * (sp + (f.get('selfpromo_frac') or 0)):.0f}%"
        age = f.get("age_days")
        age_s = "-" if age is None else f"{age:.0f}"
        status = "yes" if d.kept else f"NO {d.reason}"
        print(
            f"  {d.video_id:<11}  {status[:20]:<20}  {age_s:>6}  {lang:<8}  {sp_s:>5}  "
            f"{d.candidate.title[:40]}"
        )


def print_stage_c(gate: GateResult, decisions: list[Decision], secs: float, stats) -> None:
    if gate.score is None:
        print(f"\ninterest gate: {gate.mode} (no Interests or no current-video data)")
    else:
        print(
            f"\ninterest gate: {gate.score:.2f} vs tau {gate.tau:.2f} ({gate.rep}, best "
            f"{gate.best_interest!r}) -> {gate.mode} mode"
        )
    kept = sum(d.kept for d in decisions)
    print(
        f"Stage C: kept {kept} of {len(decisions)}  ({gate.rep}; vectors "
        f"{stats.get('computed', 0)} computed, {stats.get('cached', 0)} cached, {secs:.1f} s)"
    )
    print(
        f"  {'video_id':<11}  {'kept':<22}  {'dont':>5}  {'int':>5}  {'seed':>5}  cb   pr   title"
    )

    def n(x):
        return "    -" if x is None else f"{x:5.2f}"

    for d in sorted(decisions, key=lambda d: (d.kept, -(d.features.get("dont_show_sim") or 0))):
        f = d.features
        status = "yes" if d.kept else f"NO {d.reason}"
        print(
            f"  {d.video_id:<11}  {status[:22]:<22}  {n(f.get('dont_show_sim'))}  "
            f"{n(f.get('interest_sim'))}  {n(f.get('seed_sim'))}  {f['clickbait']:.2f} "
            f"{f['promo']:.2f} {d.candidate.title[:38]}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch a watch page (+ searches) and list candidates")
    ap.add_argument("video", help="video id or watch URL")
    ap.add_argument("--search", action="append", default=[], help="search query (repeatable)")
    ap.add_argument(
        "--check-thumbs", action="store_true", help="check midpoint thumbnails (first 3 up-next)"
    )
    ap.add_argument("--no-db", action="store_true", help="don't log this run to sidekick.db")
    ap.add_argument(
        "--no-enrich", action="store_true", help="skip Stage B (watch pages + SponsorBlock)"
    )
    ap.add_argument(
        "--enrich-cap", type=int, default=ENRICH_CAP, help="max survivors to enrich (default 50)"
    )
    ap.add_argument("--no-stage-c", action="store_true", help="skip the interest gate and Stage C")
    args = ap.parse_args(argv)

    paths = ensure_home()
    vid = video_id_from(args.video)
    store = None if args.no_db else Store.open(paths)
    profile = load_profile(paths)
    detector = load_detector(paths)
    cid = None
    all_cands: list[Candidate] = []
    if store:
        cid = store.start_compute(
            vid, trigger="cli", profile_hash=store.save_profile_version(profile)
        )
    t0 = time.monotonic()
    pages = 0

    def finish(status: str, error: str | None = None) -> None:
        if store and cid:
            store.finish_compute(
                cid,
                status=status,
                rounds_used=1 if args.search else 0,
                pages_fetched=pages,
                duration_ms=int((time.monotonic() - t0) * 1000),
                error=error,
            )
            store.close()

    f = None
    try:
        with Fetcher(FetchConfig(), paths=paths) as f:
            page = f.watch(vid)
            m = page.meta
            if store:
                if m:
                    store.upsert_meta(m)
                store.log_candidates(cid, page.upnext, source="upnext", round=0)
            if m:
                print(
                    f"current: {m.video_id}  {m.title[:60]!r}  ({m.channel}, "
                    f"{_fmt_duration(m.duration_s)}, {_fmt_views(m.views)} views, "
                    f"{m.publish_date[:10] or '?'}, {m.category or '?'})"
                )
            else:
                print(f"current: {vid}  (no metadata found)")
            all_cands.extend(page.upnext)
            print_table("up-next", page.upnext)
            if page.nonvideo_skipped:
                print(f"     ({page.nonvideo_skipped} non-video items skipped)")
            for q in args.search:
                sp = f.search(q)
                if store:
                    store.log_query(
                        cid, round=1, query_text=q, method="manual", n_results=len(sp.candidates)
                    )
                    store.log_candidates(cid, sp.candidates, source="query", round=1, query=q)
                all_cands.extend(sp.candidates)
                print_table(f"search {q!r}", sp.candidates)
                if sp.shorts_skipped:
                    print(f"     ({sp.shorts_skipped} Shorts items skipped)")
            if args.check_thumbs:
                print("\nmidpoint thumbnails (open the URLs in Chrome to eyeball them):")
                for c in [vid, *[c.video_id for c in page.upnext[:3]]]:
                    t = f.check_thumbnail(c)
                    print(f"  {t.status} {t.content_type:<11} {t.size:>7,} B  {t.url}")

            decisions = run_stage_a(
                all_cands,
                profile,
                current_video_id=vid,
                downvoted=store.downvoted_video_ids() if store else (),
                detector=detector,
            )
            if store and cid:
                log_decisions(store, cid, decisions)
            print_stage_a(decisions, lang_checked=detector is not None)

            if not args.no_enrich:
                survivors = [d.candidate for d in decisions if d.kept]
                with SponsorBlock() as sb:
                    b_decisions, stats = run_stage_b(
                        survivors,
                        profile,
                        fetcher=f,
                        store=store,
                        sponsorblock=sb,
                        detector=detector,
                        cap=args.enrich_cap,
                        current_category=m.category if m else None,
                    )
                if store and cid:
                    log_decisions(store, cid, b_decisions, stage="B")
                print_stage_b(b_decisions, stats)
                survivors = [d.candidate for d in b_decisions if d.kept]
            else:
                survivors = [d.candidate for d in decisions if d.kept]
            pages = f.pages_fetched

            if store and not args.no_stage_c:
                t_c = time.monotonic()
                th = thresholds_mod.load(paths)
                space = build_space(
                    store, profile, paths=paths, extra_ids=[vid, *[c.video_id for c in survivors]]
                )
                gate = interest_gate(store, space, vid, th.for_rep(space.name)["gate"])
                c_decisions = run_stage_c(store, space, survivors, current_id=vid, thresholds=th)
                if cid:
                    store.set_gate(
                        cid,
                        mode=gate.mode,
                        score=gate.score,
                        interest=gate.best_interest,
                        rep=gate.rep,
                    )
                    log_decisions(store, cid, c_decisions, stage="C")
                print_stage_c(gate, c_decisions, time.monotonic() - t_c, space.stats)
                if th.source == "defaults":
                    print("  (default thresholds; `just tune` sets them from your labels)")
    except BlockedError as e:
        pages = f.pages_fetched if f else pages
        finish("blocked", str(e))
        print(f"BLOCKED: {e}", file=sys.stderr)
        return 3
    except PageCapReached as e:
        pages = f.pages_fetched if f else pages
        finish("error", str(e))
        print(f"STOPPED: {e}", file=sys.stderr)
        return 4
    except BaseException as e:
        pages = f.pages_fetched if f else pages
        finish("error", repr(e))
        raise
    finish("ok")
    print(f"\n{pages} page(s) in {time.monotonic() - t0:.1f}s; raw log in {paths.raw_dir}")
    if cid:
        print(f"logged as compute {cid} in {paths.db} (`just db` to inspect)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
