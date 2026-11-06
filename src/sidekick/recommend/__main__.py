"""``just recommend``: one full compute for a video, printed as SideKick's 10 picks.

Examples::

    just recommend bFZ-0FH5hfs
    just recommend "https://www.youtube.com/watch?v=wQXdPZikag0" --show 20
    just recommend 5eW3gNkNsA8 --backfill          # run the search round even with enough videos
    just recommend bFZ-0FH5hfs --min-pool 20 --enrich-cap 15

It fetches the watch page (signed out), filters the up-next list (Stages A–C), searches for
more when fewer than ``--min-pool`` videos survive, scores everything with the v0 score and
prints the top 10 with each score's parts and how long each round took. The whole compute is
logged (`just db`); the ranking is stage ``R`` in ``decisions``.
"""

from __future__ import annotations

import argparse
import sys

from sidekick.config import ensure_home
from sidekick.features.lang import load_detector
from sidekick.profile import load_profile
from sidekick.recommend import Config, Result, RoundLog, recommend
from sidekick.scrape.fetcher import BlockedError, FetchConfig, Fetcher, PageCapReached
from sidekick.scrape.record import video_id_from
from sidekick.scrape.sponsorblock import SponsorBlock
from sidekick.store.db import Store


def print_round(r: RoundLog, out=print) -> None:
    if r.round == 0:
        head = f"round 0  up-next {r.n_candidates}"
    else:
        qs = " + ".join(f"{q!r} ({n})" for q, _, n in r.queries)
        head = f"round {r.round}  search {qs}: {r.n_new} new"
    out(
        f"{head} → A {r.a_kept} → B {r.a_kept - r.b_dropped} (fetched {r.b_fetched}, cached "
        f"{r.b_cached}, over budget {r.b_over_cap}) → C {r.c_kept}   {r.seconds:.1f} s"
    )


def _src(lists: list[str]) -> str:
    up = "up" if "upnext" in lists else ""
    q = sum(x.startswith("query:") for x in lists)
    return "+".join(x for x in (up, f"q{q}" if q else "") if x)


def print_result(res: Result, show: int, out=print) -> None:
    if res.backfill_skipped:
        out(f"backfill: skipped ({res.backfill_skipped})")
    out(f"\ntop {len(res.shown)} (of {len(res.ranked)} survivors):")
    out(
        f"  {'#':>2} {'score':>5} {'topic':>5} {'seed':>5} {'int':>5} {'rec':>4} {'src':<5} "
        "channel | title"
    )
    rows = res.shown + [r for r in res.ranked if r.rank is None][: max(0, show - len(res.shown))]
    for i, r in enumerate(rows):
        if i == len(res.shown):
            out("  -- not shown --")
        p = r.parts
        rank = f"{r.rank:>2}" if r.rank else " -"
        note = "" if r.rank else f"  [{r.reason}]"
        out(
            f"  {rank} {r.score:5.2f} {p['topic']:5.2f} {p['seed']:5.2f} {p['interest']:5.2f} "
            f"{p['recurrence']:4.1f} {_src(p['lists']):<5} {r.candidate.channel[:18]} | "
            f"{r.candidate.title[:58]}{'' if p['enriched'] else ' (not enriched)'}{note}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="just recommend", description=__doc__.split("\n")[0])
    ap.add_argument("video", help="video id or watch URL")
    ap.add_argument("--min-pool", type=int, default=Config.min_pool, help="backfill below this")
    ap.add_argument(
        "--enrich-cap", type=int, default=Config.enrich_cap, help="enrichment budget per compute"
    )
    ap.add_argument("--backfill", action="store_true", help="always run the search round")
    ap.add_argument("--show", type=int, default=15, help="rows to print (top 10 + runners-up)")
    args = ap.parse_args(argv)

    paths = ensure_home()
    vid = video_id_from(args.video)
    profile = load_profile(paths)
    cfg = Config(min_pool=args.min_pool, enrich_cap=args.enrich_cap, force_backfill=args.backfill)
    with Store.open(paths) as store, Fetcher(FetchConfig(), paths=paths) as f, SponsorBlock() as sb:
        try:
            res = recommend(
                store,
                profile,
                video_id=vid,
                fetcher=f,
                sponsorblock=sb,
                detector=load_detector(paths),
                paths=paths,
                config=cfg,
                on_round=print_round,
            )
        except BlockedError as e:
            print(f"BLOCKED: {e}", file=sys.stderr)
            return 3
        except PageCapReached as e:
            print(f"STOPPED: {e}", file=sys.stderr)
            return 4
    m, g, a = res.current, res.gate, res.anchor
    if m:
        print(f"\ncurrent: {m.video_id}  {m.title[:70]!r}  ({m.channel}, {m.category or '?'})")
    if g:
        best = f", {g.best_interest!r}" if g.best_interest else ""
        score = "-" if g.score is None else f"{g.score:.2f}"
        print(f"mode: {g.mode} (gate {score} vs τ {g.tau:.2f}{best})")
    if a:
        print(f"topic ({a.lead_source}): {a.lead[:80]!r}   query: {a.query()!r}")
    print_result(res, args.show)
    print(f"\n{res.pages} page(s) in {res.seconds:.1f} s; compute {res.compute_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
