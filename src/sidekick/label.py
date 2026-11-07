"""``just label``: hand-label logged videos for the M3 experiments (E1, E2).

A stop-gap until the M6 labeling app. It walks through videos that past computes saw and asks
one question per video, saving answers to the ``labels`` table:

- **Seed question** (``kind=seed``), asked once per current video: *does this video match your
  profile's Interests lines* (printed with the question)? ``g`` yes · ``m`` partly · ``b`` no.
  Not "do I like it": a comedy set you enjoy is still "no" if comedy isn't an Interest. This is
  the ground truth for the interest gate's threshold τ (E2). ``--seeds`` asks it again for every
  current video; the newest seed label per video is the one that counts.
- **Candidate question** (``kind=candidate``): *while watching the seed video, would you want
  this one shown next?* ``g`` good · ``m`` meh · ``b`` bad, plus an optional short reason
  ("political", "clickbait", "off-topic", ...). These score the representations and soft
  filters (E1) and later the ranker.

Filter decisions are deliberately **not** shown, so they can't bias the label. Videos that
Stage A dropped for format reasons (current video, Short, live, upcoming) are skipped, since
SideKick never shows them; videos dropped by duration, language, channel or keyword rules
*are* included, which is how false removals get measured.

Order: round-robin across seed videos, random within each seed, so a short session still
covers every seed. Already-labeled (seed, video) pairs are never asked again. Keys:
``g``/``m``/``b`` label, ``s`` skip, ``u`` undo the last label, ``q`` quit (everything answered
so far is already saved).

Examples::

    just label                  # 40 videos (default)
    just label --n 20 --seed bFZ-0FH5hfs
    just label --stats          # counts only, no questions
    just label --seeds          # redo the seed question for every current video
    just label --top10          # only SideKick's top 10 and YouTube's first 10 up-next of the
                                # newest `just recommend` per seed (mixed, unmarked), for P@10

``--stats`` also prints precision@10 of SideKick's top 10 against YouTube's own up-next
(``sidekick.recommend.evaluate``).
"""

from __future__ import annotations

import argparse
import random
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from sidekick.config import get_paths
from sidekick.parse.util import parse_age_days
from sidekick.profile import load_profile
from sidekick.store.db import Store

# Stage A reasons that mean "SideKick would never show this", so labeling it is wasted effort.
_NEVER_SHOWN = ("current_video", "short", "live", "upcoming")

LABELS = {"g": "good", "m": "meh", "b": "bad"}


@dataclass
class Item:
    seed_id: str
    video_id: str
    compute_id: str  # latest compute of this seed that saw the video


def _fmt_duration(s: int | None) -> str:
    if s is None:
        return "?"
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _fmt_views(v: int | None) -> str:
    if v is None:
        return "?"
    for div, suf in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if v >= div:
            return f"{v / div:.1f}{suf}"
    return str(v)


def seeds(store: Store) -> list[str]:
    """Current videos of past computes, oldest first."""
    rows = store.conn.execute(
        """SELECT current_video_id, MIN(ts) AS first FROM computes
           WHERE current_video_id IS NOT NULL
             AND EXISTS (SELECT 1 FROM candidates k WHERE k.compute_id = computes.compute_id)
           GROUP BY current_video_id ORDER BY first"""
    )
    return [r[0] for r in rows]


def latest_seed_label(store: Store, seed_id: str) -> str | None:
    row = store.conn.execute(
        "SELECT label FROM labels WHERE kind = 'seed' AND video_id = ? "
        "ORDER BY ts DESC, label_id DESC LIMIT 1",
        (seed_id,),
    ).fetchone()
    return row[0] if row else None


def seed_is_labeled(store: Store, seed_id: str) -> bool:
    row = store.conn.execute(
        "SELECT 1 FROM labels WHERE kind = 'seed' AND video_id = ? LIMIT 1", (seed_id,)
    ).fetchone()
    return row is not None


def pool(store: Store, seed_id: str) -> list[Item]:
    """Unlabeled, showable candidates seen while ``seed_id`` was the current video."""
    rows = store.conn.execute(
        """
        SELECT k.video_id, MAX(k.compute_id) AS compute_id
        FROM candidates k JOIN computes c ON c.compute_id = k.compute_id
        WHERE c.current_video_id = :seed AND k.video_id != :seed
        GROUP BY k.video_id
        """,
        {"seed": seed_id},
    ).fetchall()
    never = _never_shown(store, seed_id)
    done = {
        r[0]
        for r in store.conn.execute(
            """SELECT l.video_id FROM labels l JOIN computes c ON c.compute_id = l.compute_id
               WHERE l.kind = 'candidate' AND c.current_video_id = ?""",
            (seed_id,),
        )
    }
    out = []
    for r in rows:
        vid = r["video_id"]
        v = store.get_video(vid)
        if vid in never or vid in done or (v and v["is_short"]):
            continue
        out.append(Item(seed_id, vid, r["compute_id"]))
    return out


def _never_shown(store: Store, seed_id: str) -> set[str]:
    rows = store.conn.execute(
        """SELECT d.video_id, d.reason FROM decisions d JOIN computes c
               ON c.compute_id = d.compute_id
           WHERE c.current_video_id = ? AND d.kept = 0 AND d.reason IS NOT NULL""",
        (seed_id,),
    )
    out = set()
    for vid, reason in rows:
        if any(r in _NEVER_SHOWN for r in reason.split(";")):
            out.add(vid)
    return out


def plan(store: Store, n: int, *, seed: str | None = None, rng: random.Random) -> list[Item]:
    """Pick up to ``n`` items, round-robin across seeds and shuffled within each seed."""
    seed_list = [seed] if seed else seeds(store)
    pools = []
    for s in seed_list:
        p = pool(store, s)
        rng.shuffle(p)
        pools.append(p)
    picked: list[Item] = []
    while len(picked) < n and any(pools):
        for p in pools:
            if p and len(picked) < n:
                picked.append(p.pop())
    return picked


def plan_top10(store: Store, n: int | None, *, rng: random.Random) -> list[Item]:
    """Unlabeled videos of both top-10 lists, shuffled within each seed (so neither list is
    recognizable by order), round-robin across seeds."""
    from sidekick.recommend.evaluate import to_label

    by_seed: dict[str, list[Item]] = {}
    for seed, vid, cid in to_label(store):
        by_seed.setdefault(seed, []).append(Item(seed, vid, cid))
    pools = list(by_seed.values())
    for p in pools:
        rng.shuffle(p)
    picked: list[Item] = []
    while any(pools) and (n is None or len(picked) < n):
        for p in pools:
            if p and (n is None or len(picked) < n):
                picked.append(p.pop())
    return picked


def describe(store: Store, video_id: str) -> list[str]:
    v = store.get_video(video_id) or {}
    views = store.conn.execute(
        "SELECT views FROM video_snapshots WHERE video_id = ? AND views IS NOT NULL "
        "ORDER BY source = 'watch' DESC, ts DESC LIMIT 1",
        (video_id,),
    ).fetchone()
    age = None
    if v.get("published_at"):
        try:
            pub = datetime.fromisoformat(v["published_at"])
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=UTC)
            age = (datetime.now(UTC) - pub).days
        except ValueError:
            age = None
    if age is None:
        age = parse_age_days(v.get("published_text") or "")
    age_s = f"{age:.0f} d old" if age is not None else "age ?"
    lines = [
        f"  {v.get('title') or '(no title)'}",
        f"  {v.get('channel') or '?'} · {_fmt_duration(v.get('duration_s'))} · "
        f"{_fmt_views(views[0] if views else None)} views · {age_s}",
    ]
    desc = " ".join((v.get("description") or "").split())
    if desc:
        lines += textwrap.wrap(
            desc[:280] + ("…" if len(desc) > 280 else ""),
            96,
            initial_indent="  ",
            subsequent_indent="  ",
        )
    lines.append(f"  https://youtu.be/{video_id}")
    return lines


_SEED_WORDS = {"good": "yes", "meh": "partly", "bad": "no"}


def _seed_header(interests: list[str], out: Callable[[str], None]) -> None:
    out("\n" + "=" * 100)
    out("CURRENT VIDEO: does it match one of your profile's Interests? (not: do you like it)")
    out("  Interests: " + (" · ".join(interests) if interests else "(none in profile.md)"))


def ask_seed(
    store: Store,
    seed_id: str,
    *,
    interests: list[str],
    session: str,
    input_fn: Callable[[str], str],
    out: Callable[[str], None],
) -> tuple[str, int | None]:
    """Ask the seed question once. Returns (key pressed, label_id or None)."""
    _seed_header(interests, out)
    for line in describe(store, seed_id):
        out(line)
    current = latest_seed_label(store, seed_id)
    if current:
        out(f"  currently: {_SEED_WORDS[current]}")
    a = _ask("  [g] yes  [m] partly  [b] no  [s] skip  [q] quit > ", input_fn, "gmbsq")
    if a in LABELS:
        return a, store.add_label(seed_id, LABELS[a], kind="seed", session=session)
    return a, None


def _ask(prompt: str, input_fn: Callable[[str], str], allowed: str) -> str:
    while True:
        try:
            ans = input_fn(prompt).strip().lower()
        except EOFError:
            return "q"
        if ans[:1] in allowed and ans:
            return ans[:1]


def run(
    store: Store,
    items: list[Item],
    *,
    session: str,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
    interests: list[str] | None = None,
) -> dict[str, int]:
    """Ask about each item. Returns counts of what was saved this session."""
    counts = {"seed": 0, "candidate": 0, "skipped": 0}
    history: list[tuple[int, str, int]] = []  # (label_id, kind, item index) for undo
    asked_seed: set[str] = set()
    i = 0
    while i < len(items):
        it = items[i]
        if it.seed_id not in asked_seed and not seed_is_labeled(store, it.seed_id):
            asked_seed.add(it.seed_id)
            a, lid = ask_seed(
                store,
                it.seed_id,
                interests=interests or [],
                session=session,
                input_fn=input_fn,
                out=out,
            )
            if a == "q":
                break
            if lid is not None:
                history.append((lid, "seed", i))
                counts["seed"] += 1
        asked_seed.add(it.seed_id)
        seed_title = (store.get_video(it.seed_id) or {}).get("title") or it.seed_id
        out("\n" + "-" * 100)
        out(f"[{i + 1}/{len(items)}] while watching: {seed_title[:80]}")
        out("would you want this shown next?")
        for line in describe(store, it.video_id):
            out(line)
        a = _ask(
            "  [g] good  [m] meh  [b] bad  [s] skip  [u] undo  [q] quit > ", input_fn, "gmbsuq"
        )
        if a == "q":
            break
        if a == "s":
            counts["skipped"] += 1
            i += 1
            continue
        if a == "u":
            if history:
                lid, kind, idx = history.pop()
                store.delete_label(lid)
                counts[kind] -= 1
                out("  undone.")
                if kind == "seed":
                    asked_seed.discard(items[idx].seed_id)
                i = idx
            else:
                out("  nothing to undo.")
            continue
        try:
            reason = input_fn("  reason (optional, Enter to skip) > ").strip()
        except EOFError:
            reason = ""
        lid = store.add_label(
            it.video_id,
            LABELS[a],
            kind="candidate",
            compute_id=it.compute_id,
            reason=reason,
            session=session,
        )
        history.append((lid, "candidate", i))
        counts["candidate"] += 1
        i += 1
    return counts


def relabel_seeds(
    store: Store,
    *,
    interests: list[str],
    session: str,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> int:
    """Ask the seed question again for every current video. Returns how many were saved."""
    n = 0
    for seed_id in seeds(store):
        a, lid = ask_seed(
            store, seed_id, interests=interests, session=session, input_fn=input_fn, out=out
        )
        if a == "q":
            break
        n += lid is not None
    return n


def stats(store: Store) -> list[str]:
    # seeds: only the newest label per video counts
    rows = store.conn.execute(
        """SELECT kind, label, COUNT(*) FROM labels l
           WHERE kind = 'candidate' OR label_id = (
               SELECT label_id FROM labels m WHERE m.kind = 'seed' AND m.video_id = l.video_id
               ORDER BY ts DESC, label_id DESC LIMIT 1)
           GROUP BY kind, label ORDER BY kind, label"""
    ).fetchall()
    if not rows:
        return ["no labels yet"]
    lines = ["labels:"]
    for kind, label, n in rows:
        lines.append(f"  {kind:<10} {label:<5} {n:>4}")
    per_seed = store.conn.execute(
        """SELECT c.current_video_id, COUNT(*) FROM labels l
           JOIN computes c ON c.compute_id = l.compute_id
           WHERE l.kind = 'candidate' GROUP BY c.current_video_id"""
    ).fetchall()
    if per_seed:
        lines.append("candidate labels per current video:")
        for seed_id, n in per_seed:
            title = (store.get_video(seed_id) or {}).get("title") or ""
            lines.append(f"  {seed_id}  {n:>4}  {title[:60]}")
    from sidekick.recommend.evaluate import report as top10_report

    top = top10_report(store)
    if top:
        lines += top
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="just label", description=__doc__.split("\n")[0])
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help="how many candidate videos (default 40; all with --top10)",
    )
    ap.add_argument("--seed", help="only candidates seen while this video was current")
    ap.add_argument("--session", help="session name stored with each label (default: now)")
    ap.add_argument("--stats", action="store_true", help="print label counts and exit")
    ap.add_argument(
        "--seeds", action="store_true", help="redo the seed question for every current video"
    )
    ap.add_argument(
        "--top10",
        action="store_true",
        help="label SideKick's and YouTube's top 10 of the newest `just recommend` per seed",
    )
    args = ap.parse_args(argv)

    paths = get_paths()
    if not paths.db.exists():
        print(f"no database yet at {paths.db} (`just fetch <id>` creates it)")
        return 1
    interests = load_profile(paths).interests
    session = args.session or datetime.now(UTC).strftime("%Y%m%dT%H%MZ")
    with Store.open(paths) as store:
        if args.stats:
            print("\n".join(stats(store)))
            return 0
        if args.seeds:
            n = relabel_seeds(store, interests=interests, session=session)
            print(f"\nsaved {n} seed label(s) (session {session})")
            print("\n".join(stats(store)))
            return 0
        if args.top10:
            items = plan_top10(store, args.n, rng=random.Random())
        else:
            items = plan(store, args.n or 40, seed=args.seed, rng=random.Random())
        if not items:
            hint = "`just recommend`" if args.top10 else "`just fetch`"
            print(f"nothing left to label. Run {hint} on more videos, or see --stats.")
            return 0
        n_seeds = len({it.seed_id for it in items})
        print(f"{len(items)} videos from {n_seeds} current video(s). Answers save as you go.")
        counts = run(store, items, session=session, interests=interests)
        print(
            f"\nsaved {counts['candidate']} candidate + {counts['seed']} seed label(s), "
            f"skipped {counts['skipped']} (session {session})"
        )
        print("\n".join(stats(store)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
