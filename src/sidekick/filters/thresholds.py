"""Thresholds for the interest gate and Stage C, per representation.

Defaults are starting guesses. ``just tune`` replaces them from your labels (precision-first:
each soft-filter threshold sits just above the highest score of any video you labeled *good*,
so none of those would be dropped) and saves them to ``~/.sidekick/thresholds.json``. The file
is private because it is derived from private labels. Missing keys fall back to the defaults.

Similarity thresholds are on the **centered** scale (see ``embed.center``); clickbait and promo
scores are in [0, 1] and don't depend on the representation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sidekick.config import Paths, get_paths

FILE_NAME = "thresholds.json"

# Starting points. e5 gate τ = 0.10 sits between the "yes" seeds (≥ 0.15) and the one "no"
# seed (0.04–0.06) in the 2026-11-04 `just sim --eval` run (centered e5). The rest are guesses.
DEFAULT_SIM = {
    "e5-small-qp": {"gate": 0.10, "dont_show": 0.30, "downvoted": 0.40},
    "e5-small-q": {"gate": 0.10, "dont_show": 0.30, "downvoted": 0.40},
    "tfidf": {"gate": 0.05, "dont_show": 0.30, "downvoted": 0.50},
}
DEFAULT_TEXT = {"clickbait": 0.80, "promo": 0.67}  # used until `just tune --write`

# `just tune` never goes below these. With a few dozen labels, "just above the highest good
# score" can land near 0 when no good video happens to score high, which would make a filter
# fire on many unlabeled videos. The floors keep a small label set from doing that.
# Clickbait 0.65 (was 0.50): on 2026-11-04 real titles the bait ones scored 0.72-1.00, and 0.50
# would have dropped "Bayesian vs. Frequentist Statistics ... MADE EASY!!!" (0.60).
# Don't-show 0.20 (was 0.15): on 2026-11-04, with the line "political commentary and election
# news", political/news videos scored 0.29-0.40 in both a lecture and a comedy context, while no
# non-political video scored above 0.16. At 0.15 a stand-up clip was dropped on the floor alone.
FLOORS = {"dont_show": 0.20, "downvoted": 0.25, "clickbait": 0.65, "promo": 0.34}


@dataclass
class Thresholds:
    sim: dict[str, dict[str, float]] = field(default_factory=lambda: json.loads(_DS))
    clickbait: float = DEFAULT_TEXT["clickbait"]
    promo: float = DEFAULT_TEXT["promo"]
    source: str = "defaults"  # "defaults" or the tuning timestamp
    notes: dict[str, Any] = field(default_factory=dict)

    def for_rep(self, rep: str) -> dict[str, float]:
        base = DEFAULT_SIM.get(rep, DEFAULT_SIM["tfidf"])
        return {**base, **self.sim.get(rep, {})}


_DS = json.dumps(DEFAULT_SIM)


def path(paths: Paths | None = None) -> Path:
    return (paths or get_paths()).home / FILE_NAME


def load(paths: Paths | None = None) -> Thresholds:
    p = path(paths)
    if not p.exists():
        return Thresholds()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Thresholds(source="defaults (thresholds.json unreadable)")
    t = Thresholds()
    for rep, vals in (raw.get("sim") or {}).items():
        t.sim[rep] = {**t.sim.get(rep, {}), **{k: float(v) for k, v in vals.items()}}
    t.clickbait = float(raw.get("clickbait", t.clickbait))
    t.promo = float(raw.get("promo", t.promo))
    t.source = str(raw.get("source", "file"))
    t.notes = raw.get("notes") or {}
    return t


def save(t: Thresholds, paths: Paths | None = None) -> Path:
    p = path(paths)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(t), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return p
