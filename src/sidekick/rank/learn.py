"""M6c: learned rankers, evaluated on current videos they never saw.

Two models, both trained on the labeled rows of the *other* current videos' pools:

- **Logistic regression** (pointwise): P(good | features). Missing values are imputed with
  the training median plus an "is missing" indicator, then standardized, so the coefficients
  are comparable ("one standard deviation more of this feature → this much more log-odds of
  good"). Small L2 penalty (C = 0.3): with a few hundred rows, a strong prior toward 0 keeps
  it from fitting noise.
- **LightGBM lambdarank** (listwise): boosted trees trained to order each pool for NDCG
  directly, with the same gains as the metric (good 2, meh 1, bad 0). Shallow trees
  (7 leaves, ≥ 10 rows per leaf) and 150 rounds at 0.05, again because the data is small.
  Missing values are handled natively.

**Protocol (Russell, 2026-11-15):** leave one current video out. Each of the 13 folds trains
on 12 current videos' newest pools and scores the 13th; the metric is averaged over the 13
held-out pools, with a bootstrap interval over them. A time split (older current videos train,
newer test) is the second check. Rows of one current video are never split between train and
test, because videos labeled in one context aren't independent.

Anything tuned (the MMR window) is chosen inside each fold on the training pools only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from sidekick.rank import metrics as M
from sidekick.rank import rerank as R
from sidekick.rank.dataset import FEATURES, Pool

WINDOWS: Sequence[float | None] = (0.0, 0.02, 0.05, 0.1, None)  # None = plain MMR
FEATURE_NAMES = list(FEATURES)


def _xy(pools: list[Pool]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    X, y, groups = [], [], []
    for p in pools:
        rows = p.labeled
        X += p.matrix(rows)
        y += [r.gain for r in rows]
        groups.append(len(rows))
    return np.array(X, dtype=float), np.array(y, dtype=int), groups


@dataclass
class Model:
    name: str
    predict: Callable[[np.ndarray], np.ndarray]
    importance: dict[str, float] = field(default_factory=dict)

    def scorer(self) -> R.Scorer:
        def score(pool: Pool) -> list[float]:
            return [float(v) for v in self.predict(np.array(pool.matrix(), dtype=float))]

        return score


def fit_logreg(train: list[Pool], C: float = 0.3) -> Model:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, _ = _xy(train)
    pipe = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=5000),
    )
    pipe.fit(X, (y == 2).astype(int))
    coef = pipe[-1].coef_[0][: len(FEATURE_NAMES)]  # indicator columns come after
    return Model(
        "logistic regression",
        lambda X: pipe.predict_proba(X)[:, 1],
        {n: float(c) for n, c in zip(FEATURE_NAMES, coef, strict=True)},
    )


def fit_lgbm(train: list[Pool], seed: int = 0) -> Model:
    import lightgbm as lgb

    X, y, groups = _xy(train)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=10,
        label_gain=[0, 1, 2],
        random_state=seed,
        deterministic=True,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(X, y, group=groups)
    gains = model.booster_.feature_importance(importance_type="gain")
    total = float(gains.sum()) or 1.0
    return Model(
        "LightGBM lambdarank",
        lambda X: model.predict(X),
        {n: float(g) / total for n, g in zip(FEATURE_NAMES, gains, strict=True)},
    )


FITTERS: dict[str, Callable[[list[Pool]], Model]] = {
    "logistic regression": fit_logreg,
    "LightGBM lambdarank": fit_lgbm,
}


def best_window(train: list[Pool], vectors: R.Vectors | None) -> float | None:
    """The MMR window with the best mean NDCG@10 on the training pools (v0 scores)."""
    best, best_v = None, -1.0
    for w in WINDOWS:
        rows = M.per_pool(train, R.picker(R.v0_scores, vectors, window=w))
        v = float(np.mean([r["ndcg"] for r in rows if r["ndcg"] is not None]))
        if v > best_v + 1e-9:
            best, best_v = w, v
    return best


@dataclass
class Eval:
    """Per-pool results (aligned with the test pools) for every ranker variant."""

    seeds: list[str] = field(default_factory=list)
    results: dict[str, list[dict]] = field(default_factory=dict)
    importance: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    windows: list[float | None] = field(default_factory=list)

    def add(self, name: str, res: list[dict]) -> None:
        self.results.setdefault(name, []).extend(res)


def evaluate_split(
    train: list[Pool], test: list[Pool], vectors: R.Vectors | None, out: Eval
) -> None:
    out.seeds += [p.seed_id for p in test]
    out.add(
        "v0 score only", M.per_pool(test, R.picker(R.v0_scores, vectors, caps=False, mmr=False))
    )
    out.add("v0 + caps", M.per_pool(test, R.picker(R.v0_scores, vectors, mmr=False)))
    out.add("v0 + caps + MMR (shipped)", M.per_pool(test, R.picker(R.v0_scores, vectors)))
    w = best_window(train, vectors)
    out.windows.append(w)
    out.add(
        "v0 + caps + MMR, tuned window",
        M.per_pool(test, R.picker(R.v0_scores, vectors, window=w)),
    )
    for name, fit in FITTERS.items():
        model = fit(train)
        out.importance.setdefault(name, []).append(model.importance)
        sc = model.scorer()
        out.add(name, M.per_pool(test, R.picker(sc, vectors, caps=False, mmr=False)))
        out.add(f"{name} + caps", M.per_pool(test, R.picker(sc, vectors, mmr=False)))


def leave_one_out(pools: list[Pool], vectors: R.Vectors | None) -> Eval:
    out = Eval()
    for _, train, test in M.leave_one_seed_out(pools):
        if train and test:
            evaluate_split(train, test, vectors, out)
    return out


def time_split_eval(pools: list[Pool], vectors: R.Vectors | None, test_frac: float = 0.3) -> Eval:
    out = Eval()
    train, test = M.time_split(pools, test_frac)
    if train and test:
        evaluate_split(train, test, vectors, out)
    return out


def mean_importance(imps: list[dict[str, float]]) -> list[tuple[str, float, float]]:
    """(feature, mean, sd) across folds, largest absolute mean first."""
    out = []
    for f in FEATURE_NAMES:
        vals = [d.get(f, 0.0) for d in imps]
        out.append((f, float(np.mean(vals)), float(np.std(vals))))
    return sorted(out, key=lambda t: -abs(t[1]) if not math.isnan(t[1]) else 0)


# -- M6c part 2: cap sizes (Russell chose option B, 2026-11-15) ---------------------------------

PERFORMER_CAPS: Sequence[int | None] = (2, 3, 4, None)  # "other" context; shipped: 2
LECTURE_CAPS: Sequence[int | None] = (5, 8, None)  # lecture context; shipped: 5
TOLERANCE = 0.01  # prefer the tightest caps within this much training NDCG of the best


def cap_grid() -> list[dict[str, int | None]]:
    """Tightest first (more variety first), so ties and near-ties go to tighter caps."""
    grid = [{"other": o, "lecture": lec} for o in PERFORMER_CAPS for lec in LECTURE_CAPS]
    big = 10**6
    return sorted(
        grid, key=lambda c: ((c["other"] or big) + (c["lecture"] or big), c["other"] or big)
    )


def cap_label(c: dict[str, int | None]) -> str:
    def f(v):
        return "none" if v is None else str(v)

    return f"performer {f(c['other'])}, course {f(c['lecture'])}"


def score_caps(pools: list[Pool], caps: dict[str, int | None]) -> dict[str, float]:
    """Mean NDCG@10 / P@10 / bad@10 and variety of v0 + these caps (no MMR) over ``pools``."""
    rank = R.picker(R.v0_scores, None, mmr=False, cap_values=caps)
    res = M.per_pool(pools, rank)
    var = [R.variety(rank(p)) for p in pools]
    return {
        "ndcg": float(np.mean([r["ndcg"] for r in res])),
        "p": float(np.mean([r["p"] for r in res])),
        "bad": float(np.mean([r["bad"] for r in res])),
        "groups": float(np.mean([v[0] for v in var])),
        "largest": float(np.mean([v[1] for v in var])),
    }


def choose_caps(train: list[Pool], tolerance: float = TOLERANCE) -> dict[str, int | None]:
    """The tightest caps whose training NDCG is within ``tolerance`` of the best."""
    grid = cap_grid()
    scores = [score_caps(train, c)["ndcg"] for c in grid]
    best = max(scores)
    for c, v in zip(grid, scores, strict=True):
        if v >= best - tolerance:
            return c
    return grid[-1]


@dataclass
class CapEval:
    chosen: list[dict[str, int | None]] = field(default_factory=list)
    results: dict[str, list[dict]] = field(default_factory=dict)


def caps_leave_one_out(pools: list[Pool]) -> CapEval:
    """Nested choice of caps per fold, scored on the held-out current video, next to the
    shipped caps and no caps."""
    out = CapEval()
    shipped = {"other": dv_caps("other"), "lecture": dv_caps("lecture")}
    for _, train, test in M.leave_one_seed_out(pools):
        if not train or not test:
            continue
        c = choose_caps(train)
        out.chosen.append(c)
        for name, caps in (
            ("chosen caps (nested)", c),
            ("shipped caps (performer 2, course 5)", shipped),
            ("no caps", {"other": None, "lecture": None}),
        ):
            rank = R.picker(R.v0_scores, None, mmr=False, cap_values=caps)
            out.results.setdefault(name, []).extend(M.per_pool(test, rank))
    return out


def dv_caps(kind: str) -> int:
    from sidekick.recommend import diversity as dv

    return dv.CAPS[kind]
