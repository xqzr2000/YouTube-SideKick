# E4 — Does SideKick's ranking beat YouTube's own up-next?

*Part 1 (M4, 2026-11-06): the hand-weighted v0 score against YouTube's up-next. Part 2 (M6):
v0 against a learned ranker. Data: one user's labels on one machine (private; aggregate
numbers only). Reproduce with `just recommend <id>`, `just label --top10` and
`just label --stats`.*

## Question

SideKick earns its place only if its 10 videos are better than what YouTube already puts
next to the player. Part 1 measures that for the v0 ranker. v0 combines:

- topic anchor, seed and interest similarity, recurrence, same channel, persistence and views
  (fixed weights);
- Stage C soft filters;
- MMR with group caps.

## Setup

- **Seeds (7):**
  - three on-interest: a university Bayesian-statistics lecture, a reinforcement-learning
    course lecture, an introductory chemometrics talk on PCA;
  - four off-interest: a stand-up comedy clip, a local TV newscast, a restaurant cooking show,
    a jazz guitar concert.
- **Lists:** for the newest `just recommend` of each seed, two lists are compared:
  - **SideKick**'s top 10;
  - **YouTube**'s first 10 up-next on the signed-out watch page, skipping what nobody would be
    shown (the current video, Shorts, live streams, premieres).
- **Labels:** `just label --top10` asked one question about every video in either list:
  *while watching the seed, would you want this shown next?* good / meh / bad. The two lists
  were mixed and unmarked, with no scores shown, so neither could be recognized. Of the 140
  list entries, 23 were in both lists. All 140 have labels.
- **Metrics:** **P@10** = share of the 10 labeled good; **bad@10** = share labeled bad. Both
  matter, because removing bad videos is half of what SideKick is for. Uncertainty comes from
  a percentile bootstrap over seeds (20 000 resamples), because seeds, not videos, are the
  independent units.

## Results

| seed | SideKick P@10 | SideKick bad@10 | YouTube P@10 | YouTube bad@10 | in both lists |
|---|---|---|---|---|---|
| Bayesian lecture | 60 % | 10 % | 20 % | 10 % | 4 |
| RL course lecture | **100 %** | 0 % | 60 % | 20 % | 4 |
| comedy clip | 80 % | 20 % | 80 % | 20 % | 4 |
| chemometrics talk | 10 % | **0 %** | 10 % | 80 % | 0 |
| local newscast | 40 % | 60 % | 30 % | 50 % | 3 |
| cooking show | 80 % | 0 % | 70 % | 10 % | 4 |
| jazz concert | 90 % | 10 % | 70 % | 30 % | 4 |
| **all** | **66 %** | **14 %** | **49 %** | **31 %** | 23 |

- **P@10 +17 points**, 95 % bootstrap interval over seeds [+6, +29]. SideKick is ahead on 5
  seeds and tied on 2, never behind (one-sided sign test p ≈ 0.03).
- **bad@10 −17 points** [−40, 0]. The bad share falls on 4 seeds, is unchanged on 2, and is
  one video worse on the newscast.
- **The largest gains are on the lectures** (+40 points on both). This is where YouTube drifts
  to general "smart" content. The topic anchor and the same-course cap keep SideKick on the
  course.

**Where it doesn't help yet:**

- **Chemometrics talk: 10 % good on both lists, for different reasons.** YouTube's list was
  80 % bad: a company-valuation talk, French politics, unrelated chemistry. SideKick removed
  all of that, but it filled every slot with generic PCA explainers from the search round,
  which were labeled meh ("drifted away from the course"). The one chemometrics video labeled
  good ranked 15th. It had the pool's highest similarity to the "chemometrics" interest, but
  interest similarity carries 15 % of the v0 score in interest mode, against 35 % for the
  topic anchor. Here the topic is "PCA", so the method beat the domain.
- **Local newscast: 60 % bad.** The labels accept other local newscasts and tunnel-related
  stories, and reject national politics and other stories. That is consistent with the news
  scope set in M4: stop pundits, not solve news. Same-story videos are missing from both
  lists (see the topic-anchor findings).

## Decision

- **v0 ships as the M5 ranker.** It beats YouTube's own list on precision and on removing bad
  videos, with no seed worse on precision.
- **No weight changes now.** The chemometrics case suggests interest similarity should count
  for more than topic in interest mode, but that is one seed. Hand-tuning weights on 7 seeds
  would overfit. The learned ranker (M6, part 2 of this report) will set them from these
  labels plus clicks from the panel.

## Threats to validity

- **Small and single-user:** 7 seeds, one labeler, one machine. The interval is wide, and the
  seed set doesn't represent everything one watches.
- **Signed-out up-next as the baseline:** in M5 the panel uses the signed-in up-next from the
  open tab, which is personalized and may be a stronger baseline.
- **One compute per seed:** up-next changes from run to run, and so does SideKick's search
  pool. A different run could move a seed by ±10–20 points.
- **The labeler built the profile:** "good" partly means "matches what I told SideKick". That
  is the intended use, but it favours SideKick over YouTube's broader list.
- **Labels around self-promotion and "influencer" content are noisy** (the same pitch got
  good once and bad once). They mostly move videos between good and meh, not into bad.

---

# Part 2 (M6) — v0 vs a learned ranker, and what the diversity step costs

*Data: 13 current videos, each with its newest ranked pool fully labeled in the M6 labeling app
(394 candidates: 192 good / 113 meh / 89 bad). Commands: `just eval`, `just eval --learn`,
`just eval --caps` (2026-11-15).*

## Question

1. Does a ranker learned from the labels beat v0 on current videos it never saw?
2. What do the M4 diversity rules (MMR with group caps: performer ≤ 2, course ≤ 5) cost?

## Setup

- **Metric:** NDCG@10 with gains good 2 / meh 1 / bad 0, per pool, averaged over pools; P@10
  (good share of the top 10) and bad@10 alongside. Unlabeled videos would be dropped after
  ranking (condensed lists); here every pool is complete.
- **Uncertainty:** 95 % bootstrap intervals over current videos; differences are paired.
- **Protocol (agreed before the run):** leave one current video out (13 folds) as the main
  test, a time split (9 older train, 4 newer test) as the check. Anything tuned (the MMR
  window, the caps) is chosen inside each fold on its training pools only.
- **Rankers:**
  - v0's score, with and without the M5 pick step;
  - logistic regression (31 features, median imputation + missing indicators, C 0.3);
  - LightGBM lambdarank (same features, gains [0, 1, 2], 7 leaves, 150 × 0.05).
- **Replay:** the pick step is redone from logged scores, diversity groups and cached e5
  vectors. It reproduces the logged top 10 on 85 % of the pools.

## Results

**What v0 knows** (all 13 pools; no training involved):

| ranker | NDCG@10 | P@10 | bad@10 |
|---|---|---|---|
| v0 as shipped (v0 + MMR + caps) | 0.816 [0.724, 0.903] | 65 % | 16 % |
| v0 score only | **0.895** [0.830, 0.953] | 73 % | **5 %** |
| seed similarity only | 0.872 | 72 % | 6 % |
| topic only | 0.847 | 68 % | 10 % |
| interest only | 0.700 | 55 % | 18 % |
| random order | 0.661 | 50 % | 22 % |

**Learned rankers, leave one current video out:**

| ranker | NDCG@10 | P@10 | bad@10 |
|---|---|---|---|
| v0 score only | **0.895** [0.830, 0.953] | 73 % | 5 % |
| v0 + caps (M5) | 0.825 [0.742, 0.905] | 65 % | 15 % |
| v0 + caps + MMR (M5, as shipped) | 0.823 | 65 % | 16 % |
| logistic regression | 0.874 [0.793, 0.946] | 72 % | 5 % |
| LightGBM lambdarank | 0.878 [0.789, 0.958] | 75 % | 7 % |
| logistic regression + caps | 0.811 | 62 % | 13 % |
| LightGBM lambdarank + caps | 0.821 | 67 % | 13 % |

Paired: v0 score only − v0 + caps **+0.070 [0.014, 0.140]**; v0 + caps − v0 + caps + MMR
+0.003 [−0.021, 0.028]; learned + caps − v0 + caps −0.014 (LR) and −0.004 (LightGBM), both
intervals across 0. The time split agrees: v0 score only 0.847, LightGBM 0.818, LR 0.809.

**Cap sizes** (v0 scores, no MMR; descriptive over all 13 pools):

| performer cap | NDCG@10 | bad@10 | groups in top 10 | largest group |
|---|---|---|---|---|
| 2 (M5) | 0.825 | 15 % | 9.0 | 1.8 |
| 3 | 0.845 | 12 % | 8.3 | 2.4 |
| 4 | 0.851 | 12 % | 7.8 | 3.0 |
| none | **0.895** | **5 %** | 6.1 | 4.8 |

The course cap (5 / 8 / none) changed nothing: in the 3 lecture pools no course reached 5 in
the top 10. Chosen inside each leave-one-out fold (the tightest caps within 0.01 NDCG of the
best on the training pools): **no performer cap, course cap 5, in all 13 folds**; held out,
+0.070 [0.014, 0.140] over the M5 caps.

**What the models lean on** (mean over folds): logistic regression, same channel +1.36 per SD,
seed percentile +0.81, age +0.35, sponsor share −0.35; seed_sim −0.27 (collinear with the seed
percentile, so the two split the effect). LightGBM: seed percentile 33 % of split gain, then
raw seed similarity, description length, same channel and age.

## Decision

- **The learned rankers don't ship.** On held-out current videos neither beats v0's score
  (0.874 / 0.878 vs 0.895). With 13 contexts, 31 features are too many to learn reliably, and
  v0's hand-set weights already carry the main signal (similarity and topic). Revisit at
  about 30 labeled current videos.
- **The pick step changes (Russell, option B: "let the data pick"):** v0 score order with the
  course cap of 5, **no performer cap**, **MMR off**. That is exactly the configuration that
  was measured, rather than leaving MMR on with no caps (untested, and the setup in which MMR
  promoted a "binaural beats" outlier).
- **Expected effect:** NDCG@10 0.825 → 0.895 and bad@10 15 % → 5 % on these pools; lists get
  less varied (largest group 1.8 → 4.8 videos on average).

## Threats to validity

- **Item labels can't see list-level variety.** "Would I want this video next?" is asked one
  video at a time, so NDCG is biased against diversity by construction. Whether five clips of
  one comedian make a worse list is a product question. Watch for it in use (👎 notes, clicks
  on the panel), and re-add a looser cap if lists feel repetitive.
- **Small and single-user:** 13 current videos, one labeler; wide intervals.
- **Same labels chose and judged the caps** in the descriptive grid; the nested
  leave-one-out estimate (+0.070) is the honest one.
- **The test pools were picked by v0:** candidates come from v0's pools, so a ranker that
  would have searched differently can't be judged here.
