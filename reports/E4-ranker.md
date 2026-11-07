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
