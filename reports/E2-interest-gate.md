# E2 — Can the interest gate tell on-interest videos from the rest?

*M3 · 2026-11-04/05 · data: one user's seed labels on one machine (private; aggregate numbers
only) · reproduce with `just sim --eval --md`, `just tune` and NB03 section 2*

## Question

Every compute starts with the **interest gate**: is the video being watched close enough to one
of the profile's Interests lines (*interest mode*: backfill may blend in those interests) or
not (*filter-only mode*: only clean the up-next list and stay on the video's own topic)? The
gate exists so a cooking video doesn't get Bayesian-statistics recommendations mixed in. E2
asks whether a similarity threshold τ can make that call, and where τ should sit.

## Setup

- **Gate score:** the best similarity between the current video's text and any Interests line
  ("reinforcement learning", "Bayesian statistics", "chemometrics").
- **Ground truth:** `just label` seed question, *"does this current video match one of your
  profile's Interests?"* (yes / partly / no), shown together with the Interests lines. An early
  round without them got "yes" for a comedy clip, so the prompt was changed and the seeds were
  relabeled.
- **Seeds (7):** yes: a university Bayesian-statistics lecture, a reinforcement-learning course
  lecture, an introductory chemometrics talk on PCA for spectroscopy. No: a stand-up comedy
  clip, a local TV news broadcast, a restaurant cooking show, a jazz guitar concert.
- **Representations:** as in E1, raw and centered.

## Results

| representation | variant | lowest *yes* | highest *no* | gap |
|---|---|---|---|---|
| tfidf | raw | 0.00 | 0.00 | 0.00 |
| tfidf | centered | −0.01 | −0.00 | −0.00 |
| lsa | raw | 0.01 | 0.02 | −0.01 |
| lsa | centered | −0.03 | 0.01 | −0.04 |
| e5-small-q | raw | 0.84 | 0.81 | 0.03 |
| e5-small-q | centered | 0.21 | 0.02 | **0.19** |
| e5-small-qp | raw | 0.85 | 0.80 | 0.04 |
| e5-small-qp | centered | 0.23 | 0.03 | **0.20** |

Live computes logged the same picture: on-interest lecture 0.37–0.40 (interest mode), comedy
0.02–0.03, news −0.07, cooking −0.03 and jazz −0.02 (all filter-only), with τ ≈ 0.10–0.11 set by
`just tune` as the midpoint between the lowest yes and the highest no seed.

## Findings

1. **Centered e5 separates all 7 seeds with a wide margin** (gap 0.19–0.20: yes ≥ 0.21, no ≤
   0.03). Raw e5 separates them too, but by only 0.03–0.04, because of e5's bunched-up cosines,
   which would make τ fragile.
2. **TF-IDF and LSA can't do this job at all.** A three-word interest line and a video's text
   rarely share words ("chemometric" vs "chemometrics" don't even match without stemming), so
   every seed scores ≈ 0.
3. **Off-interest seeds sit below zero once centered.** News, cooking and jazz score −0.07 to
   −0.02, comfortably away from τ, while the closest *no* is the comedy clip at 0.02–0.03.

## Decision

- The gate uses **e5-small-qp, centered**, with τ set by `just tune` at the midpoint between the
  lowest *yes* and highest *no* seed (≈ 0.10–0.13 on current data). Without the e5 model the
  gate falls back to TF-IDF, which on this evidence almost never enters interest mode. That
  fails safe: filter-only is the conservative mode.

## Limits and next steps

- **Seven seeds, all easy cases.** The *yes* seeds are core-topic lectures and the *no* seeds are
  unrelated genres. The hard cases are untested: a general machine-learning lecture, a physics
  video, analytical chemistry that isn't chemometrics. Those should be labeled *partly*. Next
  data collection: 3–5 such borderline seeds.
- **This measures the gate's decision, not its effect.** Whether interest mode improves the
  final list depends on what backfill does with it, which arrives in M4. E2 part 2 (M4)
  compares the shown lists' labels in interest vs filter-only mode for on-interest seeds.
