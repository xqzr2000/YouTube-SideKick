# E3 — How much can one compute fetch and still finish in 90 s?

*M4 · 2026-11-05/06 · data: one user's computes on the primary dev machine (Windows 11 + WSL2,
Ryzen 5 3400G); aggregate numbers only · reproduce with `just e3 --md` and
`just recommend <id> --cold`*

## Question

The side panel promises a refresh in **≤ 90 s**. In a compute, almost all of the time goes to
fetching pages. There is the current watch page, one or more search pages, and one watch page
for each candidate SideKick *enriches*: description, exact duration, category, language
check, SponsorBlock. Enrichment makes filtering and ranking better, so E3 asks two things:

1. how many pages fit in 90 s;
2. what a smaller enrichment budget costs in what gets shown.

The spec framed E3 as "backfill rounds and `min_pool`". The data below show that the
enrichment budget is the setting that matters.

## Setup

- **No new sweep.** Every compute already logs `pages_fetched` and `duration_ms`, so the
  fit uses the 30 finished CLI computes from 2026-11-03 to 2026-11-06. That's 18 `just fetch`,
  9 `just recommend` (warm or partly warm cache) and 3 `just recommend --cold`. A cold sweep
  over several budgets and seeds would have cost about 350 signed-out requests, the largest
  burst so far.
- **Model:** least squares, `seconds = a + b · pages`. The decision uses the **one-sided 95 %
  prediction bound** for a single new compute, not the fitted mean, because the 90 s is a
  promise about the slow computes.
- **Cold checks:** three seeds with the 24 h page cache skipped (a statistics lecture, a
  reinforcement-learning course lecture, a stand-up comedy clip). Embeddings and models stay
  cached.
- **Quality cost:** each of the 12 logged `just recommend` computes is replayed at budgets
  10–30. The budget goes out in pre-score order, round by round, so a Stage B decision's log
  order is its budget position. For each budget the replay counts:
  - shown videos (top 10) that would have gone unenriched;
  - of those, the ones labeled *good* in that seed's context;
  - pages saved.

## Results

**Timing is a straight line in pages.**

| | |
|---|---|
| intercept *a* | 2.0 s (models, parsing, scoring) |
| slope *b* | **2.60 s per page** (1–2 s polite delay + request, parse, SponsorBlock) |
| R² | 0.98 |
| residual SD | 3.5 s |
| most pages under 90 s (95 % bound) | **31** (mean 83 s, bound 89 s) |

The two slowest computes relative to the line ran about 10 s over (13 pages in 46 s, 20 pages
in 64 s). Per-stage times aren't logged, so the cause is open. Likely candidates are a
first-time e5 embedding of many new videos and SponsorBlock latency.

**The cold checks land on the line.**

| seed | pages | measured | line | 95 % bound |
|---|---|---|---|---|
| statistics lecture | 27 | 72.2 s | 72 s | 79 s |
| RL course lecture | 27 | 70.3 s | 72 s | 79 s |
| comedy clip | 27 | 71.4 s | 72 s | 79 s |

In each one, round 0 (enriching 19–20 up-next videos) took 51–53 s and the round-1 search took
12–16 s.

**Budget.** A cold compute fetches 1 current page + its searches + one page per enriched
candidate. With the median search count of the logged recommends (1), the largest budget
under the bound is **29**. With the worst case of 4 searches (topic + interest query in round 1,
one query each in rounds 2–3) it is **26**.

**What a smaller budget costs** (12 computes, 120 shown videos, 42 of them labeled good):

| budget | cold s (mean / bound) | shown left unenriched | labeled-good left unenriched | pages saved (as logged) |
|---|---|---|---|---|
| 10 | 33 / 39 | 60 / 120 | 13 / 42 | 121 |
| 15 | 46 / 52 | 46 / 120 | 10 / 42 | 78 |
| 20 | 59 / 65 | 32 / 120 | 6 / 42 | 33 |
| **25** | **72 / 79** | **5 / 120** | **0 / 42** | 0 |
| 30 | 85 / 92 | 1 / 120 | 0 / 42 | 0 |

**Backfill rounds.** The round-1 search pays for itself. In the three cold checks, 3, 5 and 3
of the 10 shown videos came only from search, and 2, 2 and 0 more from both lists. Rounds 2–3
never ran: round 1 already left 26–30 survivors, above `min_pool` 10. On these seeds the
`min_pool` question doesn't arise.

## Decision

- **Keep the enrichment budget at 25.**
  - Cold computes: about 72 s on average, 95 % bound 79 s.
  - Worst case (4 searches, 30 pages): bound ≈ 87 s, still under 90.
  - Going down to 20 saves about 13 s, but 32 of 120 shown videos (6 labeled good) go out
    without enrichment.
  - Going up to 30 changes almost nothing that's shown and breaks the bound.
- **Keep the 60 s deadline for rounds 2–3** as a second guard. Total pages are already capped
  at about 30 by construction: the budget is shared across rounds, and snowball reuses
  fetched pages.
- **No `min_pool` change:** rounds 2–3 didn't trigger on any seed so far.

## Follow-up

The 5 shown-but-unenriched videos all come from search. Round 0 spends 19–20 of the 25 slots on
up-next before the search results are seen, so 4–6 search results per cold compute go
unenriched. Some still rank high: one comedian clip at #2, and a running-coach video at #10 in
the comedy context. That one came from a query built from the joke's subject.

A cheaper allocation is to run the round-1 search *before* enrichment and pre-score the
combined pool. The page count stays the same, and the budget goes to the best of both lists,
with recurrence known. Precision@10 on labeled top-10s is the measure, before and after.

## Threats to validity

- **One machine:** the Crostini (ChromeOS) numbers are still to come, and the 90 s budget is
  set against the weaker machine.
- **Few cold computes:** only three, all with 27 pages. The line is fitted mostly on warm and
  partly warm computes; the cold checks confirm it but don't widen its range.
- **"Unenriched" is not "dropped":** an unenriched video can still be shown, just with less
  information. Only 42 of 120 shown videos have *good* labels, and the rest aren't all
  labeled, so "labeled-good left unenriched" undercounts.
- **External latency can change:** YouTube's response time and the polite delay set the slope.
  A slower network or a bot page would move it.
