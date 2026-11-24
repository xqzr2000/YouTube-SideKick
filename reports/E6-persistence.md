# E6 — Why does YouTube keep recommending what I don't want?

*M7 · data up to 2026-11-24 · command: `just e6` · one user, one machine (WSL2)*

## Question

YouTube's up-next lists kept carrying videos the profile says not to show: late-night
political clips under a statistics lecture, sponsor-heavy science videos under a reinforcement
learning course. **Is that the same few videos stuck to a context, or something else?** And
does a 👎 make it stop?

## Setup

- **Data:** every YouTube list SideKick logged, with no new requests:
  - the signed-out up-next of each fetched watch page (`upnext`, 852 sightings);
  - the signed-in list read from the open tab in panel computes (`tab`, 63 sightings).

  In all: 915 sightings, 43 computes, 15 current videos, 507 distinct videos.
- **Classes per sighting**, from what is known in *that* context:
  - **unwanted** (249): labeled bad for this current video, dropped by Stage C in this compute
    (Don't-show, pundit, promo, clickbait, like a 👎'd video), or 👎'd;
  - **wanted** (305): labeled good for this current video;
  - **meh** (200) and **unknown** (161).
- **Measures:**
  - **Stickiness:** share of a compute's YouTube videos still in YouTube's lists the next
    time the same current video is computed.
  - **Spread:** the number of different current videos a video appeared under.
  - **Channels:** how many current videos each channel's unwanted videos appeared under.
  - **Medians per class:** up-next position, age when seen, and views.
- **Uncertainty:** 95 % cluster bootstrap over current videos (resample current videos, pool
  their counts), because sightings under one current video aren't independent.

## Results

**1. Unwanted videos are less sticky, not more.**

| class | still there next compute |
|---|---|
| unwanted | **32 %** [20, 42] (n 137) |
| wanted | **64 %** [56, 81] (n 187) |
| meh | 57 % [25, 76] (n 123) |
| unknown | 13 % [6, 20] (n 106) |

For the same current video, YouTube keeps its on-topic picks (the next lecture of a course,
the same comedian) and rotates the rest. The "it keeps coming back" impression doesn't come
from one stubborn video in one context.

**2. Unwanted videos spread across contexts.**

| class | videos | under ≥ 2 current videos | mean current videos |
|---|---|---|---|
| unwanted | 164 | **13 %** | **1.26** |
| wanted | 130 | 8 % | 1.12 |
| meh | 82 | 9 % | 1.15 |
| unknown | 131 | 1 % | 1.01 |

**3. A few channels account for the cross-context push.**

| channel | current videos with its unwanted videos | all its sightings unwanted? |
|---|---|---|
| Jimmy Kimmel Live | **9 of 15** | yes (26 of 26) |
| Veritasium | 4 | 13 of 14 |
| NDC Conferences | 4 | 5 of 5 |
| 3Blue1Brown | 3 | 10 of 19 |
| MIT OpenCourseWare | 3 | 8 of 39 |

Two single clips show the pattern at its strongest. "Trump Blames Everyone But Himself…" was in
11 computes under 6 different current videos, and "Trump Rants & Raves…" was in 8 computes
under 8, from statistics lectures to cooking. For both, the most common reason was a Stage C
Don't-show drop.

The channel table also shows the limit of a channel view. 3Blue1Brown and MIT OpenCourseWare
are unwanted in some contexts and wanted in others (MIT OCW: 8 of 39 sightings), so "unwanted"
is about the context, not the channel.

**4. Unwanted videos are fresh and popular.**

| class | up-next position | age when seen | views |
|---|---|---|---|
| unwanted | 10 | **8 days** | **443 k** |
| wanted | 9 | 688 days | 269 k |
| meh | 11 | 297 days | 302 k |
| unknown | 12 | 128 days | 297 k |

YouTube doesn't hide them low in the list (median position 10 vs 9). What sets them apart is
recency and reach: the median unwanted video was a week old, and the median wanted one almost
two years. That matches E4 part 2's feature screen, where older videos were more often good
(AUC 0.64).

**5. Signed in vs signed out: inconclusive.** The signed-in list was 37 % [15, 70] unwanted,
the signed-out list 27 % [20, 35]. That's only 63 signed-in sightings from 3 panel computes
(+4 points on the paired difference), far too few to say personalization makes it better or
worse.

**6. After a 👎: too early to tell.** Ten videos have been 👎'd. Three were seen again in
YouTube's lists once each after the 👎, and no other video from their channels was. That's a
handful of computes since the 👎s, so it says nothing yet. Three of the ten have no stored
title or channel, probably early API tests.

## Answer

YouTube "keeps recommending" unwanted videos through **breadth, not stickiness**. Within one
context it is the wanted videos that recur. The unwanted ones are mostly **fresh, popular
videos from a few channels, shown under unrelated current videos**: a late-night show reaches
9 of 15 contexts, and a single clip turns up under 8 different videos within days. That fits
a system mixing a context-free "what's popular and new right now" signal into every up-next
list, alongside the context-specific "watch next" one.

## Decision

- **No change to the filters now.** Stage C already catches the main case: every Kimmel
  sighting was dropped or labeled bad. The cost is enrichment budget: each such video is
  pre-scored, possibly fetched, and only then dropped.
- **A candidate improvement for later: a channel prior.** All 26 of Kimmel's sightings were
  unwanted, under 9 different current videos, and its most persistent clips were dropped by
  the profile's political Don't-show line, which applies whatever the context. A channel whose videos keep getting dropped that way across contexts
  could be dropped in Stage A, before any fetch. Spec §14 lists "channel-level priors" as
  later work. It must stay context-aware: MIT OpenCourseWare and 3Blue1Brown are unwanted in
  some contexts and wanted in others.
- **Freshness as a signal** is already in the M6 dataset (`age_days_log`). This report adds a
  reason it works: fresh content is how context-free pushes arrive.

## Threats to validity

- **One user, 15 current videos, 43 computes,** most from the CLI. Panel (signed-in) data is
  thin, so section 5 is open.
- **Unwanted is partly defined by SideKick's own filters** (Stage C drops). A video the filter
  drops counts as unwanted even if it was never labeled. That matches the profile's intent, but
  it ties this experiment to the filter's thresholds.
- **Labels are per context,** and many sightings have none (161 unknown). Unknown videos are
  mostly one-offs (13 % stickiness, 1 % spread), so they don't change the picture, but they
  dilute the class counts.
- **Computes cluster in time:** many were made on a few evenings. Stickiness over days or
  weeks may differ from stickiness over hours.
- **Signed-out lists are the bulk of the data;** the signed-in list may behave differently
  once there are more panel computes.
