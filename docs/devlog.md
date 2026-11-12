# Dev log

A running record of how YouTube-SideKick is being built: what was done each session, the
decisions made and why, and the dead ends and how they were fixed. Newest entry last.

**Roles.** The repo owner (xqzr2000) makes the decisions, reviews and merges every pull request,
and runs everything on the real machines (Linux backend under WSL2 or ChromeOS, Chrome) and
reports the results. Claude (Anthropic's AI assistant) writes the code, tests and docs on
branches and opens the pull requests. Claude's environment cannot reach YouTube, so all
evidence from real pages comes from the owner's runs.

| PR | Milestone | What |
|---|---|---|
| #1 | M0 | Repo scaffold: uv, just, ruff, pytest, CI, `SIDEKICK_HOME` config, `/health`, setup docs |
| #2 | M1 | Models, watch/search parsers, raw log, `just record`, `just parse-check`, synthetic fixtures |
| #3 | M1 | Fix: `just` recipes lost argument quoting |
| #4 | M1 | Parser for `lockupViewModel` up-next items; `parse-check --skeleton` |
| #5 | M1 | Fix: up-next view counts (bare `33K` + accessibility label) |
| #6 | M1 | Polite fetcher, `just fetch`, midpoint thumbnails, age helper, explore loader, synthetic sample, NB01 |
| #7 | — | Dev log |
| #8 | M2 | Profile parser + hashing, SQLite store (schema v1), compute logging, `just profile`, `just db` |
| #9 | M2 | Stage A hard filters, fastText language ID, `just models` |
| #10 | M2 | Stage B enrichment (24 h cache), SponsorBlock, schema v2 |
| #11 | M2 | NB02 (what gets filtered and why), `load_decisions` |
| #12 | M3 | `just label` hand labels (current-video and candidate questions), schema v3 |
| #13 | M3 | TF-IDF / LSA / multilingual-e5-small representations, embeddings cache (schema v4), `just sim` |
| #14 | M3 | `just sim --eval`: representations scored against labels; centering; e5 query/passage variant |
| #15 | M3 | Interest gate + Stage C soft filters, `just tune` (precision-first), language drops moved to Stage B, schema v5 |
| #16 | M3 | `just tune` clickbait floor 0.50 → 0.65 |
| #17 | M3 | `just tune` Don't-show floor 0.15 → 0.20 |
| #18 | M3 | NB03, bootstrap intervals, `just sim --eval --md` |
| #19 | M3 | E1 and E2 reports, this entry |
| #20 | M3 | Language rule no longer applies to Music-category videos (language kept as a feature) |
| #21 | M3 | …and not in a music context either (current video is Music), for concerts filed under Entertainment |
| #22 | M4 | Topic anchor (lead story, entities, keyphrases), spaCy `en_core_web_sm`, `just topic` |
| #23 | M4 | Topic anchor fixes from the first real run: show-name segments, channel tags, series outside news, places, `just topic --text` |
| #24 | M4 | News context: Don't-show matches drop only non-news and commentary (pundit) titles; `just tune` ignores news-context labels for Don't-show |
| #25 | M4 | `just recommend`: one full compute with backfill round 1, a cheap pre-score for the enrichment budget, v0 score, per-channel cap |
| #26 | M4 | Round 1 always searches; rounds 2–3 (Rocchio + snowball) under a deadline; MMR with group caps (same course ≤ 5, same performer/show ≤ 2) |
| #27 | M4 | Fix the caps: performer names trimmed and matched across the pool; same-course lectures not redundant in MMR |
| #28 | M4 | Diversity groups also match performer names in the channel |
| #29 | M4 | `just e3` (time vs pages, budget under 90 s, budget replay), `just recommend --cold` |
| #30 | M4 | E3 report; `just label --top10` and precision@10 against YouTube's up-next |
| #31 | M4 | E4 part 1 report (v0 vs YouTube's up-next); M4 closed |
| #32 | M5 | Backend API: computes as background jobs with progress steps, feedback (click, 👎 + note) |
| #33 | M5 | Chrome extension (WXT + TypeScript): context menu, side panel, settings; CI job |
| #34 | M5 | English only: Stage A script rule; English and cross-lingual round-1 queries; ChromeOS zip setup |
| #35 | M5 | M5 closed; ChromeOS paused; README status; this entry |

---

## Session 1 — 2026-11-02: planning (no code)

**Done**
- Designed v1 over several rounds of Q&A and wrote the spec; created this repo (MIT).
- Studied Andrej Karpathy's HackerTube (launched 2026-10-31) for ideas.

**Decisions and reasons**
- *A data science project first, a browser tool second.* The goal is to show data collection,
  feature engineering, classical NLP, ranking, evaluation and preference drift.
- *No LLM inference; small local embedding models only.* Classical, cheap, runs on a CPU.
- *No accounts or API keys; polite scraping of public pages.* The terms-of-service gray area is
  stated in the README.
- *Rank by the user's own signals, not engagement.* Profile rules, 👎 notes and clicks drive the
  ranking; views and likes are weak features only.
- *Borrowed from HackerTube's data pipeline:* recurrence scoring, midpoint thumbnails, local
  state, a small embedding model and a published sample dataset.
- *A 👎 requires a short note,* so every negative comes with text the models can learn from.
- *An interest gate* keeps personal interests from taking over recommendations for unrelated
  videos.

## Session 2 — 2026-11-02: second dev environment, M0 scaffold

**Done**
- Added a Windows 11 + WSL2 development environment alongside ChromeOS + Crostini.
- PR #1: uv project, justfile, ruff + pytest + GitHub Actions, private-folder config, FastAPI
  `/health`, setup docs for both environments. Verified on WSL2, including `/health` from
  Chrome on Windows.

**Decisions and reasons**
- *Windows 11 + WSL2 becomes the primary dev box; ChromeOS stays supported.*
- *Edge was considered and dropped.* Chrome runs on both machines, so a single Chrome extension
  build serves both and only the backend host differs.
- *Private data stays on each machine, with no sync.* The public synthetic sample is what makes
  results reproducible.
- *The repo and private folder live inside the Linux filesystem (not `/mnt/c` on Windows)* for
  fast file I/O and safe SQLite locking.
- *Changes arrive only as reviewed pull requests;* the dev machines only clone and pull.
- *Minimal tooling to start* (ruff, pytest, CI), in line with a ~3 h/week budget.
- *A real `/health` endpoint in M0,* so browser → backend reachability is tested with the actual
  server rather than a stand-in.

**Dead ends and fixes**
- An early reachability check looked like `/health` was failing (404). The backend had never
  started: `just` and `uv` weren't installed yet, and the 404 came from a plain file server used
  for an earlier test. Installing the tools and running from the repo fixed it.

## Session 3 — 2026-11-02/03: M1 parsers, fetcher, NB01

**Done**
- PR #2: data models; parsers for watch pages (metadata and up-next) and search pages; a gzip
  raw log with 90-day pruning; `just record` to save real pages to the private folder and
  `just parse-check` to report which fields were extracted.
- The owner recorded 4 real watch pages and 2 searches; all came back with no consent page.
- PRs #3–#5 fixed problems those real pages exposed (below). Afterwards every real page parsed
  20/20 up-next videos with title, channel, duration, views and age.
- PR #6: polite fetcher, `just fetch`, midpoint thumbnails, an age helper, a data loader,
  a synthetic sample and the first notebook (NB01, candidate pools). 54 tests; CI also runs NB01.
- The owner ran `just fetch` live (2 pages in under 10 s), confirmed the midpoint thumbnails show
  mid-video frames, and opened NB01.

**Decisions and reasons**
- *M1 split into two PRs (parser, then fetcher)* so parsing could be checked on real pages
  before anything was built on top of it.
- *Real pages never enter the public repo.* They contain third-party titles and descriptions;
  tests use hand-built synthetic pages instead.
- *A browser-like User-Agent (overridable),* the option least likely to trigger bot pages.
- *The parser finds data by key name, not fixed paths,* to survive small layout changes.
- *The raw log keeps the extracted JSON, not full HTML:* about 10× smaller and still enough to
  replay the parser later.
- *Politeness defaults:* 1–2 s between requests, at most 60 pages per compute, and a 30-minute
  back-off after a consent page, bot check or HTTP 429.
- *Midpoint thumbnails (`hq2.jpg`)* instead of click-optimized custom thumbnails.
- *NB01 reads real data when present and falls back to a synthetic sample* stored in the same
  format as the raw log, so it runs the same code path for anyone who clones the repo.

**Dead ends and fixes**
- *Quoted arguments were split.* `just record ... --search "dry bar comedy"` would have reached
  Python as three words. Caught while reviewing the command before running it; fixed in PR #3.
- *0 up-next videos on every real page.* The parser targeted an older layout
  (`compactVideoRenderer`); YouTube now serves up-next entirely as `lockupViewModel`. Fixed in
  PR #4, which also added a tool to print the structure of a real item.
- *Up-next views 0/20.* Real pages show bare counts like `33K`; the word "views" appears only
  in an accessibility label. A targeted diagnostic that printed every count-like string with its
  JSON path found it; fixed in PR #5.
- *The synthetic sample was not being committed* because `.gitignore` excluded `*.jsonl.gz`;
  added an exception for `data/sample/`.
- *`just nb` seemed to hang.* It starts a notebook server that runs until stopped; you open the
  printed URL in the browser.

**Findings so far**
- Up-next for an MIT Bayesian statistics lecture was mostly on topic, but 2 of 20 items were
  same-day political late-night clips: exactly what SideKick should filter out.
- 5 of 12 search results also appeared in up-next, early support for a recurrence feature.
- Search pages carried more Shorts than videos (25 vs 12, and 65 vs 25).

## Session 4 — 2026-11-03: M2 profile, store, Stage A and B filters, NB02

**Done**
- PR #8: `profile.md` parser (interests, rules, don't-show-me lines including dated 👎 notes),
  a whitespace-insensitive profile hash, an atomic writer for 👎 notes, and the SQLite store with
  all eleven tables from the spec. `just fetch` now logs every run as a compute.
- PR #9: Stage A hard filters and fastText language ID (`lid.176.ftz`, downloaded once).
- PR #10: Stage B enrichment from watch pages with a 24-hour cache, language on title +
  description, and sponsor share from SponsorBlock. Schema v2 adds the sponsor columns and
  migrates the existing database in place.
- PR #11: NB02, a notebook over the logged decisions: the funnel per compute, drop reasons,
  durations against the rules, title-only vs title + description language, sponsor share, and
  what still gets through.
- The owner checked each PR on real pages before the next one was built: 4 computes on the MIT
  Bayesian statistics lecture, 165 view snapshots, 132 decisions. 109 tests.

**Decisions and reasons**
- *M2 split into four PRs* (profile + store, Stage A, Stage B, notebook) so each could be
  checked on real data before anything was built on it. The notebook got its own PR because the
  enrichment PR already carried network code and a schema migration.
- *The profile parser is lenient.* A hand-edited file should never crash a compute; problems
  become warnings that `just profile` shows.
- *Compute ids are timestamps plus a random suffix* (`20261104T035843Z-ace1e3`), so the two
  machines' databases can be merged later without id clashes.
- *One decision per unique video per stage, with every reason kept* (`short;duration<min`), so
  the "why was this dropped" counts are complete rather than first-rule-wins.
- *Language acts only at ≥ 0.80 confidence.* Short English titles often score low with fastText
  (`MCMC` 0.12), while non-English titles score 0.90–0.99; a low-confidence title is kept and
  re-checked in Stage B with the description.
- *`fasttext-predict`* (an inference-only wheel) instead of `fasttext`, which needs a compiler.
- *Stage B drops only on hard facts* (language, live, premiere, exact duration). Sponsor share,
  age and views are features for later stages, not filters.
- *SponsorBlock via the hash-prefix endpoint*: only 4 hex characters of the video id's hash are
  sent. "Unknown" (API down) is stored as missing, never as 0, and the client stops after 3
  failures so a dead API can't eat the time budget.
- *Cheap up-next sightings never overwrite exact watch-page fields*, and view snapshots record
  whether the count was exact or rounded.

**Dead ends and fixes**
- *Tests started calling the real SponsorBlock* once enrichment became the default in
  `just fetch`. The CLI tests now inject a mock client, and no test touches the network.
- *The mock served the same watch page for every id*, which hid a real case: YouTube can return
  a page for a different video (redirect or removed video). Stage B now treats a mismatched id
  as a failed fetch instead of filing that data under the wrong id.
- *NB02 copied Stage A counts into the Stage B bar* for computes without enrichment; it now
  leaves the bar out.

**Findings**
- *Up-next changes from run to run.* Within about 20 minutes the same lecture's up-next gained
  and lost videos; political clips appeared in every run (one late-night clip in all four, and
  up to 3 of 20 in one run). None of them break a hard rule, so they reach the soft filters.
- *Time goes to page fetches.* A cold compute with 25 enrichments took 68 s for 27 pages
  (~2.5 s a page, mostly the polite delay); the same compute minutes later took 22 s with 17 of
  24 videos from the cache. At the spec's cap of 50 enrichments a cold compute would exceed the
  90 s budget, so the cap and the backfill rounds have to be tuned together (experiment E3).
- *Descriptions make language ID confident.* Most videos moved from 0.4–0.8 on the title alone
  to 0.85–0.97 with the description; a few lecture pages with boilerplate descriptions stay low.
- *SponsorBlock answered every request.* Most educational videos have no marked segments; the
  highest seen was 12%.
- *One borderline drop worth a look:* a two-hour lecture by a well-known statistics educator was
  removed by `max_duration: 90m`, the kind of false removal the evaluation will track.

## Session 5 — 2026-11-04/05: M3 labels, text representations, interest gate, Stage C, E1/E2

**Done**
- **Hand labels** (`just label`): two questions, with filter decisions hidden so the labels
  can't echo them. For each current video: *does it match one of your Interests?* For each
  candidate: *while watching this, would you want that shown next?* By the end there were
  95 candidate labels (42 good / 31 meh / 22 bad) and 7 current videos (3 on-interest, 4 not).
- **Text representations:** TF-IDF and LSA fitted on our own videos, and `multilingual-e5-small`
  (int8 ONNX through fastembed, no torch) with cached vectors that are recomputed when a
  video's text changes.
- **Interest gate and Stage C soft filters** in `just fetch`: Don't-show-me similarity,
  similarity to downvoted videos, a clickbait score and a promo score. Every score is logged
  whether it drops the video or not. `just tune` sets the thresholds from the labels.
- **NB03 and the first two experiment reports:** [E1](../reports/E1-representations.md) (which
  representation) and [E2](../reports/E2-interest-gate.md) (the gate).

**Decisions and reasons**
- *Measure before choosing.* Raw e5 cosines sit at 0.74–0.89 for almost any pair, so instead of
  hand-picking cut-offs, `just sim --eval` scores each representation against the labels with
  threshold-free AUCs (later with bootstrap intervals). That showed **mean-centering** triples
  the gate's margin, which is why it's used for every comparison with a profile line.
- *Precision-first soft filters with floors* (owner's call): each threshold sits just above the
  highest score of any video labeled good, so no good video is dropped. A floor keeps a small
  label set from pushing a threshold toward zero, and `just tune` also reports how many
  *unlabeled* videos each threshold would drop.
- *Language drops only in Stage B* (owner's call): on the title alone, a lecture titled with a
  course code was confidently classed as German.
- *Hand-weighted clickbait and promo scores for now:* about ten bad labels can't train a
  classifier. Their parts are logged so M6 can learn the weights.

**Dead ends and fixes**
- *The e5 model can't be downloaded in Claude's environment* (Hugging Face is blocked). A 28 KB
  stand-in model with the same shape exercises the whole fastembed path in CI. The real model
  ran first on the owner's machine (124 MB download in 7 s, ~130–170 ms per video on CPU).
- *The first seed answers were all "yes"*, including a comedy clip, because the question didn't
  show the Interests lines. The prompt now lists them, and `just label --seeds` re-asks.
- *A Don't-show-me line dropped three non-political stand-up clips.* The line mixed a topic and
  a genre ("political late-night clips"), and the embedding matched the genre. Rewording it to a
  single concept ("political commentary and election news") fixed it without code changes; the
  clickbait and Don't-show floors were then raised from evidence (#16, #17).

**Findings** (details in the reports)
- *The interest gate works on clear cases:* centered e5 separates 3 on-interest from 4
  off-interest current videos with a 0.19–0.20 margin; TF-IDF and LSA can't (≈ 0 for every
  seed).
- *Similarity to the current video* ranks good above the rest at AUC ≈ 0.75–0.78 for every
  representation. Series share words, so TF-IDF keeps up with e5.
- *Don't-show-me is context-dependent.* While watching local news, the owner labeled two news
  reports as good even though they score among the highest on the political line. A
  context-free line can't express "not unless I'm already watching this". Next experiment: a
  context-relative score (E1b).
- *Timing:* cold computes without backfill took 42–82 s; warm ones about 20 s, of which the gate
  and Stage C take 2.5–6 s.

## Session 6 — 2026-11-05: the topic anchor (M4 starts)

**Done**
- **Topic anchor** (`sidekick.topic`, `just topic`): for any video, what it's about. Channel
  boilerplate (description lines repeated across a channel's videos) and the channel's title
  template ("… at Six for Tuesday") are removed first. The lead is the title plus the first
  description sentence, or, for a full newscast whose title says nothing specific, its first
  real chapter. Entities and keyphrases come from the lead only; tokens are weighted by IDF over
  our own videos. `just topic <id>` shows how every logged candidate matches the anchor next to
  its label and the current filter decision. Nothing filters on it yet.
- **spaCy `en_core_web_sm`** for entities and part-of-speech keyphrases: a ~13 MB statistical
  pipeline (no text generation), downloaded once by `just models` from spaCy's GitHub releases,
  checked against a pinned sha256 and loaded from the models folder. Without it a heuristic
  tagger takes over.

**Decisions and reasons**
- *Replace the context-relative Don't-show idea (E1b) with a topic anchor* (owner's call). While
  watching news the owner wants that story, not "news" in general and not "no politics". A
  positive rule tied to the current video expresses that; a context-free Don't-show line can't.
- *In a news context, the topic is the lead story only*, and *pundit/commentary videos are not
  shown there* (owner's call). The pundit rule comes after "same story?" labels exist.
- *A topic anchor for every compute*, strict in a news context and lighter elsewhere (owner's
  call). The lecture labels point the same way: "good" meant same series or tight topic, "meh"
  meant drift. Title-template words are kept as `series` for that.
- *Part-of-speech runs, not spaCy noun chunks:* headlines drop their verbs ("Port strike talks
  collapse as union rejects offer"), and the parser then tags "collapse" as the head noun, so
  noun chunks miss "port strike talks". Runs of adjectives and nouns don't.
- *Interpretable parts, no threshold yet:* token overlap, entity overlap, series overlap,
  publish-date gap and e5 similarity of the topic texts are shown separately; which ones
  separate "same story" from the rest is for labels to decide.

**Dead ends and fixes**
- *Title-Case headlines confuse the entity recognizer* ("Kimmel Breaks Down" comes out as a
  person), so entities are one signal beside keyphrases, matched at the token level.
- *Bare month names* ("November" in a newscast title) leaked into keyphrases; month and weekday
  names are now filler.

**Later the same session: first real run (#22 merged)**
- spaCy downloaded and loaded on the owner's machine; e5 topic vectors took 0.6–1.3 s for 21–52
  videos.
- *The news lead came out right,* but **none of the news seed's 20 up-next videos was about the
  same story** (other newscasts, U.S. midterm coverage, true crime). YouTube's up-next doesn't
  surface the same story, so a strict news topic lock only works together with search backfill
  on the topic's query. e5 similarity of topic texts didn't separate anything either
  (0.76–0.87 for every candidate).
- *The lecture anchor works:* for the MIT lecture the next lecture of the course ranked first
  (e5 0.98) and labeled-good videos sat near the top.
- **PR #23** fixes what the real titles exposed:
  - *Repeated channel tags* ("BC news") gave every CTV newscast a 0.35 match with no shared
    story. Tags on 2+ of a channel's videos no longer count in news.
  - *The lecture lost its topic:* words common to the course's titles were treated as a
    template and removed, leaving "ii". Outside news, the series is part of the topic (0.3×),
    and the title is never emptied.
  - *Show names inside titles* ("… | CTV News Vancouver at Six for November 3", "CBC Vancouver
    News at 6, Nov. 04: …", "ABC World News Tonight … Full Broadcast - Nov. 4") are recognized
    as segments and dropped in news; a title that only names the show has no topic.
  - *"Prof." ended a sentence;* abbreviations and initials no longer do.
  - *The search query lost the place,* because "Vancouver" is in the show name. spaCy reads
    "CTV News Vancouver" as one organization, so spans of show names are probed as
    "in <span>"; the query is now "tunnel turnaround Vancouver".
  - `just topic <id> --text` shows how each title segment, description line and tag was read,
    to check the cleaning on real pages.

**Closing the news problem (#24)**
- *A search on the topic query mostly found the wrong thing.* "tunnel turnaround Vancouver" returned
  general tunnel videos, most of them years old; the two clips about the actual story were
  under the 3-minute minimum. The headline is a pun and CTV's description is pure boilerplate,
  so the words that name the story never appear in the video's own text.
- *Decision (owner): stop there.* The goal was never to solve news, only to stop a news video
  from being followed by commentary and pundits, and to stop news reports being dropped as
  "political". The topic anchor stays as built: a ranking feature and a query source for M4.
- **PR #24:** when the current video is news, a Don't-show-me match drops a candidate only if it
  isn't news itself (late-night shows are filed under Entertainment) or its title reads like
  commentary (clickbait ≥ 0.30, reason `pundit:<line>`). On that compute, commentary titles
  ("Trump DOOMED the GOP", "BREAKING: MAJOR UPDATE on FINAL Senate races", "ELECTION UPDATE:
  Dems FLIP…") scored 0.36–0.40 and newscasts 0.00. `just tune` no longer counts news-context
  labels for the Don't-show threshold, so the good-labeled news reports can't switch the filter
  off everywhere else.

**Back to the main path: `just recommend` (#25)**
- *#24 checked on the news seed:* Kimmel dropped as non-news, MS NOW "Trump DOOMED the GOP" as
  a pundit, CBC/CNN/CTV reports kept; `just tune` kept the Don't-show threshold at 0.20 and left
  out 6 news-context labels.
- **`sidekick.recommend`** runs one compute end to end: the up-next list through Stages A–C,
  then, while fewer than `min_pool` (10) videos survive, a search round. The queries come from
  the topic anchor, plus one crossed with the best Interests line in interest mode. Each round
  ranks its Stage A survivors by a **cheap pre-score** (topic match on the title, how many lists
  surfaced it, YouTube's position), so the enrichment budget (25 per compute) goes to the likely
  picks. Everything that survives Stage C gets the **v0 score**: hand-set weights on topic
  match, similarity to the current video (as a percentile within the pool), interest match (in
  interest mode), recurrence, same channel, persistence across earlier computes and a weak views
  term. A per-channel cap (3) stands in for MMR. The ranking is logged as decision stage `R`.
- *Decisions:* percentiles rather than raw cosines, so e5's bunched-up scores and the TF-IDF
  fallback share a scale; enrichment budget spent by pre-score rather than first-seen order,
  because one search round with full enrichment took 85 s on the news seed; no interest-crossed
  search when the topic query already contains the Interests line (it would repeat the same
  results and inflate recurrence); query words keep their spelling ("Bayes rule", not the stem
  "baye rule").

**First real `just recommend` runs, and #26**
- *Results (cold cache, all under 90 s):* MIT lecture 50.9 s, next lecture first and Bayesian
  videos after it; Chi Jin with the search round forced 74.2 s (52 s up-next + 17 s search), 3
  of the course's lectures then search finds (Simons RL tutorial, Poupart's partially
  observable RL module, Stanford CS224R); Dry Bar comedy 64.0 s, 4 videos of the same comedian
  on top. Enrichment (18–19 watch pages) is almost all of the time.
- *Decisions (owner):* the search round **always runs** (it added good picks for +17 s). In a
  **lecture context the same course may take up to 5 slots** (the cap of 3 had pushed out two
  lectures labeled good); **elsewhere one performer or show gets at most 2**, counted across
  channels (the comedian posts through two).
- **PR #26:** round 1 always; rounds 2–3 only while fewer than 10 survive and the compute is
  under 60 s, with a Rocchio-expanded query (the topic query plus terms shared by the best 3
  survivors) and snowball lists (up-next of the best 2 survivors, taken from watch pages
  enrichment already fetched, so no extra requests); MMR (λ = 0.75, centered vectors) with the
  group caps in place of the per-channel cap. The group is the channel in a lecture context,
  else the person named in the title (spaCy PERSON) or the channel.
- *Second real run (#26, warm cache, search always on): 48.6 s, 42.4 s, 32.2 s.* Two caps didn't
  hold. **Dry Bar:** 6 of the top 10 were the same comedian, because spaCy returned his name with
  varying tails ("Chad Thornsberry - Full Special", "… Stand") or not at all, so each variant was
  its own group. **Chi Jin:** only 3 course lectures, not up to 5, because MMR counted lectures
  of the same course as near-duplicates. **PR #27:** entity names are cut to their leading
  capitalized name words, and every name found in the pool is matched in every title; in a
  lecture context MMR measures redundancy only against other courses.

- *Third real run (#27, warm cache): 21.5 s and 39.8 s.* **Chi Jin:** the 3 course lectures in
  the pool all made the top 10, and no hidden candidate scored above 0.49, so 3 is what the pool
  had; the cap of 5 no longer holds them back. **Dry Bar:** 3 of the comedian's videos were in
  the top 10, against a cap of 2. Two came from the title-name match, and five more were held
  back by the cap. The third, "Running A 5k", has his name only in its channel ("Chad
  Thornsberry - …"). **PR #28:** pool names are also matched against the channel when the title
  has none.
- *Fourth run (#28): exactly 2 of the comedian's videos in the top 10, 5 more held back by the
  cap.* The ranking caps are accepted.
- **E3 tool (#29).** Compute time is almost all page fetches, and every compute already logs
  pages and time, so E3 starts from the logs rather than a sweep of new requests. `just e3`
  fits `seconds = a + b · pages` and uses the one-sided 95 % prediction bound, not the mean,
  to find the most pages that fit in 90 s. That gives the largest enrichment budget, with the
  typical and the worst-case search count. It then replays every logged `just recommend` at
  budgets 10–30 and counts how many shown and labeled-good videos would have gone unenriched.
  A few `just recommend <id> --cold` runs (page cache skipped, logged as `cli-cold`) check the
  line where it matters, at the cold end.

## Session 7 — 2026-11-06: E3 report, top-10 label pass

- **E3 on real data:** 30 computes (3 of them cold). Time = 2.0 s + **2.60 s per page**, R²
  0.98, residual SD 3.5 s. The three cold checks (27 pages: 72, 70 and 71 s) landed on the line
  (72 s). The 95 % prediction bound allows 31 pages, so the budget **stays at 25**: 72 s mean,
  79 s bound, and still under 90 s with the worst case of 4 searches. A budget of 20 would save
  about 13 s but leave 32 of 120 shown videos unenriched. The round-1 search supplied 3–5 of
  each top 10; rounds 2–3 never triggered. Write-up: `reports/E3-timing.md`.
- *Finding for later:* round 0 spends 19–20 of the 25 enrichment slots before the search
  results arrive, so 4–6 search results per cold compute go out unenriched, and some still
  rank high. Pre-scoring up-next and search together before enriching would cost no extra
  pages. It waits for a precision@10 baseline to measure against.
- **`just label --top10` + precision@10 in `just label --stats`.** For the newest
  `just recommend` of each seed it asks only about SideKick's top 10 and YouTube's first 10
  up-next (current video, Shorts, live and premieres skipped), mixed and unmarked, so neither
  list is recognizable. P@10 is good ÷ labeled, with the bad share alongside. YouTube's own
  list is the baseline that SideKick has to beat.
- **E4 part 1: SideKick vs YouTube's up-next.** 7 seeds, both top-10 lists labeled blind (140
  entries, 23 in both lists). SideKick: **66 % good, 14 % bad**. YouTube's first 10 up-next:
  49 % good, 31 % bad. That's +17 points [+6, +29] by bootstrap over seeds; SideKick is ahead
  on 5 seeds, tied on 2 and never behind. Lectures gain the most (+40 points). Weak spots:
  - the chemometrics talk (10 % good): YouTube's off-topic junk is replaced by generic PCA
    explainers, labeled meh, because topic outweighs interest in the v0 weights;
  - the newscast (60 % bad): national politics leaks through, as expected under the news
    scope.
- **Decision: v0 ships as the M5 ranker, and M4 is closed.** No hand-tuning on 7 seeds; the
  M6 learned ranker sets the weights from these labels plus panel clicks. Write-up:
  `reports/E4-ranker.md`.

## Session 7, continued — M5 starts: the backend API

- **Three decisions for M5 (Russell):**
  - start from **both up-next lists**: the signed-in list the extension reads from the tab,
    and the signed-out list on the page the backend fetches anyway (no extra requests, and a
    video in both gets the recurrence credit);
  - show **progress steps** while a compute runs;
  - after a **card click**, recompute for the new video automatically (other navigation
    doesn't trigger a compute).
- **PR #32: backend API.**
  - `POST /computes` starts a background job and returns at once. The panel polls
    `GET /computes/{job_id}` for steps ("Up-next: checking videos (7 fetched)",
    "Searching “…”", "Ranking") and then the 10 cards (midpoint thumbnail, title, channel,
    duration).
  - Computes run one at a time. A request for the running video joins it; one for another
    video is queued, and a newer request replaces the queued one, so fast clicking computes
    only the last.
  - `POST /feedback` takes clicks and 👎. A 👎 needs a note, which is appended to `profile.md`
    with the video's title and channel.
  - The shown 10 are logged as impressions. The tab's up-next is logged as `source = tab`.
  - Without a video id, the API falls back to the last video SideKick computed for; with no
    history it returns `empty`.
- `just e3` now fits on panel computes too (any trigger except the synthetic sample).
- **PR #33: the Chrome extension** (`extension/`, WXT 0.21 + TypeScript, no UI framework).
  - **Background:** the right-click menu "SideKick's Selection" and the toolbar icon open the
    side panel. It remembers the last watch page, for the off-watch-page fallback.
  - **Content script:** reads the open video and its signed-in up-next. It walks
    `/watch?v=` links in the right-hand column and sorts each item's text by shape (`12:34`
    is a duration, `… views`, `… ago`), so it doesn't depend on exact class names. Shorts,
    playlists and mixes are skipped.
  - **Panel:** progress steps, then 10 cards. A card click plays the video in the same tab,
    logs the click, waits for the new up-next to render, and recomputes. 👎 opens a note box
    whose Save stays disabled until there's text.
  - **Settings page:** the backend URL, with a connection test.
  - **Tests:** Vitest on synthetic markup in both up-next layouts, the API client and
    polling. CI gets an extension job (typecheck, test, build).
  - `just ext-install`, `just ext-build`, and `just ext-win` (WSL2: copies the build to a
    Windows folder for **Load unpacked**).

## Session 7, end — M5 on both machines; English only

- **M5 works on Windows 11 + WSL2 + Chrome** (Russell). It also works on **ChromeOS + Crostini +
  Chrome**: the repo installed from a zip download, all tests pass, `localhost:8000` reaches
  the backend, and panel computes finish under 90 s. The Crostini setup doc now follows that
  route: uv, `just` from PyPI (`rust-just`), Node 22 via nvm, and a copy of the extension build
  outside the hidden `.output` folder.
- **Crostini timing (10 panel computes, 10–14 pages):** about 7.8 s + 2.98 s/page, against
  2.0 s + 2.60 s/page on WSL2 (R² 0.82, residual SD 9.8 s). The fit doesn't reach the 27-page
  cold case yet, so two cold checks there come next.
- **Finding: 44 of 100 panel cards were never enriched.** Merging the tab's up-next with the
  signed-out list gives round 0 about 35 candidates against a budget of 25, and only enriched
  videos get Stage B's language check.
- **The visible symptom:** a Japanese video's panel was full of Japanese videos. Russell wants
  recommendations "about the content, but only in English". **PR #34:**
  - **Stage A drops titles written mostly in a non-Latin script** (`script:japanese`,
    `script:cjk`, `script:korean`, `script:cyrillic`…) when every profile language is
    Latin-script. It needs no model and runs before any fetch, so these videos no longer use
    up enrichment slots. Kanji, kana and Hangul count 3× a Latin letter: "【MV】夜に駆ける /
    YOASOBI" is Japanese, "The Great Guitars - 東京 1980" isn't. It applies to music too
    (Russell): a French or German concert passes, a Japanese-titled song doesn't.
  - **A non-English current video gets English round-1 queries instead of the topic query:**
    - `english`: the English words from its own title, tags and description;
    - `crosslingual`: the multilingual e5 vector of the video against the English titles
      already in the database. The nearest titles vote for their shared words. No
      translation model; it's experimental and will be judged on labels.

## Session 8 — 2026-11-12: M5 closed, ChromeOS paused, M6 planned

- **Status check:** `main` at #34, no open PRs, 234 Python and 16 extension tests pass, and no
  private data in the repo.
- **#34 checked on a real Japanese video** (a talk on Bayesian statistics for a statistics
  exam):
  - the panel saw the non-English title and searched in English. The title had no English
    words, so the only query was the cross-lingual one, and the English titles already in the
    database voted for `Bayesian Statistics`;
  - Stage A dropped 10 of the 20 up-next videos before any fetch, and the top 10 was all
    English and all on Bayesian statistics;
  - 27 pages in 76.8 s, on the timing line.
- **Two findings for M6:**
  - **Diversity can promote an outlier.** A "binaural beats for studying" video from up-next
    took #9 with score 0.20, ahead of three Bayesian videos at 0.22–0.25. MMR rewards being
    unlike what's already picked, so in a pool that is mostly one topic the odd one out gains
    the most. The fix is a relevance floor for MMR, measured with NDCG in M6.
  - **The topic part is 0 for every candidate when the current video isn't English**, because
    the topic anchor uses an English spaCy model. The ranking then rests on e5 similarity,
    which is multilingual. The learned ranker gets a feature for it.
- **E3 rechecked** on 39 computes: 1.5 s + 2.69 s/page (R² 0.98); budget 25 stays. At 25,
  about 10 % of shown videos go out unenriched on WSL2. The 44 % seen on the Chromebook came
  from its fresh, empty page cache, so round 0 stays as it is.
- **ChromeOS paused** (owner's decision): development continues on WSL2 only, and the 90 s
  budget is judged there. The ChromeOS guide stays, marked as last checked on 2026-11-06.
- **M6 plan, decisions** (owner took the recommended answers):
  - labels stay good / meh / bad, scored 2 / 1 / 0 for NDCG@10;
  - train and test split by seed in time order, with leave-one-seed-out as a second check (the
    spec's rule against random splits; seeds, not videos, are the independent units);
  - no separate clickbait classifier: with about 50 bad labels, the Stage C scores go into the
    ranker as features;
  - panel clicks are an online check only, not training positives, since a click depends on
    the card's position.
- **M6 order:** M6a a labeling app (marimo), M6b the evaluation dataset and metrics, M6c the
  learned ranker (logistic regression, then LightGBM) with the MMR fix, M6d the E4 part 2
  report.

