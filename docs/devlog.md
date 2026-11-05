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

