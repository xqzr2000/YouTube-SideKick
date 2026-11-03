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
