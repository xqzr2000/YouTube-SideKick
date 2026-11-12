# YouTube-SideKick

A Chrome side-panel extension that cleans up YouTube's recommendations for the video you're watching.

Inspired by Andrej Karpathy's [HackerTube project](https://karpathy.bearblog.dev/hacker-tube/), which replaces YouTube's recommendation feed with videos that Hacker News commenters actually recommended.

https://github.com/user-attachments/assets/4c383c72-8002-434b-b403-070a67a8c993


## Overview

On a YouTube watch page, right-click and choose **"SideKick's Selection"** (or click the toolbar icon) to open the side panel. It shows 10 videos to watch next, picked for the video you're on. A local Python backend (running on Linux) handles the processing.

On the surface, SideKick is a browser extension. At its core, it's a data science project: I'm building a recommendation engine from scratch.

## Status

The extension works end to end (M0–M5 done). The data-science half continues:

- **M1–M2:** parsers, polite fetcher, profile, SQLite store, Stage A/B filters, enrichment
  (NB01, NB02).
- **M3:** hand labels, TF-IDF / LSA / e5 text representations, interest gate, Stage C soft
  filters (NB03, [E1](reports/E1-representations.md), [E2](reports/E2-interest-gate.md)).
- **M4:** topic anchor, `just recommend` with backfill rounds, v0 score, MMR with group caps
  ([E3 timing](reports/E3-timing.md)).
- **M5:** backend API and the Chrome extension; English-only rule for non-Latin titles, with
  English and cross-lingual searches for non-English videos.
- **First result ([E4 part 1](reports/E4-ranker.md)):** on 7 seed videos, 66 % of SideKick's
  top 10 were labeled good, against 49 % of YouTube's own first 10 up-next (bad: 14 % vs 31 %).
- **Next, M6:** a labeling app, offline evaluation (NDCG@10) and a learned ranker. **M7:**
  taste drift and why YouTube keeps resurfacing some videos (E5, E6), and a model card.

See `docs/` for setup:

- [Windows 11 + Chrome + WSL2](docs/setup-windows-wsl2.md) (the development machine)
- [ChromeOS + Chrome + Crostini](docs/setup-chromeos.md) (worked on 2026-11-06; paused since
  2026-11-12)

How it's being built, session by session: [dev log](docs/devlog.md).

Quick start (inside the Linux environment): `just sync && just test && just dev`, then open
`http://localhost:8000/health` in Chrome.

Useful commands: `just models` (one-time downloads: ~1 MB language ID, ~120 MB
multilingual-e5-small, ~13 MB spaCy `en_core_web_sm`), `just profile` (how your
`profile.md` is read), `just fetch <video-id>` (fetch, run Stage A + B and log one
compute), `just db` (what's in the database), `just label` (hand-label logged videos for the
experiments), `just sim` (what the text representations find similar; `--eval` scores them against your
labels), `just tune` (set the interest-gate and Stage C thresholds from your labels),
`just topic <video-id>` (the topic anchor of a video and how its candidates match it).
`just recommend <video-id>` runs one full compute (filters, a search round, more rounds when
too few videos survive, v0 score, MMR) and prints SideKick's top 10 (`--cold` skips the page
cache). `just e3` fits compute time against pages fetched and works out the largest
enrichment budget that stays under 90 s ([E3 report](reports/E3-timing.md)).
`just label --top10` labels SideKick's top 10 and YouTube's first 10 up-next, mixed, and
`just label --stats` then compares their precision@10.
`just label-app` opens the labeling app (M6) at `http://localhost:2718`: it labels each
current video's whole ranked pool, shown or not, with scores and ranks hidden.

**Backend API (M5, for the side panel; `just dev` serves it on port 8000):**
- `POST /computes` starts a compute. Body: `{"video_id", "upnext": [...], "trigger"}`, where
  `upnext` is the signed-in list from the open tab. It returns a job to poll.
- `GET /computes/{job_id}` reports progress steps, then the 10 cards. `GET /computes/latest`
  returns the running or newest job.
- `POST /feedback` records a `click` or a `downvote`. A downvote needs a note, which is
  appended to `profile.md`.

Computes run one at a time. A request for another video waits in a queue, and a newer request
replaces the waiting one.

**Chrome extension (M5, `extension/`, WXT + TypeScript):** the right-click menu **SideKick's
Selection** (or the toolbar icon) opens a side panel. The panel reads the video and its
signed-in up-next list from the tab, shows progress while the backend computes, then shows 10
cards. A click plays the video in the same tab and recomputes; 👎 asks for a note that goes
into your profile. To build and load it, see section 7 of
[the WSL2 setup](docs/setup-windows-wsl2.md#7-the-chrome-extension-sidekicks-selection-m5)
(`just ext-install`, `just ext-win`).

## Scraping disclaimer

SideKick fetches public YouTube pages for personal, low-volume use (polite delays, a per-run
page cap, no bulk crawling). It uses no API key. Scraping YouTube may violate its Terms of
Service, which is a gray area; use it at your own risk.

## Privacy

Your profile, database and raw logs live in a private folder (`~/.sidekick/`, override with
`SIDEKICK_HOME`) and are never part of this repo. `data/sample/` holds synthetic data only.

## License

MIT
