# YouTube-SideKick

A Chrome side-panel extension that cleans up YouTube's recommendations for the video you're watching.

Inspired by Andrej Karpathy's [HackerTube project](https://karpathy.bearblog.dev/hacker-tube/), which replaces YouTube's recommendation feed with videos that Hacker News commenters actually recommended.

## Overview

Select text on a YouTube page, right-click, and choose **"SideKick's Selection"** to open the side panel. A local Python backend (running on Linux) handles the processing.

On the surface, SideKick is a browser extension. At its core, it's a data science project: I'm building a recommendation engine from scratch.

## Status

Early development (M1 done: parsers, polite fetcher, NB01; M2 done: profile, SQLite store,
Stage A/B filters, enrichment, NB02; M3 done: hand labels, TF-IDF / LSA / e5 text
representations, interest gate, Stage C soft filters, NB03, E1/E2; M4 in progress: the topic
anchor, `just recommend` with the first backfill round and a v0 score). See `docs/` for setup:

- [Windows 11 + Chrome + WSL2](docs/setup-windows-wsl2.md) (primary dev box)
- [ChromeOS + Chrome + Crostini](docs/setup-chromeos.md)

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
`just recommend <video-id>` runs one full compute (filters, a search round when too few videos
survive, v0 score) and prints SideKick's top 10.

## Scraping disclaimer

SideKick fetches public YouTube pages for personal, low-volume use (polite delays, a per-run
page cap, no bulk crawling). It uses no API key. Scraping YouTube may violate its Terms of
Service, which is a gray area; use it at your own risk.

## Privacy

Your profile, database and raw logs live in a private folder (`~/.sidekick/`, override with
`SIDEKICK_HOME`) and are never part of this repo. `data/sample/` holds synthetic data only.

## License

MIT
