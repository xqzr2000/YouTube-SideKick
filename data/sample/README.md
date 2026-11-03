# Sample data

Synthetic sample dataset so anyone can run the notebooks and CI without scraping YouTube.
Nothing here comes from a real user's profile or watch history, and no real YouTube content
is included: ids, titles, channels and counts are made up.

- `pages.jsonl.gz` — 8 synthetic pages (4 watch pages with 20 up-next items each, 4 searches)
  in the same format as the raw log (`~/.sidekick/raw/*.jsonl.gz`), so it replays through the
  same parser. Regenerate with `uv run python scripts/make_sample.py`.
