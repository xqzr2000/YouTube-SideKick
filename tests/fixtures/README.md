# Test fixtures

`watch_synthetic.html` and `search_synthetic.html` are **hand-built, synthetic** pages that mimic
the shape of YouTube's embedded JSON (`ytInitialData`, `ytInitialPlayerResponse`). They contain no
real YouTube content. Real recordings stay in the private folder (`~/.sidekick/fixtures/`) and are
never committed; use `just record` and `just parse-check` to compare the parser against them.
