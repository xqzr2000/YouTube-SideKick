set shell := ["bash", "-cu"]

# List available recipes
default:
    @just --list

# Install/sync the Python environment
sync:
    uv sync --all-groups

# Run the backend with auto-reload (http://localhost:8000/health)
dev:
    uv run uvicorn sidekick.api.app:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
    uv run pytest

# Lint and check formatting
lint:
    uv run ruff check .
    uv run ruff format --check .

# Format code
fmt:
    uv run ruff check --fix .
    uv run ruff format .

# Create the private folder and a starter profile.md if missing (does not overwrite)
init:
    uv run python -m sidekick.config

# Show how SideKick reads your profile.md (interests, rules, don't-show-me lines, warnings)
profile:
    uv run python -m sidekick.profile

# Download the models into ~/.sidekick/models: fastText language ID (~1 MB),
# multilingual-e5-small (~120 MB) and spaCy en_core_web_sm (~13 MB, topic entities)
models:
    uv run python -m sidekick.features.lang --download
    uv run python -m sidekick.embed --download
    uv run python -m sidekick.topic --download

# Show what the text representations find similar (e.g. `just sim`, `just sim <id>`, `just sim "text"`)
[positional-arguments]
sim *args:
    uv run python -m sidekick.embed "$@"

# Show row counts and the latest computes in ~/.sidekick/sidekick.db
db:
    uv run python -m sidekick.store.db

# One full compute: filter up-next, search, score, pick 10 with MMR, print them
# (e.g. `just recommend <id or watch URL>`, `just recommend <id> --no-search --show 20`)
[positional-arguments]
recommend *args:
    uv run python -m sidekick.recommend "$@"

# E3: compute time vs pages fetched, the largest enrichment budget under 90 s, and what a
# smaller budget costs on logged computes (`just e3 --md` for reports/E3-timing.md)
[positional-arguments]
e3 *args:
    uv run python -m sidekick.e3 "$@"

# Show the topic anchor of a video and how its candidates match it
# (e.g. `just topic`, `just topic <id or watch URL>`, `just topic <id> --k 30 --no-e5`)
[positional-arguments]
topic *args:
    uv run python -m sidekick.topic "$@"

# Hand-label logged videos for E1/E2 (e.g. `just label`, `just label --n 20`, `just label --stats`)
[positional-arguments]
label *args:
    uv run python -m sidekick.label "$@"

# Set gate + Stage C thresholds from your labels (dry run; `just tune --write` saves them)
[positional-arguments]
tune *args:
    uv run python -m sidekick.tune "$@"

# Save raw YouTube pages to ~/.sidekick/fixtures (e.g. `just record ID1 ID2 --search "query"`)
[positional-arguments]
record *args:
    uv run python -m sidekick.scrape.record "$@"

# Show what the parsers extract from saved pages (e.g. `just parse-check ~/.sidekick/fixtures/*.html`)
# Add --skeleton to print the key structure of the first up-next item instead
[positional-arguments]
parse-check *files:
    uv run python -m sidekick.parse.check "$@"

# Fetch a video's watch page (+ --search "q", --check-thumbs) and list the candidates
[positional-arguments]
fetch *args:
    uv run python -m sidekick.scrape.cli "$@"

# Open a notebook in marimo (NB01 candidate pools by default; `just nb notebooks/02_filters.py`
# for NB02 filters, `just nb notebooks/03_representations.py` for NB03 representations)
nb notebook="notebooks/01_candidates.py":
    uv run --group nb marimo edit --headless {{notebook}}

# --- Chrome extension (M5). Needs Node 22+ in this Linux environment (docs/setup-*.md) ---

# Install the extension's dependencies (once, and after package changes)
ext-install:
    cd extension && npm ci

# Typecheck and unit-test the extension
ext-test:
    cd extension && npm run typecheck && npm test

# Build the extension into extension/.output/chrome-mv3
ext-build:
    cd extension && npm run build
    @echo "Built: extension/.output/chrome-mv3 (WSL2: run 'just ext-win' to copy it where Windows Chrome can load it)"

# WSL2 only: build, then copy the extension to a Windows folder (default %USERPROFILE%\SideKick-extension)
[positional-arguments]
ext-win *dest: ext-build
    #!/usr/bin/env bash
    set -euo pipefail
    win_home="$(wslpath "$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')")"
    dest="${1:-$win_home/SideKick-extension}"
    rm -rf "$dest" && cp -r extension/.output/chrome-mv3 "$dest"
    echo "Copied to $(wslpath -w "$dest")"
    echo "First time: chrome://extensions -> Developer mode -> Load unpacked -> that folder."
    echo "After a rebuild: click the reload icon on the SideKick card in chrome://extensions."

