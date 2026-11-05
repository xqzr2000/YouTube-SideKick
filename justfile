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
