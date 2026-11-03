set shell := ["bash", "-cu"]

# List available recipes
default:
    @just --list

# Install/sync the Python environment
sync:
    uv sync

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

# Save raw YouTube pages to ~/.sidekick/fixtures (e.g. `just record ID1 ID2 --search "query"`)
record *args:
    uv run python -m sidekick.scrape.record {{args}}

# Show what the parsers extract from saved pages (e.g. `just parse-check ~/.sidekick/fixtures/*.html`)
parse-check *files:
    uv run python -m sidekick.parse.check {{files}}
