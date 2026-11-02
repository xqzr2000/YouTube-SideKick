"""FastAPI backend. M0 only exposes /health so Chrome-to-Linux reachability can be tested."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sidekick import __version__

app = FastAPI(title="YouTube-SideKick", version=__version__)

# The extension calls the backend from a chrome-extension:// origin (and from youtube.com
# if a content script fetches directly). Local-only service, so allow these origins.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://[a-z]+|https://(www\.)?youtube\.com|http://(localhost|127\.0\.0\.1)(:\d+)?)$",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
