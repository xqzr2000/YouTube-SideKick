"""FastAPI backend for the side panel (M5).

- ``GET  /health`` — reachability check.
- ``POST /computes`` — start a compute for a video. The body has the video id, the up-next list
  the extension read from the open tab, and the trigger (``open`` / ``refresh`` / ``click``).
  Without a video id it uses the last video SideKick computed for; with no history the reply
  is ``{"status": "empty"}`` and the panel shows blanks. Returns a job (see ``api.jobs``).
- ``GET  /computes/latest`` — the running job, else the newest finished one.
- ``GET  /computes/{job_id}`` — progress steps while running; the 10 cards when done.
- ``POST /feedback`` — ``click`` (the only positive signal) or ``downvote``. A downvote needs a
  non-empty note, which is appended to ``profile.md`` under "Don't show me" with the date and
  the video's title and channel. It takes effect at the next compute.

The panel shows no reasons on cards; everything is logged in the database instead (decisions,
impressions = the 10 shown, feedback).
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sidekick import __version__
from sidekick.api.jobs import Card, Job, JobResult, JobRunner
from sidekick.config import ensure_home
from sidekick.models import Candidate
from sidekick.parse.util import parse_count, parse_duration
from sidekick.profile import load_profile
from sidekick.profile.store import append_dont_show
from sidekick.store.db import Store

VIDEO_ID = r"^[A-Za-z0-9_-]{11}$"
MAX_TAB_ITEMS = 40

app = FastAPI(title="YouTube-SideKick", version=__version__)

# The extension calls the backend from a chrome-extension:// origin (and from youtube.com
# if a content script fetches directly). Local-only service, so allow these origins.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://[a-z]+|https://(www\.)?youtube\.com|http://(localhost|127\.0\.0\.1)(:\d+)?)$",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class TabItem(BaseModel):
    """One up-next entry as the extension reads it from the page (text as shown)."""

    video_id: str = Field(pattern=VIDEO_ID)
    title: str = ""
    channel: str = ""
    duration: str = ""  # badge text, e.g. "12:34"
    views: str = ""  # e.g. "1.2M views"
    age: str = ""  # e.g. "3 years ago"
    badges: list[str] = []  # e.g. ["LIVE"], ["SHORTS"], ["Premiere"]

    def candidate(self) -> Candidate:
        b = " ".join(self.badges).lower()
        return Candidate(
            self.video_id,
            title=self.title.strip(),
            channel=self.channel.strip(),
            duration_s=parse_duration(self.duration) if self.duration else None,
            views=parse_count(self.views) if self.views else None,
            published_text=self.age.strip(),
            is_live="live" in b,
            is_upcoming="upcoming" in b or "premiere" in b,
            is_short="short" in b,
            source="tab",
        )


class ComputeIn(BaseModel):
    video_id: str | None = Field(default=None, pattern=VIDEO_ID)
    upnext: list[TabItem] = []
    trigger: Literal["open", "refresh", "click"] = "open"


class FeedbackIn(BaseModel):
    type: Literal["click", "downvote"]
    video_id: str = Field(pattern=VIDEO_ID)
    compute_id: str | None = None
    note: str | None = None


def run_compute(job: Job, step) -> JobResult:
    """The real compute: the same pipeline as ``just recommend``, plus impressions."""
    from sidekick.features.lang import load_detector
    from sidekick.recommend import recommend
    from sidekick.scrape.fetcher import FetchConfig, Fetcher, midpoint_thumbnail_url
    from sidekick.scrape.sponsorblock import SponsorBlock

    paths = ensure_home()
    profile = load_profile(paths)
    with Store.open(paths) as store, Fetcher(FetchConfig(), paths=paths) as f, SponsorBlock() as sb:
        res = recommend(
            store,
            profile,
            video_id=job.video_id,
            fetcher=f,
            sponsorblock=sb,
            detector=load_detector(paths),
            paths=paths,
            trigger=job.trigger,
            tab_upnext=job.tab_upnext,
            on_step=step,
        )
        store.log_impressions(res.compute_id, [r.candidate.video_id for r in res.shown])
        cards = []
        for r in res.shown:
            c = r.candidate
            v = store.get_video(c.video_id) or {}
            cards.append(
                Card(
                    video_id=c.video_id,
                    title=v.get("title") or c.title,
                    channel=v.get("channel") or c.channel,
                    duration_s=v.get("duration_s") or c.duration_s,
                    thumbnail=midpoint_thumbnail_url(c.video_id),
                    rank=r.rank or len(cards) + 1,
                )
            )
    return JobResult(
        compute_id=res.compute_id,
        current_title=res.current.title if res.current else "",
        mode=res.gate.mode if res.gate else None,
        cards=cards,
    )


app.state.runner = JobRunner(run_compute)


def _last_video() -> str | None:
    with Store.open(ensure_home()) as store:
        row = store.conn.execute(
            "SELECT current_video_id FROM computes WHERE current_video_id IS NOT NULL "
            "ORDER BY ts DESC, rowid DESC LIMIT 1"
        ).fetchone()
    return row[0] if row else None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/computes")
def start_compute(body: ComputeIn) -> dict:
    vid = body.video_id or _last_video()
    if vid is None:
        return {"job_id": None, "status": "empty", "video_id": None, "steps": [], "cards": []}
    tab = [t.candidate() for t in body.upnext[:MAX_TAB_ITEMS]] if body.video_id else []
    job = app.state.runner.submit(vid, trigger=body.trigger, tab_upnext=tab)
    return job.to_dict()


@app.get("/computes/latest")
def latest_compute() -> dict:
    job = app.state.runner.latest()
    if job is None:
        return {"job_id": None, "status": "none", "steps": [], "cards": []}
    return job.to_dict()


@app.get("/computes/{job_id}")
def get_compute(job_id: str) -> dict:
    job = app.state.runner.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job (the backend may have restarted)")
    return job.to_dict()


@app.post("/feedback")
def feedback(body: FeedbackIn) -> dict:
    paths = ensure_home()
    note = (body.note or "").strip()
    if body.type == "downvote" and not note:
        raise HTTPException(422, "a 👎 needs a note")
    with Store.open(paths) as store:
        line = None
        if body.type == "downvote":
            v = store.get_video(body.video_id) or {}
            line = append_dont_show(
                note, title=v.get("title") or "", channel=v.get("channel") or "", paths=paths
            )
        known = (
            body.compute_id
            and store.conn.execute(
                "SELECT 1 FROM computes WHERE compute_id = ?", (body.compute_id,)
            ).fetchone()
        )
        fid = store.log_feedback(
            body.video_id,
            body.type,
            compute_id=body.compute_id if known else None,
            note=note or None,
        )
    return {"ok": True, "feedback_id": fid, "profile_line": line}
