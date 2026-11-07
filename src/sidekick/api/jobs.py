"""Computes for the side panel, one at a time, in a background thread.

A compute takes 20–80 s, so ``POST /computes`` starts a **job** and returns at once; the panel
polls ``GET /computes/{job_id}`` for progress steps and, at the end, the 10 cards.

Only one compute runs at a time (one polite fetcher, one SQLite writer):

- a request for the video that is already being computed joins that job;
- a request for another video while one runs is **queued**; a newer request replaces the
  queued one (marked ``superseded``), so clicking through several videos quickly computes only
  the last;
- when the running job ends, the queued one starts.

Finished jobs stay in memory (the last 20) so a reopened panel can show the latest result.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sidekick.models import Candidate

KEEP = 20

# a run function gets the job and a step(key, text) callback and returns a JobResult
RunFn = Callable[["Job", Callable[[str, str], None]], "JobResult"]


@dataclass
class Step:
    key: str
    text: str
    seconds: float  # since the job started


@dataclass
class Card:
    video_id: str
    title: str
    channel: str
    duration_s: int | None
    thumbnail: str
    rank: int


@dataclass
class JobResult:
    compute_id: str | None = None
    current_title: str = ""
    mode: str | None = None
    cards: list[Card] = field(default_factory=list)


@dataclass
class Job:
    job_id: str
    video_id: str
    trigger: str
    tab_upnext: list[Candidate] = field(default_factory=list)
    status: str = "queued"  # queued | running | done | blocked | error | superseded
    steps: list[Step] = field(default_factory=list)
    result: JobResult | None = None
    error: str | None = None
    created: float = field(default_factory=time.monotonic)
    started: float | None = None
    finished: float | None = None

    def step(self, key: str, text: str) -> None:
        t = round(time.monotonic() - (self.started or self.created), 1)
        for s in self.steps:
            if s.key == key:
                s.text, s.seconds = text, t
                return
        self.steps.append(Step(key, text, t))

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return round((self.finished or time.monotonic()) - self.started, 1)

    def to_dict(self) -> dict[str, Any]:
        r = self.result
        return {
            "job_id": self.job_id,
            "video_id": self.video_id,
            "trigger": self.trigger,
            "status": self.status,
            "elapsed_s": self.elapsed,
            "steps": [s.__dict__ for s in self.steps],
            "compute_id": r.compute_id if r else None,
            "current_title": r.current_title if r else "",
            "mode": r.mode if r else None,
            "cards": [c.__dict__ for c in r.cards] if r else [],
            "error": self.error,
        }


class JobRunner:
    def __init__(self, run_fn: RunFn, *, threaded: bool = True) -> None:
        self.run_fn = run_fn
        self.threaded = threaded
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.current: Job | None = None
        self.pending: Job | None = None
        self.lock = threading.Lock()

    def submit(self, video_id: str, *, trigger: str, tab_upnext: list[Candidate]) -> Job:
        with self.lock:
            cur = self.current
            if cur is not None and cur.video_id == video_id:
                return cur
            job = Job(uuid.uuid4().hex[:12], video_id, trigger, list(tab_upnext))
            self._remember(job)
            if cur is None:
                self.current = job
                start = True
            else:
                if self.pending is not None:
                    self.pending.status = "superseded"
                self.pending = job
                start = False
        if start:
            self._start(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def latest(self) -> Job | None:
        """The running job, else the newest finished one."""
        with self.lock:
            if self.current is not None:
                return self.current
            done = [j for j in self.jobs.values() if j.status not in ("queued", "superseded")]
            return done[-1] if done else None

    def _remember(self, job: Job) -> None:
        self.jobs[job.job_id] = job
        while len(self.jobs) > KEEP:
            old_id, old = next(iter(self.jobs.items()))
            if old is self.current or old is self.pending:
                break
            self.jobs.pop(old_id)

    def _start(self, job: Job) -> None:
        if self.threaded:
            threading.Thread(target=self._run, args=(job,), daemon=True).start()
        else:
            self._run(job)

    def _run(self, job: Job) -> None:
        job.status, job.started = "running", time.monotonic()
        try:
            job.result = self.run_fn(job, job.step)
            job.status = "done"
        except Exception as e:  # reported to the panel; the compute itself is logged
            job.status = "blocked" if type(e).__name__ == "BlockedError" else "error"
            job.error = str(e) or type(e).__name__
        finally:
            job.finished = time.monotonic()
            with self.lock:
                nxt, self.pending = self.pending, None
                self.current = nxt
            if nxt is not None:
                self._start(nxt)
