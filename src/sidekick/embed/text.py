"""What text stands for a video when it is embedded or vectorized.

One function, used by every representation, so experiment E1 compares models and not inputs.
URLs, timestamps and hashtag clutter are removed, because lecture and channel descriptions are
full of them and they say nothing about the topic.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

DESC_CHARS = 1000  # descriptions past this are mostly links, credits and sponsor blurbs

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_TS_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_WS_RE = re.compile(r"\s+")


def clean(text: str) -> str:
    text = _URL_RE.sub(" ", text or "")
    text = _TS_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def video_text(v: Mapping[str, Any]) -> str:
    """``title. channel. keywords. description[:1000]`` from a ``videos`` row (or similar dict).

    Missing parts are skipped, so an un-enriched video is just its title and channel.
    """
    kw = v.get("keywords") or []
    if isinstance(kw, str):
        try:
            kw = json.loads(kw)
        except ValueError:
            kw = [kw]
    parts = [
        clean(v.get("title") or ""),
        clean(v.get("channel") or ""),
        clean(" ".join(str(k) for k in kw)),
        clean((v.get("description") or "")[:DESC_CHARS]),
    ]
    return ". ".join(p for p in parts if p)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
