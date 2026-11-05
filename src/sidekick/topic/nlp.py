"""Named entities and noun phrases for topic extraction: spaCy ``en_core_web_sm`` on CPU.

``en_core_web_sm`` is a small statistical pipeline (~12 MB: tagger, parser, named-entity
recognizer). It labels spans of text (people, places, organizations, events); it generates
nothing. It is a one-time public download from spaCy's GitHub releases into
``~/.sidekick/models/spacy/`` (``just models``), checked against a pinned sha256, and never
committed. spaCy loads it straight from that folder, so it isn't installed as a package.

Without the model, a heuristic tagger takes over (runs of Capitalized words as entities,
stopword-split phrases as noun phrases), so topic extraction still runs, only worse.

Titles in Title Case confuse the recognizer ("Kimmel Breaks Down" comes out as a PERSON), so
callers treat entities as one signal next to keyphrases, not as ground truth.
"""

from __future__ import annotations

import hashlib
import io
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sidekick.config import Paths, get_paths

MODEL = "en_core_web_sm"
MODEL_VERSION = "3.8.0"  # must match the installed spaCy minor version (3.8.x)
MODEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    f"{MODEL}-{MODEL_VERSION}/{MODEL}-{MODEL_VERSION}-py3-none-any.whl"
)
MODEL_SHA256 = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"

# Entity labels that say nothing about *which* story it is.
SKIP_LABELS = frozenset(
    {"DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT", "LANGUAGE"}
)

PHRASE_POS = frozenset({"ADJ", "NOUN", "PROPN"})
MAX_PHRASE = 4

_CAP_RUN_RE = re.compile(r"(?:\b[A-Z][\w.'’-]*(?:\s+(?:of|the|and|de|la)\s+|\s+)?)+")
_SPLIT_RE = re.compile(r"[^\w'’-]+")


@dataclass(frozen=True)
class Entity:
    text: str
    label: str


class Tagger(Protocol):
    name: str

    def entities(self, text: str) -> list[Entity]: ...

    def phrases(self, text: str) -> list[str]: ...


def model_dir(paths: Paths | None = None) -> Path:
    return (paths or get_paths()).models_dir / "spacy" / f"{MODEL}-{MODEL_VERSION}"


def is_downloaded(paths: Paths | None = None) -> bool:
    return (model_dir(paths) / "config.cfg").exists()


class SpacyTagger:
    """spaCy pipeline wrapper. ``nlp`` can be any loaded pipeline (tests pass a blank one)."""

    def __init__(self, nlp, name: str = MODEL):
        self.nlp = nlp
        self.name = name

    @classmethod
    def load(cls, path: Path) -> SpacyTagger:
        import spacy

        # The lemmatizer isn't used and needs lookup tables; leave it out.
        return cls(spacy.load(path, exclude=["lemmatizer"]))

    def entities(self, text: str) -> list[Entity]:
        if not text:
            return []
        doc = self.nlp(text)
        return [Entity(e.text.strip(), e.label_) for e in doc.ents if e.text.strip()]

    def phrases(self, text: str) -> list[str]:
        """Runs of adjectives / nouns / proper nouns, at most ``MAX_PHRASE`` words.

        A part-of-speech pattern rather than spaCy's noun chunks: headlines drop their verbs
        ("Port strike talks collapse as union rejects offer"), so the parser often makes a mess
        of them, while the tags stay usable.
        """
        if not text:
            return []
        doc = self.nlp(text)
        if not doc.has_annotation("TAG"):  # no tagger in this pipeline (tests)
            return heuristic_phrases(text)
        out, cur = [], []
        for tok in doc:
            if tok.pos_ in PHRASE_POS and not tok.is_stop:
                cur.append(tok.text)
                if len(cur) == MAX_PHRASE:
                    out.append(" ".join(cur))
                    cur = []
            else:
                if cur:
                    out.append(" ".join(cur))
                cur = []
        if cur:
            out.append(" ".join(cur))
        return out


class HeuristicTagger:
    """Fallback without the model: Capitalized runs as entities, stopword-split phrases."""

    name = "heuristic"

    def entities(self, text: str) -> list[Entity]:
        out = []
        for m in _CAP_RUN_RE.finditer(text or ""):
            span = m.group(0).strip(" .,'’-")
            before = (text or "")[: m.start()].rstrip()
            sentence_start = not before or before[-1] in ".:!?"
            if span and (len(span.split()) > 1 or not sentence_start):  # "Talks collapsed…"
                out.append(Entity(span, "CAP"))
        return out

    def phrases(self, text: str) -> list[str]:
        return heuristic_phrases(text)


def heuristic_phrases(text: str) -> list[str]:
    """RAKE-style candidates: runs of words between stopwords and punctuation."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    out, cur = [], []
    for raw in re.split(r"([,.;:!?()|\[\]\"“”–—]|\s+-\s+)", text or ""):
        for w in _SPLIT_RE.split(raw):
            if not w:
                continue
            if w.casefold() in ENGLISH_STOP_WORDS:
                if cur:
                    out.append(" ".join(cur))
                cur = []
            else:
                cur.append(w)
        if cur:
            out.append(" ".join(cur))
        cur = []
    return out


def load_tagger(paths: Paths | None = None) -> Tagger:
    """spaCy if its model is downloaded (and spaCy is installed), else the heuristic tagger."""
    p = model_dir(paths)
    if (p / "config.cfg").exists():
        try:
            return SpacyTagger.load(p)
        except ImportError:
            pass
    return HeuristicTagger()


def download_model(paths: Paths | None = None) -> Path:
    """Fetch the model wheel, check its sha256, unpack the model folder into the models dir."""
    import httpx

    dest = model_dir(paths)
    r = httpx.get(MODEL_URL, follow_redirects=True, timeout=120)
    r.raise_for_status()
    digest = hashlib.sha256(r.content).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(f"sha256 mismatch for {MODEL_URL}: {digest}; not saved")
    prefix = f"{MODEL}/{MODEL}-{MODEL_VERSION}/"
    tmp = dest.with_name(dest.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for info in z.infolist():
            if not info.filename.startswith(prefix) or info.is_dir():
                continue
            rel = info.filename[len(prefix) :]
            target = (tmp / rel).resolve()
            if not str(target).startswith(str(tmp.resolve())):  # no path escapes
                raise RuntimeError(f"unexpected path in wheel: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))
    if not (tmp / "config.cfg").exists():
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError("model folder not found in the wheel; not saved")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.replace(dest)
    return dest
