"""Language ID with fastText's ``lid.176.ftz`` (176 languages, ~1 MB, CPU, no tokens).

The model is a one-time public download into ``~/.sidekick/models/`` (``just models``). It is
never committed. Without it, language checks are skipped (and the reason is logged), so the rest
of the pipeline still runs.

Short texts such as titles are noisy, so callers should only act on confident predictions
(see ``stage_a.LANG_MIN_CONF``).

``python -m sidekick.features.lang --download`` fetches the model;
``python -m sidekick.features.lang "some text"`` prints its prediction.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sidekick.config import Paths, ensure_home, get_paths

MODEL_NAME = "lid.176.ftz"
MODEL_URL = f"https://dl.fbaipublicfiles.com/fasttext/supervised-models/{MODEL_NAME}"
_MIN_MODEL_BYTES = 500_000  # the real file is ~917 KB; anything much smaller is an error page
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class LangGuess:
    lang: str  # ISO 639-1/3 code as fastText labels it, e.g. "en", "de", "zh"
    conf: float


class LangDetector(Protocol):
    def detect(self, text: str) -> LangGuess | None: ...


def clean_for_lid(text: str) -> str:
    """fastText wants one line; lowercasing keeps ALL-CAPS titles from confusing it."""
    return _WS_RE.sub(" ", text).strip().lower()


class FastTextLang:
    def __init__(self, model_path: Path):
        import fasttext  # fasttext-predict: inference-only wheel, no numpy needed

        self.model_path = model_path
        self._model = fasttext.load_model(str(model_path))

    def detect(self, text: str) -> LangGuess | None:
        t = clean_for_lid(text)
        if not t:
            return None
        labels, probs = self._model.predict(t, k=1)
        if not labels:
            return None
        return LangGuess(lang=labels[0].removeprefix("__label__"), conf=float(min(probs[0], 1.0)))


def model_path(paths: Paths | None = None) -> Path:
    return (paths or get_paths()).models_dir / MODEL_NAME


def load_detector(paths: Paths | None = None) -> LangDetector | None:
    """The fastText detector, or None if the model hasn't been downloaded."""
    p = model_path(paths)
    return FastTextLang(p) if p.exists() else None


def download_model(paths: Paths | None = None) -> Path:
    import httpx

    dest = model_path(paths)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with httpx.stream("GET", MODEL_URL, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    if tmp.stat().st_size < _MIN_MODEL_BYTES:
        size = tmp.stat().st_size
        tmp.unlink()
        raise RuntimeError(f"download looks wrong ({size} bytes); not saved")
    tmp.replace(dest)
    return dest


def main(argv: list[str]) -> int:
    paths = ensure_home()
    if argv[:1] == ["--download"]:
        p = model_path(paths)
        if p.exists():
            print(f"already there: {p} ({p.stat().st_size:,} B)")
        else:
            p = download_model(paths)
            print(f"saved {p} ({p.stat().st_size:,} B)")
        det = load_detector(paths)
        for sample in ("17. Bayesian Statistics", "Statistik für Anfänger", "贝叶斯统计"):
            print(f"  {sample!r:32} -> {det.detect(sample)}")
        return 0
    det = load_detector(paths)
    if det is None:
        print(f"no model at {model_path(paths)}; run `just models`", file=sys.stderr)
        return 1
    for text in argv or ["17. Bayesian Statistics"]:
        print(f"{text!r} -> {det.detect(text)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
