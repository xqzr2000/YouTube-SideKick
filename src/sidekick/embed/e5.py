"""``multilingual-e5-small`` sentence embeddings on CPU (ONNX through fastembed; no torch).

The model is Xenova's int8-quantized ONNX export of ``intfloat/multilingual-e5-small``
(384 dimensions, ~118 MB, 100 languages). It is a one-time public download from Hugging Face
into ``~/.sidekick/models/fastembed/`` (``just models``) and is never committed. Without it,
e5 is simply unavailable and the TF-IDF / LSA representations still work.

e5 models expect a prefix. The default (``e5-small-q``) treats everything as symmetric
similarity and prefixes every text with ``query: ``. The variant ``e5-small-qp``
(``E5Rep.with_passages()``) is the retrieval convention instead: profile lines and free text get
``query: ``, video texts get ``passage: ``. That fits a three-word profile line matched against
a long video text. Which one separates your labels better is measured by ``just sim --eval``.
Vectors are mean-pooled and L2-normalized, so cosine similarity is a dot product.

``fastembed`` doesn't ship this model in its built-in list, so it is registered as a custom
model (``TextEmbedding.add_custom_model``) the first time it's used.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sidekick.config import Paths, get_paths

NAME = "e5-small-q"  # value of embeddings.model: every text "query: "
NAME_QP = "e5-small-qp"  # videos "passage: ", profile lines / free text "query: "
HF_REPO = "Xenova/multilingual-e5-small"
MODEL_FILE = "onnx/model_quantized.onnx"
DIM = 384
PREFIX = "query: "
DOC_PREFIX = "passage: "
BATCH = 16  # small batches keep peak memory low on the 8 GB Chromebook

_registered = False


def cache_dir(paths: Paths | None = None) -> Path:
    return (paths or get_paths()).models_dir / "fastembed"


def is_downloaded(paths: Paths | None = None) -> bool:
    """True if the ONNX file is in the local cache (no network access)."""
    return any(cache_dir(paths).glob(f"**/{MODEL_FILE}"))


def _register() -> None:
    global _registered
    if _registered:
        return
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    known = {m["model"] for m in TextEmbedding.list_supported_models()}
    if HF_REPO not in known:
        TextEmbedding.add_custom_model(
            model=HF_REPO,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=HF_REPO),
            dim=DIM,
            model_file=MODEL_FILE,
            description="multilingual-e5-small, int8 ONNX (Xenova export)",
            license="mit",
            size_in_gb=0.12,
        )
    _registered = True


class E5Rep:
    """The e5 representation. ``fit`` is a no-op; vectors are cacheable per video."""

    cacheable = True

    def __init__(
        self,
        paths: Paths | None = None,
        *,
        threads: int | None = None,
        specific_model_path: str | None = None,
        doc_prefix: str = PREFIX,
        _model=None,
    ) -> None:
        self.doc_prefix = doc_prefix
        self.name = NAME if doc_prefix == PREFIX else NAME_QP
        if _model is not None:
            self._model = _model
            return
        from fastembed import TextEmbedding

        _register()
        kwargs = {"specific_model_path": specific_model_path} if specific_model_path else {}
        self._model = TextEmbedding(
            model_name=HF_REPO, cache_dir=str(cache_dir(paths)), threads=threads, **kwargs
        )

    def with_passages(self) -> E5Rep:
        """The query/passage variant, sharing this loaded model (no second load)."""
        return E5Rep(doc_prefix=DOC_PREFIX, _model=self._model)

    def fit(self, corpus: Sequence[str]) -> E5Rep:
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Profile lines and free text (``query: ``)."""
        return self._embed(texts, PREFIX)

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray:
        """Video texts (``query: `` for e5-small-q, ``passage: `` for e5-small-qp)."""
        return self._embed(texts, self.doc_prefix)

    def _embed(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        texts = [prefix + (t or "") for t in texts]
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        vecs = np.asarray(list(self._model.embed(texts, batch_size=BATCH)), dtype=np.float32)
        # fastembed normalizes already; re-normalize so a custom model can't break cosine = dot
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)


def load(paths: Paths | None = None) -> E5Rep | None:
    """The e5 representation if the model is downloaded, else None (never downloads)."""
    if not is_downloaded(paths):
        return None
    return E5Rep(paths)


def download(paths: Paths | None = None) -> E5Rep:
    """Download (first time) and load the model."""
    cache_dir(paths).mkdir(parents=True, exist_ok=True)
    return E5Rep(paths)
