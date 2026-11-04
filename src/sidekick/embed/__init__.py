"""Text representations for videos and profile lines (spec §5 steps 4 and 7, experiment E1).

Three representations share one interface: ``fit(corpus)`` then ``encode(texts)``, which
returns L2-normalized float32 rows, so cosine similarity is a dot product.

==========  ============================================================  ========
name        what                                                           cached
==========  ============================================================  ========
``tfidf``   word + bigram TF-IDF fitted on our own videos                  no
``lsa``     TF-IDF → 64-dim truncated SVD, fitted on our own videos        no
``e5-…``    ``multilingual-e5-small`` int8 ONNX (needs ``just models``)    yes
==========  ============================================================  ========

The two corpus-fitted ones are refitted each time (milliseconds at our scale). e5 vectors are
cached in the ``embeddings`` table with a hash of the text they came from, so a video is
re-embedded only when its text changes (e.g. its description arrives with enrichment).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from sidekick.embed.text import text_hash, video_text
from sidekick.store.db import Store, now_iso


class Representation(Protocol):
    name: str
    cacheable: bool

    def fit(self, corpus: Sequence[str]) -> Representation: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def corpus(store: Store, extra: Sequence[str] = ()) -> list[str]:
    """Texts for fitting TF-IDF / LSA: every video in the database plus ``extra`` lines."""
    rows = store.conn.execute("SELECT * FROM videos ORDER BY video_id")
    return [video_text(dict(r)) for r in rows] + [t for t in extra if t]


def video_texts(store: Store, video_ids: Sequence[str]) -> list[str]:
    out = []
    for vid in video_ids:
        v = store.get_video(vid)
        out.append(video_text(v) if v else "")
    return out


def _get_cached(store: Store, model: str, video_id: str, h: str) -> np.ndarray | None:
    row = store.conn.execute(
        "SELECT vector, text_hash FROM embeddings WHERE video_id = ? AND model = ?",
        (video_id, model),
    ).fetchone()
    if row is None or row["text_hash"] != h:
        return None
    return np.frombuffer(row["vector"], dtype="<f4").copy()


def _put_cached(store: Store, model: str, video_id: str, h: str, vec: np.ndarray) -> None:
    store.conn.execute(
        """INSERT INTO embeddings (video_id, model, dim, vector, created_at, text_hash)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(video_id, model) DO UPDATE SET
               dim = excluded.dim, vector = excluded.vector,
               created_at = excluded.created_at, text_hash = excluded.text_hash""",
        (video_id, model, int(vec.shape[0]), vec.astype("<f4").tobytes(), now_iso(), h),
    )


def encode_videos(
    store: Store, rep: Representation, video_ids: Sequence[str]
) -> tuple[np.ndarray, dict[str, int]]:
    """Vectors for ``video_ids`` (rows in the same order) plus ``{"cached", "computed"}``.

    Cacheable representations read and write the ``embeddings`` table; the others just encode.
    """
    texts = video_texts(store, video_ids)
    enc = getattr(rep, "encode_docs", rep.encode)  # e5 variants prefix video texts differently
    if not rep.cacheable:
        return enc(texts), {"cached": 0, "computed": len(texts)}
    hashes = [text_hash(t) for t in texts]
    vecs: list[np.ndarray | None] = [
        _get_cached(store, rep.name, vid, h) for vid, h in zip(video_ids, hashes, strict=True)
    ]
    todo = [i for i, v in enumerate(vecs) if v is None]
    if todo:
        new = enc([texts[i] for i in todo])
        with store.conn:
            for i, vec in zip(todo, new, strict=True):
                _put_cached(store, rep.name, video_ids[i], hashes[i], vec)
                vecs[i] = vec
    stats = {"cached": len(texts) - len(todo), "computed": len(todo)}
    if not vecs:
        return np.zeros((0, 0), dtype=np.float32), stats
    return np.vstack(vecs).astype(np.float32), stats


def center(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Subtract a mean vector and re-normalize rows (the standard fix for embedding anisotropy:
    e5 cosines bunch up around 0.7–0.9 because all vectors share a common direction)."""
    y = x - mean
    n = np.linalg.norm(y, axis=1, keepdims=True)
    return (y / np.where(n == 0, 1, n)).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine for L2-normalized rows: ``a`` (n×d), ``b`` (m×d) → n×m."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0] if b.ndim == 2 else 0), dtype=np.float32)
    return a @ b.T


def max_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each row of ``a``, its best similarity to any row of ``b`` (NaN if ``b`` is empty)."""
    if b.shape[0] == 0:
        return np.full(a.shape[0], np.nan, dtype=np.float32)
    return cosine(a, b).max(axis=1)


def available(store: Store, profile_lines: Sequence[str] = (), *, paths=None, e5=None) -> list:
    """Fitted representations ready to use: tfidf, lsa, and e5 if its model is downloaded.

    ``e5`` can be passed in (tests, or to reuse a loaded model); otherwise it's loaded from
    ``~/.sidekick/models`` if present.
    """
    from sidekick.embed import e5 as e5mod
    from sidekick.embed.tfidf import LsaRep, TfidfRep

    docs = corpus(store, profile_lines)
    reps: list = [TfidfRep().fit(docs), LsaRep().fit(docs)]
    e5_rep = e5 if e5 is not None else e5mod.load(paths)
    if e5_rep is not None:
        reps.append(e5_rep)
    return reps
