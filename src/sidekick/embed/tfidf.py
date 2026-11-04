"""TF-IDF and LSA representations, fitted on our own logged videos (the E1 baselines).

Both are refitted per use on whatever corpus they're given (all videos in the database plus
the profile lines). That takes milliseconds at our scale (tens to a few thousand documents), so
their vectors are not cached: a refit changes the vocabulary and the vectors with it.

- ``tfidf``: sparse word + bigram TF-IDF, English stop words removed, sublinear TF,
  L2-normalized. Exact word overlap only: "late-night" ≠ "talk show".
- ``lsa``: ``tfidf`` → truncated SVD (up to 64 dimensions) → L2-normalized. Folds co-occurring
  words together, so it can match texts that share no exact words, but only through what our
  own small corpus has seen.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

LSA_DIMS = 64


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        token_pattern=r"(?u)\b[^\W\d_][\w-]+\b",  # words of 2+ chars, no bare numbers
    )


class TfidfRep:
    name = "tfidf"
    cacheable = False

    def __init__(self) -> None:
        self.vec = _vectorizer()
        self.fitted = False

    def fit(self, corpus: Sequence[str]) -> TfidfRep:
        docs = [d for d in corpus if d and d.strip()]
        self.vec.fit(docs or ["empty"])
        self.fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fit() first")
        x = self.vec.transform(list(texts))
        return np.asarray(normalize(x).toarray(), dtype=np.float32)


class LsaRep:
    name = "lsa"
    cacheable = False

    def __init__(self, dims: int = LSA_DIMS, seed: int = 0) -> None:
        self.dims = dims
        self.seed = seed
        self.tfidf = TfidfRep()
        self.svd: TruncatedSVD | None = None
        self.fitted = False

    def fit(self, corpus: Sequence[str]) -> LsaRep:
        self.tfidf.fit(corpus)
        x = self.tfidf.vec.transform([d for d in corpus if d and d.strip()] or ["empty"])
        k = min(self.dims, x.shape[0] - 1, x.shape[1] - 1)
        # a corpus this tiny has nothing to fold together: fall back to plain TF-IDF
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed).fit(x) if k >= 1 else None
        self.fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fit() first")
        if self.svd is None:
            return self.tfidf.encode(texts)
        z = self.svd.transform(self.tfidf.vec.transform(list(texts)))
        return np.asarray(normalize(z), dtype=np.float32)
