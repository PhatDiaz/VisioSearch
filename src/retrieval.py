"""Pure NumPy retrieval and evaluation utilities.

No machine-learning downloads are needed to test these functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SearchHit:
    image_index: int
    cosine_similarity: float


def unit_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows. Zero-length embeddings raise an error."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    if vectors.ndim != 2:
        raise ValueError('Expected a 1D or 2D embedding array.')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError('Zero-length embedding cannot be normalized.')
    return vectors / norms


def search(
    query: np.ndarray,
    image_embeddings: np.ndarray,
    top_k: int = 8,
    allowed_indices: Iterable[int] | None = None,
) -> list[SearchHit]:
    """Cosine rank a single text vector against pre-normalized image vectors.

    Stable ranking breaks score ties in gallery order. The query is normalized here.
    """
    if top_k < 1:
        raise ValueError('top_k must be positive')
    gallery = np.asarray(image_embeddings, dtype=np.float32)
    if gallery.ndim != 2 or not len(gallery):
        raise ValueError('Image gallery must be a non-empty 2D matrix.')
    q = unit_normalize(np.asarray(query, dtype=np.float32))[0]
    if q.shape[0] != gallery.shape[1]:
        raise ValueError('Text and image embedding dimensions differ.')
    if allowed_indices is None:
        ids = np.arange(len(gallery), dtype=np.int64)
    else:
        ids = np.asarray(list(allowed_indices), dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= len(gallery)):
            raise ValueError('Gallery filter contains invalid indices.')
        ids = np.unique(ids)  # unique and ascending for deterministic tie breaks
    if len(ids) == 0:
        return []
    scores = gallery[ids] @ q
    best = np.argsort(-scores, kind='stable')[: min(top_k, len(ids))]
    return [SearchHit(int(ids[i]), float(scores[i])) for i in best]


def ground_truth_ranks(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return 1-based rank for each target image, stable on exact ties."""
    scores = np.asarray(scores)
    targets = np.asarray(targets, dtype=np.int64)
    if scores.ndim != 2 or targets.shape != (scores.shape[0],):
        raise ValueError('Scores/target shapes do not match.')
    if np.any(targets < 0) or np.any(targets >= scores.shape[1]):
        raise ValueError('Targets outside gallery bounds.')
    truths = scores[np.arange(len(targets)), targets]
    earlier_index = np.arange(scores.shape[1])[None, :] < targets[:, None]
    ahead = (scores > truths[:, None]) | ((scores == truths[:, None]) & earlier_index)
    return 1 + ahead.sum(axis=1)


def recall_at_k(ranks: np.ndarray, ks: Iterable[int] = (1, 5, 10)) -> dict[str, float]:
    ranks = np.asarray(ranks)
    if ranks.ndim != 1 or not len(ranks) or np.any(ranks < 1):
        raise ValueError('Rank array must be nonempty and contain positive integers.')
    return {f'R@{int(k)}': round(float(100 * np.mean(ranks <= int(k))), 2) for k in ks}
