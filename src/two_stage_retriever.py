"""Exact frozen K20 z-score fusion from final evaluation notebook cells 5, 8, 9."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
from threading import RLock
from time import perf_counter
from typing import Callable

import numpy as np

from src.pipeline_artifacts import ALPHA_BLIP, CANDIDATE_COUNT, PIPELINE_NAME, RuntimeArtifacts


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Scores must be a non-empty, finite one-dimensional array.')
    return (values - values.mean()) / (values.std() + 1e-8)


def fuse_scores(cosine: np.ndarray, positive_logits: np.ndarray) -> np.ndarray:
    if np.shape(cosine) != np.shape(positive_logits):
        raise ValueError('SigLIP and BLIP scores must refer to the same candidate positions.')
    return (1 - ALPHA_BLIP) * zscore(cosine) + ALPHA_BLIP * zscore(positive_logits)


def top20_candidates(query_vector: np.ndarray, image_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query_vector, dtype=np.float32)
    gallery = np.asarray(image_vectors, dtype=np.float32)
    if (gallery.ndim != 2 or query.shape != (gallery.shape[1],) or len(gallery) < CANDIDATE_COUNT
            or not np.isfinite(query).all()):
        raise ValueError('Text/image vector dimensions are incompatible with Top-20 retrieval.')
    # The encoder already performed the notebook's L2 normalization, exactly once.
    scores = (query[None, :] @ gallery.T)[0]
    order = np.argsort(-scores, kind='stable')[:CANDIDATE_COUNT]
    return order, scores[order]


@dataclass(frozen=True)
class TwoStageHit:
    image_index: int
    cosine_similarity: float
    blip_logit: float
    fused_score: float


@dataclass(frozen=True)
class SearchTimings:
    siglip_seconds: float
    blip_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class SearchResponse:
    hits: tuple
    timings: SearchTimings
    pipeline: str = PIPELINE_NAME
    cache_hit: bool = False


class TwoStageRetriever:
    def __init__(self, artifacts: RuntimeArtifacts, device: str = 'auto', *,
                 encoder_factory: Callable | None = None, reranker_factory: Callable | None = None,
                 cache_queries: int = 64):
        if cache_queries < 1:
            raise ValueError('Query cache capacity must be positive.')
        self.artifacts, self.device = artifacts, device
        self._encoder_factory, self._reranker_factory = encoder_factory, reranker_factory
        self._encoder = self._reranker = None
        self._results: OrderedDict[str, tuple[TwoStageHit, ...]] = OrderedDict()
        self._cache_queries = cache_queries
        self._lock = RLock()

    def _load_encoder(self):
        if self._encoder is None:
            if self._encoder_factory:
                self._encoder = self._encoder_factory()
            else:
                from src.models.siglip_lora_loader import load_siglip_lora
                self._encoder = load_siglip_lora(self.artifacts.adapter_dir, self.device)
        return self._encoder

    def _load_reranker(self):
        if self._reranker is None:
            if self._reranker_factory:
                self._reranker = self._reranker_factory()
            else:
                from src.models.blip_reranker import BlipReranker
                self._reranker = BlipReranker(self.device,
                    pair_batch_size=int(os.getenv('TOPIC11_BLIP_PAIR_BATCH', '12')),
                    vision_batch_size=int(os.getenv('TOPIC11_BLIP_VISION_BATCH', '8')))
        return self._reranker

    @staticmethod
    def _synchronize(device: str) -> None:
        if str(device).startswith('cuda'):
            import torch
            torch.cuda.synchronize(device)

    def search(self, query: str, top_k: int = 10) -> SearchResponse:
        query = query.strip()
        if not query:
            raise ValueError('Enter a description to search your library.')
        if top_k not in (5, 10, 20):
            raise ValueError('Choose 5, 10 or 20 results.')
        started = perf_counter()
        with self._lock:
            if query in self._results:
                self._results.move_to_end(query)
                return SearchResponse(self._results[query][:top_k], SearchTimings(0, 0, perf_counter() - started),
                                      cache_hit=True)
            encoder = self._load_encoder()
            vector = encoder.encode(query)
            candidates, cosine = top20_candidates(vector, self.artifacts.vectors)
            self._synchronize(encoder.device)
            stage1_done = perf_counter()
            reranker = self._load_reranker()
            logits = reranker.score(query, [self.artifacts.image_paths[int(i)] for i in candidates])
            if np.shape(logits) != (CANDIDATE_COUNT,):
                raise ValueError('BLIP must return exactly one positive logit per candidate.')
            fused = fuse_scores(cosine, logits)
            order = np.argsort(-fused, kind='stable')
            hits = tuple(TwoStageHit(int(candidates[i]), float(cosine[i]), float(logits[i]), float(fused[i])) for i in order)
            self._synchronize(reranker.device)
            finished = perf_counter()
            self._results[query] = hits
            while len(self._results) > self._cache_queries:
                self._results.popitem(last=False)
            return SearchResponse(hits[:top_k], SearchTimings(stage1_done - started,
                                                            finished - stage1_done, finished - started))
