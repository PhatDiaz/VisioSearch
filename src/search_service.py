"""Cached two-stage retrieval service plus an explicit, separately named v1 fallback."""
from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import streamlit as st

from src.retrieval import SearchHit, search
from src.storage import read_embeddings, read_metadata
from src.pipeline_artifacts import ArtifactError, MANIFEST_PATH, PIPELINE_NAME, load_runtime_artifacts
from src.two_stage_retriever import TwoStageRetriever, SearchResponse, SearchTimings

DEFAULT_MODEL = PIPELINE_NAME


def active_pipeline() -> str:
    mode = os.getenv('TOPIC11_PIPELINE', 'two_stage')
    if mode not in ('two_stage', 'pretrained_siglip'):
        raise ArtifactError('TOPIC11_PIPELINE must be two_stage or the explicit pretrained_siglip fallback.')
    return mode


@st.cache_data(show_spinner=False)
def _metadata_rows(directory: str, stamp: tuple[int, int]) -> list[dict]:
    return read_metadata(directory)


def metadata_rows(directory: str) -> list[dict]:
    stat = (Path(directory) / 'metadata.jsonl').stat()
    return _metadata_rows(directory, (stat.st_mtime_ns, stat.st_size))


@st.cache_resource(show_spinner=False)
def embedding_matrix(directory: str, model_name: str, num_images: int):
    return read_embeddings(directory, model_name, num_images)


@st.cache_resource(show_spinner=False)
def cached_encoder(name: str):
    from src.modeling import load_model
    return load_model(name)


@st.cache_data(show_spinner=False, max_entries=256)
def query_embedding(query: str, name: str):
    from src.modeling import text_embeddings
    model, processor, device = cached_encoder(name)
    return text_embeddings([query], model, processor, device)[0]


def find_matches_pretrained(query: str, directory: str, num_images: int, top_k: int = 10,
                            model_name: str = 'SigLIP') -> list[SearchHit]:
    """Explicit legacy fallback; never selected automatically after a model error."""
    query = query.strip()
    if not query:
        raise ValueError('Enter a description to search your library.')
    if top_k not in (5, 10, 20):
        raise ValueError('Choose 5, 10 or 20 results.')
    vectors = embedding_matrix(directory, model_name, num_images)
    return search(query_embedding(query, model_name), vectors, top_k=top_k)


@st.cache_resource(show_spinner=False)
def cached_two_stage_retriever(directory: str, num_images: int, manifest_path: str, device: str = 'auto',
                                signature: tuple = ()) -> TwoStageRetriever:
    artifacts = load_runtime_artifacts(directory, manifest_path)
    if num_images != len(artifacts.image_ids):
        raise ArtifactError('The UI metadata does not match the verified LoRA image gallery.')
    return TwoStageRetriever(artifacts, device)


def prepare_search_index(directory: str, num_images: int):
    if active_pipeline() == 'pretrained_siglip':
        return embedding_matrix(directory, 'SigLIP', num_images)
    manifest = os.getenv('TOPIC11_RUNTIME_MANIFEST', str(MANIFEST_PATH))
    manifest_path = Path(manifest).resolve()
    # An artifact change must revalidate hashes rather than reusing an incompatible model/index cache.
    paths = [Path(directory) / 'metadata.jsonl', manifest_path]
    try:
        config = json.loads(manifest_path.read_text(encoding='utf-8'))
        root = manifest_path.parents[2]
        for relative in config.get('runtime_sha256', {}) if isinstance(config, dict) else ():
            if not isinstance(relative, str):
                continue
            path = (root / relative).resolve()
            if path.is_relative_to(root):
                paths.append(path)
    except (OSError, ValueError, TypeError, IndexError):
        pass  # The loader below supplies the actionable artifact error.
    signature = []
    for path in paths:
        try:
            stat = path.stat()
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            signature.append((str(path), None, None))
    return cached_two_stage_retriever(directory, num_images, str(manifest_path),
                                      os.getenv('TOPIC11_DEVICE', 'auto'), tuple(signature))


def retrieve(query: str, directory: str, num_images: int, top_k: int = 10) -> SearchResponse:
    if active_pipeline() == 'pretrained_siglip':
        started = perf_counter()
        hits = find_matches_pretrained(query, directory, num_images, top_k)
        elapsed = perf_counter() - started
        return SearchResponse(tuple(hits), SearchTimings(elapsed, 0, elapsed), pipeline='Pretrained SigLIP (explicit fallback)')
    return prepare_search_index(directory, num_images).search(query, top_k)


def find_matches(query: str, directory: str, num_images: int, top_k: int = 10,
                 model_name: str | None = None) -> list:
    if model_name is not None:
        return find_matches_pretrained(query, directory, num_images, top_k, model_name)
    return list(retrieve(query, directory, num_images, top_k).hits)
