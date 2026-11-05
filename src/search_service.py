"""Cached access to the existing photo index and pretrained retrieval pipeline."""
from __future__ import annotations

import streamlit as st

from src.retrieval import SearchHit, search
from src.storage import read_embeddings, read_metadata

DEFAULT_MODEL = 'SigLIP'


@st.cache_data(show_spinner=False)
def metadata_rows(directory: str) -> list[dict]:
    return read_metadata(directory)


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


def find_matches(query: str, directory: str, num_images: int, top_k: int = 10,
                 model_name: str = DEFAULT_MODEL) -> list[SearchHit]:
    """Search the complete gallery using the original cosine ranking function."""
    query = query.strip()
    if not query:
        raise ValueError('Enter a description to search your library.')
    if top_k not in (5, 10, 20):
        raise ValueError('Choose 5, 10 or 20 results.')
    vectors = embedding_matrix(directory, model_name, num_images)
    return search(query_embedding(query, model_name), vectors, top_k=top_k)
