"""VisioSearch: a focused natural-language photo search application.

Launch: streamlit run app.py
Academic dashboard: streamlit run research_app.py --server.port 8502
"""
from __future__ import annotations

import csv
from html import escape
from io import StringIO
import logging
import os
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))

from src import photo_ui
from src.search_service import DEFAULT_MODEL, active_pipeline, metadata_rows, prepare_search_index, retrieve
from src.pipeline_artifacts import ArtifactError

INDEX_DIR = Path(os.getenv('TOPIC11_INDEX_DIR', str(ROOT / 'data' / 'flickr8k')))
LOGGER = logging.getLogger(__name__)
EXAMPLES = [
    ('Dogs at the beach', 'A dog running on the beach'),
    ('A snowy adventure', 'People skiing on a snowy mountain'),
    ('Life on two wheels', 'People riding bicycles'),
    ('By the water', 'People kayaking on the water'),
]


def choose_example(query: str) -> None:
    st.session_state['search_query'] = query
    st.session_state['example_pending'] = True


def search_controls() -> tuple[str, int, bool]:
    with st.container(key='search_panel'):
        with st.form('search_form', border=False):
            text, action = st.columns([6, 1.25], vertical_alignment='center')
            with text:
                query = st.text_input('Describe the image you are looking for',
                                      placeholder="Describe the image you're looking for...",
                                      key='search_query', label_visibility='collapsed')
            with action:
                submitted = st.form_submit_button('Search', type='primary', width='stretch')
            with st.container(key='search_options'):
                count, hint = st.columns([1.25, 3.75])
                with count:
                    top_k = st.selectbox('Number of results', [5, 10, 20], index=1,
                                         format_func=lambda value: f'{value} results',
                                         key='result_count', label_visibility='collapsed')
        with st.container(key='examples'):
            with st.container(key='example_buttons'):
                for column, (label, example) in zip(st.columns(4, gap='small'), EXAMPLES):
                    column.button(label, key=f'example_{label}', on_click=choose_example,
                                  args=(example,), width='stretch')
    return query, top_k, submitted or st.session_state.pop('example_pending', False)


def run_search(query: str, top_k: int, image_count: int) -> None:
    query = query.strip()
    if not query:
        st.warning('Describe an image first, or choose an example to get started.')
        return
    try:
        with st.spinner('Finding your images…'):
            result = retrieve(query, str(INDEX_DIR), image_count, top_k)
            response = dict(query=query, hits=list(result.hits), elapsed=result.timings.total_seconds,
                            pipeline=result.pipeline, timings=vars(result.timings), cache_hit=result.cache_hit)
        # Keep the last successful response, so image dialogs never re-run retrieval.
        st.session_state['search_response'] = response
    except ArtifactError as exc:
        LOGGER.exception('Two-stage pipeline artifacts or model cache are unavailable')
        st.error(str(exc))
    except (FileNotFoundError, OSError):
        LOGGER.exception('Photo search could not load local model or index files')
        st.error('The search model could not be loaded. Check your connection for the first download, '
                 'or prepare the model cache using the command in README.md. Then try again.')
    except Exception:
        LOGGER.exception('Photo search failed')
        st.error('We could not complete that search. Please try again. '
                 'If it keeps happening, check the library files and the app log.')


def export_results(response: dict, rows: list[dict]) -> bytes:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=['query', 'model', 'rank', 'image_id', 'split',
                                               'cosine_similarity', 'blip_itm_logit', 'fused_score'])
    writer.writeheader()
    for rank, hit in enumerate(response['hits'], 1):
        row = rows[hit.image_index]
        writer.writerow(dict(query=response['query'], model=response.get('pipeline', 'Unspecified legacy pipeline'), rank=rank,
                             image_id=row['image_id'], split=row['split'],
                             cosine_similarity=hit.cosine_similarity,
                             blip_itm_logit=getattr(hit, 'blip_logit', ''),
                             fused_score=getattr(hit, 'fused_score', '')))
    return buffer.getvalue().encode('utf-8')


def show_results(response: dict, rows: list[dict]) -> None:
    hits = response['hits']
    if not hits:
        st.info('No images were found. Try describing your idea in a different way.')
        return
    st.markdown(f'''<section class="results-heading">
        <h2><span>Results for</span> “{escape(response['query'])}”</h2>
        <p>{len(hits)} matches · {len(rows):,} images searched · {response['elapsed']:.2f}s</p>
        </section>''', unsafe_allow_html=True)
    photo_ui.result_gallery(hits, rows, str(INDEX_DIR))
    with st.container(key='results_actions'):
        download, note = st.columns([1.3, 3.7])
        with download:
            st.download_button('Export results', export_results(response, rows),
                               file_name='visiosearch_results.csv', mime='text/csv')
        with note:
            st.markdown('<p class="score-note">Match scores describe relevance to your words; they are not probabilities.</p>',
                        unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title='VisioSearch — Find your picture', page_icon='🔎',
                       layout='wide', initial_sidebar_state='collapsed')
    if 'light_mode' not in st.session_state:
        st.session_state['light_mode'] = st.query_params.get('theme') == 'light'
    styles = (ROOT / 'assets' / 'styles.css').read_text(encoding='utf-8')
    if st.session_state['light_mode']:
        styles += (ROOT / 'assets' / 'light.css').read_text(encoding='utf-8')
    st.markdown(f'<style>{styles}</style>', unsafe_allow_html=True)
    try:
        rows = metadata_rows(str(INDEX_DIR))
        prepare_search_index(str(INDEX_DIR), len(rows))
    except ArtifactError as exc:
        LOGGER.exception('Two-stage artifacts failed validation')
        photo_ui.header(len(rows))
        photo_ui.hero()
        st.error(str(exc))
        st.code('python scripts/prepare_two_stage.py --download-models', language='bash')
        return
    except (OSError, ValueError, KeyError):
        LOGGER.exception('Photo library is unavailable')
        photo_ui.header(None)
        photo_ui.hero()
        st.error('Your photo library is not ready yet. Restore the prepared Flickr8k index to start searching.')
        st.markdown('Copy the exported Flickr8k index into this project, or prepare it with:')
        st.code('python scripts/build_gallery.py --output-dir data/flickr8k', language='bash')
        st.caption('README.md includes Windows setup instructions and model preparation steps.')
        return

    photo_ui.header(len(rows))
    photo_ui.hero()
    if active_pipeline() == 'pretrained_siglip':
        st.warning('The original pretrained SigLIP fallback is active. Two-stage retrieval is explicitly disabled.')
    if len(rows) != 8000:
        st.warning(f'This library contains {len(rows):,} of 8,000 images. Restore the full index to search the complete collection.')
    query, top_k, submitted = search_controls()
    if submitted:
        run_search(query, top_k, len(rows))
    response = st.session_state.get('search_response')
    expected_pipeline = DEFAULT_MODEL if active_pipeline() == 'two_stage' else 'Pretrained SigLIP (explicit fallback)'
    if response and response.get('pipeline') != expected_pipeline:
        # Invalidate responses from an older app version instead of relabeling their scores.
        del st.session_state['search_response']
        response = None
    if response:
        show_results(response, rows)
    else:
        photo_ui.empty_state()
    st.markdown('''<footer class="site-footer"><span>VisioSearch · A different way to look.</span>
        <span>Photography from Flickr8k</span></footer>''', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
