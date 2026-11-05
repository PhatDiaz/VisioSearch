"""VisioSearch — academic full-gallery text-to-image retrieval web application.

Launch: streamlit run app.py
Offline image embeddings are required; see README.md.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.retrieval import search
from src.storage import image_file, read_embeddings, read_metadata

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
INDEX_DIR = Path(os.getenv('TOPIC11_INDEX_DIR', str(ROOT / 'data/flickr8k')))
RESULTS_DIR = Path(os.getenv('TOPIC11_RESULTS_DIR', str(ROOT / 'results')))

st.set_page_config(
    page_title='VisioSearch | Natural Language Image Retrieval',
    page_icon='🔎',
    layout='wide',
    initial_sidebar_state='collapsed',
)

st.markdown('''<style>
:root { --accent:#4de0c1; }
.block-container { max-width:1450px; padding-top:1.35rem; padding-bottom:5rem; }
h1, h2, h3 { letter-spacing:-.025em; }
[data-testid="stAppViewContainer"] { background:radial-gradient(ellipse 65% 40% at 78% -16%, #17374b 0%, #0a1323 70%); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMetricValue"] { font-weight:800; }
.hero { padding:2.1rem 2.3rem; margin: .6rem 0 1.5rem; border:1px solid #254359;
  border-radius:22px; background:linear-gradient(117deg,rgba(26,59,76,.92),rgba(13,25,44,.88)); }
.eyebrow { color:#4de0c1; font-size:.72rem; font-weight:750; letter-spacing:.22em; text-transform:uppercase; }
.hero h1 { color:#f6fbff; font-size:clamp(2.4rem,4vw,4.15rem); margin:.45rem 0 .6rem; line-height:1.06; }
.hero p { color:#a8bdd1; font-size:1rem; line-height:1.6; max-width:780px; }
.pill { display:inline-block; font-size:.75rem; padding:.36rem .75rem; margin-right:.45rem; margin-top:.65rem;
  border-radius:99px; background:#203e4b; color:#d5fff4; border:1px solid #306456; }
.kicker { color:#91aec8; font-size:.85rem; text-transform:uppercase; letter-spacing:.1em; }
.card-title { color:#f4faff; font-weight:700; font-size:1.08rem; }
.stTabs [data-baseweb="tab-list"] { gap:1.3rem; }
.stTabs [data-baseweb="tab"] { font-weight:650; }
div[data-testid="stForm"] { border:1px solid #2a4359; border-radius:19px; background:rgba(14,29,48,.68); padding:1.2rem 1.3rem; }
div[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius:16px; }
div[data-testid="stImage"] img { border-radius:12px; }
.smallprint { color:#91a6b9; font-size:.82rem; }
</style>''', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def metadata_rows(directory: str):
    return read_metadata(directory)


@st.cache_resource(show_spinner=False)
def embedding_matrix(directory: str, model_name: str, num_images: int):
    return read_embeddings(directory, model_name, num_images)


@st.cache_resource(show_spinner=False)
def cached_encoder(name: str):
    from src.modeling import load_model
    return load_model(name)


@st.cache_data(show_spinner=False)
def get_query_embedding(query: str, name: str) -> np.ndarray:
    from src.modeling import text_embeddings
    model, processor, device = cached_encoder(name)
    return text_embeddings([query], model, processor, device)[0]


def find_matches(query: str, model_name: str, count: int, split_name: str, rows: list[dict]):
    vectors = embedding_matrix(str(INDEX_DIR), model_name, len(rows))
    text_vector = get_query_embedding(query, model_name)
    selected = None if split_name == 'All 8,000' else [i for i, item in enumerate(rows) if item['split'] == split_name.lower()]
    return search(text_vector, vectors, count, selected)


def load_image(row: dict):
    with Image.open(image_file(INDEX_DIR, row)) as im:
        return im.convert('RGB')


def hit_gallery(hits, rows, label: str | None = None, columns: int = 4):
    if label:
        st.subheader(label)
    if not hits:
        st.warning('No matching images in this gallery filter.')
        return
    cols = st.columns(columns, gap='medium')
    for rank, hit in enumerate(hits, 1):
        record = rows[hit.image_index]
        with cols[(rank-1) % columns]:
            with st.container(border=True):
                st.image(load_image(record), use_container_width=True)
                st.markdown(f'**#{rank:02d}** &nbsp; · &nbsp; **{hit.cosine_similarity:.4f}** cosine', unsafe_allow_html=True)
                st.caption(f"{record['image_id']}  •  {record['split'].upper()}")
                with st.expander('Reference captions', expanded=False):
                    for caption in record['captions']:
                        st.write('•', caption)


def show_benchmark():
    path = RESULTS_DIR / 'benchmark_results.csv'
    if not path.is_file():
        st.info('Benchmarks have not been measured yet. Build the index, then run `python scripts/evaluate.py`.')
        st.code('python scripts/evaluate.py --index-dir data/flickr8k', language='bash')
        return
    df = pd.read_csv(path)
    if df.empty:
        st.warning('No benchmark rows found.')
        return
    option_labels = {'test_only': 'Canonical test gallery (1,000)', 'all_8000': 'Full gallery (8,000)'}
    for gallery in ['test_only', 'all_8000']:
        sub = df[df['gallery'] == gallery]
        if sub.empty:
            continue
        st.subheader(option_labels[gallery])
        st.caption('Queries: 5,000 captions from the independent test split. Values are Recall (%), not accuracy probabilities.')
        st.dataframe(sub[['model','gallery_images','text_queries','R@1','R@5','R@10']], use_container_width=True, hide_index=True)
        tidy = sub.melt(id_vars='model', value_vars=['R@1','R@5','R@10'], var_name='Metric', value_name='Recall (%)')
        st.bar_chart(tidy, x='Metric', y='Recall (%)', color='model', stack=False)
        st.divider()
    st.download_button('Download benchmark CSV', path.read_bytes(), file_name='benchmark_results.csv', mime='text/csv')
    st.caption('Do not compare 1,000-image and 8,000-image scores as if gallery difficulty were identical.')


def show_failures(rows: list[dict]):
    path = RESULTS_DIR / 'three_failure_cases_siglip.csv'
    if not path.is_file():
        st.info('Failure cases are generated during test evaluation. Run `python scripts/evaluate.py`.')
        return
    failures = pd.read_csv(path).head(3)
    if failures.empty:
        st.info('No Top-1 failures were found in the saved evaluation.')
        return
    for _, entry in failures.iterrows():
        with st.container(border=True):
            st.markdown(f'**Query #{int(entry.query_id)}**  ·  Ground-truth rank **{int(entry.ground_truth_rank)}**')
            st.markdown(f'*“{entry["query"]}”*')
            left, right = st.columns(2)
            with left:
                st.image(load_image(rows[int(entry.ground_truth_index)]), use_container_width=True)
                st.caption('Ground-truth image')
            with right:
                st.image(load_image(rows[int(entry.top1_index)]), use_container_width=True)
                st.caption('Incorrect Top-1 retrieved image')
            explanation = entry.get('one_line_explanation', '')
            if isinstance(explanation, str) and explanation.strip():
                st.write('**Analysis:**', explanation)
            else:
                st.caption('Analysis not annotated yet. Compare the two images and record a grounded one-line explanation in the failure CSV.')


def show_methodology(num_images: int):
    st.subheader('Method and research protocol')
    st.markdown('''
**Research task** — Zero-shot text-to-image retrieval over Flickr8k. Each image has five English captions.

**Image index** — Both pretrained models encode the same set of gallery images offline. Embeddings are L2-normalized and saved as NumPy arrays.

**Live search** — A user enters an English text query. The chosen model converts it into a text embedding, then ranks gallery embeddings using cosine similarity.

**Evaluation** — Standard Recall@1, Recall@5 and Recall@10 over 5,000 caption queries from the held-out test split, first with a 1,000-image gallery and optionally an 8,000-image gallery.

**Scientific separation** — Test captions are never used to train the models or tune prompts, and scores from different gallery sizes are reported separately.

**Limitations** — The dataset is a proxy for a personal photo library; one paired Flickr8k ground truth per caption does not measure all semantically acceptable images. Semantic similarity is not a calibrated probability of relevance.
    ''')
    st.caption(f'Gallery currently loaded: {num_images:,} images. Models: CLIP ViT-B/32 and SigLIP Base Patch16-224.')


st.markdown('''<div class="hero">
<div class="eyebrow">Computer Vision · Graduate Research Project 11</div>
<h1>Visio<span style="color:#4de0c1">Search</span></h1>
<p>Search an entire photo library using natural language. Explore 8,000 Flickr8k images with pretrained vision-language models, inspect cosine similarity, and compare retrieval quality in a reproducible research dashboard.</p>
<span class="pill">Zero-shot image retrieval</span><span class="pill">CLIP · SigLIP</span><span class="pill">Flickr8k</span>
</div>''', unsafe_allow_html=True)

try:
    rows = metadata_rows(str(INDEX_DIR))
except (FileNotFoundError, ValueError) as exc:
    st.warning('The 8,000-image gallery has not been prepared on this computer.')
    st.markdown('**Step 1 — Build the gallery and embeddings**')
    st.code('python scripts/build_gallery.py --output-dir data/flickr8k', language='bash')
    st.markdown('**Step 2 — Restart or refresh this app**')
    st.code('streamlit run app.py', language='bash')
    st.caption('The first build needs an internet connection and downloads the Flickr8k dataset plus model checkpoints. See README.md for Colab GPU preparation.')
    st.stop()

counts = Counter(x['split'] for x in rows)
stats = st.columns(4)
for col, label, val in zip(stats,
                           ['Images indexed','Train split','Validation split','Test split'],
                           [len(rows),counts['train'],counts['dev'],counts['test']]):
    with col:
        st.metric(label, f'{val:,}')
if len(rows) != 8000:
    st.warning(f'Development gallery detected: {len(rows):,}/8,000 images. Do not present it as the full dataset.')

search_tab, comparison_tab, failure_tab, method_tab = st.tabs(
    ['🔍 Search images', '📊 Model evaluation', '🧩 Failure explorer', '📚 Methodology']
)

with search_tab:
    st.markdown('### Explore the image library')
    st.caption('Describe what you want to see in English. Search scores are cosine similarity, not probabilities.')
    examples = ['A dog playing in the snow', 'Two people riding bicycles',
                'A child climbing a playground', 'People kayaking on the water']
    with st.form('search_form', clear_on_submit=False):
        query = st.text_input('Describe an image', value='A dog running on the beach',
                              placeholder='e.g. A cyclist jumping over a ramp')
        x1, x2, x3 = st.columns([1,1,1])
        with x1:
            model_selection = st.selectbox('Vision-language model', ['SigLIP','CLIP'], index=0)
        with x2:
            split_selection = st.selectbox('Search gallery', ['All 8,000','Train','Dev','Test'])
        with x3:
            num_results = st.slider('Results to show', 4, 20, 8, 4)
        compare_models = st.checkbox('Compare CLIP and SigLIP side by side', value=False)
        submitted = st.form_submit_button('Search the library', type='primary', use_container_width=True)
    st.markdown('**Try searching for:** ' + ' · '.join(f'`{x}`' for x in examples))
    if submitted:
        if not query.strip():
            st.warning('Enter an English search query first.')
        else:
            st.session_state['last_search'] = dict(query=query.strip(), name=model_selection,
                                                    split=split_selection, top_k=num_results,
                                                    compare=compare_models)
    if 'last_search' in st.session_state:
        params = st.session_state['last_search']
        model_names = ['CLIP','SigLIP'] if params['compare'] else [params['name']]
        try:
            with st.spinner('Encoding the query and ranking the photo library…'):
                searches = {name: find_matches(params['query'], name, params['top_k'], params['split'], rows)
                            for name in model_names}
            st.markdown(f"**Search:** {params['query']}  ·  **Gallery:** {params['split']}")
            if params['compare']:
                a, b = st.columns(2)
                with a:
                    hit_gallery(searches['CLIP'], rows, 'CLIP · Baseline', columns=2)
                with b:
                    hit_gallery(searches['SigLIP'], rows, 'SigLIP · Main model', columns=2)
            else:
                hit_gallery(searches[params['name']], rows, f"{params['name']} · Top {len(searches[params['name']])}")
            export = []
            for name, hits in searches.items():
                for rank, hit in enumerate(hits, 1):
                    record = rows[hit.image_index]
                    export.append({'query': params['query'], 'model': name, 'rank': rank,
                                   'image_id': record['image_id'], 'split': record['split'],
                                   'cosine_similarity': hit.cosine_similarity})
            st.download_button('Export these search results',
                               pd.DataFrame(export).to_csv(index=False).encode('utf-8'),
                               file_name='visiosearch_results.csv', mime='text/csv')
        except Exception as exc:
            st.error('Search could not be completed. Confirm the model files and embeddings are available.')
            st.exception(exc)
    else:
        st.info('Your gallery is ready. Enter a description and click **Search the library** to explore real model results.')

with comparison_tab:
    show_benchmark()
with failure_tab:
    show_failures(rows)
with method_tab:
    show_methodology(len(rows))

st.divider()
st.markdown('<div class="smallprint">VisioSearch · Topic 11 · Computer Vision Final Project · CLIP / SigLIP · Results shown only after actual computation.</div>', unsafe_allow_html=True)
