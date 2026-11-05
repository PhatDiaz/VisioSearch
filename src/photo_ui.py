"""Photo presentation helpers. Retrieval and scientific metrics live elsewhere."""
from __future__ import annotations

from html import escape
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from src.storage import image_file


@st.cache_data(show_spinner=False, max_entries=256)
def thumbnail(directory: str, relative_path: str) -> bytes:
    """Crop display copies only; the original gallery files stay untouched."""
    with Image.open(image_file(directory, {'path': relative_path})) as source:
        image = ImageOps.fit(source.convert('RGB'), (640, 480), method=Image.Resampling.LANCZOS)
    output = BytesIO()
    image.save(output, format='JPEG', quality=85)
    return output.getvalue()


def remember_theme() -> None:
    st.query_params['theme'] = 'light' if st.session_state['light_mode'] else 'dark'


def header(image_count: int | None) -> None:
    count = f'{image_count:,} images indexed' if image_count is not None else 'Library unavailable'
    theme = 'light' if st.session_state.get('light_mode', False) else 'dark'
    with st.container(key='header_row'):
        brand, theme_control = st.columns([5, 1.1], vertical_alignment='center')
        with theme_control:
            st.toggle('Light mode', key='light_mode', on_change=remember_theme,
                      help='Switch between light and dark appearance.')
        with brand:
            st.markdown(f'''
    <header class="app-header">
      <a class="brand" href="/?theme={theme}" target="_self" aria-label="VisioSearch home">
        <span class="brand-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none">
          <rect x="3.5" y="3.5" width="17" height="17" rx="5" stroke="currentColor" stroke-width="1.6"/>
          <circle cx="10.6" cy="10.6" r="3.7" stroke="currentColor" stroke-width="1.6"/>
          <path d="m13.4 13.4 3.5 3.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
        </svg></span><span>Visio<span class="brand-accent">Search</span></span>
      </a>
      <span class="library-status"><span class="status-dot"></span>{count}</span>
    </header>''', unsafe_allow_html=True)


def hero() -> None:
    st.markdown('''<section class="search-hero">
      <div class="eyebrow"><span></span> A new way to explore your library</div>
      <h1>You have the words.<br><span class="hero-accent">Find the picture.</span></h1>
      <p>Discover images with natural language.</p>
    </section>''', unsafe_allow_html=True)


def empty_state() -> None:
    st.markdown('''<section class="empty-state">
      <div class="empty-icon" aria-hidden="true"><svg viewBox="0 0 48 48" fill="none">
        <rect x="9" y="8" width="30" height="32" rx="6" stroke="currentColor" stroke-width="1.5"/>
        <circle cx="29" cy="18" r="3" stroke="currentColor" stroke-width="1.5"/>
        <path d="m10 33 10-10 8 8 5-5 6 7" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>
      </svg></div>
      <h2>A picture starts with a few words.</h2>
      <p>A place, a moment, a feeling. Describe what you have in mind,<br class="desktop-break">
      or try one of the ideas above.</p>
      <span class="empty-note">Your next discovery is one search away.</span>
    </section>''', unsafe_allow_html=True)


@st.dialog('Image details', width='large')
def image_details(directory: str, record: dict, rank: int, score: float) -> None:
    path = image_file(directory, record)
    try:
        original = path.read_bytes()
        with Image.open(BytesIO(original)) as source:
            size = source.size
        st.image(original, width='stretch')
    except (OSError, UnidentifiedImageError, ValueError):
        st.warning('This image is currently unavailable. Please try another result.')
        return
    st.markdown(f'**Result #{rank:02d}** · **{score:.4f}** cosine similarity')
    st.caption('Similarity describes how closely the image matches your words. It is not a probability.')
    first, second, third = st.columns(3)
    first.markdown(f'**Image ID**\n\n`{record["image_id"]}`')
    second.markdown(f'**Collection**\n\nFlickr8k · {escape(record["split"].title())}')
    third.markdown(f'**Original size**\n\n{size[0]} × {size[1]} px')
    st.markdown('#### Reference captions')
    for caption in record.get('captions', []):
        st.write(caption)
    st.download_button('Download original image', original, file_name=path.name,
                       mime='image/jpeg', key=f'download_{record["image_id"]}')


def result_gallery(hits, rows: list[dict], directory: str) -> None:
    # CSS keeps DOM order identical to ranking at every screen size.
    with st.container(key='results_grid'):
        for rank, hit in enumerate(hits, 1):
            record = rows[hit.image_index]
            with st.container(border=True, key=f'photo_{rank}'):
                try:
                    st.image(thumbnail(directory, record['path']), width='stretch')
                except (OSError, UnidentifiedImageError, ValueError):
                    st.warning('Image unavailable')
                st.markdown(f'''<div class="photo-meta">
                    <span class="rank">#{rank:02d}</span>
                    <span class="score">{hit.cosine_similarity:.4f}<span> similarity</span></span>
                    </div>''', unsafe_allow_html=True)
                if st.button('View image', key=f'view_{rank}', width='stretch'):
                    image_details(directory, record, rank, hit.cosine_similarity)
