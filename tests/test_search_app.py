"""Search-page integration checks against the actual prepared Flickr8k index."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'data' / 'flickr8k'
SIGLIP_CACHE = ROOT / '.cache' / 'huggingface' / 'hub' / 'models--google--siglip-base-patch16-224' / 'snapshots'
HAS_SIGLIP = any(SIGLIP_CACHE.glob('*/model.safetensors')) or any(SIGLIP_CACHE.glob('*/pytorch_model.bin'))


def assert_healthy(app):
    assert not app.exception, [item.message for item in app.exception]
    assert not app.error, [item.value for item in app.error]


@pytest.mark.skipif(not (INDEX / 'embeddings_siglip.npy').is_file() or not HAS_SIGLIP,
                    reason='Requires the prepared gallery and cached SigLIP checkpoint')
def test_real_search_topk_details_and_examples(monkeypatch):
    from src import search_service
    from src import modeling

    def forbidden(*args, **kwargs):
        raise AssertionError('The homepage must not load an encoder before searching')

    monkeypatch.setenv('TOPIC11_INDEX_DIR', str(INDEX))
    import requests
    monkeypatch.setattr(requests.sessions.Session, 'request', forbidden)
    original_encoder = search_service.cached_encoder
    monkeypatch.setattr(search_service, 'cached_encoder', forbidden)
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
    assert_healthy(app)
    assert not app.tabs
    assert not app.dataframe
    assert len(app.selectbox) == 1
    assert app.selectbox[0].options == ['5 results', '10 results', '20 results']
    assert not any(button.label == 'View image' for button in app.button)
    assert any('8,000 images indexed' in item.value for item in app.markdown)

    app.button[0].click().run()
    assert_healthy(app)
    assert app.warning
    assert 'search_response' not in app.session_state
    monkeypatch.setattr(search_service, 'cached_encoder', original_encoder)

    def no_image_encoding(*args, **kwargs):
        raise AssertionError('Live search must never regenerate gallery embeddings')

    monkeypatch.setattr(modeling, 'image_embeddings', no_image_encoding)
    query = 'A dog running on the beach'
    app.text_input[0].set_value(query)
    for top_k in (5, 10, 20):
        app.selectbox[0].select(top_k)
        app.button[0].click().run()
        assert_healthy(app)
        response = app.session_state['search_response']
        assert len(response['hits']) == top_k
        assert len([b for b in app.button if b.label == 'View image']) == top_k
        vectors = search_service.embedding_matrix(str(INDEX), 'SigLIP', 8000)
        assert vectors.shape == (8000, 768)
        query_vector = search_service.query_embedding(query, 'SigLIP')
        scores = vectors @ query_vector
        expected = np.argsort(-scores, kind='stable')[:top_k]
        assert [hit.image_index for hit in response['hits']] == expected.tolist()
        assert np.allclose([hit.cosine_similarity for hit in response['hits']], scores[expected])

    response = app.session_state['search_response']
    app.toggle(key='light_mode').set_value(True).run()
    assert_healthy(app)
    assert app.query_params['theme'] == 'light'
    assert app.text_input[0].value == query
    assert app.session_state['search_response'] == response
    app.button(key='view_1').click().run()
    assert_healthy(app)
    assert any(button.label == 'Download original image' for button in app.get('download_button'))
    rows = search_service.metadata_rows(str(INDEX))
    best = rows[response['hits'][0].image_index]
    for caption in best['captions']:
        assert any(caption in item.value for item in app.markdown)
    assert app.session_state['search_response'] == response
    app.toggle(key='light_mode').set_value(False).run()
    assert_healthy(app)
    assert app.query_params['theme'] == 'dark'
    assert app.session_state['search_response'] == response

    # Start another session to verify example clicks populate and submit the query.
    example_app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
    example_app.button(key='example_Dogs at the beach').click().run()
    assert_healthy(example_app)
    assert example_app.text_input[0].value == query
    assert len(example_app.session_state['search_response']['hits']) == 10


def test_missing_index_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv('TOPIC11_INDEX_DIR', str(tmp_path))
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    assert not app.exception
    assert any('library is not ready' in item.value for item in app.error)
    assert app.code
    assert not any(button.label == 'Search' for button in app.button)


@pytest.mark.skipif(not (INDEX / 'embeddings_siglip.npy').is_file(), reason='Requires the prepared Flickr8k gallery')
def test_model_error_is_friendly_and_can_be_retried(monkeypatch):
    from src import search_service
    monkeypatch.setenv('TOPIC11_INDEX_DIR', str(INDEX))

    def unavailable(*args, **kwargs):
        raise OSError('Test: model cache is unavailable')

    search_service.query_embedding.clear()
    monkeypatch.setattr(search_service, 'cached_encoder', unavailable)
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    app.text_input[0].set_value('A missing model test query')
    app.button[0].click().run()
    assert not app.exception
    assert any('model could not be loaded' in item.value for item in app.error)
    assert 'search_response' not in app.session_state
