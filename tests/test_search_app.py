"""Search-page integration checks against the actual prepared Flickr8k index."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'data' / 'flickr8k'
SIGLIP_CACHE = ROOT / '.cache' / 'huggingface' / 'hub' / 'models--google--siglip-base-patch16-224' / 'snapshots'
HAS_SIGLIP = any(SIGLIP_CACHE.glob('*/model.safetensors')) or any(SIGLIP_CACHE.glob('*/pytorch_model.bin'))
BLIP_CACHE = ROOT / '.cache/huggingface/hub/models--Salesforce--blip-itm-base-coco/snapshots'
HAS_TWO_STAGE = (ROOT / 'models/two_stage/runtime_manifest.json').is_file() and (
    any(BLIP_CACHE.glob('*/model.safetensors')) or any(BLIP_CACHE.glob('*/pytorch_model.bin')))


def assert_healthy(app):
    assert not app.exception, [item.message for item in app.exception]
    assert not app.error, [item.value for item in app.error]


@pytest.mark.skipif(not (INDEX / 'embeddings_siglip.npy').is_file() or not HAS_SIGLIP or not HAS_TWO_STAGE,
                    reason='Requires the prepared two-stage artifacts and cached models')
def test_real_search_topk_details_and_examples(monkeypatch):
    from src import search_service
    from src import modeling
    from src.two_stage_retriever import TwoStageRetriever, fuse_scores

    def forbidden(*args, **kwargs):
        raise AssertionError('The homepage must not load an encoder before searching')

    monkeypatch.setenv('TOPIC11_INDEX_DIR', str(INDEX))
    import requests
    monkeypatch.setattr(requests.sessions.Session, 'request', forbidden)
    original_encoder = TwoStageRetriever._load_encoder
    original_reranker = TwoStageRetriever._load_reranker
    monkeypatch.setattr(TwoStageRetriever, '_load_encoder', forbidden)
    monkeypatch.setattr(TwoStageRetriever, '_load_reranker', forbidden)
    monkeypatch.setenv('TOPIC11_PIPELINE', 'two_stage')
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
    monkeypatch.setattr(TwoStageRetriever, '_load_encoder', original_encoder)
    monkeypatch.setattr(TwoStageRetriever, '_load_reranker', original_reranker)

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
        retriever = search_service.prepare_search_index(str(INDEX), 8000)
        vectors = retriever.artifacts.vectors
        assert vectors.shape == (8000, 768)
        query_vector = retriever._encoder.encode(query)
        scores = vectors @ query_vector
        candidates = np.argsort(-scores, kind='stable')[:20]
        all_hits = retriever.search(query, 20).hits
        assert {hit.image_index for hit in all_hits} == set(candidates)
        logit_lookup = {hit.image_index: hit.blip_logit for hit in all_hits}
        logits = np.array([logit_lookup[int(i)] for i in candidates], dtype='float32')
        fused = fuse_scores(scores[candidates], logits)
        expected = candidates[np.argsort(-fused, kind='stable')[:top_k]]
        assert [hit.image_index for hit in response['hits']] == expected.tolist()
        assert np.allclose([hit.cosine_similarity for hit in response['hits']], scores[expected])
        assert 'BLIP ITM' in response['pipeline']

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
    import csv
    from io import StringIO
    from app import export_results
    exported = list(csv.DictReader(StringIO(export_results(response, rows).decode('utf-8'))))
    assert len(exported) == 20
    assert exported[0]['model'] == response['pipeline']
    assert float(exported[0]['cosine_similarity']) == response['hits'][0].cosine_similarity
    assert float(exported[0]['blip_itm_logit']) == response['hits'][0].blip_logit
    assert float(exported[0]['fused_score']) == response['hits'][0].fused_score
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

    from src.pipeline_artifacts import ArtifactError
    from types import SimpleNamespace
    def unavailable(*args, **kwargs):
        raise ArtifactError('BLIP ITM is not cached. Run python scripts/prepare_two_stage.py --download-models.')

    monkeypatch.setattr(search_service, 'cached_two_stage_retriever', lambda *args: SimpleNamespace(search=unavailable))
    monkeypatch.setenv('TOPIC11_PIPELINE', 'two_stage')
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    app.text_input[0].set_value('A missing model test query')
    app.button[0].click().run()
    assert not app.exception
    assert any('BLIP ITM is not cached' in item.value for item in app.error)
    assert 'search_response' not in app.session_state


@pytest.mark.skipif(not (INDEX / 'metadata.jsonl').is_file(), reason='Requires gallery metadata')
def test_missing_two_stage_artifact_never_activates_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv('TOPIC11_INDEX_DIR', str(INDEX))
    monkeypatch.setenv('TOPIC11_RUNTIME_MANIFEST', str(tmp_path / 'absent.json'))
    monkeypatch.setenv('TOPIC11_PIPELINE', 'two_stage')
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    assert not app.exception
    assert any('Two-stage artifacts are not prepared' in item.value for item in app.error)
    assert not any(button.label == 'Search' for button in app.button)


@pytest.mark.skipif(not (INDEX / 'embeddings_siglip.npy').is_file(), reason='Requires legacy gallery')
def test_pretrained_fallback_is_explicit_and_visible(monkeypatch):
    monkeypatch.setenv('TOPIC11_INDEX_DIR', str(INDEX))
    monkeypatch.setenv('TOPIC11_PIPELINE', 'pretrained_siglip')
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    assert_healthy(app)
    assert any('fallback is active' in item.value for item in app.warning)
