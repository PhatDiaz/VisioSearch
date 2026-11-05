"""Frozen fusion, provenance, candidate mapping and cache behavior without model downloads."""
import ast
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.pipeline_artifacts import (ArtifactError, RuntimeArtifacts, file_sha256, load_runtime_artifacts,
                                    validate_adapter_config, validate_vectors)
from src.two_stage_retriever import TwoStageRetriever, fuse_scores, top20_candidates, zscore

ROOT = Path(__file__).resolve().parents[1]


def test_fusion_uses_float64_population_zscores_and_raw_logits():
    cosine = np.array([.2, .21, .6, .18], dtype=np.float32)
    logits = np.array([10, -8, .4, 3], dtype=np.float32)
    c, b = cosine.astype(np.float64), logits.astype(np.float64)
    expected = .25 * ((c - np.mean(c)) / (np.std(c, ddof=0) + 1e-8))
    expected += .75 * ((b - np.mean(b)) / (np.std(b, ddof=0) + 1e-8))
    np.testing.assert_array_equal(fuse_scores(cosine, logits), expected)
    assert fuse_scores(cosine, logits).dtype == np.float64
    assert np.argmax(expected) == 0  # BLIP can reorder the strongest cosine match.


@pytest.mark.skipif(not (ROOT / 'research_artifacts/Topic11_BLIP_Final_8K_Evaluation.ipynb').is_file(),
                    reason='Original research notebook is not included in Git')
def test_fusion_and_stable_order_match_the_actual_notebook_functions():
    notebook = json.loads((ROOT / 'research_artifacts/Topic11_BLIP_Final_8K_Evaluation.ipynb').read_text(encoding='utf-8'))
    source = ''.join(notebook['cells'][9]['source'])
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                 and node.name in ('zscore', 'rerank_one')]
    namespace = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<notebook fusion functions>', 'exec'), namespace)
    rng = np.random.default_rng(42)
    for cosine, logits in [(rng.normal(size=20).astype('float32'), rng.normal(size=20).astype('float32')),
                           (np.ones(20), np.ones(20)),
                           (np.linspace(0, 1e-10, 20), np.zeros(20))]:
        fusion = .25 * namespace['zscore'](cosine) + .75 * namespace['zscore'](logits)
        np.testing.assert_array_equal(fuse_scores(cosine, logits), fusion)
        expected = np.argsort(-fusion, kind='stable')
        for truth in range(20):
            rank, top1 = namespace['rerank_one'](cosine, logits, np.arange(20), truth, truth + 1, 20, .75)
            assert top1 == int(expected[0])
            assert rank == int(np.flatnonzero(expected == truth)[0]) + 1


def test_constant_scores_and_stable_candidate_ties():
    np.testing.assert_array_equal(zscore(np.ones(20)), np.zeros(20))
    vectors = np.tile(np.array([1, 0], dtype=np.float32), (30, 1))
    indices, cosine = top20_candidates(np.array([1, 0]), vectors)
    assert indices.tolist() == list(range(20))
    np.testing.assert_array_equal(cosine, np.ones(20))
    np.testing.assert_array_equal(np.argsort(-fuse_scores(cosine, np.ones(20)), kind='stable'), np.arange(20))


@pytest.mark.parametrize('bad', [np.array([np.nan]), np.array([np.inf]), np.array([]), np.ones((2, 2))])
def test_nonfinite_or_invalid_scores_fail(bad):
    with pytest.raises(ValueError):
        zscore(bad)


def test_bad_score_and_embedding_shapes_fail():
    with pytest.raises(ValueError):
        fuse_scores(np.ones(20), np.ones(19))
    with pytest.raises(ValueError):
        top20_candidates(np.ones(3), np.ones((8000, 2)))
    with pytest.raises(ArtifactError):
        validate_vectors(np.ones((7999, 768), dtype='float32'))
    with pytest.raises(ArtifactError):
        validate_vectors(np.ones((8000, 768), dtype='float32'))
    vectors = np.zeros((8000, 768), dtype='float32')
    vectors[:, 0] = 1
    validate_vectors(vectors)
    vectors[0, 0] = np.nan
    with pytest.raises(ArtifactError):
        validate_vectors(vectors)


@pytest.fixture
def fake_runtime(tmp_path):
    vectors = np.zeros((8000, 768), dtype=np.float32)
    vectors[:, 1] = 1
    values = np.linspace(.01, .99, 8000, dtype=np.float32)
    vectors[:, 0], vectors[:, 1] = values, np.sqrt(1 - values ** 2)
    return RuntimeArtifacts(tmp_path / 'adapter', tmp_path, vectors,
                            tuple(f'image:{i}' for i in range(8000)),
                            tuple(tmp_path / f'{i}.jpg' for i in range(8000)), {})


def test_top20_paths_fused_mapping_topk_and_models_loaded_once(fake_runtime):
    query = np.zeros(768, dtype=np.float32)
    query[0] = 1
    encoder = SimpleNamespace(device='cpu', encode=Mock(return_value=query))
    reranker = SimpleNamespace(device='cpu', score=Mock(return_value=np.arange(20, dtype='float32')))
    encoder_loader, blip_loader = Mock(return_value=encoder), Mock(return_value=reranker)
    retriever = TwoStageRetriever(fake_runtime, encoder_factory=encoder_loader, reranker_factory=blip_loader)
    encoder_loader.assert_not_called()
    blip_loader.assert_not_called()
    result = retriever.search('a new query', 5)
    candidates, cosine = top20_candidates(query, fake_runtime.vectors)
    assert len(candidates) == 20
    assert reranker.score.call_args.args[1] == [fake_runtime.image_paths[i] for i in candidates]
    expected = np.argsort(-fuse_scores(cosine, np.arange(20)), kind='stable')
    assert [hit.image_index for hit in result.hits] == candidates[expected[:5]].tolist()
    for hit in result.hits:
        original_position = int(np.flatnonzero(candidates == hit.image_index)[0])
        assert hit.blip_logit == original_position
        assert hit.cosine_similarity == cosine[original_position]
    ten = retriever.search('a new query', 10)
    assert ten.cache_hit and len(ten.hits) == 10
    assert ten.hits[:5] == result.hits
    assert ten.timings.siglip_seconds == ten.timings.blip_seconds == 0
    retriever.search('another query', 20)
    assert encoder.encode.call_count == reranker.score.call_count == 2
    encoder_loader.assert_called_once()
    blip_loader.assert_called_once()
    for bad in ('', '  '):
        with pytest.raises(ValueError):
            retriever.search(bad)
    with pytest.raises(ValueError):
        retriever.search('query', 21)


def test_model_failure_does_not_fall_back_or_cache_results(fake_runtime):
    retriever = TwoStageRetriever(fake_runtime, encoder_factory=Mock(side_effect=ArtifactError('missing LoRA')))
    with pytest.raises(ArtifactError, match='missing LoRA'):
        retriever.search('test')
    assert not retriever._results


def test_cpu_device_fallback(monkeypatch):
    import torch
    from src.modeling import choose_device
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    assert choose_device('auto') == 'cpu'


def test_missing_runtime_manifest_is_actionable(tmp_path):
    with pytest.raises(ArtifactError, match='prepare_two_stage'):
        load_runtime_artifacts(tmp_path, tmp_path / 'absent.json')


def test_mixed_adapter_config_is_rejected():
    config = {'base_model_name_or_path': 'google/siglip-base-patch16-224', 'peft_type': 'LORA',
              'r': 8, 'lora_alpha': 16, 'target_modules': ['q_proj', 'v_proj']}
    validate_adapter_config(config)
    for bad in ({'r': 16}, {'lora_alpha': 32}, {'base_model_name_or_path': 'wrong/model'}):
        with pytest.raises(ArtifactError):
            validate_adapter_config(config | bad)
