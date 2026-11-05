"""BLIP positive-logit semantics and optional real-adapter/model integration checks."""
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import torch

from src.models.blip_reranker import BlipReranker
from src.models.siglip_lora_loader import inference_autocast, runtime_device
from src.pipeline_artifacts import ArtifactError, MANIFEST_PATH

ROOT = Path(__file__).resolve().parents[1]
BLIP_CACHE = ROOT / '.cache/huggingface/hub/models--Salesforce--blip-itm-base-coco/snapshots'
SIGLIP_CACHE = ROOT / '.cache/huggingface/hub/models--google--siglip-base-patch16-224/snapshots'
HAS_MODELS = MANIFEST_PATH.is_file() and (any(BLIP_CACHE.glob('*/pytorch_model.bin'))
                                        or any(BLIP_CACHE.glob('*/model.safetensors'))) and (
    any(SIGLIP_CACHE.glob('*/pytorch_model.bin')) or any(SIGLIP_CACHE.glob('*/model.safetensors')))


def test_blip_returns_raw_class_one_logits_and_reuses_vision_tokens(tmp_path):
    paths = []
    for i, shade in enumerate([32, 128, 224]):
        path = tmp_path / f'{i}.jpg'
        Image.new('RGB', (4, 4), (shade, 0, 0)).save(path)
        paths.append(path)

    class Batch(dict):
        def to(self, device):
            return self

    calls = []
    def processor(**kwargs):
        calls.append(kwargs)
        if 'images' in kwargs:
            values = [image.getpixel((0, 0))[0] / 255 for image in kwargs['images']]
            return Batch(pixel_values=torch.tensor(values).reshape(-1, 1, 1))
        assert kwargs['padding'] is True and kwargs['max_length'] == 64 and kwargs['truncation'] is True
        return Batch(input_ids=torch.ones(len(kwargs['text']), 2, dtype=torch.long),
                     attention_mask=torch.ones(len(kwargs['text']), 2, dtype=torch.long))

    def vision(pixel_values):
        assert not torch.is_grad_enabled()
        return SimpleNamespace(last_hidden_state=pixel_values.expand(-1, 2, 4))

    def text_encoder(**kwargs):
        assert not torch.is_grad_enabled()
        assert torch.all(kwargs['encoder_attention_mask'] == 1)
        return SimpleNamespace(last_hidden_state=kwargs['encoder_hidden_states'])

    def itm_head(features):
        return torch.stack([-features[:, 0], 3 + features[:, 0] * 2], dim=1)

    reranker = object.__new__(BlipReranker)
    reranker.device, reranker.processor = 'cpu', processor
    reranker.model = SimpleNamespace(vision_model=Mock(side_effect=vision),
                                    text_encoder=Mock(side_effect=text_encoder), itm_head=itm_head)
    reranker.pair_batch_size, reranker.vision_batch_size, reranker.cache_images = 2, 8, 3
    reranker._vision, reranker._lock = OrderedDict(), RLock()
    actual = reranker.score('a query', paths)
    expected = 3 + np.array([Image.open(path).getpixel((0, 0))[0] / 255 for path in paths], dtype='float16').astype('float32') * 2
    np.testing.assert_array_equal(actual, expected)
    assert np.all(actual > 1)  # Raw positive logits, never softmax probabilities.
    assert all(value.dtype == np.float16 for value in reranker._vision.values())
    reranker.score('another query', paths)
    assert reranker.model.vision_model.call_count == 1
    assert reranker.model.text_encoder.call_count == 4  # Two batches for each query.
    reranker.cache_images = 1
    reranker.score('third query', paths)
    assert len(reranker._vision) == 1


def test_explicit_cuda_without_cuda_is_actionable(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ArtifactError, match='CUDA was requested'):
        runtime_device('cuda')
    assert runtime_device('cpu') == 'cpu'


@pytest.mark.skipif(not HAS_MODELS, reason='Requires cached models and prepared LoRA artifacts')
def test_real_lora_weights_are_loaded_frozen_and_active():
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from src.search_service import prepare_search_index
    retriever = prepare_search_index(str(ROOT / 'data/flickr8k'), 8000)
    encoder = retriever._load_encoder()
    expected = load_file(str(retriever.artifacts.adapter_dir / 'adapter_model.safetensors'))
    actual = get_peft_model_state_dict(encoder.model)
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key].cpu(), expected[key]) for key in expected)
    assert not encoder.model.training
    assert not any(parameter.requires_grad for parameter in encoder.model.parameters())
    tokens = encoder.processor(text=['A DOG RUNNING ON THE BEACH'], padding='max_length', truncation=True, return_tensors='pt')
    assert tokens['input_ids'].shape == (1, 64)
    tuned = encoder.encode('A dog running on the beach')
    with encoder.model.disable_adapter():
        original = encoder.encode('A dog running on the beach')
    assert not np.allclose(tuned, original, atol=1e-5)
    assert np.allclose(np.linalg.norm(tuned), 1)
    assert retriever is prepare_search_index(str(ROOT / 'data/flickr8k'), 8000)


@pytest.mark.skipif(not HAS_MODELS, reason='Requires cached models and prepared LoRA artifacts')
def test_real_cached_blip_path_matches_standard_itm_forward():
    from src.search_service import prepare_search_index
    retriever = prepare_search_index(str(ROOT / 'data/flickr8k'), 8000)
    reranker = retriever._load_reranker()
    path = retriever.artifacts.image_paths[4278]
    query = 'A dog running on the beach'
    actual = float(reranker.score(query, [path])[0])
    with Image.open(path) as image:
        sample = reranker.processor(images=image.convert('RGB'), text=query, return_tensors='pt').to(reranker.device)
    with torch.inference_mode(), inference_autocast(reranker.device):
        expected = reranker.model(**sample, use_itm_head=True).itm_score[0, 1].float().item()
    assert abs(actual - expected) < .25  # Same tolerance as the notebook's cached-vision sanity check.
    assert not any(parameter.requires_grad for parameter in reranker.model.parameters())
