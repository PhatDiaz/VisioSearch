"""Loading and using the exact CLIP and SigLIP checkpoints from Topic 11."""
from __future__ import annotations

import numpy as np

from src.retrieval import unit_normalize

MODELS = {
    'SigLIP': 'google/siglip-base-patch16-224',
    'CLIP': 'openai/clip-vit-base-patch32',
}


def choose_device(requested: str = 'auto') -> str:
    import torch
    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _features_to_tensor(output):
    import torch
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        return output.pooler_output
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f'Unexpected feature output type: {type(output)!r}')


def load_model(name: str, device: str = 'auto', *, local_files_only: bool = False):
    """Return (model, processor, device), preferring a cached local checkpoint.

    A complete cached snapshot needs no HTTP requests, including checks for optional
    tokenizer files. Only an absent/incomplete cache falls back to downloading.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, CLIPModel, SiglipModel
    if name not in MODELS:
        raise ValueError(f'Unknown model {name}')
    identifier = MODELS[name]
    runtime = choose_device(device)
    model_class = CLIPModel if name == 'CLIP' else SiglipModel
    try:
        snapshot = snapshot_download(identifier, local_files_only=True)
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        model = model_class.from_pretrained(snapshot, local_files_only=True)
    except OSError:
        if local_files_only:
            raise
        processor = AutoProcessor.from_pretrained(identifier)
        model = model_class.from_pretrained(identifier)
    model = model.to(runtime).eval()
    return model, processor, runtime


def image_embeddings(images, model, processor, device: str, batch_size: int = 24) -> np.ndarray:
    import torch
    blocks = []
    for start in range(0, len(images), batch_size):
        batch = images[start: start + batch_size]
        inputs = processor(images=batch, return_tensors='pt').to(device)
        with torch.inference_mode():
            output = _features_to_tensor(model.get_image_features(pixel_values=inputs['pixel_values']))
        blocks.append(output.detach().float().cpu().numpy())
    return unit_normalize(np.concatenate(blocks, axis=0))


def text_embeddings(texts, model, processor, device: str, batch_size: int = 96) -> np.ndarray:
    import torch
    blocks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        # SigLIP's text encoder was trained with max-length padding.
        inputs = processor(text=batch, return_tensors='pt', padding='max_length', truncation=True).to(device)
        kwargs = {'input_ids': inputs['input_ids']}
        if 'attention_mask' in inputs:
            kwargs['attention_mask'] = inputs['attention_mask']
        with torch.inference_mode():
            output = _features_to_tensor(model.get_text_features(**kwargs))
        blocks.append(output.detach().float().cpu().numpy())
    return unit_normalize(np.concatenate(blocks, axis=0))
