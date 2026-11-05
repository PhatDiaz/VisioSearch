"""Notebook-equivalent BLIP positive-class logits with bounded FP16 vision cache."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import RLock

import numpy as np
from PIL import Image

from src.models.siglip_lora_loader import inference_autocast, runtime_device
from src.pipeline_artifacts import ArtifactError, BLIP_MODEL


class BlipReranker:
    def __init__(self, device: str = 'auto', *, local_files_only: bool = True,
                 pair_batch_size: int = 12, vision_batch_size: int = 8, cache_images: int = 64):
        from huggingface_hub import snapshot_download
        from transformers import BlipForImageTextRetrieval, BlipProcessor
        self.device = runtime_device(device)
        if min(pair_batch_size, vision_batch_size, cache_images) < 1:
            raise ValueError('BLIP batch/cache sizes must be positive.')
        try:
            snapshot = snapshot_download(BLIP_MODEL, local_files_only=True)
            self.processor = BlipProcessor.from_pretrained(snapshot, local_files_only=True)
            self.model = BlipForImageTextRetrieval.from_pretrained(snapshot, local_files_only=True)
        except OSError as exc:
            if local_files_only:
                raise ArtifactError('BLIP ITM is not cached. Run python scripts/prepare_two_stage.py --download-models.') from exc
            self.processor = BlipProcessor.from_pretrained(BLIP_MODEL)
            self.model = BlipForImageTextRetrieval.from_pretrained(BLIP_MODEL)
        self.model = self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.pair_batch_size, self.vision_batch_size = pair_batch_size, vision_batch_size
        self.cache_images = cache_images
        self._vision: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = RLock()

    def score(self, text: str, paths: list[Path]) -> np.ndarray:
        import torch
        with self._lock, torch.inference_mode():
            keys = [str(path.resolve()) for path in paths]
            missing = list(dict.fromkeys(key for key in keys if key not in self._vision))
            # Retain this query's tokens separately, even if its batch exceeds cache capacity.
            tokens_by_key = {key: self._vision[key] for key in keys if key in self._vision}
            for start in range(0, len(missing), self.vision_batch_size):
                block = missing[start:start + self.vision_batch_size]
                images = []
                for key in block:
                    with Image.open(key) as image:
                        images.append(image.convert('RGB'))
                inputs = self.processor(images=images, return_tensors='pt').to(self.device)
                with inference_autocast(self.device):
                    tokens = self.model.vision_model(pixel_values=inputs['pixel_values']).last_hidden_state
                # The evaluation notebook persisted vision tokens in FP16. Preserve that round trip on CPU too.
                arrays = tokens.float().cpu().numpy().astype(np.float16)
                for key, array in zip(block, arrays):
                    tokens_by_key[key] = array.copy()
                    self._vision[key] = tokens_by_key[key]
            for key in keys:
                if key in self._vision:
                    self._vision.move_to_end(key)
            while len(self._vision) > self.cache_images:
                self._vision.popitem(last=False)
            outputs = []
            dtype = torch.float16 if self.device.startswith('cuda') else torch.float32
            for start in range(0, len(keys), self.pair_batch_size):
                block = keys[start:start + self.pair_batch_size]
                image_embeds = torch.from_numpy(np.stack([tokens_by_key[key] for key in block])).to(self.device, dtype=dtype)
                token_batch = self.processor(text=[text] * len(block), padding=True, truncation=True,
                                             max_length=64, return_tensors='pt').to(self.device)
                image_mask = torch.ones(image_embeds.shape[:2], device=self.device, dtype=torch.long)
                with inference_autocast(self.device):
                    encoded = self.model.text_encoder(input_ids=token_batch['input_ids'],
                                attention_mask=token_batch.get('attention_mask'), encoder_hidden_states=image_embeds,
                                encoder_attention_mask=image_mask, return_dict=True)
                    logits = self.model.itm_head(encoded.last_hidden_state[:, 0, :])
                # Positive-class RAW logit, not softmax or a contrastive matching score.
                outputs.append(logits[:, 1].float().cpu().numpy())
            scores = np.concatenate(outputs).astype(np.float32)
            if not np.isfinite(scores).all():
                raise ArtifactError('BLIP returned non-finite image-text matching logits.')
            return scores
