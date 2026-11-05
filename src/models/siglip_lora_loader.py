"""Load the validation-selected SigLIP adapter and reproduce notebook text encoding."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.modeling import load_model, _features_to_tensor, choose_device
from src.pipeline_artifacts import ArtifactError, validate_adapter_config


def inference_autocast(device: str):
    import torch
    return torch.autocast(device_type='cuda', dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()


def runtime_device(requested: str) -> str:
    import torch
    device = choose_device(requested)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ArtifactError('CUDA was requested but is unavailable. Use auto/cpu or install a CUDA-enabled PyTorch build.')
    return device


@dataclass
class SiglipLoRAEncoder:
    model: Any
    processor: Any
    device: str

    def encode(self, text: str) -> np.ndarray:
        import torch
        tokens = self.processor(text=[text], padding='max_length', truncation=True, return_tensors='pt').to(self.device)
        kwargs = {'input_ids': tokens['input_ids']}
        if 'attention_mask' in tokens:
            kwargs['attention_mask'] = tokens['attention_mask']
        with torch.inference_mode(), inference_autocast(self.device):
            features = _features_to_tensor(self.model.get_base_model().get_text_features(**kwargs))
        array = features.float().cpu().numpy().astype(np.float32)
        # Notebook normalization, once; do not normalize again during cosine ranking.
        array = array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)
        if array.shape != (1, 768) or not np.isfinite(array).all() or np.linalg.norm(array) < .99:
            raise ArtifactError('SigLIP LoRA returned an invalid text embedding.')
        return array[0]


def load_siglip_lora(adapter_dir: str | Path, device: str = 'auto', *, local_files_only: bool = True) -> SiglipLoRAEncoder:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor
    adapter_dir = Path(adapter_dir)
    try:
        validate_adapter_config(json.loads((adapter_dir / 'adapter_config.json').read_text(encoding='utf-8')))
        if not (adapter_dir / 'adapter_model.safetensors').is_file():
            raise OSError('Missing adapter weights')
        base, _, runtime = load_model('SigLIP', runtime_device(device), local_files_only=local_files_only)
        processor = AutoProcessor.from_pretrained(str(adapter_dir), local_files_only=True)
        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False).to(runtime).eval()
    except (OSError, ValueError) as exc:
        raise ArtifactError('Cannot load SigLIP LoRA. Run python scripts/prepare_two_stage.py --download-models after installing requirements.txt.') from exc
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = model.peft_config['default']
    if config.r != 8 or config.lora_alpha != 16:
        raise ArtifactError('Loaded LoRA configuration does not match batch16_r8_e5.')
    return SiglipLoRAEncoder(model, processor, runtime)
