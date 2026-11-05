"""Prepare verified two-stage artifacts once, without replacing existing research data."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))

from src.pipeline_artifacts import (ArtifactError, CHECKPOINT, ADAPTER_FILES, file_sha256,
                                    load_runtime_artifacts, validate_adapter_config,
                                    validate_locked_config, validate_vectors)

def write_new_or_identical(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ArtifactError(f'Refusing to overwrite a different existing artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects another preparation process's output.
    with path.open('xb') as output:
        output.write(payload)


def prepare(root: Path = ROOT, artifact_dir: Path | None = None) -> Path:
    root = root.resolve()
    source = artifact_dir or root / 'research_artifacts'
    ablation = source / 'Topic11_Ablation_Final_Outputs.zip'
    final = source / 'Topic11_BLIP_Final_8K_Results.zip'
    notebook = source / 'Topic11_BLIP_Final_8K_Evaluation.ipynb'
    for path in (ablation, final, notebook):
        if not path.is_file():
            raise ArtifactError(f'Missing research artifact: {path}')
    index_dir = root / 'data' / 'flickr8k'
    with zipfile.ZipFile(ablation) as saved, zipfile.ZipFile(final) as reference:
        lock = json.loads(reference.read('locked_final_config.json'))
        audit = json.loads(reference.read('final_config_and_audit.json'))
        validate_locked_config(lock)
        validate_locked_config(audit['locked_selection'])
        if json.loads(saved.read('locked_selection.json'))['name'] != CHECKPOINT:
            raise ArtifactError('The ablation archive selected a different LoRA checkpoint.')
        prefix = 'models/siglip_lora_ablation_best/'
        adapter = {name: saved.read(prefix + name) for name in ADAPTER_FILES}
        validate_adapter_config(json.loads(adapter['adapter_config.json']))
        vectors = saved.read('data/flickr8k/embeddings_siglip_ablation_best.npy')
        validate_vectors(np.load(BytesIO(vectors), allow_pickle=False))
        fingerprints = {
            'metadata_sha256': file_sha256(index_dir / 'metadata.jsonl'),
            'embeddings_sha256': hashlib.sha256(vectors).hexdigest(),
            'adapter_sha256': hashlib.sha256(adapter['adapter_model.safetensors']).hexdigest(),
        }
        for key, value in fingerprints.items():
            if value != audit[key]:
                raise ArtifactError(f'{key} does not match the final evaluation audit. No artifacts were written.')
    # All research inputs are checked before writing anything into runtime directories.
    outputs = {f'models/siglip_lora_{CHECKPOINT}/{name}': payload for name, payload in adapter.items()}
    vector_name = f'data/flickr8k/embeddings_siglip_lora_{CHECKPOINT}.npy'
    outputs[vector_name] = vectors
    for name, payload in outputs.items():
        destination = root / name
        if destination.exists() and destination.read_bytes() != payload:
            raise ArtifactError(f'Refusing to replace existing artifact: {destination}')
    for name, payload in outputs.items():
        write_new_or_identical(root / name, payload)
    manifest = {
        'schema_version': 1, 'locked_config': lock,
        'adapter_dir': f'models/siglip_lora_{CHECKPOINT}', 'image_vectors': vector_name,
        'metadata_sha256': fingerprints['metadata_sha256'],
        'runtime_sha256': {name: hashlib.sha256(payload).hexdigest() for name, payload in outputs.items()},
        'source_sha256': {path.name: file_sha256(path) for path in (ablation, final, notebook)},
    }
    manifest_path = root / 'models' / 'two_stage' / 'runtime_manifest.json'
    write_new_or_identical(manifest_path, (json.dumps(manifest, indent=2) + '\n').encode('utf-8'))
    load_runtime_artifacts(index_dir, manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, default=ROOT / 'research_artifacts')
    parser.add_argument('--download-models', action='store_true')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    manifest = prepare(artifact_dir=args.artifact_dir)
    print(f'Prepared and verified runtime manifest: {manifest}', flush=True)
    if args.download_models:
        from src.models.siglip_lora_loader import load_siglip_lora
        from src.models.blip_reranker import BlipReranker
        artifacts = load_runtime_artifacts(ROOT / 'data' / 'flickr8k', manifest)
        encoder = load_siglip_lora(artifacts.adapter_dir, args.device, local_files_only=False)
        print(f'SigLIP LoRA ready on {encoder.device}.', flush=True)
        del encoder
        import gc
        gc.collect()
        reranker = BlipReranker(device=args.device, local_files_only=False)
        print(f'BLIP ITM ready on {reranker.device}.', flush=True)


if __name__ == '__main__':
    main()
