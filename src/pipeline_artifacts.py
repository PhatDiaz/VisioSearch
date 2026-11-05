"""Validated, immutable runtime inputs for the frozen two-stage experiment."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from src.storage import read_metadata, image_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = 'google/siglip-base-patch16-224'
BLIP_MODEL = 'Salesforce/blip-itm-base-coco'
CHECKPOINT = 'batch16_r8_e5'
CANDIDATE_COUNT = 20
ALPHA_BLIP = 0.75
PIPELINE_NAME = 'SigLIP LoRA batch16_r8_e5 + BLIP ITM K20 alpha0.75'
MANIFEST_PATH = PROJECT_ROOT / 'models' / 'two_stage' / 'runtime_manifest.json'
ADAPTER_FILES = ('adapter_config.json', 'adapter_model.safetensors', 'preprocessor_config.json',
                 'tokenizer_config.json', 'special_tokens_map.json', 'spiece.model')


class ArtifactError(RuntimeError):
    """A required runtime input is missing or differs from the frozen experiment."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_locked_config(config: dict) -> None:
    expected = {'checkpoint': CHECKPOINT, 'base_model': BASE_MODEL, 'reranker': BLIP_MODEL,
                'candidate_count': CANDIDATE_COUNT, 'top_k_candidates': CANDIDATE_COUNT,
                'alpha_blip': ALPHA_BLIP}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ArtifactError('The runtime configuration differs from the locked Top-20 / alpha=0.75 experiment.')


def validate_adapter_config(config: dict) -> None:
    if (config.get('base_model_name_or_path') != BASE_MODEL or config.get('peft_type') != 'LORA'
            or config.get('r') != 8 or config.get('lora_alpha') != 16
            or set(config.get('target_modules', [])) != {'q_proj', 'v_proj'}):
        raise ArtifactError('Expected the batch16_r8_e5 SigLIP LoRA adapter (rank 8, alpha 16, q_proj/v_proj).')


def validate_vectors(vectors: np.ndarray) -> None:
    if vectors.shape != (8000, 768) or vectors.dtype != np.float32:
        raise ArtifactError(f'Expected float32 LoRA embeddings of shape (8000, 768), got {vectors.shape}/{vectors.dtype}.')
    if not np.isfinite(vectors).all():
        raise ArtifactError('Fine-tuned image embeddings contain non-finite values.')
    if np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1)) >= .005:
        raise ArtifactError('Fine-tuned image embeddings are not L2-normalized as required by the notebook.')


@dataclass(frozen=True)
class RuntimeArtifacts:
    adapter_dir: Path
    index_dir: Path
    vectors: np.ndarray
    image_ids: tuple[str, ...]
    image_paths: tuple[Path, ...]
    manifest: dict


def load_runtime_artifacts(index_dir: str | Path, manifest_path: str | Path = MANIFEST_PATH) -> RuntimeArtifacts:
    """Check hashes before associating any embedding row with gallery metadata."""
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ArtifactError('Two-stage artifacts are not prepared. Run: python scripts/prepare_two_stage.py') from exc
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
        raise ArtifactError('Unsupported runtime manifest. Prepare the two-stage artifacts again.')
    required = {'locked_config', 'runtime_sha256', 'metadata_sha256', 'adapter_dir', 'image_vectors'}
    if (not required.issubset(manifest) or not isinstance(manifest['runtime_sha256'], dict)
            or not isinstance(manifest['locked_config'], dict)
            or not all(isinstance(manifest[key], str) for key in ('metadata_sha256', 'adapter_dir', 'image_vectors'))):
        raise ArtifactError('Runtime manifest is incomplete or malformed. Run scripts/prepare_two_stage.py.')
    validate_locked_config(manifest.get('locked_config', {}))
    try:
        root = manifest_path.resolve().parents[2]
    except IndexError as exc:
        raise ArtifactError('Place the runtime manifest in <project>/models/two_stage/runtime_manifest.json.') from exc
    # Runtime paths are relative to the project and cannot escape it.
    resolved = {}
    for relative, expected_hash in manifest['runtime_sha256'].items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ArtifactError('Runtime manifest contains malformed file fingerprints.')
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ArtifactError(f'Missing runtime artifact: {relative}. Run scripts/prepare_two_stage.py.')
        if file_sha256(path) != expected_hash:
            raise ArtifactError(f'Runtime artifact fingerprint mismatch: {relative}. The checkpoint/index pair is unsafe.')
        resolved[relative] = path
    index_dir = Path(index_dir).resolve()
    metadata_path = index_dir / 'metadata.jsonl'
    if not metadata_path.is_file() or file_sha256(metadata_path) != manifest['metadata_sha256']:
        raise ArtifactError('Gallery metadata order does not match the fine-tuned embeddings. Restore the original Flickr8k metadata.')
    rows = read_metadata(index_dir)
    if len(rows) != 8000 or len({row['image_id'] for row in rows}) != 8000:
        raise ArtifactError('The two-stage pipeline requires the complete, unique 8,000-image gallery.')
    if Counter(row['split'] for row in rows) != {'train': 6000, 'dev': 1000, 'test': 1000}:
        raise ArtifactError('Gallery split counts differ from the frozen Flickr8k metadata.')
    if any(not image_file(index_dir, row).is_file() for row in rows):
        raise ArtifactError('Some gallery images are missing. Restore all original Flickr8k JPEGs.')
    try:
        adapter_dir = (root / manifest['adapter_dir']).resolve()
        vector_path = resolved[manifest['image_vectors']]
        config_path = adapter_dir / 'adapter_config.json'
        if (not adapter_dir.is_relative_to(root)
                or any(adapter_dir / name not in resolved.values() for name in ADAPTER_FILES)):
            raise ArtifactError('The selected adapter must be included in the verified runtime manifest.')
        validate_adapter_config(json.loads(config_path.read_text(encoding='utf-8')))
        vectors = np.load(vector_path, mmap_mode='r', allow_pickle=False)
    except (OSError, ValueError, KeyError) as exc:
        raise ArtifactError('Cannot load the verified LoRA checkpoint and image index.') from exc
    validate_vectors(vectors)
    return RuntimeArtifacts(adapter_dir, index_dir, vectors, tuple(row['image_id'] for row in rows),
                            tuple(image_file(index_dir, row) for row in rows), manifest)
