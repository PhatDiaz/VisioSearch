"""Runtime provenance, non-overwriting extraction, and saved-result audit checks."""
from io import StringIO
import csv
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from PIL import Image

from src.pipeline_artifacts import (ADAPTER_FILES, ArtifactError, MANIFEST_PATH, file_sha256,
                                    load_runtime_artifacts)
from scripts.prepare_two_stage import write_new_or_identical

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture_manifest(tmp_path):
    adapter = tmp_path / 'models/adapter'
    adapter.mkdir(parents=True)
    for name in ADAPTER_FILES:
        (adapter / name).write_bytes(b'unit-test artifact')
    (adapter / 'adapter_config.json').write_text(json.dumps({
        'base_model_name_or_path': 'google/siglip-base-patch16-224', 'peft_type': 'LORA',
        'r': 8, 'lora_alpha': 16, 'target_modules': ['q_proj', 'v_proj']}))
    index = tmp_path / 'data/flickr8k'
    index.mkdir(parents=True)
    Image.new('RGB', (4, 3), 'orange').save(index / 'unit.jpg')
    rows = [{'image_id': f'image:{i}', 'path': 'unit.jpg',
             'split': 'train' if i < 6000 else 'dev' if i < 7000 else 'test'} for i in range(8000)]
    (index / 'metadata.jsonl').write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    vector_path = index / 'unit_vectors.npy'
    vectors = np.zeros((8000, 768), dtype='float32')
    vectors[:, 0] = 1
    np.save(vector_path, vectors)
    paths = [*(adapter / name for name in ADAPTER_FILES), vector_path]
    manifest = dict(schema_version=1, adapter_dir='models/adapter', image_vectors='data/flickr8k/unit_vectors.npy',
                    metadata_sha256=file_sha256(index / 'metadata.jsonl'),
                    runtime_sha256={p.relative_to(tmp_path).as_posix(): file_sha256(p) for p in paths},
                    locked_config={'checkpoint': 'batch16_r8_e5', 'base_model': 'google/siglip-base-patch16-224',
                                   'reranker': 'Salesforce/blip-itm-base-coco', 'candidate_count': 20,
                                   'top_k_candidates': 20, 'alpha_blip': .75})
    destination = tmp_path / 'models/two_stage/runtime_manifest.json'
    destination.parent.mkdir()
    destination.write_text(json.dumps(manifest))
    return index, destination, manifest


def test_verified_memorymap_and_metadata_order(fixture_manifest):
    index, path, _ = fixture_manifest
    runtime = load_runtime_artifacts(index, path)
    assert isinstance(runtime.vectors, np.memmap)
    assert not runtime.vectors.flags.writeable
    assert runtime.image_ids[0] == 'image:0' and runtime.image_ids[-1] == 'image:7999'
    assert runtime.image_paths[25] == index / 'unit.jpg'
    lines = (index / 'metadata.jsonl').read_text().splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    (index / 'metadata.jsonl').write_text('\n'.join(lines))
    with pytest.raises(ArtifactError, match='metadata order'):
        load_runtime_artifacts(index, path)


def test_mixed_weights_or_unverified_tokenizer_rejected(fixture_manifest):
    index, path, manifest = fixture_manifest
    config_name = 'models/adapter/tokenizer_config.json'
    del manifest['runtime_sha256'][config_name]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match='verified runtime manifest'):
        load_runtime_artifacts(index, path)
    manifest['runtime_sha256'][config_name] = 'wrong hash'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match='fingerprint mismatch'):
        load_runtime_artifacts(index, path)


def test_locked_weight_or_missing_image_rejected(fixture_manifest):
    index, path, manifest = fixture_manifest
    manifest['locked_config']['alpha_blip'] = .5
    path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match='locked'):
        load_runtime_artifacts(index, path)
    manifest['locked_config']['alpha_blip'] = .75
    path.write_text(json.dumps(manifest))
    (index / 'unit.jpg').unlink()
    with pytest.raises(ArtifactError, match='images are missing'):
        load_runtime_artifacts(index, path)


def test_cached_service_revalidates_changed_runtime_files(fixture_manifest, monkeypatch):
    from src.search_service import prepare_search_index
    index, path, _ = fixture_manifest
    monkeypatch.setenv('TOPIC11_PIPELINE', 'two_stage')
    monkeypatch.setenv('TOPIC11_RUNTIME_MANIFEST', str(path))
    original = prepare_search_index(str(index), 8000)
    assert original is prepare_search_index(str(index), 8000)
    (path.parents[1] / 'adapter/adapter_model.safetensors').write_bytes(b'changed checkpoint')
    with pytest.raises(ArtifactError, match='fingerprint mismatch'):
        prepare_search_index(str(index), 8000)


def test_metadata_cache_does_not_reuse_an_obsolete_row_mapping(fixture_manifest):
    from src.search_service import metadata_rows
    index, _, _ = fixture_manifest
    assert metadata_rows(str(index))[0]['image_id'] == 'image:0'
    path = index / 'metadata.jsonl'
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first['image_id'] = 'changed:image:0'
    lines[0] = json.dumps(first)
    path.write_text('\n'.join(lines))
    assert metadata_rows(str(index))[0]['image_id'] == 'changed:image:0'


def test_preparation_is_idempotent_and_never_overwrites(tmp_path):
    path = tmp_path / 'new' / 'artifact'
    write_new_or_identical(path, b'original')
    write_new_or_identical(path, b'original')
    with pytest.raises(ArtifactError, match='Refusing to overwrite'):
        write_new_or_identical(path, b'replacement')
    assert path.read_bytes() == b'original'


def test_malformed_manifest_is_actionable(tmp_path):
    path = tmp_path / 'bad.json'
    for payload in ([], {'schema_version': 1}, {'schema_version': 2}):
        path.write_text(json.dumps(payload))
        with pytest.raises(ArtifactError, match='manifest'):
            load_runtime_artifacts(tmp_path, path)


@pytest.mark.skipif(not (ROOT / 'research_artifacts/Topic11_BLIP_Final_8K_Results.zip').is_file(),
                    reason='Research archive is intentionally outside Git')
def test_saved_reference_metrics_and_ground_truth_mapping_are_consistent():
    from src.storage import read_metadata
    archive_path = ROOT / 'research_artifacts/Topic11_BLIP_Final_8K_Results.zip'
    with zipfile.ZipFile(archive_path) as archive:
        records = list(csv.DictReader(StringIO(archive.read('per_query_final_test.csv').decode('utf-8'))))
        summary = list(csv.DictReader(StringIO(archive.read('final_test_summary.csv').decode('utf-8'))))
    assert len(records) == 5000
    for key, column in [('siglip_rank', 0), ('reranked_rank', 1)]:
        for k in (1, 5, 10):
            measured = round(100 * sum(int(row[key]) <= k for row in records) / 5000, 2)
            assert measured == float(summary[column][f'R@{k}'])
    metadata = ROOT / 'data/flickr8k/metadata.jsonl'
    if metadata.is_file():
        rows = read_metadata(metadata.parent)
        for record in records:
            row = rows[int(record['ground_truth_global_index'])]
            assert row['image_id'] == record['ground_truth_image_id']
            assert row['captions'][int(record['caption_index'])] == record['query']


@pytest.mark.skipif(not MANIFEST_PATH.is_file() or not (ROOT / 'research_artifacts/Topic11_BLIP_Final_8K_Results.zip').is_file(),
                    reason='Prepared runtime artifacts or original audit archive absent')
def test_real_export_fingerprints_match_final_audit():
    with zipfile.ZipFile(ROOT / 'research_artifacts/Topic11_BLIP_Final_8K_Results.zip') as archive:
        audit = json.loads(archive.read('final_config_and_audit.json'))
    runtime = load_runtime_artifacts(ROOT / 'data/flickr8k')
    assert file_sha256(runtime.adapter_dir / 'adapter_model.safetensors') == audit['adapter_sha256']
    assert file_sha256(ROOT / runtime.manifest['image_vectors']) == audit['embeddings_sha256']
