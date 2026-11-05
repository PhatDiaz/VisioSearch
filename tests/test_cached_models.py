"""Opt-in real-model regression check, skipped when prepared checkpoints are absent."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache' / 'huggingface'


def _checkpoints_present():
    for repo in ['models--google--siglip-base-patch16-224', 'models--openai--clip-vit-base-patch32']:
        directory = CACHE / 'hub' / repo
        revision = directory / 'refs' / 'main'
        if not revision.is_file():
            return False
        snapshot = directory / 'snapshots' / revision.read_text().strip()
        if not any((snapshot / name).is_file() for name in ['model.safetensors', 'pytorch_model.bin']):
            return False
    return True


@pytest.mark.skipif(not _checkpoints_present(), reason='Requires both downloaded model checkpoints')
def test_cached_models_search_without_any_http_requests():
    environment = os.environ.copy()
    environment['HF_HOME'] = str(CACHE)
    environment.pop('HF_HUB_OFFLINE', None)
    environment.pop('TRANSFORMERS_OFFLINE', None)
    source = textwrap.dedent('''
        import numpy as np
        import requests
        from src.modeling import load_model, text_embeddings

        def forbidden_request(*args, **kwargs):
            raise AssertionError('Cached models must load without HTTP requests')

        requests.sessions.Session.request = forbidden_request
        for name, dimension in [('SigLIP', 768), ('CLIP', 512)]:
            model, processor, device = load_model(name, device='cpu')
            vector = text_embeddings(['A dog running on the beach'], model, processor, device)
            assert vector.shape == (1, dimension)
            assert np.isfinite(vector).all()
            assert np.allclose(np.linalg.norm(vector, axis=1), 1)
            del model, processor
    ''')
    completed = subprocess.run([sys.executable, '-c', source], cwd=ROOT, env=environment,
                               capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
