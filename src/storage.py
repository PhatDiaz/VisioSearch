"""On-disk index, metadata and image-path validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_metadata(root: str | Path) -> list[dict]:
    path = Path(root) / 'metadata.jsonl'
    if not path.exists():
        raise FileNotFoundError(f'Gallery metadata not found: {path}')
    with path.open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError('Gallery metadata is empty.')
    ids = [x['image_id'] for x in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate image IDs in gallery metadata.')
    return rows


def image_file(root: str | Path, row: dict) -> Path:
    root = Path(root).resolve()
    relative = Path(row['path'])
    absolute = (root / relative).resolve()
    if not absolute.is_relative_to(root):
        raise ValueError('Image path escapes the gallery directory.')
    return absolute


def read_embeddings(root: str | Path, name: str, num_images: int | None = None) -> np.ndarray:
    path = Path(root) / f'embeddings_{name.lower()}.npy'
    data = np.load(path, allow_pickle=False, mmap_mode='r')
    if data.ndim != 2 or (num_images is not None and len(data) != num_images):
        raise ValueError(f'Unexpected embedding array shape in {path}: {data.shape}')
    return data


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
