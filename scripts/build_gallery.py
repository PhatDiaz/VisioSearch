"""Prepare the full 8,000-image Flickr8k gallery and its pretrained embeddings.

Run from the repository root:
    python scripts/build_gallery.py --output-dir data/flickr8k

First pass downloads the dataset and models. Subsequent passes reuse the saved
JPEGs and skip completed embedding arrays (unless --overwrite is set).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.modeling import MODELS, image_embeddings, load_model
from src.storage import image_file, read_metadata, write_jsonl

SPLITS = ('train', 'dev', 'test')
EXPECTED = {'train': 6000, 'dev': 1000, 'test': 1000}


# The dataset publisher exposes an auto-converted Parquet revision that contains
# the SAME original 6k / 1k / 1k splits, embedded images and five captions.
# Using a handful of Parquet shards avoids the thousands of HTTP requests made
# by load_dataset("intro/flickr8k") against the per-image imagefolder source.
DATASET_REPO = "intro/flickr8k"
PARQUET_REVISION = "refs/convert/parquet"


def load_parquet_split(split: str):
    """Load a dataset split from cached, prepackaged Parquet shards.

    Hugging Face Hub caches the downloaded shards across reruns within a
    persisted environment. The repo and branch are fixed to keep provenance.
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    if split not in SPLITS:
        raise ValueError(f'Unknown Flickr8k split: {split}')
    files = HfApi().list_repo_files(
        DATASET_REPO, repo_type='dataset', revision=PARQUET_REVISION
    )
    prefix = f'default/{split}/'
    shards = sorted(path for path in files if path.startswith(prefix) and path.endswith('.parquet'))
    if not shards:
        raise RuntimeError(
            f'No Parquet shards for {split} at {DATASET_REPO}@{PARQUET_REVISION}. '
            'Verify the converted Parquet branch exists before retrying.'
        )
    print(f'{split}: downloading/caching {len(shards)} Parquet shard(s) instead of {EXPECTED[split]} individual JPEGs.', flush=True)
    cached = []
    for filename in shards:
        # A transient 429/503 can still happen even with few requests.
        for attempt in range(3):
            try:
                cached.append(str(hf_hub_download(
                    DATASET_REPO, filename, repo_type='dataset', revision=PARQUET_REVISION
                )))
                break
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f'Could not download shard {filename}. '
                        'Hugging Face may be rate-limiting downloads. '
                        'Rerun this cell later to reuse cached shards. '
                        f'Original error: {type(exc).__name__}: {exc}'
                    ) from exc
                time.sleep(3 * (attempt + 1))
    return load_dataset('parquet', data_files={split: cached}, split=split)


def decode_image(value):
    """Both HF Image objects and raw Parquet image dictionaries are supported."""
    if isinstance(value, Image.Image):
        return value.convert('RGB')
    if isinstance(value, dict):
        if value.get('bytes') is not None:
            with Image.open(io.BytesIO(value['bytes'])) as im:
                return im.convert('RGB')
        if value.get('path'):
            with Image.open(value['path']) as im:
                return im.convert('RGB')
    raise TypeError(f'Unsupported image value from Parquet: {type(value).__name__}')


def prepare_images(output: Path, max_images: int | None, force: bool) -> list[dict]:
    """Prepare Flickr8k JPEGs/captions from batched Parquet files (not 8k HTTP requests)."""
    existing = output / 'metadata.jsonl'
    if existing.is_file() and not force:
        rows = read_metadata(output)
        print(f'Found {len(rows):,} existing metadata rows in {existing}')
        missing = [row for row in rows if not image_file(output, row).is_file()]
        if not missing:
            return rows
        print(f'{len(missing)} JPEGs missing: rebuilding metadata and images.')

    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir(parents=True, exist_ok=True)
    rows = []
    for split in SPLITS:
        if max_images is not None and len(rows) >= max_images:
            break
        print(f'Loading Flickr8k {split} as Parquet ...', flush=True)
        ds = load_parquet_split(split)
        if len(ds) != EXPECTED[split]:
            raise ValueError(f'Unexpected Flickr8k {split} size: {len(ds)} (expected {EXPECTED[split]})')
        required = {'image', *(f'caption_{j}' for j in range(5))}
        if not required.issubset(set(ds.column_names)):
            raise ValueError(f'Missing expected columns. Got {ds.column_names}')
        for offset in tqdm(range(len(ds)), desc=f'Preparing {split}'):
            if max_images is not None and len(rows) >= max_images:
                break
            entry = ds[offset]
            file_name = f'{split}_{offset:05d}.jpg'
            relative = Path('images') / file_name
            target = output / relative
            if not target.is_file() or force:
                img = ImageOps.exif_transpose(decode_image(entry['image']))
                img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                img.save(target, 'JPEG', quality=82, optimize=True)
            rows.append({
                'image_id': f'{split}:{offset:05d}',
                'split': split,
                'path': relative.as_posix(),
                'captions': [str(entry[f'caption_{j}']).strip() for j in range(5)],
            })
        del ds
        print(f'Completed {split}: {sum(r["split"] == split for r in rows):,} images', flush=True)
    if max_images is None and len(rows) != 8000:
        raise ValueError(f'Expected 8,000 images, got {len(rows)}')
    write_jsonl(existing, rows)
    print(f'Prepared {len(rows):,} gallery images. Metadata: {existing}')
    return rows


def build_embeddings(output: Path, rows: list[dict], names: list[str], batch_size: int, device: str, overwrite: bool):
    for name in names:
        filename = output / f'embeddings_{name.lower()}.npy'
        if filename.exists() and not overwrite:
            previous = np.load(filename, allow_pickle=False, mmap_mode='r')
            if previous.ndim == 2 and len(previous) == len(rows):
                print(f'Skipping {name}; existing matrix {previous.shape} found.')
                continue
        print(f'Embedding {len(rows):,} images with {name} ...')
        model, processor, actual_device = load_model(name, device)
        encoded = []
        # Images are decoded in small batches, avoiding a list of 8000 PIL objects.
        for start in tqdm(range(0, len(rows), batch_size), desc=f'{name} batches'):
            segment = rows[start:start + batch_size]
            with_images = []
            for row in segment:
                with Image.open(image_file(output, row)) as im:
                    with_images.append(im.convert('RGB'))
            encoded.append(image_embeddings(with_images, model, processor, actual_device, batch_size))
        matrix = np.concatenate(encoded, axis=0).astype(np.float32)
        if matrix.shape[0] != len(rows):
            raise AssertionError('Index image count mismatch')
        np.save(filename, matrix)
        print(f'{name} saved: {matrix.shape} → {filename}')
        del model, processor, encoded, matrix
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ImportError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='data/flickr8k')
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=['SigLIP', 'CLIP'])
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-images', type=int, help='Development smoke test only: creates a partial gallery!')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--images-only', action='store_true')
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error('--max-images must be >= 1')
    if args.batch_size < 1:
        parser.error('--batch-size must be >= 1')
    output = Path(args.output_dir)
    rows = prepare_images(output, args.max_images, args.overwrite)
    if not args.images_only:
        build_embeddings(output, rows, args.models, args.batch_size, args.device, args.overwrite)
    manifest = {
        'dataset': 'intro/flickr8k',
        'download_method': f'parquet/{PARQUET_REVISION}',
        'source': 'Flickr8k',
        'models': {name: MODELS[name] for name in args.models},
        'num_images': len(rows),
        'split_counts': {name: sum(x['split'] == name for x in rows) for name in SPLITS},
        'retrieval_type': 'zero-shot, pretrained, cosine similarity',
        'partial_gallery': len(rows) != 8000,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Done. Index root:', output.resolve())

if __name__ == '__main__':
    main()
