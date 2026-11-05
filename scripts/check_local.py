"""Validate a prepared local gallery; optionally test both real text encoders."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[1] / '.cache' / 'huggingface'))
from src.modeling import MODELS, load_model, text_embeddings
from src.retrieval import recall_at_k, search
from src.storage import image_file, read_embeddings, read_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-dir', default='data/flickr8k')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--check-models', action='store_true',
                        help='Download/load both checkpoints and run a real query')
    args = parser.parse_args()
    root = Path(args.index_dir)
    rows = read_metadata(root)
    counts = Counter(row['split'] for row in rows)
    if counts != {'train': 6000, 'dev': 1000, 'test': 1000}:
        raise ValueError(f'Unexpected split counts: {counts}')
    if any(len(row['captions']) != 5 for row in rows):
        raise ValueError('Each image must have five reference captions')
    for row in rows:
        with Image.open(image_file(root, row)) as image:
            image.verify()
    print(f'Gallery OK: {len(rows)} readable images, 40000 captions, {dict(counts)}', flush=True)

    for name in MODELS:
        vectors = read_embeddings(root, name, len(rows))
        if not np.isfinite(vectors).all():
            raise ValueError(f'{name}: non-finite embedding values')
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError(f'{name}: embeddings are not unit normalized')
        print(f'{name} embeddings OK: {vectors.shape}', flush=True)

    results = Path(args.results_dir)
    benchmarks = pd.read_csv(results / 'benchmark_results.csv')
    for _, benchmark in benchmarks.iterrows():
        table = pd.read_csv(results / f'per_query_{benchmark.model.lower()}_{benchmark.gallery}.csv')
        if len(table) != 5000 or int(benchmark.text_queries) != 5000:
            raise ValueError('Expected 5000 test queries for each protocol')
        expected_size = {'test_only': 1000, 'all_8000': 8000}[benchmark.gallery]
        if int(benchmark.gallery_images) != expected_size or benchmark.query_split != 'test':
            raise ValueError('Unexpected benchmark protocol')
        expected_queries = [caption for row in rows if row['split'] == 'test' for caption in row['captions']]
        expected_targets = [i for i, row in enumerate(rows) if row['split'] == 'test' for _ in row['captions']]
        if table['query'].tolist() != expected_queries or table.ground_truth_index.tolist() != expected_targets:
            raise ValueError('Saved queries or ground truth differ from the test split')
        for metric, value in recall_at_k(table.ground_truth_rank.to_numpy()).items():
            if not np.isclose(value, benchmark[metric]):
                raise ValueError(f'Benchmark does not match per-query ranks: {benchmark.model}, {metric}')
        print(f'{benchmark.model} / {benchmark.gallery}: saved metrics match 5000 per-query ranks', flush=True)

    if args.check_models:
        for name in MODELS:
            model, processor, device = load_model(name)
            query = text_embeddings(['A dog running on the beach'], model, processor, device)[0]
            hits = search(query, read_embeddings(root, name, len(rows)), top_k=4)
            if len(hits) != 4:
                raise ValueError(f'{name}: expected four search results')
            best = hits[0]
            print(f'{name} live query OK ({device}): {rows[best.image_index]["image_id"]}, '
                  f'cosine={best.cosine_similarity:.4f}', flush=True)
            del model, processor
            import gc
            gc.collect()
    print('Local checks passed.', flush=True)


if __name__ == '__main__':
    main()
