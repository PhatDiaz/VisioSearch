"""Reproducible Flickr8k evaluation on 1k-test and optional 8k-all galleries.

The caption queries always come from the held-out test split (5,000 captions).
The standard 1,000-image test gallery is reported separately from the 8k gallery.
No model training, model selection, or test-derived prompt tuning occurs here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.modeling import MODELS, load_model, text_embeddings
from src.retrieval import ground_truth_ranks, recall_at_k
from src.storage import image_file, read_embeddings, read_metadata


def evaluate_one(model_name, index, rows, queries, targets, outputs, root, device, batch_size):
    matrix = read_embeddings(root, model_name, len(rows))
    if len(matrix) != len(rows):
        raise ValueError('Gallery embedding count differs from metadata')
    model, processor, runtime = load_model(model_name, device)
    output_vectors = []
    for start in tqdm(range(0, len(queries), batch_size), desc=f'{model_name} text'):
        output_vectors.append(text_embeddings(queries[start:start+batch_size], model, processor, runtime, batch_size))
    query_vectors = np.concatenate(output_vectors)
    if query_vectors.shape[1] != matrix.shape[1]:
        raise ValueError('Text/image feature dimensions differ')
    del model, processor
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError:
        pass

    summaries = []
    for gallery_name, gallery_ids in index.items():
        gallery_vecs = np.asarray(matrix[gallery_ids], dtype=np.float32)
        index_lookup = {int(gid): j for j, gid in enumerate(gallery_ids)}
        local_truth = np.asarray([index_lookup[int(i)] for i in targets], dtype=np.int64)
        ranks = np.zeros(len(queries), dtype=np.int32)
        top1_global = np.zeros(len(queries), dtype=np.int64)
        for start in tqdm(range(0, len(queries), batch_size), desc=f'{model_name} / {gallery_name}'):
            stop = min(start+batch_size, len(queries))
            scores = query_vectors[start:stop] @ gallery_vecs.T
            ranks[start:stop] = ground_truth_ranks(scores, local_truth[start:stop])
            top1_global[start:stop] = gallery_ids[np.argmax(scores, axis=1)]
        metrics = recall_at_k(ranks)
        summaries.append({
            'model': model_name, 'gallery': gallery_name, 'gallery_images': int(len(gallery_ids)),
            'query_split': 'test', 'text_queries': int(len(queries)), **metrics,
        })
        table = pd.DataFrame({
            'query_id': np.arange(len(queries)),
            'query': queries,
            'ground_truth_id': [rows[int(i)]['image_id'] for i in targets],
            'ground_truth_index': targets,
            'ground_truth_rank': ranks,
            'top1_image_id': [rows[int(i)]['image_id'] for i in top1_global],
            'top1_index': top1_global,
        })
        table.to_csv(outputs / f'per_query_{model_name.lower()}_{gallery_name}.csv', index=False)
        if model_name == 'SigLIP' and gallery_name == 'test_only':
            select_failures(table, rows, root, outputs)
        print(f'{model_name} | {gallery_name} | {metrics}')
    return summaries


def select_failures(frame, rows, root, outputs):
    errors = frame[frame.ground_truth_rank > 1]
    chosen = []
    for selector in [errors[errors.ground_truth_rank == 2],
                     errors[(errors.ground_truth_rank >= 3) & (errors.ground_truth_rank <= 10)],
                     errors[errors.ground_truth_rank > 10], errors]:
        for _, row in selector.iterrows():
            if row.query_id not in chosen:
                chosen.append(int(row.query_id))
                break
    chosen = chosen[:3]
    if not chosen:
        return
    results = frame.set_index('query_id').loc[chosen].reset_index().copy()
    results['one_line_explanation'] = ''  # Human visual annotation, not invented.
    results.to_csv(outputs / 'three_failure_cases_siglip.csv', index=False)

    # Side-by-side images with labels; query is in CSV to avoid truncating long text.
    width, image_h, footer_h = 960, 290, 54
    collage = Image.new('RGB', (width, len(results) * (image_h + footer_h)), '#101a2f')
    painter = ImageDraw.Draw(collage)
    for j, (_, item) in enumerate(results.iterrows()):
        for side, (label, idx) in enumerate([('GROUND TRUTH', int(item.ground_truth_index)),
                                            ('TOP-1 RETRIEVED', int(item.top1_index))]):
            with Image.open(image_file(root, rows[idx])) as src:
                im = src.convert('RGB')
                im.thumbnail((450, image_h-10))
            left = side * 480 + (480-im.width)//2
            top = j*(image_h+footer_h)+(image_h-im.height)//2
            collage.paste(im, (left, top))
            painter.text((side*480+15, j*(image_h+footer_h)+image_h+5), label, fill='white')
        painter.text((15, j*(image_h+footer_h)+image_h+27),
                     f"Query #{int(item.query_id)} | GT rank: {int(item.ground_truth_rank)}", fill='#a7c8ed')
    collage.save(outputs / 'three_failure_cases_siglip.png')


def plot_results(results, outputs):
    import matplotlib.pyplot as plt
    df = pd.DataFrame(results)
    for gallery in df.gallery.unique():
        sub = df[df.gallery == gallery]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        names = ['R@1', 'R@5', 'R@10']
        x = np.arange(3)
        for ix, model_name in enumerate(['CLIP', 'SigLIP']):
            row = sub[sub.model == model_name]
            if len(row):
                vals = row.iloc[0][names].to_numpy(dtype=float)
                ax.bar(x + (ix-.5)*.34, vals, width=.34, label=model_name)
        ax.set_xticks(x, names); ax.set_ylim(0, 100)
        ax.set_ylabel('Recall (%)')
        ax.set_title(f'Flickr8k — {gallery.replace("_", " ")} gallery, 5,000 test queries')
        ax.legend(); ax.grid(axis='y', alpha=.2)
        fig.tight_layout()
        fig.savefig(outputs / f'benchmark_{gallery}.png', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-dir', default='data/flickr8k')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=['CLIP', 'SigLIP'])
    parser.add_argument('--test-only', action='store_true', help='Evaluate only the canonical 1k-image gallery')
    args = parser.parse_args()
    root = Path(args.index_dir)
    outputs = Path(args.output_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    rows = read_metadata(root)
    test_ids = np.asarray([i for i, row in enumerate(rows) if row['split'] == 'test'], dtype=np.int64)
    if len(test_ids) != 1000:
        raise ValueError(f'Expected 1000 test images, got {len(test_ids)}. Build the full gallery first.')
    if len(rows) != 8000 and not args.test_only:
        raise ValueError(f'Expected 8000 images for full-gallery evaluation, got {len(rows)}')
    galleries = {'test_only': test_ids}
    if not args.test_only:
        galleries['all_8000'] = np.arange(len(rows), dtype=np.int64)
    # Query ordering intentionally fixed by test image, then caption number.
    queries = [caption for idx in test_ids for caption in rows[int(idx)]['captions']]
    targets = np.repeat(test_ids, 5)
    if len(queries) != 5000:
        raise ValueError('Flickr8k test must provide 5 captions for each of 1000 images')
    results = []
    for model_name in args.models:
        results.extend(evaluate_one(model_name, galleries, rows, queries, targets, outputs, root, args.device, args.batch_size))
    pd.DataFrame(results).to_csv(outputs / 'benchmark_results.csv', index=False)
    plot_results(results, outputs)
    manifest = {
        'query_split': 'test', 'gallery_definitions': {name: len(ids) for name, ids in galleries.items()},
        'num_queries': len(queries), 'ground_truth': 'paired Flickr8k image',
        'metric': 'one relevant image per caption; Recall@K',
        'zero_shot': True, 'test_tuning': False,
        'models': {name: MODELS[name] for name in args.models},
        'results': results,
    }
    (outputs / 'benchmark_results.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Evaluation finished. Results:', (outputs / 'benchmark_results.csv').resolve())

if __name__ == '__main__':
    main()
