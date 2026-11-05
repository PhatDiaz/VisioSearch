"""Measure real full-gallery inference; optionally compare a small saved-reference sample."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import zipfile
from io import StringIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))

from src.pipeline_artifacts import MANIFEST_PATH, load_runtime_artifacts
from src.two_stage_retriever import TwoStageRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', default='A dog running on the beach')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--reference-queries', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 0 <= args.reference_queries <= 5000:
        parser.error('--reference-queries must be between 0 and 5000.')
    if args.output and args.output.exists():
        parser.error('The output file already exists. Choose a new path to preserve earlier measurements.')
    artifacts = load_runtime_artifacts(ROOT / 'data' / 'flickr8k', MANIFEST_PATH)
    retriever = TwoStageRetriever(artifacts, args.device)
    reports = []
    for query in [args.query, args.query, 'People riding bicycles']:
        response = retriever.search(query, 10)
        report = dict(query=query, pipeline=response.pipeline, cache_hit=response.cache_hit,
                      timings=vars(response.timings), device=retriever._encoder.device,
                      results=[dict(image_id=artifacts.image_ids[hit.image_index], **vars(hit)) for hit in response.hits])
        reports.append(report)
        print(json.dumps(report), flush=True)
    if args.reference_queries:
        with zipfile.ZipFile(ROOT / 'research_artifacts/Topic11_BLIP_Final_8K_Results.zip') as archive:
            references = list(csv.DictReader(StringIO(archive.read('per_query_final_test.csv').decode('utf-8'))))
            summary = list(csv.DictReader(StringIO(archive.read('final_test_summary.csv').decode('utf-8'))))
        ranks = [int(row['reranked_rank']) for row in references]
        recorded = {f'R@{k}': round(100 * sum(rank <= k for rank in ranks) / len(ranks), 2) for k in (1, 5, 10)}
        print('Saved reference CSV metrics (NOT a fresh integrated evaluation):', recorded, flush=True)
        for key, value in recorded.items():
            assert value == float(summary[-1][key])
        integrated_ranks = []
        for row in references[:args.reference_queries]:
            response = retriever.search(row['query'], 20)
            stage1 = max(response.hits, key=lambda hit: hit.cosine_similarity)
            truth = int(row['ground_truth_global_index'])
            assert artifacts.image_ids[truth] == row['ground_truth_image_id']
            # Ground truth is used only here, after search, for evaluation.
            rank = next((i for i, hit in enumerate(response.hits, 1) if hit.image_index == truth), 21)
            integrated_ranks.append(rank)
            report = dict(reference_query_index=int(row['query_index']), query=row['query'],
                          expected_top1=row['reranked_top1_id'], actual_top1=artifacts.image_ids[response.hits[0].image_index],
                          expected_siglip_top1=row['siglip_top1_id'], actual_siglip_top1=artifacts.image_ids[stage1.image_index],
                          integrated_top20_rank=rank, timings=vars(response.timings))
            reports.append(report)
            print(json.dumps(report), flush=True)
        if len(integrated_ranks) == 5000:
            measured = {f'R@{k}': round(100 * sum(rank <= k for rank in integrated_ranks) / 5000, 2)
                        for k in (1, 5, 10)}
            reports.append(dict(fresh_integrated_full_gallery_metrics=measured, text_queries=5000, gallery_images=8000))
            print('Fresh integrated full-gallery evaluation:', measured, flush=True)
        else:
            print(f'Only {len(integrated_ranks)} reference queries ran; no full-test Recall claim is made.', flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
