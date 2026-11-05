"""Offline checks for the batched Parquet preparation path."""
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts import build_gallery as builder


class SmallDataset:
    column_names = ['image', 'caption_0', 'caption_1', 'caption_2', 'caption_3', 'caption_4']

    def __init__(self, size, dict_image=False):
        self.size = size
        self.dict_image = dict_image

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        im = Image.new('RGB', (18, 12), 'orange')
        if self.dict_image:
            buf = io.BytesIO()
            im.save(buf, format='PNG')
            image = {'bytes': buf.getvalue(), 'path': None}
        else:
            image = im
        return {'image': image, **{f'caption_{j}': f'caption {i} {j}' for j in range(5)}}


class TestParquetPreparation(unittest.TestCase):
    def test_parquet_shards_are_selected_instead_of_imagefolder(self):
        paths = [
            'default/train/0000.parquet', 'default/train/0001.parquet',
            'default/dev/0000.parquet', 'default/test/0000.parquet', 'README.md',
        ]
        downloaded = []
        parquet_loads = []

        class FakeApi:
            def list_repo_files(self, repo_id, repo_type, revision):
                self.assertion_dummy = (repo_id, repo_type, revision)
                return paths

        def fake_download(repo_id, filename, **kwargs):
            downloaded.append(filename)
            return f'/fake-cache/{filename}'

        def fake_load_dataset(kind, data_files, split):
            parquet_loads.append((kind, data_files, split))
            return SmallDataset(2)

        fake_hf = types.SimpleNamespace(HfApi=FakeApi, hf_hub_download=fake_download)
        fake_ds = types.SimpleNamespace(load_dataset=fake_load_dataset)
        with patch.dict(sys.modules, {'huggingface_hub': fake_hf, 'datasets': fake_ds}):
            ds = builder.load_parquet_split('train')
        self.assertEqual(len(ds), 2)
        self.assertEqual(downloaded, ['default/train/0000.parquet','default/train/0001.parquet'])
        self.assertEqual(parquet_loads[0][0], 'parquet')
        self.assertEqual(parquet_loads[0][2], 'train')
        self.assertEqual(parquet_loads[0][1]['train'], ['/fake-cache/default/train/0000.parquet','/fake-cache/default/train/0001.parquet'])

    def test_build_images_preserves_captions_and_reuses_cache(self):
        sizes = {'train':2,'dev':1,'test':1}
        counts = []

        def fake_loader(split):
            counts.append(split)
            return SmallDataset(sizes[split], dict_image=True)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(builder, 'load_parquet_split', side_effect=fake_loader), \
                 patch.object(builder, 'EXPECTED', sizes), \
                 patch.object(builder, 'tqdm', side_effect=lambda values, **kwargs: values):
                rows = builder.prepare_images(Path(tmp), max_images=4, force=False)
                self.assertEqual(len(rows), 4)
                self.assertEqual([r['split'] for r in rows], ['train','train','dev','test'])
                self.assertTrue(all(len(r['captions'])==5 for r in rows))
                self.assertTrue(all(builder.image_file(tmp,row).is_file() for row in rows))
                self.assertTrue((Path(tmp)/'metadata.jsonl').exists())
                cached = builder.prepare_images(Path(tmp), max_images=4, force=False)
                self.assertEqual(cached, rows)
                self.assertEqual(counts, ['train','dev','test'])


if __name__ == '__main__':
    unittest.main()
