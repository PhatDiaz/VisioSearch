# VisioSearch — Natural-Language Photo Search

**Computer Vision Final Project · Topic 11 · SigLIP LoRA + BLIP ITM · Flickr8k (8,000 images)**

VisioSearch is a focused photo-search application built with Streamlit. Describe an
image in English, press **Enter** or **Search**, and explore real Flickr8k matches.
The homepage uses the selected SigLIP LoRA checkpoint to retrieve 20 candidates
from **precomputed fine-tuned image embeddings**, then BLIP ITM reranks them.
It includes clickable examples, 5/10/20 results, a responsive
grid, original-image dialogs with captions and metadata, and CSV export.

The academic interface is preserved separately in `research_app.py`, including
CLIP/SigLIP comparison, model evaluation, failure analysis and methodology. It is
not included in the main search navigation.

**What it is:** A working project implementation (source code and experiment scripts).
**What it is not yet:** A prebuilt 8,000-image index or a completed scientific experiment. Dataset and model weights are downloaded and embeddings are computed when you run the commands below. No unmeasured results are presented as real.

## Current branch: the frozen two-stage pipeline

The development branch `feature/two-stage-retrieval` now uses:

1. `google/siglip-base-patch16-224` with LoRA **batch16_r8_e5**, rank **8**, alpha **16**.
2. The matching `embeddings_siglip_ablation_best.npy` from the supplied ablation ZIP.
3. Exact cosine ranking against all **8,000** image vectors; stable Top-**20** selection.
4. Frozen `Salesforce/blip-itm-base-coco` image-text matching for those 20 actual images.
5. The notebook's **raw positive-class logit**, without sigmoid or softmax, followed by:

```python
def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-8)

fusion = 0.25 * zscore(candidate_cosine) + 0.75 * zscore(blip_positive_logits)
order = np.argsort(-fusion, kind="stable")
```

Both arrays are normalized over the **same 20 candidates**, independent of whether
the user requests 5, 10 or 20 results. Candidate positions are mapped back to the
original global image indices. No captions, ground-truth IDs or split labels enter
the live ranking calculation. Captions are read only for the image-details view.

The existing design and light/dark controls are retained. Cards now show the fused
**match score**; the dialog and CSV retain the separate cosine and BLIP logit. Fused
scores can be negative and are not probabilities. No research tabs or model controls
were added to the homepage.

### Windows / VS Code: prepare and launch this version

Use the existing `.venv`, or install Python 3.11/3.12 and create one with
`py -3.11 -m venv .venv`. Select `.venv\Scripts\python.exe` in VS Code.
Keep the original 8,000 JPEGs and `metadata.jsonl` in `data/flickr8k`, and place the
following **unmodified** files in `research_artifacts/`:

- `Topic11_Ablation_Final_Outputs.zip`
- `Topic11_BLIP_Final_8K_Evaluation.ipynb`
- `Topic11_BLIP_Final_8K_Results.zip`

From a PowerShell terminal in this project:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/prepare_two_stage.py --download-models
.\.venv\Scripts\python.exe -m streamlit run app.py
```

With the virtual environment activated, the launch command is still:

```powershell
python -m streamlit run app.py
```

The preparation step runs once, checks the selected checkpoint and the final audit's
SHA-256 fingerprints, and writes these new runtime files:

```text
models/siglip_lora_batch16_r8_e5/
    adapter_config.json, adapter_model.safetensors
    preprocessor_config.json, tokenizer_config.json
    special_tokens_map.json, spiece.model
models/two_stage/runtime_manifest.json
data/flickr8k/embeddings_siglip_lora_batch16_r8_e5.npy
```

The base SigLIP and BLIP checkpoints use `.cache/huggingface` (or an existing
`HF_HOME`). Preparation is idempotent and refuses to replace different existing
artifacts. ZIPs are never extracted when the web app starts. Original data,
pretrained embeddings, LoRA exports, notebooks and result files remain unchanged.
`models/`, `research_artifacts/`, data, arrays and weights are ignored by Git.

Models load lazily on the first search. Models/indexes are Streamlit resources;
metadata, query results and BLIP vision tokens are cached. The vision cache holds
at most 64 images in FP16 on CPU, reproducing the notebook's saved-token round trip.
On CUDA, inference uses the notebook's FP16 autocast; CPU uses FP32 computation with
the same FP16 vision-cache round trip. Near-tie scores can differ across hardware.
Artifact changes invalidate the resource cache and trigger hash checks again.

### Checks, measurements and regression limits

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m compileall -q app.py src scripts
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_two_stage.py --reference-queries 3 --output .local/check-new.json
```

The original audit and result CSV report **63.14 / 83.08 / 88.06%** Recall@1/5/10.
These are supplied research results, **not a newly reproduced web-app benchmark**.
The results ZIP contains the per-query ranks but omits the original candidate
arrays and BLIP logits, so a complete score-by-score replay is unavailable.
Tests validate saved CSV consistency and compare fusion with the notebook functions.
The diagnostic additionally compares real predicted IDs with saved references.

Verification executed during this integration: **39 tests passed**, including
real adapter-weight equality and activation, frozen model parameters, BLIP's cached
ITM path against standard forward, Top-5/10/20 UI results, metadata/index alignment,
cache invalidation, missing-input errors and explicit fallback. Syntax compilation
and dependency checks passed. A temporary Streamlit server returned HTTP 200 / `ok`
and was then stopped. All 20 protected original-file fingerprints were unchanged;
the gallery still contains 8,000 JPEGs. Fresh-browser visual QA and CUDA inference
were not rerun; the established CSS/layout files were not changed.

Actual local CPU measurements (PyTorch `2.15.0+cpu`, Transformers `4.57.6`, PEFT
`0.18.1`):

| Search | SigLIP | BLIP + fusion | Pipeline total |
|---|---:|---:|---:|
| First query, including model loading | 5.38 s | 9.44 s | 14.83 s |
| New query, loaded models | 0.05 s | 9.28 s | 9.33 s |
| Repeated identical query | cached | cached | <0.001 s |

These measure pipeline time, excluding browser rendering. Three first test-caption
queries matched both saved SigLIP and final Top-1 IDs; this small sample does not
establish full-test Recall. A complete fresh evaluation is opt-in and may take hours
on CPU:

```powershell
.\.venv\Scripts\python.exe scripts/check_two_stage.py --reference-queries 5000 --output .local/full-test-new.json
```

Only a completed 5,000-query run prints fresh integrated Recall@K; that full run was
**not executed** during this integration. Do not replace the reference claims with
integrated-model claims until it is completed. The project had inspected earlier
test results; the full gallery also contains LoRA training images. BLIP's upstream
pretraining overlap is unknown. Exact-image Recall does not credit acceptable
alternative photos.

### Missing artifacts, CPU/GPU and explicit fallback

- **Missing LoRA files/BLIP cache:** install requirements, then run
  `scripts/prepare_two_stage.py --download-models`. The app fails with an actionable
  message; it never switches models automatically after an error.
- **Metadata/hash mismatch:** restore the exact original metadata and matching
  exports. Do not reorder rows, rename IDs or substitute the pretrained embeddings.
- **CPU latency:** BLIP is the main cost. CUDA is chosen automatically when
  `torch.cuda.is_available()` is true. Installing a GPU driver alone does not make
  a CPU-only PyTorch build CUDA-enabled; use the official
  [PyTorch installation selector](https://pytorch.org/get-started/locally/) for a
  suitable Windows CUDA build. CUDA inference was not exercised on this CPU build.
- **GPU memory pressure:** reduce only compute batch sizes, then restart:

  ```powershell
  $env:TOPIC11_BLIP_VISION_BATCH = "4"
  $env:TOPIC11_BLIP_PAIR_BATCH = "4"
  ```

  This keeps Top-20, raw-logit scoring and alpha=0.75 unchanged. Increase the bounded
  vision cache only with appropriate RAM; do not change fusion to hide slow inference.
- **Device selection:** `TOPIC11_DEVICE` accepts `auto`, `cpu` or `cuda`; an explicit
  unavailable CUDA request fails with a clear message.
- **Deliberate v1 fallback**, for troubleshooting only:

  ```powershell
  $env:TOPIC11_PIPELINE = "pretrained_siglip"
  .\.venv\Scripts\python.exe -m streamlit run app.py
  ```

  A visible homepage warning and CSV model label identify this mode. Remove that
  environment variable to return to the default two-stage pipeline. The unchanged
  `research_app.py` and `scripts/evaluate.py` still run the original pretrained
  CLIP/SigLIP experiments, not this final two-stage evaluation.

New code is in `src/pipeline_artifacts.py`, `src/models/`, `src/two_stage_retriever.py`,
`scripts/prepare_two_stage.py`, `scripts/check_two_stage.py` and `tests/test_two_stage*.py`.
The integration adapts `src/search_service.py`, `app.py`, `src/photo_ui.py`, requirements,
Git ignores and UI integration tests. No commit or push is performed automatically.

## 1. Original v1 scientific protocol (retained baseline)

The sections below describe the original pretrained-only baseline and its data
preparation. The current two-stage model, fine-tuned index and final 8K evaluation
protocol are specified above; do not confuse the two experimental settings.

Dataset: [`intro/flickr8k`](https://huggingface.co/datasets/intro/flickr8k), with 8,000 images, each with 5 captions.

| Split | Images | Captions | Use in this project |
|---|---:|---:|---|
| Train | 6,000 | 30,000 | Search-gallery images only; **not** fine-tuning |
| Dev | 1,000 | 5,000 | Search-gallery images; previous progress validation |
| Test | 1,000 | 5,000 | Held-out test queries and ground truth |
| **Total** | **8,000** | **40,000** | Full gallery of 8,000 images |

**Two evaluation protocols are deliberately kept separate:**

- **Canonical test:** 5,000 test-caption queries searching only the **1,000 test images**. Report these as the primary final test metrics.
- **Harder full-gallery evaluation:** the *same* 5,000 test queries searching **all 8,000 images**. Report separately as an extra evaluation under a different gallery size.

**Live app:** search all 8,000 images with *any new English-language query*, using
SigLIP. Split filters and CLIP/SigLIP comparison remain in `research_app.py` for
research use. Interactive searches are **not** a quantitative evaluation by themselves.

The pretrained models are `openai/clip-vit-base-patch32` (baseline) and `google/siglip-base-patch16-224` (main). There is no Flickr8k fine-tuning. The models can have unknown overlap with data seen in their original pretraining; "held-out" refers to **our** development/evaluation workflow.

Do not compare scores between the 200-image validation pilot, 1,000-image test, and 8,000-image retrieval gallery as if they have the same evaluation difficulty. Do not use the test split to tune the models or query prompt engineering. SigLIP was selected as main model during validation-stage development; do not pick it based on test outcomes.

## 2. Folder structure

```text
Topic11_8K_Search_App/
├── app.py                     # User-facing SigLIP photo-search homepage
├── research_app.py            # Preserved academic dashboard and model comparison
├── assets/styles.css          # Dark theme and responsive image grid
├── start.bat / start.ps1       # Windows launchers
├── scripts/
│   ├── build_gallery.py       # Downloads Parquet shards, caches 8k JPGs, encodes both models
│   └── evaluate.py            # 1k test and 8k-gallery benchmarking
├── src/
│   ├── modeling.py            # Pretrained CLIP/SigLIP feature extractors
│   ├── retrieval.py           # Cosine search, rank and Recall@K math
│   ├── search_service.py      # Cached metadata, embeddings and text-query encoding
│   ├── photo_ui.py            # Image thumbnails, gallery and original-image dialog
│   └── storage.py             # Metadata and embedding persistence
├── tests/test_retrieval.py    # Self-contained correctness tests
├── notebooks/                # Optional GPU preparation notebook
├── data/flickr8k/            # Generated on first run; NOT included in zip
│   ├── metadata.jsonl
│   ├── images/                # 8,000 display-ready JPEGs
│   ├── embeddings_clip.npy
│   └── embeddings_siglip.npy
└── results/                  # Generated benchmarks and failure images
    ├── benchmark_results.csv
    ├── three_failure_cases_siglip.csv
    └── ...
```

## 3. Quick start (Mac, Windows, Linux)

Recommended: Python 3.11. A GPU is highly recommended **for building the 8k index**; the app itself can query with CPU if the index is already built.

```bash
# From the project directory
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Index ALL 8,000 Flickr8k images for both models (needs internet + GPU recommended)
python scripts/build_gallery.py --output-dir data/flickr8k

# Evaluate without model tuning: 1k test gallery AND full 8k gallery
python scripts/evaluate.py --index-dir data/flickr8k

# Launch the professional browser interface (not Colab notebook output)
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The first search loads SigLIP and may
download its checkpoint if it is not cached. Opening the homepage does not load
the model. Subsequent searches reuse the encoder; image embeddings are never
recomputed during a search. Cached checkpoints load without contacting Hugging Face.

On Windows, after installing the dependencies into `.venv`, you can also double-click
`start.bat` or run `.\start.ps1`. The launcher uses the project's Python environment
directly and stores model downloads in `.cache/huggingface` unless `HF_HOME` is set.
`sentencepiece` is required by the SigLIP tokenizer and is included in `requirements.txt`.

### Windows / VS Code

Open this folder in VS Code and use its PowerShell terminal. Install Python 3.11
or 3.12 if needed, then create the environment (skip this step if `.venv` exists):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you installed Python 3.12, replace `-3.11` with `-3.12`. Choose
`.venv\Scripts\python.exe` as the VS Code Python interpreter. No activation is
required when running the explicit Python path.

With the prepared gallery in `data/flickr8k`, double-click `start.bat`, or run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

Open **http://localhost:8501**. Enter an English description, select 5, 10 or 20
results and press Enter. **View image** opens the uncropped original, its dimensions,
dataset identifier and five reference captions. Press Ctrl+C in the app terminal
to stop it. CPU execution is supported; a CUDA installation is not required.

Use **Light mode** in the top-right corner to switch between light and dark
appearance. The selected theme is kept in the page URL, including after refresh.
Switching themes preserves the current query and search results.

To open the preserved academic dashboard separately:

```powershell
.\.venv\Scripts\python.exe -m streamlit run research_app.py --server.port 8502
```

The UI requires Streamlit 1.49 or newer within the 1.x series. No additional
runtime packages were added for the redesign. To prepare/check both cached models:

```powershell
.\.venv\Scripts\python.exe scripts/check_local.py --check-models
```

If you already have the extracted `Topic11_8K_Outputs` directory from Colab, copy
its `data/flickr8k` directory into the project's `data/` directory and its `results`
directory into the project root. You can then skip indexing and evaluation.

Validate the exported artifacts before launching:

```bash
python scripts/check_local.py
```

This checks every image, split sizes, caption counts, normalized embedding arrays,
and agreement between saved benchmark metrics and per-query ranks. Add
`--check-models` to download/load both checkpoints and run a new text query with each.
This validation does not rerun the complete scientific benchmark.

**If your machine lacks a GPU**: run the indexing/evaluation on Google Colab using the GPU notebook in `notebooks/`; then transfer the generated `data/flickr8k` and `results` directories to this project on your computer. Only run `streamlit run app.py` locally afterwards.

## 4. Colab GPU preparation (recommended)

1. Open `notebooks/Topic11_Prepare_8000_Gallery.ipynb` in Google Colab and choose a GPU runtime.
2. Upload the **full project zip** when the notebook requests it (a copy of this source project, not the previous Colab-only notebook).
3. Execute the cells in order. The notebook uses **intro/flickr8k Parquet shards** rather than requesting thousands of image files individually. It builds 8,000 embeddings and both evaluation protocols.
4. The notebook downloads a **small `Topic11_Evaluation_Results.zip`** containing test metrics and failure-case figures. Send this file for analysis.
5. The separate **large `Topic11_8K_Outputs.tar.gz`** contains the indexed gallery for local Streamlit use.
6. Use the optional Google Drive export cell to move the generated `.tar.gz` to Drive. Browser downloads of the large archive can fail.
7. On your computer, extract the archive into the project root. It should create `data/flickr8k/` and `results/`.
8. Start the UI with `streamlit run app.py`.

Do not commit the generated image gallery or model weights to Git. Generated JPEGs/embeddings may occupy considerable disk space.

## 5. Commands and debugging

Smoke test for the indexing script without doing all 8k **(not valid for the final evaluation)**:

```bash
python scripts/build_gallery.py --output-dir data/smoke --models SigLIP --max-images 12
python -m pip install pytest
python -m pytest -q
```

The 12-image smoke index is intentionally separate from the real project index. The full evaluation requires all 1,000 test images.

If the GPU runs out of memory:

```bash
python scripts/build_gallery.py --output-dir data/flickr8k --batch-size 8
```

The builder reuses the cached gallery and completed model `.npy` files. A failed partial model `.npy` will not be written because each model is saved only after encoding all images.

To re-encode model features after intentionally changing model code, pass `--overwrite`; this also rebuilds the image cache, so use carefully.

## 6. Search interface and academic outputs

`app.py` contains one English-language search page with no research tabs or model
controls. SigLIP searches every indexed image. Desktop uses four gallery columns,
tablet uses three, and phone uses two (one on very narrow screens). Display-only
thumbnails are cropped to 4:3 in memory; original files remain unchanged. Every
result shows its rank and cosine score. **View image** opens the full original,
captions and metadata, with an original-image download. **Export results** saves
the current query, ranks, image IDs and cosine scores as CSV.

The unchanged scientific scripts and files remain available for reporting:

- `research_app.py`: the previous four-tab dashboard, including model comparison,
  Recall@K, three failure pairs and methodology.
- `scripts/evaluate.py`: canonical 1k test and full 8k gallery evaluation.
- `results/`: measured benchmark CSV/JSON, per-query ranks and failure figures.

Missing indexes, model-loading errors and empty queries produce English messages.
The main page loads metadata and memory-maps the index without loading a model;
query vectors, models and thumbnails are cached with bounded query/image caches.
The scientific scripts, model checkpoints, cosine ranking and ground truth are
unchanged. No placeholder search results are used.

## 7. Notes for a Master's-level report

Strong deliverables go beyond a beautiful interface:

- Evaluation on the **correct held-out split** with split sizes and exact query/gallery sizes stated.
- Transparent model configuration, fixed checkpoints and reproducible outputs.
- Separate metrics for 1k and 8k galleries, not mixed together.
- Three concrete failure examples with image pairs and reasoned comments.
- A functioning real-time text-search interface (not a screenshot of Colab output).
- Discussion of why cosine similarity is a ranking score, *not* a calibrated probability.
- Acknowledge that Flickr8k is a proxy rather than a real private photo library and that one annotated ground-truth image per caption does not capture every semantically acceptable image.

## 8. Status and limitations

Local validation has been completed on Windows with Python 3.12:

- All **11 automated tests** passed, including cached models without HTTP requests,
  lazy model loading, real SigLIP results for 5/10/20 images, exact ranking against
  the full 8k embedding matrix, image details, examples and missing-file/model errors.
- Syntax compilation passed for both apps and all source modules.
- All 8,000 original images and both embedding arrays were validated. Every saved
  Recall value matched its corresponding 5,000 per-query ranks.
- Chrome browser checks covered Enter submission, a four-column 1440 px desktop
  grid, a two-column 390 px phone viewport without horizontal overflow, and the
  original-image dialog with all five captions. Both CSV and original-JPEG downloads
  were executed; the downloaded JPEG matched the source file byte for byte.
- The preserved academic app rendered all four tabs and both benchmark tables.
- SHA-256 checks confirmed that all 19 protected metadata, embedding, result and
  scientific-backend files remained unchanged.

Physical phones, other browser engines and a clean Python installation still need
manual verification. Browser responsiveness was tested using Chrome device emulation.

The prepared gallery and evaluation files were imported from the supplied
`Topic11_8K_Outputs` directory. Full indexing and the complete 5,000-query model
evaluation were **not rerun locally**. The two checkpoints are cached under
`.cache/huggingface`; the app and local checker respect an existing `HF_HOME` override.

Model loading now tries the cached snapshot directory first, with network access
disabled for that load. This prevents repeated Hugging Face metadata checks from
stalling search when internet access is blocked. An absent or incomplete cache
still triggers the usual first-time download. The first search after a server restart
can take several seconds to import and load the encoder. Later queries reuse it.
The cache regression test rejects every HTTP request while loading and querying
both real models; the homepage integration test also blocks HTTP requests.

Run the available checks without rebuilding the index:

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m compileall -q app.py research_app.py src
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_local.py
```

Real-model integration tests skip when their cached checkpoints or prepared gallery
are absent; they never create fake search results for the app.

For official API references: [Hugging Face CLIP](https://huggingface.co/docs/transformers/model_doc/clip), [Hugging Face SigLIP](https://huggingface.co/docs/transformers/model_doc/siglip), [Streamlit caching](https://docs.streamlit.io/develop/concepts/architecture/caching).

## 9. Fix for HTTP 429 during image downloads

The original code used `load_dataset("intro/flickr8k", split=...)`, which caused thousands of independent image HTTP downloads and triggered `429 Too Many Requests`. The current `scripts/build_gallery.py` uses **the same dataset** from Hugging Face's `refs/convert/parquet` branch instead: the images and captions are stored in a small number of Parquet shards. The usual 6,000/1,000/1,000 split and five captions per image remain the basis of this experiment.

**Important:** Upload the **new project ZIP**, not the ZIP from the failing Colab run. The new notebook verifies the updated downloader before execution and uses `subprocess.run(..., check=True)` to stop when a command fails. If a temporary 429 persists on a large shard or on the model checkpoints, retry only the affected cell after the service limit resets; cached shards can be reused.
