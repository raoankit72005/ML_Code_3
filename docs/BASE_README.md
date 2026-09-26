> **Google Colab:** Open the [single-GPU notebook](https://colab.research.google.com/github/raoankit72005/ML_Code_1/blob/main/colab/ML_Code_1_Colab.ipynb). See [Colab setup and limits](colab/README.md).

# ML_Code_1 — tracked hybrid business entity resolution

Original TSV data → cluster split → PyTorch/LoRA multilingual encoder → disk embeddings → country/source FAISS retrieval + lexical blocking → pair features → LightGBM → macro F0.5 threshold → complete test outputs.

This is the second-attempt hybrid project. The older lexical-only pipeline remains available via `run.py`. The new entry point is **`hybrid.py`**.

## SageMaker quick start

Use an activated, dedicated Python 3.11+ environment on your **Linux `ml.g5.xlarge` GPU app**. The CUDA LightGBM build also needs `nvcc` and `g++`; the NVIDIA driver must support the PyTorch CUDA 12.6 wheel. The setup script checks prerequisites and performs a real LightGBM CUDA smoke test. It cannot install or repair your app's system drivers.

```bash
cd ML_Code_1
bash scripts/setup_hybrid.sh
python hybrid.py --input ML_Dataset.zip --work work_hybrid --skip-test
```

Use the original challenge ZIP, not a sampled archive. This command trains and validates. Continue with the same input, code and configs to produce full test predictions:

```bash
python hybrid.py --input ML_Dataset.zip --work work_hybrid
```

Completed stages are skipped. Every test S1, including France and empty-match cases, must appear in both TSV outputs. The last stage audits both files against the original test input. Also run the challenge-supplied validator with reference-ID checks before uploading.

**No claims of error-free execution or improved F0.5 on the original dataset:** the target GPU and full dataset are not available in development. See [verification](docs/TESTING.md).

## Follow training live

Terminal output includes stage names, encoder loss, epoch, learning rate, processed pairs, throughput, sampled monitor recall, process RAM and GPU allocated/reserved/peak memory. LightGBM prints validation loss every 10 iterations and logs final full-validation macro F0.5 and the selected threshold.

```bash
tail -f work_hybrid/logs/encoder.jsonl
```

In a SageMaker notebook, use TensorBoard:

```python
%load_ext tensorboard
%tensorboard --logdir work_hybrid/logs/tensorboard
```

Or start it in a separate terminal and use your environment's supported notebook proxy:

```bash
python -m tensorboard.main --logdir work_hybrid/logs/tensorboard --port 6006
```

No external experiment-tracking account or API key is needed. See [tracking and restart details](docs/TRAINING_TRACKING.md).

## What fits, and what may not

The instance has only 16 GiB host RAM. Encoder activations are divided into mini-batches; training pairs are read from SQLite, embeddings are written to float16 files, FAISS loads one country/source partition at a time, and features are written to Parquet in batches. Each major stage runs in its own process so memory is released between stages.

Encoder training retries CUDA OOM by reducing the activation mini-batch, preserving its contrastive batch. Embedding inference similarly retries smaller batches. These measures do not prevent every possible OOM, driver failure or OS kill. The last completed checkpoint remains the recovery point.

LightGBM uses all non-holdout candidate pairs through a disk-backed Sequence. **It still needs resident bins/buffers in RAM/VRAM.** Its preflight estimate can reject a full-data run on this instance. If that happens, use more memory for fitting; the pipeline does not silently downsample, switch models, or pretend chunk-by-chunk boosting is equivalent. Training is single-GPU.

Disk usage can be much larger than the ZIP. Ten million 768-dimensional float16 vectors alone use 15.36 GB, before indexes, raw data and pair tables. Monitor free disk, and use persistent storage/backups for outputs and checkpoints.

## Configurations

- `configs/hybrid.json`: pinned model revision, LoRA, cached contrastive batches, encoding and FAISS settings.
- `configs/full_preparation.json`: lexical retrieval and final candidate cap (64 initially).
- `configs/full_gpu.json`: all-pair CUDA LightGBM and threshold grid.

Starting encoder settings: LoRA rank 8, contrastive batch 128, activation mini-batch 8, sequence length 128, embedding batch 64, one epoch. A10G training uses BF16 autocast when supported; otherwise FP32. These are starting settings, not validated optimal settings.

The entity-aware batch stream visits every training positive pair each epoch. If a tail batch contains one cluster, it replays a positive from another training cluster to supply a valid negative partner. No positive links are silently dropped. Same-country batches are preferred. Explicit mined hard negatives are not implemented in this version; other clusters in the contrastive batch supply negatives.

The encoder monitor uses a deterministic validation-query subset and sampled reference pool plus its known positives. It is affordable but optimistic relative to full-pool retrieval. The best monitor checkpoint is selected, including the initial pretrained-equivalent LoRA checkpoint. Full-pool candidate recall and final F0.5 are computed later. The small monitor is never presented as the final score.

For a controlled frozen-encoder experiment, copy `configs/hybrid.json`, set `encoder_mode` to `frozen`, and use a new work directory. Compare final full-pool metrics between runs; do not mix embeddings/candidates from different checkpoints.

## Model and data integrity

- One shared multilingual encoder for S1 and reference records; PyTorch performs adapter training.
- Ground-truth connected components are built before supervised training. Shared references cannot bridge training and validation clusters.
- LoRA trains only on training-cluster positives. Validation supervision is used only for selection/evaluation.
- Embeddings are normalized; IVF-PQ shortlists are reranked using original stored vectors. Tiny partitions use exact FAISS search.
- The final candidate set combines lexical and neural retrieval with diversity quotas. `candidate_pairs.tsv` records exactly what the matcher scores.
- Every final candidate gets neural cosine, including lexical-only candidates.
- Unknown countries have a retrieval fallback. Country names are not restricted to India and US.
- Missing comparisons remain NaN with missingness features.
- No external business lookup, geocoding or identity enrichment.
- Reference text, including validation-linked records, is present in the retrieval pool; validation links are excluded from encoder gradient training.

## Key outputs

| Path under `work_hybrid/` | Content |
| --- | --- |
| `logs/tensorboard/` | TensorBoard events for encoder, embeddings, retrieval and LightGBM |
| `logs/encoder.jsonl` | Append-only encoder training metrics |
| `encoder/last.pt` | Adapter, optimizer, RNG, epoch and data cursor for resume |
| `encoder/best.pt` | Best sampled-monitor encoder checkpoint |
| `train/candidate_recall.json` | Full-pool retrieval recall and oracle F0.5 ceiling |
| `model/resource_plan.json` | LightGBM RAM/VRAM sizing estimate |
| `model/model_metadata.json` | Trained model counts, threshold, score and hashes |
| `model/validation_f05.json` | Full-validation threshold sweep and country scores |
| `test/matching_results.tsv` | Final predictions |
| `test/candidate_pairs.tsv` | Final candidate lists |
| `test/submission_coverage.json` | Original test S1 coverage audit |

## Tests and publishing

```bash
python -m unittest discover -s tests -v
```

The hybrid integration test builds a tiny random local Transformer and exercises real LoRA, cached-loss backpropagation, FAISS, LightGBM, TensorBoard, restart and output validation without downloading model weights. This is a correctness test, not a quality benchmark.

If the GitHub repository has not yet been created, the included helper can create it from this code folder after you authenticate GitHub CLI as an account authorized to create `raoankit72005/ML_Code_1`:

```bash
bash scripts/publish_github.sh
```

It creates a **private** repository, commits code and pushes `main`. Run this before generating data in arbitrary unignored folders. Standard data, weights, embeddings and work directories are ignored. The original `ML-code` repository is not modified.
