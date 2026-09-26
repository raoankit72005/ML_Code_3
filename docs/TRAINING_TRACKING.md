# Tracking and recovery

## Reading the curves

| Metric | Meaning |
| --- | --- |
| `encoder/loss` | Contrastive training loss; falling loss alone does not prove better entity resolution |
| `monitor_recall` | Recall@K for fixed validation queries against a sampled reference pool; optimistic monitor only |
| `pairs_per_second` | Training-pair throughput since the current process started, including monitoring time |
| `rss_gib` | Current process resident RAM |
| `ram_available_gib` | Host-reported RAM availability; the fitting resource guard separately accounts for cgroup limits |
| `gpu_allocated_gib` | PyTorch tensor allocations, not total device usage |
| `gpu_reserved_gib` | PyTorch caching allocator reservations |
| `gpu_peak_gib` | Peak PyTorch tensor allocation since process start |
| `validation_monitor_binary_logloss` | LightGBM validation loss |
| `macro_f05` | Final threshold-tuned full-validation macro F0.5 |

Use `nvidia-smi` for total device memory and utilization. LightGBM/FAISS allocations are not represented by PyTorch allocator metrics. Use `free -h` and `df -h` for RAM and disk. The JSONL logs and TensorBoard events are saved locally and need backups along with checkpoints.

## Checkpoint semantics

Encoder `last.pt` is saved every configured checkpoint interval and at epoch end. It records adapter weights, optimizer state, epoch, batch cursor, Torch CPU/CUDA RNG state and the input/config signature. Restarting the identical command loads it and skips already-consumed batches. The stream must scan past those batches, which can take time for a late checkpoint. Work since the last checkpoint is replayed. Only load checkpoints produced by this project that you trust.

`best.pt` is selected using the fixed sampled recall monitor. The initial pretrained-equivalent adapter state is eligible, so training is not assumed to improve retrieval. Full reference-pool evaluation remains necessary.

Embeddings have a committed row count. Restart truncates any uncommitted tail, then continues from the next record. The model and index fingerprints must match. Checkpointing flushes vectors to disk before acknowledging progress.

Other preparation stages restart their incomplete stage. Completed stages are skipped. Cleaning does not support partial-file resume: if it stopped before completion, inspect the incomplete generated cleaning folder and use a fresh work directory or explicitly remove only that generated folder before retrying. Original inputs must remain unchanged.

LightGBM writes a snapshot every 100 iterations for inspection, but the training command does **not** automatically resume boosting from it. An interrupted LightGBM fit restarts fitting, reusing completed float32 caches. Its complete model/metadata is written after validation and threshold selection.

Do not edit source code or configs while a pipeline is running: fingerprints intentionally reject mixed runs. Use a new work folder for changed encoder/retrieval settings. Do not run multiple training jobs against the same work folder.

## Resource failure

An encoder CUDA OOM triggers a smaller activation mini-batch retry. An embedding CUDA OOM triggers a smaller inference batch retry. If one example still cannot fit, the process stops; restart from the last checkpoint after correcting the environment. These retries cannot recover from a process killed by the operating system, hardware errors or native library faults.

LightGBM/ANN estimated-memory rejection is deliberate. Do not bypass it by raising safety fractions merely to force a run. Every non-holdout pair is retained; a larger fitting instance may be necessary.
