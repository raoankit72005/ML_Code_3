> Legacy lexical-only workflow. For the hybrid encoder/FAISS workflow use the root README and `hybrid.py`.

# Full-data CUDA run on SageMaker

## Environment

Work in the **terminal of a Linux NVIDIA GPU SageMaker app**, using a persistent work volume. This is a terminal workflow, not a managed SageMaker Training Job entry point. The app needs working NVIDIA drivers, CUDA toolkit including `nvcc`, and a CUDA-compatible C++ compiler. Installing a CUDA runtime Python wheel alone does not provide the compiler.

In an activated Python 3.11+ environment:

```bash
cd /home/ec2-user/ML-code
git pull --ff-only
nvidia-smi
nvcc --version
g++ --version
bash scripts/setup_gpu.sh
```

The helper installs pinned Python dependencies, compiles LightGBM 4.6.0 with `USE_CUDA=ON` using two build jobs, and executes a real two-tree CUDA training test. It stops if a prerequisite or the backend test fails. It does not alter drivers or silently train on CPU. Your image's CUDA toolkit, compiler and driver must be compatible; this has to be checked on your actual app.

Official references: [LightGBM 4.6 CUDA installation](https://lightgbm.readthedocs.io/en/v4.6.0/Installation-Guide.html#build-cuda-version), [Sequence interface](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.Sequence.html).

## Run

Place the ORIGINAL `ML_Dataset.zip` in the repository directory, or pass its absolute path. The data does not belong in GitHub.

```bash
python run.py full --input ML_Dataset.zip --work work_full_gpu --skip-test
```

This cleans all supplied original sources, indexes every training reference, generates and labels candidates for every training S1, builds features, retains all training/validation pairs, checks resource estimates, fits CUDA LightGBM, scores all validation candidates, and selects the F0.5 threshold. Validation remains held out of fitting.

After examining candidate recall and country scores, continue with the same configuration:

```bash
python run.py full --input ML_Dataset.zip --work work_full_gpu
```

Completed training is reused. Full test preparation, prediction, TSV creation and original-test coverage validation follow. This includes France even though training countries are India and US.

Run the official validator from the original dataset, using its actual extracted paths:

```bash
python utils/validate_submission.py \
  --matching work_full_gpu/test/matching_results.tsv \
  --candidate work_full_gpu/test/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids
```

The `utils` and `dataset/test` paths above belong to the challenge distribution, not this Git repository. Adjust them to where you extracted the original files.

## Choose capacity from measurements

Run `python run.py doctor` before preparation. It reports currently available RAM and GPU memory and tests the requested backend. Once feature tables exist, use:

```bash
python run.py estimate --work work_full_gpu
```

Training repeats the sizing check automatically. Its JSON report includes actual candidate-pair counts, feature count, estimated resident RAM and VRAM, and float32 staging disk bytes. The model directory gets the same report during fitting.

For perspective, just staging 100 million pairs with 47 float32 features plus a float32 label uses 19.2 GB of disk (decimal), before Parquet, indexes, validation, or outputs. Neither the ZIP size nor the number of sampled training pairs is a useful estimate of the complete work-volume size. A larger India candidate budget can substantially increase this.

Defaults permit estimates up to 65% of currently available RAM and 75% of free VRAM. Estimates include heuristic headroom. They do not control the native allocator or guarantee absence of OOM under all data distributions/concurrent workloads. Keep this app dedicated to the job. If sizing fails, choose more RAM/VRAM rather than raising the budget fractions to bypass the check. Multiple small GPUs do not combine into one larger GPU in this implementation.

The loader reduces its own batch size according to available RAM. All non-holdout pairs are still used by the same model. It does not emulate incremental `partial_fit` by successively training on chunks; that would change the model objective and early-stopping behavior.

## Recovery and configuration changes

- Rerunning the same `full` command skips completed stages with matching input/config/code fingerprints.
- An interrupted preparation stage restarts that stage from its beginning. It is not a per-query checkpointing implementation.
- Completed float32 loading caches are reused by an interrupted fitting run. Boosting itself restarts; partial models are never marked complete.
- If cleaning was interrupted before its report/manifest was completed, keep the original input and choose a new work folder (or remove only the incomplete generated cleaning output after inspecting it). The cleaner does not resume partway through a source file.
- Changed code, input, or preparation settings require a new work folder. New training settings can use a new model directory with the modular `train` command.
- Do not delete generated files during a run. No automatic cleanup deletes source data or completed work. The test feature file uses a hard link where supported to avoid a duplicate full file.
- Download or back up completed models, reports and TSV outputs. Stop SageMaker compute after your job and backups finish; closing a browser tab does not stop compute.

## Common failures

| Message | Action |
| --- | --- |
| `Missing nvcc` | Use a CUDA development environment/toolkit; a GPU driver alone is insufficient |
| `CUDA Tree Learner was not enabled` | Run the source-build helper in the active Python environment |
| CUDA build/compiler error | Check toolkit/host compiler compatibility for the chosen image; do not use a CPU wheel as a purported fix |
| `GPU estimate ... exceeds permitted ...` | Use a GPU with enough free VRAM; smaller Python batches do not shrink LightGBM's resident data |
| `RAM estimate ... exceeds permitted ...` | Use more host RAM and stop competing processes |
| `Insufficient disk` | Expand the persistent work volume; changing GPU type does not necessarily expand disk |
| `Configuration, input or code changed` | Use a fresh work folder |
| Missing S1 IDs in challenge upload | Run original-test coverage audit; ensure files come from full test inference, never sample output |

## Evaluate improvements

Inspect India/US candidate recall, the oracle F0.5 ceiling, actual macro F0.5, precision/recall trade-offs in the threshold grid, and preparation stage timings. Full reference pools can lower scores compared with sampled pools because they contain more plausible distractors. Do not treat a sampled score as a guaranteed leaderboard result. Threshold tuning and early stopping share this holdout; its result is a tuned validation score, not an unbiased final-test estimate.
