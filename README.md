# ML_Code_3 — Lightning CPU/GPU entity-resolution pipeline

Based on `raoankit72005/ML_Code_1` commit `5abd2d1a6361fc4d43a4e0231bb0536daab8d227`.
Keeps **LoRA fine-tuning + multilingual embeddings + FAISS + lexical retrieval + LightGBM**.
New entry point: **pipeline.py**. Automatic machine controller: **lightning_run.py**.
Do not use the legacy `hybrid.py` or Colab launcher for the automatic workflow.

**No error-free, 12-hour or ₹4,000 guarantee.** The supplied summary has 24,229,173 total
train+test source records (not 12M). That means ~37.2 GB of float16 768-D vectors alone,
plus indexes, raw/clean data, pair tables and LightGBM staging. Up to 252,119,360 candidate
pairs at 64 per query. Budget at least substantial persistent disk headroom (several hundred
GB may be needed); monitor actual free space. The sample is inspection-only, not a leaderboard dataset.

## What is automatic

A controller on a **separate CPU Studio** starts/stops a dedicated **worker Studio**:

1. CPU (`DATA_PREP`, screenshot: 32 cores/128 GB, $1.48/hr): input staging, checks,
   parallel file cleaning, train/test indexes and cluster split.
2. One A100 40 GB ($2.19/hr): LoRA fine-tuning, one epoch by default.
3. One A100 by default; optionally two ($4.38/hr TOTAL): embeddings.
4. CPU: multithreaded FAISS, parallel candidate blocking and bulk feature extraction,
   full-data LightGBM, validation, test inference and official output validation.
5. Worker is stopped on completion or error; files stay in its persistent storage.

These phases are sequential; CPU work is parallelized within stages. This release does NOT
claim cross-machine CPU/GPU overlap or distributed encoder training. Hardware switching has
startup overhead. Existing worker cloud/account must support all chosen machines; availability
is checked by the SDK at startup and unsupported capacity fails rather than changing provider/rate silently.

## 1. Set up a dedicated worker Studio (CPU initially)

Create a default Lightning Cloud Studio (not a custom single-provider cluster), note its name and full teamspace (`owner/teamspace`). In its terminal:

```bash
git clone https://github.com/raoankit72005/ML_Code_3.git
cd ML_Code_3
bash scripts/setup_lightning.sh
```

Use Python 3.11–3.13. Setup creates `.venv`, installs a CUDA-enabled PyTorch wheel even while
on CPU, and a standard CPU LightGBM wheel. No `nvcc`/custom compiler is required. GPU phases
still require an appropriate NVIDIA driver; setup cannot repair system drivers.

Put the original challenge ZIP or extracted dataset on this worker's persistent `/teamspace`
storage. For an S3 ZIP, the runner downloads it in the prepare phase using boto3's standard
credential chain. Configure AWS access privately in the **worker** Studio's managed secrets:
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` for temporary credentials;
set `AWS_DEFAULT_REGION` as needed. An attached role is also supported. Grant only required
read access to your dataset. Never put credentials in Git or chat. SageMaker credentials are
not inherited automatically. S3 download/transfer charges are outside the compute estimate.

The S3 input must be a single ZIP object (`s3://bucket/path/dataset.zip`). For extracted S3
prefixes, sync them to persistent storage yourself and set `input` to that local directory.
Do not upload challenge data to this public GitHub repository.

**Stop the worker Studio** in the UI after setup. Keep its name for the controller.

## 2. Set up the controller on another CPU Studio

Use the free/default CPU option if available to your account. It must stay running while
orchestrating. Do NOT run the controller inside the worker it stops/restarts.

```bash
git clone https://github.com/raoankit72005/ML_Code_3.git
cd ML_Code_3
python -m pip install lightning-sdk==2026.9.18
lightning login
cp configs/lightning_launch.json configs/local_launch.json
```

Edit `configs/local_launch.json`:

- `teamspace`: actual owner/teamspace; `worker_studio`: the worker's name.
- `worker_repo`: absolute worker checkout path, usually `/teamspace/studios/this_studio/ML_Code_3`.
- `input`: original S3 ZIP URI or absolute worker data path.
- `work`: a NEW persistent worker directory. Reuse this exact directory to resume.
- Machine identifiers/rates: verify against your live machine selector. Screenshot prices are
  starting values, not live billing quotes. `embedding_rate_usd` is the TOTAL machine rate.
- `max_compute_usd`: default $35 estimated worker compute. Set your own allowance after
  accounting for INR conversion, taxes, setup, storage, transfers and controller charges.
- `max_hours` and `phase_hours`: time limits, NOT runtime promises. Defaults sum to 12h,
  but startup/release overhead reduces executable time. A long phase times out and retains
  checkpoints; it does not silently reduce data, skip fine-tuning or declare success.

For two-GPU embeddings set all three together:

```json
"embedding_machine": "A100_40GB_X_2",
"embedding_gpus": 2,
"embedding_rate_usd": 4.38
```

Set this BEFORE the first run. Changing model configuration after starting requires a new
work directory. One A100 remains the training machine. Two-GPU embedding workers load the
same selected checkpoint and write disjoint record ranges with separate durable cursors.

Controller and worker must have the same Git commit. Pull both before starting; do not
change source/config during a run. Edit `configs/lightning_model.json` before launch if needed:
`workers` defaults to 8, LightGBM to 24 threads, encoder CPU threads to 8. Match these to your
actual machine; more workers consume more memory. Max candidates remains 64, LoRA is required.

## 3. Review and launch once

```bash
python lightning_run.py --config configs/local_launch.json --plan
```

This prints the machine sequence without starting resources. When ready:

```bash
nohup python -u lightning_run.py --config configs/local_launch.json --run > controller.log 2>&1 &
tail -f controller.log
```

Monitor worker logs at `<work>/logs/prepare.log`, `train.log`, `embed.log`, `finish.log`.
Encoder JSONL and TensorBoard are under `<work>/logs/`. Each completed stage is recorded in
`pipeline_manifest.json`. The controller records phase state in local `launch_state.json`.
After a failure, fix the environment/capacity issue and rerun the same command. Completed phases
and stages are skipped; encoder resumes a checkpoint; embedding shards resume durable cursors.
Blocking/features restart their unfinished stage. Never run two controllers on the same worker.

The controller stops compute in `finally`, polls time/estimated charges and the remote guard
kills overlong commands. The guard attempts an independent stop on timeout. These are
**best-effort controls, NOT a provider-enforced billing cap**. API outages, controller termination,
price differences and storage costs can exceed estimates. If stop fails, use Lightning's UI.
If you terminate the controller, verify the worker stopped. Do not delete state to evade accounting.

## 4. Outputs

After official validation passes, full-data files are in the worker:

```
<work>/output/matching_results.tsv
<work>/output/candidate_pairs.tsv
```

Upload **matching_results.tsv** to the leaderboard. Both files retain every test S1, including
France and singletons; candidate lists are exactly those scored. The included
`utils/validate_submission.py` is the validator from your uploaded challenge resources, unchanged.
Restart the worker on the lowest-cost CPU if needed to download files, then stop it again.

Create the final code+outputs+methodology ZIP from the worker checkout:

```bash
.venv/bin/python scripts/package_submission.py --work /teamspace/studios/this_studio/work_ml3 --team YOUR_TEAM
```

The package contains the run's model config and measured metadata. It refuses sample runs.
No pretrained weights or original challenge data are committed; pinned encoder weights download
from Hugging Face. Model license: Apache-2.0; LightGBM/code: MIT. No external business lookup.

## Local execution / reproduction on a fixed machine

```bash
.venv/bin/python pipeline.py --input /absolute/path/dataset.zip --work work_local
```

This uses the same stages but **does not switch rented hardware**. To reproduce a submitted
package, add `--config configs/reproduction_model.json`. `--phase prepare|train|embed|finish`
can be run manually in order on appropriate machines; keep work paths stable. Never change
configuration between phases. Persistent storage is mandatory.

## Tests and limitations

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/sample_smoke.py --input /path/er_sample.zip --work work_smoke
```

The offline smoke test uses a **tiny random Transformer with real LoRA training**, then FAISS,
lexical retrieval, parallel features, real LightGBM and official validation on your sample.
It tests execution/format, not the production multilingual encoder's quality. Outputs go to
`sample_output/` with a warning marker and cannot be packaged. The production model can also
be exercised on a sample by adding `--sample` to `pipeline.py` or `lightning_run.py` with a
separate work/state configuration; that incurs real GPU costs.

See docs/VERIFICATION.md for executed checks and untested cloud/GPU paths. Full-data accuracy,
12-hour completion and budget compliance are not established by passing sample tests.
