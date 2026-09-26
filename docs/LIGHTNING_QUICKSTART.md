# Run this model on Lightning AI

The model pipeline is multilingual MPNet with LoRA fine-tuning, neural + lexical candidate
retrieval, and CPU LightGBM. The final threshold is selected using per-S1 macro F0.5, including
singletons and missed candidates. It writes both required TSV files and validates original test
ID coverage before declaring success.

## Machine sequence

| Phase | Default machine | Work |
| --- | --- | --- |
| prepare | DATA_PREP CPU | Stage data, check dependencies/model, clean, build indexes, split clusters |
| train | One A100 40 GB | Fine-tune LoRA and select checkpoint |
| embed | Two A100 40 GB GPUs in one machine | Embed train/test records in disjoint shards |
| finish | DATA_PREP CPU | FAISS/lexical retrieval, features, LightGBM, validation, prediction, TSV audit |

These machine names are SDK choices, not a statement about your account's capacity. Check the
machine selector in your account. The configured hourly prices are estimates copied from the
original repository; update them to the displayed **total machine** rate before starting.
For a different GPU, pass the exact SDK enum through `--train-machine` and `--embedding-machine`.
Two-GPU embeddings require two visible CUDA GPUs on that machine; fine-tuning uses one GPU.

Use two Studios: a small CPU **controller**, and a dedicated **worker** that the controller
stops/restarts on each machine. Data and checkpoints live on the worker's persistent storage.
Running the controller inside the worker would terminate the controller when hardware changes;
the launcher rejects this. Keep the controller running and do not start a second controller.

## 1. Prepare the worker on CPU

In the worker terminal:

```bash
git clone https://github.com/raoankit72005/ML_Code_3.git
cd ML_Code_3
# If these changes are still in a PR, check out that PR branch on BOTH Studios.
git checkout codex/lightning-autoswitch-ready
bash scripts/setup_lightning.sh
```

Use Python 3.11–3.13. Put `er_sample.zip` or the original dataset ZIP under
`/teamspace/studios/this_studio/`. Alternatively use an absolute persistent directory containing
all seven input TSVs. Do not put data in `/tmp` or upload it to the public GitHub repository.

An S3 input must be one ZIP object (`s3://bucket/path/dataset.zip`). Configure AWS credentials
privately on the **worker**, or use an attached role, then pass that URI as `--input` below.
Credentials from SageMaker do not automatically transfer to Lightning. Extracted S3 prefixes
must first be synced to a persistent directory.

Record the worker name and its full teamspace (`owner/teamspace`), then stop the worker in the UI.

## 2. Configure the controller

In a different CPU Studio:

```bash
git clone https://github.com/raoankit72005/ML_Code_3.git
cd ML_Code_3
git checkout codex/lightning-autoswitch-ready
python -m pip install lightning-sdk==2026.9.18
lightning login
```

Both checkouts must use the same commit and have no tracked source edits. The controller does
not need PyTorch or the dataset locally. Generate the sample-run configuration:

```bash
python scripts/configure_lightning.py \
  --teamspace YOUR_OWNER/YOUR_TEAMSPACE \
  --worker YOUR_WORKER_NAME \
  --input /teamspace/studios/this_studio/er_sample.zip \
  --sample
```

Replace the three values above. Use `--worker-repo` if the worker checkout is elsewhere.
The generator defaults to one A100 for training and two for embedding. It writes private,
git-ignored `configs/local_sample.json` and `configs/local_sample_model.json` and refuses to
overwrite an existing run configuration. `model_config` is resolved relative to the launch JSON.

For the sample, CPU processes default to 2 and LightGBM threads to 4; its production encoder
and real LoRA are unchanged. For the full dataset these default to 8 and 24. Pass `--workers`
and `--threads` to fit your actual CPU allocation. In the generated JSON, review CPU/GPU machine
names, all rates, `max_compute_usd`, `max_hours`, and per-phase `phase_hours` before launching.

## 3. Review, run and monitor

```bash
python lightning_run.py --config configs/local_sample.json --plan
nohup python -u lightning_run.py --config configs/local_sample.json --run > controller.log 2>&1 &
tail -f controller.log
```

`--plan` validates configuration and machine GPU counts without accessing a Studio or renting
compute. It does not prove that your account has capacity. `--run` starts the paid worker.

```bash
python lightning_run.py --config configs/local_sample.json --status
```

Status reads the local state and makes no cloud calls. Detailed worker logs are in
`<work>/logs/{prepare,train,embed,finish}.log`. Encoder metrics are in `<work>/logs/encoder.jsonl`.
CPU and GPU phases run sequentially; two embedding processes run concurrently only in `embed`.

## 4. Resume after interruption

Rerun the exact `--run` command with the same config, work directory and controller state.
The default state is `controller_state/<worker-work-id>.json` on the controller. Keep it.
Completed phases/stages skip; encoder and embedding cursors resume. An incomplete blocking
or feature stage restarts that stage. Do not change model settings, source commit, machine
selection or dataset mid-run.

If a time/compute limit was exhausted, increase `max_compute_usd`, `max_hours`, and/or the
exhausted phase's `phase_hours` in the **same launch JSON**, then resume. These limits are
cumulative across attempts; previous estimated charges and active phase time are retained.
Inactive time between cleanly stopped attempts does not consume the runtime allowance.
A crash with an active worker is conservatively charged until the next confirmed shutdown.
Budget changes do not permit silently changing model or machine settings.

On normal failure, Ctrl-C or SIGTERM the controller attempts to stop the worker. If it cannot
confirm shutdown it retains active state and tells you to inspect the UI. The remote phase
guard has its own timeout. SIGKILL, network/provider failures and storage costs are outside
these best-effort controls. SDK `max_runtime` is not a universal provider billing cap.

## 5. Outputs and the full run

Sample outputs live under `<work>/sample_output/` and are **not leaderboard submissions**.
`run_report.json` includes actual row counts, candidate recall, validation F0.5, threshold,
checkpoint identity, package versions and exact coverage audit. Validation-selected F0.5 is
not an unbiased test score. Inspect candidate recall before spending on a full run: the default
64-candidate cap may omit true matches.

After the sample succeeds, generate a separate full run, without `--sample`:

```bash
python scripts/configure_lightning.py \
  --teamspace YOUR_OWNER/YOUR_TEAMSPACE \
  --worker YOUR_WORKER_NAME \
  --input /teamspace/studios/this_studio/dataset.zip \
  --work /teamspace/studios/this_studio/work_ml3_full
python lightning_run.py --config configs/local_launch.json --plan
nohup python -u lightning_run.py --config configs/local_launch.json --run > controller_full.log 2>&1 &
```

Review the generated limits/rates before the final command. Never reuse sample checkpoints for
full-data training. After `audit` succeeds, the worker contains:

```text
<work>/output/matching_results.tsv
<work>/output/candidate_pairs.tsv
<work>/output/run_report.json
```

Every original **test_source1.tsv** ID has exactly one row, including France and empty matches.
Upload only `matching_results.tsv` to the leaderboard. On the worker, create the final package:

```bash
.venv/bin/python scripts/package_submission.py \
  --work /teamspace/studios/this_studio/work_ml3_full --team YOUR_TEAM
```

The ZIP contains both TSVs, source, configs, pinned dependencies, methodology, environment and
measured run report. Sample runs are refused. Starting a worker manually to download/package
files also incurs compute; stop it when finished.

## Offline verification

```bash
python -m unittest discover -s tests -v
python scripts/sample_smoke.py --input /path/to/er_sample.zip --work work_smoke
```

The smoke command uses a tiny random local Transformer with real LoRA/FAISS/LightGBM on CPU;
it verifies pipeline wiring and file format, **not production model quality or GPU execution**.
The configured sample cloud run above uses the actual multilingual pretrained model.
